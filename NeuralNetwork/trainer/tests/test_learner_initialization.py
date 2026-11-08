"""Continuation identity, real fit I/O, teacher invariants and resume safety."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from stella import engine as engine_module
from stella.catalogue_engine import CatalogueEngine
from stella.config import resolve
from stella.engine import Engine
from stella.fit_cache import identity
from stella.io_utils import atomic_json, file_hash
from stella.learner_initialization import freeze_initialization, fit_initial_model
from stella.network import QNetwork
from stella.preset_catalogue import catalogue_presets
from stella.retention import apply_retention, retention_plan


def save_model(run, model_id, value):
    model = QNetwork(input_dim=3, hidden=(2,), seed=1, heuristic_init=False)
    model.params["skip_W"][0] = value
    path = run / "checkpoints" / (model_id + ".npz")
    model.save(path, {"model_id": model_id})
    return path


def committed_state(run):
    save_model(run, "M000071", 0.0)
    save_model(run, "M000072", 0.125)
    return {"teacher_id": "M000071", "teacher_checkpoint": "checkpoints/M000071.npz",
            "last_candidate_checkpoint": "checkpoints\\M000072.npz",
            "models": {"M000071": "checkpoints/M000071.npz", "M000072": "checkpoints/M000072.npz"},
            "teacher_bank": ["M000071"] * 4960, "teacher_scores": "teacher_scores/round_000072.npz",
            "next_round": 73, "best_candidate_mean": 75.0, "composite_teacher_mean": 75.0}


def frozen_manifest(run, state, number=73, mode="last_candidate"):
    training = {"learner_initialization": mode, "epochs": 1, "checkpoint_metric": "label_regret",
                "objective_version": "action_gap_calibration_v3"}
    teacher_path = state["teacher_checkpoint"]
    return {"round": number, "teacher_id": state["teacher_id"], "teacher_checkpoint": teacher_path,
            "teacher_sha256": file_hash(run / Path(teacher_path.replace("\\", "/"))),
            "teacher_bank": list(state["teacher_bank"]),
            "teacher_models": {state["teacher_id"]: {"checkpoint": teacher_path,
                "sha256": file_hash(run / Path(teacher_path.replace("\\", "/")))}},
            "training_config": training, "compute_config": {},
            "learner_initialization": freeze_initialization(training, state, run, number)}


def test_explicit_continuation_uses_unpromoted_last_candidate_and_portable_paths(tmp_path):
    state = committed_state(tmp_path)
    previous = deepcopy(state)
    manifest = frozen_manifest(tmp_path, state)
    initial = fit_initial_model(manifest, tmp_path)
    assert initial["model_id"] == "M000072"
    assert initial["source"] == "last_candidate" and initial["source_round"] == 72
    assert initial["sha256"] == file_hash(tmp_path / "checkpoints/M000072.npz")
    assert state == previous
    assert manifest["teacher_id"] == "M000071"
    assert set(manifest["teacher_bank"]) == {"M000071"}


def test_best_and_old_manifest_keep_historical_initialization(tmp_path):
    state = committed_state(tmp_path)
    manifest = frozen_manifest(tmp_path, state, mode="best")
    assert fit_initial_model(manifest, tmp_path)["model_id"] == "M000071"
    assert freeze_initialization({}, state, tmp_path, 73) is None
    del manifest["learner_initialization"]
    del manifest["training_config"]["learner_initialization"]
    assert fit_initial_model(manifest, tmp_path)["checkpoint"] == state["teacher_checkpoint"]


def test_first_round_bootstrap_is_explicit_and_later_missing_state_fails(tmp_path):
    save_model(tmp_path, "M000000", 0.0)
    state = {"teacher_id": "M000000", "teacher_checkpoint": "checkpoints/M000000.npz",
             "models": {"M000000": "checkpoints/M000000.npz"}, "teacher_bank": ["M000000"]}
    manifest = frozen_manifest(tmp_path, state, number=1)
    assert fit_initial_model(manifest, tmp_path)["source"] == "best_bootstrap"
    with pytest.raises(ValueError, match="missing after the first"):
        freeze_initialization(manifest["training_config"], state, tmp_path, 2)
    del manifest["learner_initialization"]
    with pytest.raises(ValueError, match="requires a frozen"):
        fit_initial_model(manifest, tmp_path)


def test_changed_missing_or_unregistered_last_candidate_never_silently_rewinds(tmp_path):
    state = committed_state(tmp_path)
    manifest = frozen_manifest(tmp_path, state)
    changed = deepcopy(state)
    changed["models"]["M000072"] = "checkpoints/M000071.npz"
    with pytest.raises(ValueError, match="model registry"):
        freeze_initialization(manifest["training_config"], changed, tmp_path, 73)
    path = tmp_path / "checkpoints/M000072.npz"
    path.write_bytes(b"changed learner")
    with pytest.raises(ValueError, match="checksum changed"):
        fit_initial_model(manifest, tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        freeze_initialization(manifest["training_config"], state, tmp_path, 73)
    with pytest.raises(ValueError, match="checksum changed"):
        fit_initial_model(manifest, tmp_path)


def test_initializer_identity_is_part_of_the_frozen_fit_cache(tmp_path):
    state = committed_state(tmp_path)
    manifest = frozen_manifest(tmp_path, state)
    previous = identity(manifest, 1)
    changed = deepcopy(manifest)
    changed["learner_initialization"]["sha256"] = "0" * 64
    assert identity(changed, 1) != previous
    changed = deepcopy(manifest)
    changed["learner_initialization"]["source_round"] = 70
    with pytest.raises(ValueError, match="preceding committed"):
        fit_initial_model(changed, tmp_path)


def test_manifest_changes_only_learning_start_and_keeps_all_4960_routes(tmp_path):
    state = committed_state(tmp_path)
    engine = CatalogueEngine.__new__(CatalogueEngine)
    engine.run_dir = tmp_path
    engine.presets = catalogue_presets()
    engine.state = state
    raw = {"catalogue_mode": True, "workers": 1,
           "sampling": {"families": 4960, "max_families": 4960, "replay_max_roots": 39680},
           "training": {"objective_version": "action_gap_calibration_v3"},
           "catalogue_coverage": {}}
    engine.config = resolve(raw)
    baseline = engine.prepare_manifest(73, tmp_path / "old_round")
    raw["training"]["learner_initialization"] = "last_candidate"
    engine.config = resolve(raw)
    continued = engine.prepare_manifest(73, tmp_path / "new_round")
    for key in ("teacher_id", "teacher_checkpoint", "teacher_sha256", "teacher_bank", "teacher_models",
                "behavior_checkpoint", "behavior_sha256", "sampling_tasks", "catalogue_coverage",
                "evaluation_mode", "evaluation_candidates", "evaluation_games_per_preset"):
        assert continued[key] == baseline[key]
    assert len(continued["sampling_tasks"]) == 2 * 4960
    assert continued["training_config"]["initial_student_model_id"] == "M000072"
    assert engine.prepare_manifest(73, tmp_path / "new_round") == continued
    save_model(tmp_path, "M000072", 0.25)
    with pytest.raises(ValueError, match="checksum changed"):
        engine.prepare_manifest(73, tmp_path / "new_round")


def test_real_fit_io_uses_continuation_once_and_preserves_teacher_metadata(tmp_path, monkeypatch):
    state = committed_state(tmp_path)
    manifest = frozen_manifest(tmp_path, state)
    engine = Engine.__new__(Engine)
    engine.run_dir = tmp_path
    engine.config = {"seed": 123, "training": {}, "compute": {}}
    engine.stage = lambda *args, **kwargs: None
    engine.log = lambda message: None
    roots = [{"root_id": "frozen-root", "teacher_id": "M000071", "q": [1, 2], "x": [[0, 0, 0]]}]
    replay_arguments = []
    engine.replay = lambda *args: (replay_arguments.append(args) or roots)
    calls = []

    def fit(initial_student, supplied_roots, settings, **kwargs):
        calls.append(initial_student.checksum())
        assert supplied_roots is roots
        assert initial_student.checksum() == QNetwork.load(tmp_path / "checkpoints/M000072.npz").checksum()
        assert initial_student.frozen
        assert settings["initial_student_model_id"] == "M000072"
        assert settings["training_round"] == 73
        points = []
        for epoch in (1, 2, 3, 4, 5, 6):
            model = initial_student.clone()
            model.params["skip_W"][0] += epoch / 1000
            points.append({"epoch": epoch, "model": model, "optimizer_steps": epoch})
        return SimpleNamespace(interrupted=False, candidate_checkpoints=points,
                               summary=lambda: {"best_epoch": 3, "initial_parameter_sha256": calls[-1]})

    monkeypatch.setattr(engine_module, "fit_candidate", fit)
    directory = tmp_path / "rounds/round_000073"
    directory.mkdir(parents=True)
    teacher_hash = file_hash(tmp_path / "checkpoints/M000071.npz")
    pool = engine.fit_checkpoints(manifest, directory)
    assert len(pool["candidates"]) == 6
    assert len(list((directory / "fit_checkpoints").glob("*.npz"))) == 6
    assert replay_arguments == [(73, "M000071")]
    assert file_hash(tmp_path / "checkpoints/M000071.npz") == teacher_hash
    assert pool["fit"]["learner_initialization"] == manifest["learner_initialization"]
    metadata = QNetwork.load(directory / pool["candidates"][0]["checkpoint"]).metadata
    assert metadata["teacher_id"] == "M000071"
    assert metadata["initial_student_model_id"] == "M000072"
    assert "each preset's frozen teacher" in metadata["label_semantics"]
    assert engine.fit_checkpoints(manifest, directory) == pool
    assert len(calls) == 1 and len(replay_arguments) == 1
    save_model(tmp_path, "M000072", 0.5)
    with pytest.raises(ValueError, match="checksum changed"):
        engine.fit_checkpoints(manifest, directory)
    assert len(calls) == 1


def test_commit_preserves_unpromoted_learning_start_without_promoting_best(tmp_path):
    state = committed_state(tmp_path)
    save_model(tmp_path, "M000073", 0.25)
    engine = CatalogueEngine.__new__(CatalogueEngine)
    engine.run_dir = tmp_path
    engine.state_path = tmp_path / "run_state.json"
    engine.state = state
    engine.config = {"retention": {"history_rows": 100}}
    engine.elapsed = lambda: 1.0
    engine.publish = lambda *args, **kwargs: None
    engine.log = lambda *args, **kwargs: None
    engine.prune = lambda: None
    summary = {"round": 73, "candidate_id": "M000073", "candidate_checkpoint": "checkpoints/M000073.npz",
               "best_model_id": "M000071", "candidate_mean": 74.999, "best_candidate_mean": 75.0,
               "teacher_bank": list(state["teacher_bank"]), "teacher_scores": "teacher_scores/round_000073.npz",
               "composite_teacher_mean": 75.0, "champion_updated_count": 0, "distinct_teacher_count": 1,
               "primary_delta": -0.001}
    engine.commit(summary)
    assert engine.state["teacher_id"] == "M000071"
    assert engine.state["teacher_bank"] == summary["teacher_bank"]
    assert engine.state["last_candidate_checkpoint"] == "checkpoints/M000073.npz"
    initial = freeze_initialization({"learner_initialization": "last_candidate"}, engine.state, tmp_path, 74)
    assert initial["model_id"] == "M000073" and initial["source_round"] == 73


def completed_round_engine(run):
    state = committed_state(run)
    save_model(run, "M000073", 0.25)
    manifest = frozen_manifest(run, state)
    engine = CatalogueEngine.__new__(CatalogueEngine)
    engine.run_dir = run
    engine.state_path = run / "run_state.json"
    engine.state = state
    engine.config = {"retention": {"history_rows": 100}}
    engine.elapsed = lambda: 1.0
    engine.publish = lambda *args, **kwargs: None
    engine.log = lambda *args, **kwargs: None
    engine.apply_next_round_config = lambda *args, **kwargs: None
    engine.prune = lambda: apply_retention(retention_plan(run, engine.state, keep_rounds=2))
    summary = {"round": 73, "candidate_id": "M000073", "candidate_checkpoint": "checkpoints/M000073.npz",
               "best_model_id": "M000071", "candidate_mean": 74.999, "best_candidate_mean": 75.0,
               "teacher_bank": list(state["teacher_bank"]), "teacher_scores": "teacher_scores/round_000073.npz",
               "composite_teacher_mean": 75.0, "champion_updated_count": 0, "distinct_teacher_count": 1,
               "primary_delta": -0.001, "learner_initialization": manifest["learner_initialization"]}
    directory = run / "rounds/round_000073"
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(directory / "summary.json", summary)

    def never_train(*args, **kwargs):
        raise AssertionError("Completed summary resume must not fit or evaluate")

    engine.train = engine.collect = engine.evaluate_matrix = never_train
    return engine, directory


def test_completed_summary_reentry_validates_source_without_refit_and_retains_it(tmp_path):
    engine, directory = completed_round_engine(tmp_path)
    engine.run_round(73)  # Atomic summary existed; the commit had not completed.
    assert engine.state["next_round"] == 74
    assert engine.state["teacher_id"] == "M000071"
    assert (tmp_path / "checkpoints/M000072.npz").is_file()
    assert "M000072" not in engine.state["models"]  # Kept by manifest, not current state.
    before = deepcopy(engine.state)
    engine.run_round(73)  # Re-entry after commit/pruning remains idempotent.
    assert engine.state == before
    assert (directory / "manifest.json").is_file()


@pytest.mark.parametrize("failure", ["missing_source", "changed_source", "missing_manifest", "different_record"])
def test_completed_summary_rejects_invalid_continuation_before_committing(tmp_path, failure):
    engine, directory = completed_round_engine(tmp_path)
    previous = deepcopy(engine.state)
    if failure == "missing_source":
        (tmp_path / "checkpoints/M000072.npz").unlink()
    elif failure == "changed_source":
        save_model(tmp_path, "M000072", 0.5)
    elif failure == "missing_manifest":
        (directory / "manifest.json").unlink()
    else:
        from stella.io_utils import read_json
        summary = read_json(directory / "summary.json")
        summary["learner_initialization"]["sha256"] = "0" * 64
        atomic_json(directory / "summary.json", summary)
    with pytest.raises(ValueError, match="frozen|checksum"):
        engine.run_round(73)
    assert engine.state == previous


def test_historical_completed_summary_does_not_require_continuation_manifest(tmp_path):
    engine, directory = completed_round_engine(tmp_path)
    from stella.io_utils import read_json
    summary = read_json(directory / "summary.json")
    del summary["learner_initialization"]
    atomic_json(directory / "summary.json", summary)
    (directory / "manifest.json").unlink()
    engine.run_round(73)
    assert engine.state["next_round"] == 74
