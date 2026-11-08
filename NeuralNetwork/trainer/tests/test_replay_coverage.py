"""Real label-format replay checks on a tiny synthetic, frozen run tree.

Returns are synthetic; public states, legal actions, features, NPZ validation,
manifest traversal and replay provenance are the actual trainer code paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stella.catalogue_engine import CatalogueEngine, PROTOCOL
from stella.environment import Episode, heuristic_action, score
from stella.features import encode_actions
from stella.io_utils import read_json
from stella.preset_catalogue import catalogue_presets
from stella.presets import PRESET_VERSION
from stella.sampling import build_sampled_family, load_label, save_label


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def files(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def public_label(preset, task, *, teacher_sha="teacher-sha-1", provenance=False):
    episode = Episode(preset, task["environment_seed"])
    while not episode.done:
        if len(episode.actions) > 1 and preset.total-score(episode.state) >= 2:
            break
        episode.step(heuristic_action(episode.state, episode.actions))
    assert not episode.done
    local = dict(task)
    if provenance:
        local.update(coverage_protocol="catalogue4960_independent_trajectories_v1",
                     root_selection_schedule=["natural"])
    family = build_sampled_family(local, preset, [(episode.steps, episode.state)],
                                  episode.steps, episode.score)
    metadata = family["roots"][0]
    metadata.update(catalogue_id=task["catalogue_id"], preset_version=PRESET_VERSION,
                    schema="complete_root_q_v1", teacher_sha256=teacher_sha,
                    root_score=score(episode.state), branch_decisions=0)
    returns = np.tile(np.array([[0., 2.]], dtype=np.float32), (len(episode.actions), 1))
    metadata["complete_branches"] = int(returns.size)
    return dict(x=encode_actions(episode.state, episode.actions), q=returns.mean(1),
                returns=returns, standard_error=returns.std(1, ddof=1)/np.sqrt(2), metadata=metadata)


@pytest.fixture
def replay_case(tmp_path):
    run = tmp_path / "run"
    presets = catalogue_presets()
    current_tasks, historical_tasks = [], []
    saved = {}

    def add(number, index, split, teacher="M000001"):
        task = dict(family_id=f"r{number:06d}_c{index:04d}_{split}", catalogue_id=index,
                    teacher_id=teacher, split=split, preset=presets[index].to_dict(),
                    roots_per_family=1, rollouts_per_action=2, epsilon=0.,
                    environment_seed=number*10000+index*10+(split=="val"),
                    exploration_seed=number*100000+index*10+(split=="val"),
                    label_seed=number*1000000+index*10+(split=="val"))
        row = public_label(presets[index], task, provenance=(number == 73))
        path = run / f"rounds/round_{number:06d}/labels" / (row["metadata"]["root_id"] + ".npz")
        save_label(path, row)
        saved[task["family_id"]] = (task, path, row)
        (current_tasks if number == 73 else historical_tasks).append(task)

    for index in (0, 1):
        add(73, index, "train")
        add(73, index, "val")
    add(72, 0, "train")
    add(72, 1, "val")
    # This historic label is valid, but its per-preset teacher has since changed.
    add(72, 2, "train", teacher="M000000")
    current = dict(protocol=PROTOCOL, catalogue_coverage={"protocol": "independent_trajectories"},
                   teacher_bank=["M000001"]*4960,
                   teacher_models={"M000001": {"sha256": "teacher-sha-1"}}, sampling_tasks=current_tasks)
    write_json(run / "rounds/round_000073/manifest.json", current)
    write_json(run / "rounds/round_000072/manifest.json",
               dict(protocol=PROTOCOL, sampling_tasks=historical_tasks))
    engine = SimpleNamespace(run_dir=run,
        config={"sampling": {"replay_rounds": 2, "replay_max_roots": 100}}, control=lambda: None)
    return engine, saved


def test_coverage_replay_retains_current_val_and_matching_historical_train_only(replay_case):
    engine, saved = replay_case
    before = files(engine.run_dir)
    roots = CatalogueEngine.replay(engine, 73, teacher_id="unused-global-teacher")
    current_ids = {family for family in saved if family.startswith("r000073_")}
    assert {row["family_id"] for row in roots} == current_ids | {"r000072_c0000_train"}
    assert len(roots) == 5
    validation = [row for row in roots if row["split"] == "val"]
    assert len(validation) == 2
    assert all(row["family_id"].startswith("r000073_") for row in validation)
    assert not any(row["catalogue_id"] == 2 for row in roots)
    for row in roots:
        task, _, archived = saved[row["family_id"]]
        assert row["split"] == task["split"] == archived["metadata"]["split"]
        assert row["source_environment_seed"] == task["environment_seed"]
        assert row["source_exploration_seed"] == task["exploration_seed"]
        assert row["label_seed"] == archived["metadata"]["label_seed"]
        assert row["teacher_sha256"] == "teacher-sha-1"
        assert row["preset_group"] == f"catalogue_{task['catalogue_id']:04d}"
        assert row["source_round"] == (73 if row["family_id"] in current_ids else 72)
    # The historical val archive is still a val archive; it was not rewritten
    # or relabelled as train to obtain full current-preset gradient coverage.
    old_val = saved["r000072_c0001_val"][1]
    assert load_label(old_val)["split"] == "val"
    assert files(engine.run_dir) == before


@pytest.mark.parametrize("key,value", [("family_id", "another-family"),
                                      ("split", "val"),
                                      ("source_environment_seed", 9999999)])
def test_replay_refuses_labels_from_another_source_task(replay_case, key, value):
    engine, saved = replay_case
    _, path, row = saved["r000073_c0000_train"]
    row["metadata"][key] = value
    save_label(path, row)
    before = files(engine.run_dir)
    with pytest.raises(ValueError, match="(identity/split|provenance).*frozen"):
        CatalogueEngine.replay(engine, 73, teacher_id="unused-global-teacher")
    assert files(engine.run_dir) == before


@pytest.mark.parametrize("family", ["r000072_c0000_train", "r000073_c0000_train", "r000073_c0001_val"])
@pytest.mark.parametrize("checksum", ["stale-file-sha", None])
def test_coverage_replay_checks_current_and_historical_teacher_file_sha(replay_case, family, checksum):
    engine, saved = replay_case
    _, path, row = saved[family]
    if checksum is None:
        del row["metadata"]["teacher_sha256"]
    else:
        row["metadata"]["teacher_sha256"] = checksum
    save_label(path, row)
    before = files(engine.run_dir)
    with pytest.raises(ValueError, match="teacher checksum differs"):
        CatalogueEngine.replay(engine, 73, teacher_id="unused-global-teacher")
    assert files(engine.run_dir) == before


def test_old_val_is_excluded_before_loading_or_matching_sha(replay_case):
    engine, saved = replay_case
    _, path, row = saved["r000072_c0001_val"]
    row["metadata"]["teacher_sha256"] = "obsolete-but-preserved"
    save_label(path, row)
    before = files(engine.run_dir)
    roots = CatalogueEngine.replay(engine, 73, teacher_id="unused-global-teacher")
    assert len(roots) == 5
    assert not any(root["family_id"] == "r000072_c0001_val" for root in roots)
    assert files(engine.run_dir) == before


def test_legacy_replay_keeps_its_original_validation_policy(replay_case):
    engine, saved = replay_case
    path = engine.run_dir / "rounds/round_000073/manifest.json"
    current = read_json(path)
    del current["catalogue_coverage"]
    write_json(path, current)
    before = files(engine.run_dir)
    roots = CatalogueEngine.replay(engine, 73, teacher_id="unused-global-teacher")
    assert len(roots) == 6
    old_val = next(root for root in roots if root["family_id"] == "r000072_c0001_val")
    assert old_val["split"] == "val"
    assert files(engine.run_dir) == before


def test_actual_coverage_manifest_passes_initial_student_identity_into_fit(tmp_path, monkeypatch):
    """Exercise production manifest construction and the inherited fit caller.

    The optimizer is replaced by an argument capture; there is no training or
    4960-preset rollout. All tasks and teacher checksums are prepared normally.
    """
    import stella.engine as engine_module
    from stella.config import resolve
    from stella.network import QNetwork

    run = tmp_path / "run"
    checkpoint = run / "checkpoints/M000071.npz"
    QNetwork(input_dim=958, hidden=(8,), heuristic_init=False).save(checkpoint)
    engine = object.__new__(CatalogueEngine)
    engine.run_dir = run
    engine.presets = catalogue_presets()
    engine.config = resolve({"catalogue_mode": True, "workers": 1,
        "compute": {"backend": "numpy"},
        "sampling": {"families": 4960, "max_families": 4960, "replay_max_roots": 39680,
                     "hard_sampling": True, "hard_rollouts_per_action": 4,
                     "hard_validation_rollouts_per_action": 4},
        "training": {"objective_version": "action_gap_calibration_v3"},
        "catalogue_coverage": {"train_rollouts_per_action": 2, "val_rollouts_per_action": 2,
                               "train_hard_rollouts_per_action": 4,
                               "val_hard_rollouts_per_action": 4}})
    engine.state = {"teacher_id": "M000071", "teacher_checkpoint": "checkpoints/M000071.npz",
                    "teacher_bank": ["M000071"]*4960,
                    "models": {"M000071": "checkpoints/M000071.npz"},
                    "teacher_scores": "teacher_scores/round_000072.npz"}
    directory = run / "rounds/round_000073"
    manifest = engine.prepare_manifest(73, directory)
    assert manifest["training_config"]["initial_student_model_id"] == "M000071"
    assert manifest["training_config"]["training_round"] == 73
    assert manifest["training_config"]["validation_unit"] == "trajectory"
    assert len(manifest["sampling_tasks"]) == 9920
    assert engine.prepare_manifest(73, directory) == manifest  # Frozen resume preserves the metadata.
    captured = {}
    engine.replay = lambda number, teacher_id: []
    engine.stage = lambda *args, **kwargs: None

    class FitCaptured(Exception):
        pass

    def capture(teacher, roots, settings, **kwargs):
        captured.update(settings)
        assert all(not array.flags.writeable for array in teacher.params.values())
        raise FitCaptured

    monkeypatch.setattr(engine_module, "fit_candidate", capture)
    with pytest.raises(FitCaptured):
        engine.fit_checkpoints(manifest, directory)
    assert captured["initial_student_model_id"] == "M000071"
    assert captured["training_round"] == 73
    assert captured["validation_unit"] == "trajectory"
    assert captured["objective_version"] == "action_gap_calibration_v3"
