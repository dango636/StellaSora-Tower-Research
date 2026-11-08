"""Revision-boundary tests use synthetic run state and never load a checkpoint.

The legacy config digest was captured from the unmodified R48 resolver.  It is a
portable regression fixture: workers is explicit, and checkpoint paths are
relative strings, so no machine-dependent defaults or user files are required.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from stella.config import resolve
from stella.io_utils import digest, read_json
from tools import prepare_calibration_revision as revision_tool


LEGACY_RAW = {
    "catalogue_mode": True,
    "seed": 7319,
    "workers": 1,
    "compute": {"backend": "numpy"},
    "sampling": {
        "families": 4960,
        "max_families": 4960,
        "roots_per_family": 4,
        "rollouts_per_action": 256,
        "max_rollouts_per_action": 256,
        "replay_rounds": 2,
        "replay_max_roots": 39680,
        "hard_sampling": True,
        "balanced_phase_sampling": True,
        "validation_rollouts_multiplier": 2,
        "hard_rollouts_per_action": 512,
        "hard_validation_rollouts_per_action": 1024,
        "refinement_roots": 384,
        "refinement_rollouts_per_action": 1024,
        "refinement_validation_rollouts_per_action": 2048,
    },
    "training": {
        "learning_rate": 0.0003,
        "checkpoint_candidates": 6,
        "ranking_weight": 0.5,
        "advantage_confidence_weighting": True,
    },
    "evaluation": {
        "games_per_preset": 128,
        "max_games_per_preset": 128,
        "development_games_per_preset": 128,
    },
    "retention": {"rounds": 3, "checkpoints": 8},
    "model": {
        "hidden": [48, 32],
        "initialization": "checkpoint",
        "initial_checkpoint": "checkpoints/M000048.npz",
    },
}

# Captured from the original R48 config.py, using io_utils.digest encoding.
LEGACY_RESOLVED_SHA256 = "74790aeea5f582b90c3c1c602f5a67daa1d31298c19a0e18f827cd9f5c297439"

OVERLAY = {
    "catalogue_coverage": {"train_roots": 3, "val_roots": 1},
    "training": {
        "objective_version": "action_gap_calibration_v3",
        "validation_unit": "trajectory",
        "refresh_student_actions": True,
        "checkpoint_metric": "label_regret",
    },
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def file_snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def fake_run(tmp_path):
    run = tmp_path / "run"
    config = resolve(LEGACY_RAW)
    bank = [{"catalogue_id": index, "teacher_id": "M000048",
             "teacher_sha256": "frozen-teacher-hash"} for index in range(4960)]
    state = {
        "protocol": "catalogue4960_per_preset_champions_v1",
        "next_round": 73,
        "config_hash": digest(config),
        "teacher_bank": bank,
        "best_candidate": "M000072",
        "global_champion": "M000068",
        "models": {"M000072": {"path": "checkpoints/M000072.npz", "sha256": "frozen-model"}},
        "official_score": 101.25,
    }
    write_json(run / "config.json", LEGACY_RAW)
    write_json(run / "resolved_config.json", config)
    write_json(run / "run_state.json", state)
    write_json(run / "rounds/round_000072/manifest.json", {
        "round": 72, "config": config, "config_hash": digest(config),
        "teacher_bank": bank,
    })
    write_json(run / "rounds/round_000072/catalogue_scores.json", {
        "formal_evaluation": "fixed original protocol", "score": 101.25,
    })
    model = run / "checkpoints/M000072.npz"
    model.parent.mkdir()
    model.write_bytes(b"synthetic-checkpoint-do-not-open-or-rewrite")
    return run


def invoke_main(monkeypatch, run, profile, *, write_pending=False, revision="test_revision",
                check_code_upgrade=False):
    argv = ["prepare_calibration_revision.py", "--run-dir", str(run),
            "--profile", str(profile), "--revision", revision]
    if write_pending:
        argv.append("--write-pending")
    if check_code_upgrade:
        argv.append("--check-code-upgrade")
    monkeypatch.setattr(sys, "argv", argv)
    revision_tool.main()


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "overlay.json"
    write_json(path, OVERLAY)
    return path


def test_legacy_resolve_is_byte_identity_compatible():
    raw = copy.deepcopy(LEGACY_RAW)
    config = resolve(raw)
    assert digest(config) == LEGACY_RESOLVED_SHA256
    assert raw == LEGACY_RAW
    assert "catalogue_coverage" not in config
    assert "validation_unit" not in config["training"]
    assert "objective_version" not in config["training"]
    assert "calibration" not in config["training"]


def test_preview_changes_only_allowed_config_and_writes_nothing(fake_run):
    before = file_snapshot(fake_run)
    overlay = copy.deepcopy(OVERLAY)
    result = revision_tool.prepare_revision(fake_run, overlay, revision="test_revision")
    assert result["effective_round"] == 73
    assert result["previous_config_hash"] == digest(resolve(LEGACY_RAW))
    assert result["config_paths"] == [str(fake_run / "config.json")]
    updated = resolve(result["config"])
    previous = resolve(LEGACY_RAW)
    for section in ("seed", "compute", "sampling", "evaluation", "retention", "model", "catalogue_mode"):
        assert updated[section] == previous[section]
    assert updated["training"]["objective_version"] == "action_gap_calibration_v3"
    assert updated["catalogue_coverage"]["train_roots"] == 3
    assert overlay == OVERLAY
    assert file_snapshot(fake_run) == before


def test_skips_every_consecutive_frozen_manifest(fake_run):
    for number in (73, 74, 75):
        write_json(fake_run / f"rounds/round_{number:06d}/manifest.json",
                   {"round": number, "frozen": True})
    before = file_snapshot(fake_run)
    result = revision_tool.prepare_revision(fake_run, OVERLAY)
    assert result["effective_round"] == 76
    assert file_snapshot(fake_run) == before


@pytest.mark.parametrize("which", ["state_hash", "resolved_config", "raw_config"])
def test_identity_mismatch_is_rejected_before_any_write(fake_run, which):
    path = fake_run / {"state_hash": "run_state.json", "resolved_config": "resolved_config.json",
                       "raw_config": "config.json"}[which]
    value = read_json(path)
    if which == "state_hash":
        value["config_hash"] = "not-the-active-config"
    else:
        value["training"]["learning_rate"] *= 2
    write_json(path, value)
    before = file_snapshot(fake_run)
    with pytest.raises(ValueError, match="identity disagrees"):
        revision_tool.prepare_revision(fake_run, OVERLAY)
    assert file_snapshot(fake_run) == before


@pytest.mark.parametrize("overlay", [
    {"seed": 9}, {"compute": {"backend": "cuda"}},
    {"evaluation": {"games_per_preset": 64}}, {"sampling": {"replay_rounds": 1}},
    {"retention": {"rounds": 1}}, {"model": {"hidden": [256]}},
    {"catalogue_mode": False}, {"teacher_bank": []},
])
def test_teacher_model_and_formal_protocol_edits_are_not_allowed(fake_run, overlay):
    before = file_snapshot(fake_run)
    with pytest.raises(ValueError, match="only changes training and catalogue_coverage"):
        revision_tool.prepare_revision(fake_run, overlay)
    assert file_snapshot(fake_run) == before


@pytest.mark.parametrize("bad_state", ["protocol", "teacher_bank"])
def test_only_complete_existing_catalogue_runs_are_accepted(fake_run, bad_state):
    state_path = fake_run / "run_state.json"
    state = read_json(state_path)
    state[bad_state] = "wrong-protocol" if bad_state == "protocol" else state[bad_state][:-1]
    write_json(state_path, state)
    before = file_snapshot(fake_run)
    with pytest.raises(ValueError):
        revision_tool.prepare_revision(fake_run, OVERLAY)
    assert file_snapshot(fake_run) == before


def test_preview_cli_does_not_stage(fake_run, profile, monkeypatch, capsys):
    before = file_snapshot(fake_run)
    invoke_main(monkeypatch, fake_run, profile)
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "preview"
    assert not summary["teacher_bank_modified"]
    assert not summary["checkpoints_modified"]
    assert not summary["formal_evaluation_modified"]
    assert file_snapshot(fake_run) == before


def test_staging_writes_only_pending_and_repeating_is_idempotent(fake_run, profile, monkeypatch, capsys):
    before = file_snapshot(fake_run)
    invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    assert json.loads(capsys.readouterr().out)["status"] == "staged_for_new_round"
    pending = fake_run / "pending_config.json"
    staged = pending.read_bytes()
    after = file_snapshot(fake_run)
    assert set(after) - set(before) == {"pending_config.json"}
    assert {name: after[name] for name in before} == before
    invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    assert json.loads(capsys.readouterr().out)["status"] == "already_staged"
    assert pending.read_bytes() == staged
    assert file_snapshot(fake_run) == after


def test_existing_different_pending_is_preserved(fake_run, profile, monkeypatch):
    write_json(fake_run / "pending_config.json", {"revision": "somebody_elses_revision"})
    before = file_snapshot(fake_run)
    with pytest.raises(ValueError, match="different pending revision exists"):
        invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    assert file_snapshot(fake_run) == before


@pytest.mark.parametrize("same_change", [False, True])
def test_concurrent_pending_creation_never_overwrites_the_winner(
        fake_run, profile, monkeypatch, capsys, same_change):
    expected = revision_tool.prepare_revision(fake_run, OVERLAY, revision="test_revision")
    rival = expected if same_change else {"revision": "concurrent_revision", "marker": "preserve me"}
    before = file_snapshot(fake_run)
    original_link = revision_tool.os.link

    def create_rival_then_link(source, destination):
        write_json(destination, rival)
        return original_link(source, destination)

    monkeypatch.setattr(revision_tool.os, "link", create_rival_then_link)
    if same_change:
        invoke_main(monkeypatch, fake_run, profile, write_pending=True)
        assert json.loads(capsys.readouterr().out)["status"] == "already_staged"
    else:
        with pytest.raises(ValueError, match="created concurrently"):
            invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    assert read_json(fake_run / "pending_config.json") == rival
    after = file_snapshot(fake_run)
    assert set(after) - set(before) == {"pending_config.json"}
    assert {name: after[name] for name in before} == before


def test_active_config_change_during_preparation_rejects_staging(fake_run, profile, monkeypatch):
    import stella.io_utils as io_utils
    original_read = io_utils.read_json
    reads = 0

    def simulate_config_transition(path):
        nonlocal reads
        value = original_read(path)
        if Path(path) == fake_run / "resolved_config.json":
            reads += 1
            # Read-only code preflight and prepare_revision each read once;
            # the third read is the immediate pre-publication identity guard.
            if reads == 3:
                value["training"]["learning_rate"] *= 2
                write_json(path, value)
        return value

    monkeypatch.setattr(io_utils, "read_json", simulate_config_transition)
    with pytest.raises(ValueError, match="Active config changed"):
        invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    assert not (fake_run / "pending_config.json").exists()


def test_noop_overlay_is_already_active(fake_run, tmp_path, monkeypatch, capsys):
    profile = tmp_path / "noop.json"
    write_json(profile, {})
    before = file_snapshot(fake_run)
    assert revision_tool.prepare_revision(fake_run, {}) is None
    invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "already_active"
    assert summary["code_upgrade"]["ready_at_snapshot"]
    assert file_snapshot(fake_run) == before


def test_applied_revision_name_cannot_be_reused(fake_run):
    write_json(fake_run / "config_revisions/already_used.json", {"applied_round": 72})
    before = file_snapshot(fake_run)
    with pytest.raises(ValueError, match="already applied"):
        revision_tool.prepare_revision(fake_run, OVERLAY, revision="already_used")
    assert file_snapshot(fake_run) == before


def test_optional_launcher_config_is_registered_without_modification(fake_run, tmp_path):
    launcher = tmp_path / "launcher.json"
    write_json(launcher, {"unrelated_local_setting": True})
    before = launcher.read_bytes()
    result = revision_tool.prepare_revision(fake_run, OVERLAY, extra_config_path=launcher)
    assert result["config_paths"] == [str(fake_run / "config.json"), str(launcher)]
    assert launcher.read_bytes() == before
    result = revision_tool.prepare_revision(fake_run, OVERLAY, extra_config_path=fake_run / "config.json")
    assert len(result["config_paths"]) == 1
    with pytest.raises(ValueError, match="must already exist"):
        revision_tool.prepare_revision(fake_run, OVERLAY, extra_config_path=tmp_path / "absent.json")


def test_engine_applies_only_at_an_unfrozen_boundary_and_preserves_teacher_and_scores(fake_run):
    from stella.engine import Engine
    for number in (73, 74):
        write_json(fake_run / f"rounds/round_{number:06d}/manifest.json", {"round": number, "frozen": True})
    change = revision_tool.prepare_revision(fake_run, OVERLAY, revision="boundary_test")
    write_json(fake_run / "pending_config.json", change)
    before = file_snapshot(fake_run)
    previous_state = read_json(fake_run / "run_state.json")
    logs = []
    closes = []
    engine = SimpleNamespace(
        run_dir=fake_run, config=resolve(LEGACY_RAW), state=copy.deepcopy(previous_state),
        state_path=fake_run / "run_state.json", close_pool=lambda: closes.append(True),
        log=logs.append,
    )
    for number in (73, 74):
        Engine.apply_next_round_config(engine, number, fake_run / f"rounds/round_{number:06d}")
        assert file_snapshot(fake_run) == before
        assert not closes
    Engine.apply_next_round_config(engine, 75, fake_run / "rounds/round_000075")
    after = file_snapshot(fake_run)
    assert not (fake_run / "pending_config.json").exists()
    assert engine.config == resolve(change["config"])
    assert read_json(fake_run / "config.json") == change["config"]
    assert read_json(fake_run / "resolved_config.json") == engine.config
    record = read_json(fake_run / "config_revisions/boundary_test.json")
    assert record["applied_round"] == 75
    assert record["previous_config"] == resolve(LEGACY_RAW)
    assert len(closes) == len(logs) == 1
    state = read_json(fake_run / "run_state.json")
    allowed_state_changes = {"config_hash", "config_revision", "training_revision_effective_round"}
    assert {key: value for key, value in state.items() if key not in allowed_state_changes} == {
        key: value for key, value in previous_state.items() if key not in allowed_state_changes}
    for name, data in before.items():
        if name not in {"config.json", "resolved_config.json", "run_state.json", "pending_config.json"}:
            assert after[name] == data
    for section in ("model", "evaluation", "sampling", "retention"):
        assert engine.config[section] == resolve(LEGACY_RAW)[section]


def test_engine_rejects_pending_for_another_active_config(fake_run):
    from stella.engine import Engine
    change = revision_tool.prepare_revision(fake_run, OVERLAY)
    change["previous_config_hash"] = "stale-configuration-hash"
    write_json(fake_run / "pending_config.json", change)
    before = file_snapshot(fake_run)
    engine = SimpleNamespace(run_dir=fake_run, config=resolve(LEGACY_RAW))
    with pytest.raises(ValueError, match="does not match the active config"):
        Engine.apply_next_round_config(engine, 73, fake_run / "rounds/round_000073")
    assert file_snapshot(fake_run) == before


def frozen_calibration_round(run, number=73, *, fit_identity=True, old_source=False):
    """A frozen identity fixture, with no need for PyTorch or real weights."""
    from stella.fit_cache import identity
    directory = run / "rounds" / f"round_{number:06d}"
    manifest = {"round": number, "training_config": {
        "objective_version": "action_gap_calibration_v3"},
        "catalogue_coverage": {"train_roots_per_preset": 3, "val_roots_per_preset": 1},
        "teacher_bank": ["M000072"] * 4960, "sampling_tasks": []}
    write_json(directory / "manifest.json", manifest)
    if fit_identity:
        value = identity(manifest, resolve(LEGACY_RAW)["seed"])
        if old_source:
            value["implementation_sha256"] = "original-trainer-source-hash"
        write_json(directory / "fit_identity.json", value)
    return directory


def test_code_preflight_reports_source_mismatch_without_touching_cache(fake_run):
    frozen_calibration_round(fake_run, old_source=True)
    before = file_snapshot(fake_run)
    report = revision_tool.inspect_code_upgrade(fake_run)
    assert not report["ready_at_snapshot"]
    assert report["earliest_unfrozen_round"] == 74
    point = report["frozen_rounds"][0]
    assert point["fit_identity_status"] == "implementation_mismatch"
    assert point["different_identity_fields"] == ["implementation_sha256"]
    assert "Do not delete fit_identity.json" in report["next_step"]
    assert file_snapshot(fake_run) == before


def test_matching_fit_identity_does_not_make_frozen_round_a_code_upgrade_boundary(fake_run):
    frozen_calibration_round(fake_run)
    report = revision_tool.inspect_code_upgrade(fake_run)
    assert report["frozen_rounds"][0]["fit_identity_status"] == "compatible_with_current_source"
    assert not report["ready_at_snapshot"]


def test_pending_config_may_wait_while_original_code_finishes_frozen_fit(
        fake_run, profile, monkeypatch, capsys):
    frozen_calibration_round(fake_run, old_source=True)
    before = file_snapshot(fake_run)
    invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "staged_for_new_round"
    assert report["effective_round"] == 74
    assert not report["code_upgrade"]["ready_at_snapshot"]
    pending = read_json(fake_run / "pending_config.json")
    assert "code_upgrade" not in pending
    after = file_snapshot(fake_run)
    assert set(after) - set(before) == {"pending_config.json"}
    assert {name: after[name] for name in before} == before
    # Later cache progress must not change the staged payload or its identity.
    write_json(fake_run / "rounds/round_000073/fit.json", {"interrupted": False})
    staged = (fake_run / "pending_config.json").read_bytes()
    invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    assert json.loads(capsys.readouterr().out)["status"] == "already_staged"
    assert (fake_run / "pending_config.json").read_bytes() == staged


def test_code_upgrade_cli_blocks_frozen_round_even_before_fit_started(
        fake_run, profile, monkeypatch, capsys):
    frozen_calibration_round(fake_run, fit_identity=False)
    before = file_snapshot(fake_run)
    with pytest.raises(SystemExit) as error:
        invoke_main(monkeypatch, fake_run, profile, check_code_upgrade=True)
    assert error.value.code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "code_upgrade_blocked"
    assert report["code_upgrade"]["frozen_rounds"][0]["fit_identity_status"] == "no_fit_cache_yet"
    assert file_snapshot(fake_run) == before


def test_code_upgrade_cli_checks_even_when_overlay_is_already_active(
        fake_run, tmp_path, monkeypatch, capsys):
    frozen_calibration_round(fake_run, old_source=True)
    profile = tmp_path / "noop.json"
    write_json(profile, {})
    assert revision_tool.prepare_revision(fake_run, {}) is None
    before = file_snapshot(fake_run)
    with pytest.raises(SystemExit) as error:
        invoke_main(monkeypatch, fake_run, profile, check_code_upgrade=True)
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "code_upgrade_blocked"
    assert file_snapshot(fake_run) == before


def test_code_upgrade_cli_ready_check_is_read_only(fake_run, profile, monkeypatch, capsys):
    before = file_snapshot(fake_run)
    invoke_main(monkeypatch, fake_run, profile, check_code_upgrade=True)
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "code_upgrade_ready"
    assert report["code_upgrade"]["ready_at_snapshot"]
    assert report["code_upgrade"]["snapshot_only"]
    assert file_snapshot(fake_run) == before


def test_code_upgrade_check_and_pending_write_are_mutually_exclusive(
        fake_run, profile, monkeypatch):
    before = file_snapshot(fake_run)
    with pytest.raises(SystemExit) as error:
        invoke_main(monkeypatch, fake_run, profile, check_code_upgrade=True, write_pending=True)
    assert error.value.code == 2
    assert file_snapshot(fake_run) == before


@pytest.mark.parametrize("artifact", ["fit_identity.json", "fit.json", "fit_candidates.json",
                                    "fit_checkpoints/epoch_003.npz", "branches/action/fit_identity.json"])
def test_orphan_fit_stage_cannot_be_reassigned_to_new_config(fake_run, artifact):
    write_json(fake_run / "rounds/round_000073" / artifact, {"preserve": True})
    before = file_snapshot(fake_run)
    report = revision_tool.inspect_code_upgrade(fake_run)
    assert not report["ready_at_snapshot"]
    assert report["orphaned_stage_artifacts"]
    with pytest.raises(ValueError, match="without their frozen manifest"):
        revision_tool.prepare_revision(fake_run, OVERLAY)
    assert file_snapshot(fake_run) == before


def test_code_upgrade_check_reports_orphan_cache_without_staging(
        fake_run, profile, monkeypatch, capsys):
    write_json(fake_run / "rounds/round_000073/fit_identity.json", {"preserve": True})
    before = file_snapshot(fake_run)
    with pytest.raises(SystemExit) as error:
        invoke_main(monkeypatch, fake_run, profile, check_code_upgrade=True)
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "code_upgrade_blocked"
    assert file_snapshot(fake_run) == before


def test_summary_is_not_a_code_boundary_until_original_engine_commits(fake_run):
    directory = frozen_calibration_round(fake_run)
    score = fake_run / "teacher_scores/round_000073.npz"
    score.parent.mkdir()
    score.write_bytes(b"synthetic-existing-score-cache")
    candidate = fake_run / "checkpoints/M000073.npz"
    candidate.write_bytes(b"synthetic-existing-candidate")
    summary = dict(round=73, teacher_bank=["M000072"] * 4960,
        teacher_scores="teacher_scores/round_000073.npz", candidate_id="M000073",
        candidate_checkpoint="checkpoints/M000073.npz", best_model_id="M000072",
        best_candidate_mean=101.25, composite_teacher_mean=101.3, candidate_mean=101.2,
        champion_updated_count=0, distinct_teacher_count=1, primary_delta=-.1)
    write_json(directory / "summary.json", summary)
    before = file_snapshot(fake_run)
    report = revision_tool.inspect_code_upgrade(fake_run)
    point = report["frozen_rounds"][0]
    assert point["status"] == "completed_pending_commit"
    assert point["commit_materials"]["complete"]
    assert not report["ready_at_snapshot"]
    assert file_snapshot(fake_run) == before
    # Model the old engine having committed, rather than deleting artifacts.
    state = read_json(fake_run / "run_state.json")
    state["next_round"] = 74
    write_json(fake_run / "run_state.json", state)
    assert revision_tool.inspect_code_upgrade(fake_run)["ready_at_snapshot"]


def test_summary_with_missing_materials_is_reported_as_incomplete(fake_run):
    directory = frozen_calibration_round(fake_run)
    write_json(directory / "summary.json", {"round": 73, "completed_at": "synthetic"})
    report = revision_tool.inspect_code_upgrade(fake_run)
    point = report["frozen_rounds"][0]
    assert point["status"] == "incomplete_summary"
    assert "teacher_scores" in point["commit_materials"]["missing_fields"]
    assert not report["ready_at_snapshot"]


def test_staging_rechecks_a_round_frozen_during_preview(fake_run, profile, monkeypatch, capsys):
    original = revision_tool.inspect_code_upgrade
    calls = 0

    def freeze_before_publication(run):
        nonlocal calls
        calls += 1
        if calls == 2:
            frozen_calibration_round(fake_run, fit_identity=False)
        return original(run)

    monkeypatch.setattr(revision_tool, "inspect_code_upgrade", freeze_before_publication)
    invoke_main(monkeypatch, fake_run, profile, write_pending=True)
    report = json.loads(capsys.readouterr().out)
    assert report["effective_round"] == 74
    assert read_json(fake_run / "pending_config.json")["effective_round"] == 74
    assert (fake_run / "rounds/round_000073/manifest.json").exists()


def test_code_preflight_does_not_report_ready_if_manifest_appears_during_final_read(
        fake_run, monkeypatch):
    original = revision_tool._next_unfrozen_round

    def freeze_before_boundary_read(run, number):
        frozen_calibration_round(fake_run, fit_identity=False)
        return original(run, number)

    monkeypatch.setattr(revision_tool, "_next_unfrozen_round", freeze_before_boundary_read)
    report = revision_tool.inspect_code_upgrade(fake_run)
    assert report["boundary_changed_during_check"]
    assert report["earliest_unfrozen_round"] == 74
    assert not report["ready_at_snapshot"]
