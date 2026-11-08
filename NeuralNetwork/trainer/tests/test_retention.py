import json
from pathlib import Path

import pytest

from stella.retention import apply_retention, retention_plan


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_run(tmp_path):
    run = tmp_path / "run"
    for number in range(1, 7):
        directory = run / "rounds" / f"round_{number:06d}"
        write(directory / "manifest.json", {
            "teacher_checkpoint": f"checkpoints/M{number-1:06d}.npz",
            "teacher_scores": f"teacher_scores/round_{number-1:06d}.npz",
            "teacher_models": {"old": {"checkpoint": f"checkpoints/M{number-1:06d}_E001.npz"}}})
        if number != 2 and number < 6:
            write(directory / "summary.json", {"round": number,
                "candidate_checkpoint": f"checkpoints/M{number:06d}.npz"})
        write(directory / "labels" / "data.json", {"value": number})
    for number in range(9):
        for suffix in ("", "_E001", "_E002"):
            path = run / "checkpoints" / f"M{number:06d}{suffix}.npz"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"weight" + bytes([number]))
            write(path.with_suffix(".json"), {"model": number})
        path = run / "teacher_scores" / f"round_{number:06d}.npz"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"score" + bytes([number]))
    state = {"next_round": 6, "models": {"active": "checkpoints/M000000_E001.npz"},
             "teacher_checkpoint": "checkpoints/M000004.npz",
             "teacher_scores": "teacher_scores/round_000005.npz"}
    return run, state


def test_linux_cleanup_preserves_incomplete_rounds_and_all_retained_teachers(tmp_path):
    run, state = make_run(tmp_path)
    plan = retention_plan(run, state, keep_rounds=2)
    targets = {x["path"] for x in plan["targets"]}
    assert "rounds/round_000001" in targets and "rounds/round_000003" in targets
    assert "rounds/round_000002" not in targets  # Incomplete old work survives.
    assert "rounds/round_000006" not in targets  # Current work survives.
    assert "checkpoints/M000003_E001.npz" not in targets  # Retained round 4 teacher.
    assert "checkpoints/M000000_E001.npz" not in targets  # Current bank teacher.
    assert "checkpoints/M000002_E001.npz" not in targets  # Unfinished round's candidates.
    assert "checkpoints/M000001_E002.npz" in targets  # Unreferenced expired candidate.
    assert "teacher_scores/round_000005.npz" not in targets
    assert "teacher_scores/round_000002.npz" in targets
    assert "checkpoints/M000008.npz" not in targets  # Future file is not expired.
    report = apply_retention(plan)
    assert not (run / "rounds/round_000001").exists()
    assert (run / "rounds/round_000002/labels/data.json").is_file()
    assert (run / "checkpoints/M000003_E001.npz").read_bytes() == b"weight\x03"
    assert report["mode"] == "posix_managed_delete"
    assert not retention_plan(run, state, keep_rounds=2)["targets"]


def test_malformed_and_unmanaged_names_are_never_removed(tmp_path):
    run, state = make_run(tmp_path)
    write(run / "rounds/round_notes/summary.json", {"round": 1})
    (run / "checkpoints/Mpersonal.npz").write_bytes(b"keep")
    plan = retention_plan(run, state, keep_rounds=2)
    apply_retention(plan)
    assert (run / "rounds/round_notes/summary.json").is_file()
    assert (run / "checkpoints/Mpersonal.npz").read_bytes() == b"keep"


def test_referenced_windows_paths_are_portable(tmp_path):
    run, state = make_run(tmp_path)
    state["models"]["portable"] = "checkpoints\\M000001_E002.npz"
    plan = retention_plan(run, state, keep_rounds=2)
    assert "checkpoints/M000001_E002.npz" not in {x["path"] for x in plan["targets"]}


@pytest.mark.parametrize("link_kind", ["round", "checkpoints", "reference"])
def test_symlink_escape_aborts_before_any_mutation(tmp_path, link_kind):
    run, state = make_run(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir();(outside / "sentinel").write_bytes(b"safe")
    if link_kind == "round":
        (run / "rounds/round_000000").symlink_to(outside, target_is_directory=True)
    elif link_kind == "checkpoints":
        (run / "checkpoints/M000009.npz").symlink_to(outside / "sentinel")
    else:
        (run / "escape").symlink_to(outside, target_is_directory=True)
        state["models"]["bad"] = "escape/sentinel"
    with pytest.raises(ValueError):
        retention_plan(run, state, keep_rounds=2)
    assert (run / "rounds/round_000001").is_dir()
    assert (outside / "sentinel").read_bytes() == b"safe"


def test_corrupt_completed_summary_fails_closed(tmp_path):
    run, state = make_run(tmp_path)
    write(run / "rounds/round_000001/summary.json", {"round": 999})
    with pytest.raises(ValueError, match="round does not match"):
        retention_plan(run, state, keep_rounds=2)
    assert (run / "rounds/round_000001").is_dir()


def test_changed_target_rejected_before_deletion(tmp_path):
    run, state = make_run(tmp_path)
    plan = retention_plan(run, state, keep_rounds=2)
    path = run / "checkpoints/M000001_E002.npz"
    path.write_bytes(b"new user content must survive")
    with pytest.raises(ValueError, match="changed after planning"):
        apply_retention(plan)
    assert path.read_bytes() == b"new user content must survive"
    assert (run / "rounds/round_000001").is_dir()


def test_retained_official_fit_candidates_survive(tmp_path):
    run, state = make_run(tmp_path)
    write(run / "rounds/round_000006/fit.json", {
        "checkpoint_selection": {"candidates": [{"checkpoint": "checkpoints/M000001_E002.npz"}]}})
    plan = retention_plan(run, state, keep_rounds=2)
    assert "checkpoints/M000001_E002.npz" not in {x["path"] for x in plan["targets"]}


def test_retained_behavior_checkpoint_survives(tmp_path):
    run, state = make_run(tmp_path)
    path = run / "rounds/round_000005/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["behavior_checkpoint"] = "checkpoints/M000001_E002.npz"
    write(path, manifest)
    apply_retention(retention_plan(run, state, keep_rounds=2))
    assert (run / "checkpoints/M000001_E002.npz").read_bytes() == b"weight\x01"


def test_retained_frozen_learner_survives_after_state_advances(tmp_path):
    run, state = make_run(tmp_path)
    # This source is no longer the current last candidate and has never become
    # a label teacher. Its retained manifest must still be exactly resumable.
    state["last_candidate_checkpoint"] = "checkpoints/M000005.npz"
    path = run / "rounds/round_000005/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["learner_initialization"] = {"checkpoint": "checkpoints\\M000001_E002.npz"}
    write(path, manifest)
    apply_retention(retention_plan(run, state, keep_rounds=2))
    assert (run / "checkpoints/M000001_E002.npz").read_bytes() == b"weight\x01"
    assert (run / "checkpoints/M000005.npz").is_file()


def test_old_incomplete_epoch_candidates_survive_without_fit_metadata(tmp_path):
    run, state = make_run(tmp_path)
    branch = run / "checkpoints/M000002_Btrial_E009.npz"
    branch.write_bytes(b"interrupted training")
    apply_retention(retention_plan(run, state, keep_rounds=2))
    assert branch.read_bytes() == b"interrupted training"
    assert (run / "checkpoints/M000002_E002.npz").is_file()


def test_protected_round_descendants_preserve_transitive_dependencies(tmp_path):
    run, state = make_run(tmp_path)
    for number in (1, 3):
        weight = run / f"rounds/round_{number:06d}/fit_checkpoints/epoch.npz"
        weight.parent.mkdir()
        weight.write_bytes(b"nested teacher")
    state["models"]["nested"] = "rounds/round_000001/fit_checkpoints/epoch.npz"
    one = run / "rounds/round_000001/manifest.json"
    record = json.loads(one.read_text())
    record["behavior_checkpoint"] = "rounds/round_000003/fit_checkpoints/epoch.npz"
    write(one, record)
    three = run / "rounds/round_000003/manifest.json"
    record = json.loads(three.read_text())
    record["behavior_checkpoint"] = "checkpoints/M000000_E002.npz"
    write(three, record)
    plan = retention_plan(run, state, keep_rounds=2)
    targets = {record["path"] for record in plan["targets"]}
    assert "rounds/round_000001" not in targets
    assert "rounds/round_000003" not in targets
    assert "checkpoints/M000000_E002.npz" not in targets
    apply_retention(plan)
    assert (run / "checkpoints/M000000_E002.npz").read_bytes() == b"weight\x00"
    assert (run / "rounds/round_000003/fit_checkpoints/epoch.npz").read_bytes() == b"nested teacher"


def test_plan_only_does_not_mutate_and_has_bounded_targets(tmp_path):
    run, state = make_run(tmp_path)
    before = sorted(str(p.relative_to(run)) for p in run.rglob("*"))
    plan = retention_plan(run, state, keep_rounds=2, keep_checkpoints=12)
    assert len(plan["targets"]) < len(before)
    assert before == sorted(str(p.relative_to(run)) for p in run.rglob("*"))


def test_nested_symlink_is_unlinked_without_following(tmp_path):
    run, state = make_run(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir();(outside / "keep").write_bytes(b"keep")
    (run / "rounds/round_000001/nested").symlink_to(outside, target_is_directory=True)
    apply_retention(retention_plan(run, state, keep_rounds=2))
    assert (outside / "keep").read_bytes() == b"keep"
