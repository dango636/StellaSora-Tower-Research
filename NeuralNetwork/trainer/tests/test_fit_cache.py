import copy

import pytest

from stella.fit_cache import identity, prepare, validate_pool, validate_winner
from stella.io_utils import atomic_json, digest, file_hash


def manifest():
    return {"round": 73, "rules_version": "fixed", "preset_version": "fixed",
            "teacher_id": "M000071", "teacher_checkpoint": "checkpoints/M000071.npz",
            "teacher_sha256": "initial-sha", "teacher_bank": ["a", "b"],
            "teacher_models": {"a": {"sha256": "a"}, "b": {"sha256": "b"}},
            "catalogue_coverage": {"protocol": "independent"},
            "training_config": {"objective_version": "action_gap_calibration_v3"},
            "compute_config": {"backend": "cuda"},
            "sampling_tasks": [{"family_id": "family", "environment_seed": 1}]}


def test_old_frozen_runs_do_not_require_a_new_identity(tmp_path):
    old = {"round": 72, "training_config": {"objective_version": "old"}}
    assert identity(old, 1) is None
    assert prepare(tmp_path, None) is None
    validate_pool(tmp_path, {}, None)
    assert not list(tmp_path.iterdir())


def test_new_identity_is_stable_but_ordered_bank_and_sampling_matter(tmp_path):
    m = manifest()
    expected = identity(m, 123)
    assert prepare(tmp_path, expected) == digest(expected)
    assert prepare(tmp_path, expected) == digest(expected)
    timed = dict(m, created_at="another-time")
    assert identity(timed, 123) == expected
    changed = copy.deepcopy(m);changed["teacher_bank"].reverse()
    assert identity(changed, 123) != expected
    with pytest.raises(ValueError, match="different frozen"):
        prepare(tmp_path, identity(changed, 123))
    changed = copy.deepcopy(m);changed["sampling_tasks"][0]["environment_seed"] += 1
    assert identity(changed, 123) != expected
    assert identity(m, 124) != expected


@pytest.mark.parametrize("old_file", ["fit.json", "fit_candidates.json", "fit_checkpoints/epoch_001.npz"])
def test_unidentified_existing_calibration_cache_is_rejected(tmp_path, old_file):
    path = tmp_path / old_file;path.parent.mkdir(exist_ok=True);path.write_bytes(b"old")
    with pytest.raises(ValueError, match="Unidentified"):
        prepare(tmp_path, identity(manifest(), 123))
    assert path.read_bytes() == b"old"
    assert not (tmp_path / "fit_identity.json").exists()


def test_candidate_pool_checks_weight_bytes_and_paths(tmp_path):
    key = prepare(tmp_path, identity(manifest(), 123))
    path = tmp_path / "epoch_001.npz";path.write_bytes(b"model")
    pool = {"fit_cache_identity": key, "candidates": [
        {"checkpoint": path.name, "sha256": file_hash(path)}]}
    validate_pool(tmp_path, pool, key)
    path.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="weights changed"):
        validate_pool(tmp_path, pool, key)
    pool["candidates"][0]["checkpoint"] = "../elsewhere.npz"
    with pytest.raises(ValueError, match="confined"):
        validate_pool(tmp_path, pool, key)


def test_winner_requires_matching_identity_and_bytes(tmp_path):
    key = prepare(tmp_path, identity(manifest(), 123))
    path = tmp_path / "winner.npz";path.write_bytes(b"model")
    summary = {"fit_cache_identity": key, "candidate_sha256": file_hash(path)}
    validate_winner(path, summary, key)
    with pytest.raises(ValueError, match="Cached winner"):
        validate_winner(path, summary, "different-config")
    path.write_bytes(b"other-model")
    with pytest.raises(ValueError, match="Cached winner"):
        validate_winner(path, summary, key)
