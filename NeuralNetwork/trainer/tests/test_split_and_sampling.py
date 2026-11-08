"""New trajectory split provenance and backwards-compatible root sampling."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.environment import Episode, heuristic_action, score
from stella.fit import _settings, prepare_roots
from stella.preset_catalogue import catalogue_presets
from stella.sampling import build_sampled_family, root_selection_role, select_family_roots


def split_roots():
    roots = []
    for cat in (0, 1):
        for split, offset in (("train", 0), ("val", 1)):
            roots.append(dict(x=np.zeros((2, 958), np.float32), q=np.array([1., 2.], np.float32),
                family_id=f"cat{cat}_{split}", teacher_id=f"M{cat+10:06d}",
                catalogue_id=cat, preset_group=f"catalogue_{cat:04d}", split=split,
                source_environment_seed=cat*100+offset,
                source_exploration_seed=cat*100+offset+1000,
                label_seed=cat*100+offset+10000))
    return roots


def trajectory_settings():
    return _settings({"per_preset_teacher_bank": True, "validation_unit": "trajectory"})


def test_legacy_default_keeps_whole_preset_separation():
    roots = split_roots()
    with pytest.raises(ValueError, match="preset leaks"):
        prepare_roots(roots, 958, _settings({"per_preset_teacher_bank": True}))
    legacy = [r for r in roots if (r["catalogue_id"] == 0) == (r["split"] == "train")]
    train, val, _ = prepare_roots(legacy, 958, _settings({"per_preset_teacher_bank": True}))
    assert {r["catalogue_id"] for r in train}.isdisjoint({r["catalogue_id"] for r in val})


def test_trajectory_mode_trains_every_preset_with_independent_validation_family():
    train, val, _ = prepare_roots(split_roots(), 958, trajectory_settings())
    assert {r["catalogue_id"] for r in train} == {0, 1}
    assert {r["catalogue_id"] for r in val} == {0, 1}
    assert {r["family_id"] for r in train}.isdisjoint({r["family_id"] for r in val})


@pytest.mark.parametrize("field, message", [
    ("source_environment_seed", "environment stream"),
    ("label_seed", "label stream"),
    ("family_id", "family leaks"),
])
def test_same_statistical_source_cannot_cross_splits(field, message):
    roots = split_roots()
    roots[1][field] = roots[0][field]
    with pytest.raises(ValueError, match=message):
        prepare_roots(roots, 958, trajectory_settings())


def test_missing_provenance_is_not_treated_as_independence():
    roots = split_roots()
    del roots[1]["source_environment_seed"]
    with pytest.raises(ValueError, match="seed provenance"):
        prepare_roots(roots, 958, trajectory_settings())


def test_new_split_mode_cannot_change_a_presets_frozen_teacher():
    roots = split_roots()
    roots[1]["teacher_id"] = "M999999"
    with pytest.raises(ValueError, match="mix frozen teachers"):
        prepare_roots(roots, 958, trajectory_settings())


def test_legacy_root_selection_matches_frozen_r48_golden_sequence():
    # Golden sequence was generated once with the untouched R48 sampler.
    rows = [(i, SimpleNamespace(phase=("SHOP", "EVENT", "PAGE", "ENHANCE_PAGE")[i % 4],
                                floor=i+1, value=20-i)) for i in range(12)]
    task = dict(selection_seed=314159, roots_per_family=4, ordinal=17,
                balanced_phase_sampling=True, phase_sampling_index=5, hard_sampling=False)
    assert [root_selection_role(task, i) for i in range(4)] == ["natural", "natural", "phase", "hard"]
    assert [i for i, _ in select_family_roots(task, rows, lambda state: state.value)] == [7, 0, 9, 11]


def test_explicit_three_root_schedule_reaches_hard_selection(monkeypatch):
    rows = [(i, SimpleNamespace(phase="SHOP", floor=5, value=i)) for i in range(12)]
    calls = []
    def hard(task, remaining, gap, classify):
        calls.append(len(remaining))
        return max(remaining, key=lambda item: item[0])
    monkeypatch.setattr("stella.sampling.choose_hard_root", hard)
    task = dict(selection_seed=7, roots_per_family=3, ordinal=0, hard_sampling=True,
                root_selection_schedule=["natural", "phase", "hard"])
    chosen = select_family_roots(task, rows, lambda state: state.value)
    assert len(chosen) == 3 and len({i for i, _ in chosen}) == 3
    assert calls == [10]
    assert root_selection_role(task, 2) == "hard"
    # The old modulo schedule would call ordinal 2 "phase" and lose this slot.


def test_real_public_state_serialization_keeps_hard_budget_and_seed_provenance():
    preset = catalogue_presets()[0]
    episode = Episode(preset, 82631)
    selected = []
    while not episode.done and len(selected) < 3:
        if len(episode.actions) > 1 and score(episode.state) < preset.total:
            selected.append((episode.steps, episode.state))
        episode.step(heuristic_action(episode.state, episode.actions))
    assert len(selected) == 3
    task = dict(family_id="r000073_f00000_train", split="train", catalogue_id=0,
                teacher_id="M000010", label_seed=932, roots_per_family=3,
                rollouts_per_action=64, hard_rollouts_per_action=256, hard_sampling=True,
                root_selection_schedule=["natural", "phase", "hard"],
                coverage_protocol="catalogue4960_independent_trajectories_v1",
                environment_seed=82631, exploration_seed=1947, epsilon=0.)
    result = build_sampled_family(task, preset, selected, episode.steps, episode.score)
    roots = result["roots"]
    assert [r["root_selection_role"] for r in roots] == ["natural", "phase", "hard"]
    assert [r["rollouts_per_action"] for r in roots] == [64, 64, 256]
    assert [r.get("hard_sample_slot", False) for r in roots] == [False, False, True]
    assert all(r["source_environment_seed"] == 82631 for r in roots)
    assert all(r["source_exploration_seed"] == 1947 for r in roots)
    assert len({r["label_seed"] for r in roots}) == 3


def test_invalid_explicit_schedule_fails_before_selecting_roots():
    with pytest.raises(ValueError, match="schedule"):
        root_selection_role({"roots_per_family": 3,
                             "root_selection_schedule": ["natural", "hard"]}, 0)
