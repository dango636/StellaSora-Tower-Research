"""Regression tests for equal-preset visits and use of newly paid-for labels."""
from collections import Counter

import numpy as np
import pytest

from stella.action_calibration import CalibrationConfig, batch_pair_moments, root_pair_moments
from stella.fit_calibration import exact_preset_epoch, _systematic_resampling


def corpus(counts, *, current=73):
    roots = []
    for preset, (fresh, historical) in enumerate(counts):
        roots.extend(dict(catalogue_id=preset, source_round=current) for _ in range(fresh))
        roots.extend(dict(catalogue_id=preset, source_round=current-1) for _ in range(historical))
    return roots


def schedule(roots, k, **extra):
    extra.setdefault("sampling", "coverage_without_replacement")
    return exact_preset_epoch(roots, np.ones(len(roots)), np.random.default_rng(20261106), k,
                             current_round=73, **extra)


def test_every_new_and_replayed_root_seen_when_preset_quota_fits_corpus():
    roots = corpus([(3, 3)]*4960)
    order, report = schedule(roots, 6, require_current=True)
    assert len(order) == len(set(order)) == len(roots)
    assert Counter(roots[i]["catalogue_id"] for i in order) == dict.fromkeys(range(4960), 6)
    assert report["current_quota_enforced"]
    assert report["unique_current_roots_visited"] == 14880
    assert report["current_fraction_min"] == report["current_fraction_max"] == .5


def test_priority_can_order_roots_but_cannot_starve_others_in_one_complete_pass():
    roots = corpus([(4, 0)])
    order, report = exact_preset_epoch(roots, [1e300, 1e-300, 1, 2],
                                      np.random.default_rng(7), 4,
                                      sampling="coverage_without_replacement")
    assert set(order) == set(range(4))
    assert report["duplicate_visits"] == 0


def test_current_quota_survives_large_history_and_unequal_preset_sizes():
    roots = corpus([(2, 8), (3, 0), (5, 9)])
    order, report = schedule(roots, 6, require_current=True)
    assert Counter(roots[i]["catalogue_id"] for i in order) == {0: 6, 1: 6, 2: 6}
    fresh = Counter(roots[i]["catalogue_id"] for i in order if roots[i]["source_round"] == 73)
    assert fresh == {0: 3, 1: 6, 2: 5}
    assert report["current_fraction_min"] == .5
    for preset, n in [(0, 2), (1, 3), (2, 5)]:
        current_indices = {i for i in order if roots[i]["catalogue_id"] == preset
                           and roots[i]["source_round"] == 73}
        assert len(current_indices) == n


def test_short_preset_repeats_only_after_complete_cycles():
    roots = corpus([(3, 0)])
    order, report = schedule(roots, 8)
    counts = Counter(order)
    assert sorted(counts.values()) == [2, 3, 3]
    assert report["unique_roots_visited"] == 3


def test_short_final_global_batch_does_not_change_epoch_preset_quotas():
    roots = corpus([(3, 3)]*5)
    order, report = schedule(roots, 6)
    assert len(order) % 8 == 6
    assert Counter(roots[i]["catalogue_id"] for i in order) == dict.fromkeys(range(5), 6)
    assert report["root_visits"] == 30


def test_replacement_mode_is_retained_as_reproducible_control():
    roots = corpus([(4, 4)]*20)
    order, report = schedule(roots, 8, sampling="replacement")
    repeat, _ = schedule(roots, 8, sampling="replacement")
    np.testing.assert_array_equal(order, repeat)
    assert report["unique_roots_visited"] < len(roots)
    assert not report["current_quota_enforced"]
    assert Counter(roots[i]["catalogue_id"] for i in order) == dict.fromkeys(range(20), 8)


def test_defaults_preserve_legacy_sampling_but_use_the_tested_strong_trust():
    roots = corpus([(3, 3)])
    _, report = exact_preset_epoch(roots, np.ones(6), np.random.default_rng(3), 6)
    assert report["sampling_method"] == "replacement"
    assert CalibrationConfig().uncertainty_trust_weight == 1.0


def test_systematic_mode_covers_uniform_roots_and_honors_current_quota():
    roots = corpus([(3, 3)]*100)
    order, report = schedule(roots, 6, sampling="stratified_resampling", require_current=True)
    assert len(set(order)) == len(roots)
    assert report["current_fraction_min"] == .5
    assert report["current_quota_enforced"]


def test_systematic_counts_preserve_priority_expectation_without_multinomial_variance():
    generator = np.random.default_rng(42)
    priorities = np.array([1., 2., 7.])
    wanted = 6*priorities/priorities.sum()
    counts = []
    for _ in range(5000):
        order = _systematic_resampling(np.arange(3), priorities, 6, generator)
        visits = np.bincount(order, minlength=3)
        assert sum(visits) == 6
        assert np.all(np.abs(visits-wanted) < 1)
        counts.append(visits)
    np.testing.assert_allclose(np.mean(counts, axis=0), wanted, rtol=0, atol=.02)


def test_missing_provenance_is_reported_and_never_inferred_from_name():
    roots = [dict(catalogue_id=i//3, family_id="r000073_f00000_train") for i in range(6)]
    order, report = schedule(roots, 3)
    assert len(set(order)) == 6
    assert not report["source_round_provenance_complete"]
    assert not report["current_quota_enforced"]
    assert report["current_root_visits"] is None
    with pytest.raises(ValueError, match="source_round"):
        schedule(roots, 3, require_current=True)


def test_missing_current_preset_or_future_round_is_rejected():
    roots = corpus([(0, 3), (3, 0)])
    with pytest.raises(ValueError, match="no current"):
        schedule(roots, 4, require_current=True)
    roots[0]["source_round"] = 74
    with pytest.raises(ValueError, match="future"):
        schedule(roots, 4)


def test_zero_priority_excluded_without_masking_invalid_all_zero_preset():
    roots = corpus([(3, 0)])
    order, _ = exact_preset_epoch(roots, [1, 0, 1], np.random.default_rng(3), 5,
                                 sampling="coverage_without_replacement")
    assert set(order) == {0, 2}
    with pytest.raises(ValueError, match="positive sampling mass"):
        exact_preset_epoch(roots, [0, 0, 0], np.random.default_rng(3), 5)


@pytest.mark.parametrize("bad", [True, 1.5, "1", np.float64(1.0)])
def test_teacher_action_index_does_not_silently_truncate(bad):
    with pytest.raises(ValueError, match="teacher_action_index"):
        root_pair_moments(dict(q=[1., 2.], teacher_action_index=bad))


def test_numpy_integer_unknown_and_valid_teacher_indices_remain_supported():
    for index in [None, -1, np.int32(0), np.int64(1)]:
        result = root_pair_moments(dict(q=[1., 2.], teacher_action_index=index))
        assert result["teacher_action"] == (-1 if index is None else index)


@pytest.mark.parametrize("bad", [True, 2.7, "2", 0])
def test_max_action_width_is_not_truncated(bad):
    with pytest.raises(ValueError, match="max_actions"):
        batch_pair_moments([dict(q=[1., 2.])], max_actions=bad)


@pytest.mark.parametrize("bad", [True, -1, 1.2, "4", 2**64])
def test_explicit_sample_count_is_validated(bad):
    with pytest.raises(ValueError, match="sample_count"):
        root_pair_moments(dict(q=[1., 2.], pair_se=[[0., .1], [.1, 0.]], sample_count=bad))
