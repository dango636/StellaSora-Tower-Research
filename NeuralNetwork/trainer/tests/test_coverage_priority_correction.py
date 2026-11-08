"""Full coverage retains the former priority objective and equal preset mass."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.action_calibration import (CalibrationConfig, prepare_calibration,
                                       calibration_loss_prepared)
from stella.config import resolve
from stella.fit_calibration import (coverage_priority_weights, exact_preset_epoch,
                                   fit_candidate_calibration)
from stella.network import QNetwork


def test_full_three_and_six_root_presets_match_replacement_objective_and_gradient():
    roots = ([dict(catalogue_id=0, source_round=9) for _ in range(3)]+
             [dict(catalogue_id=1, source_round=9 if i < 3 else 8) for i in range(6)])
    priorities = np.array([1., 2., 3., 3., 1., 1., 2., 2., 1.])
    order, quota = exact_preset_epoch(roots, priorities, np.random.default_rng(8), 6,
        sampling="coverage_without_replacement", current_round=9,
        current_fraction=.5, require_current=True)
    weights, report = coverage_priority_weights(roots, priorities, order, batch_size=5)
    assert quota["root_visits"] == 12
    assert report["corrected_presets"] == 2
    assert report["fallback_presets"] == report["missed_positive_roots"] == 0
    a, b, theta = np.linspace(.4, 2., len(roots)), np.linspace(-1., 1., len(roots)), .7
    loss = .5*(a*theta-b)**2
    derivative = a*(a*theta-b)
    expected_loss = expected_derivative = 0.
    for preset in (0, 1):
        local = np.array([root["catalogue_id"] == preset for root in roots])
        probabilities = priorities[local]/priorities[local].sum()
        expected_loss += 6*np.dot(probabilities, loss[local])
        expected_derivative += 6*np.dot(probabilities, derivative[local])
        selected = np.array([roots[i]["catalogue_id"] == preset for i in order])
        assert weights[selected].sum() == pytest.approx(6., abs=1e-14)
    assert np.dot(weights, loss[order]) == pytest.approx(expected_loss, abs=1e-13)
    assert np.dot(weights, derivative[order]) == pytest.approx(expected_derivative, abs=1e-13)
    assert report["weight_effective_visits"] <= len(order)
    assert report["visit_weight_square_sum"] == pytest.approx(float(weights@weights))
    assert report["batch_loss_mass_max"] == pytest.approx(max(weights[:5].sum(), weights[5:10].sum(), weights[10:].sum())/5)


def test_partial_preset_falls_back_without_changing_any_preset_mass():
    roots = [dict(catalogue_id=preset) for preset, size in ((0, 3), (1, 4)) for _ in range(size)]
    priorities = np.array([1., 2., 3., 1., 1., 2., 5.])
    order, _ = exact_preset_epoch(roots, priorities, np.random.default_rng(19), 3,
                                 sampling="coverage_without_replacement")
    weights, report = coverage_priority_weights(roots, priorities, order)
    partial = np.array([roots[i]["catalogue_id"] == 1 for i in order])
    np.testing.assert_array_equal(weights[partial], np.ones(3))
    assert weights[~partial].sum() == pytest.approx(3.)
    assert report["corrected_presets"] == report["fallback_presets"] == 1
    assert report["missed_positive_roots"] == 1
    assert "only for fully covered presets" in report["semantics"]


def test_unvisited_zero_priority_roots_do_not_trigger_fallback_or_overflow():
    roots = [dict(catalogue_id=0) for _ in range(4)]+[dict(catalogue_id=1) for _ in range(2)]
    priorities = np.array([1e300, 0., 2e300, 0., 2e300, 1e300])
    order, _ = exact_preset_epoch(roots, priorities, np.random.default_rng(23), 2,
                                 sampling="coverage_without_replacement")
    weights, report = coverage_priority_weights(roots, priorities, order)
    assert np.isfinite(weights).all()
    assert report["corrected_presets"] == 2
    assert report["fallback_presets"] == 0
    assert report["positive_roots"] == report["covered_positive_roots"] == 4
    assert weights.sum() == pytest.approx(4.)


def test_last_short_batch_uses_fixed_batch_denominator_and_matches_full_loss_gradient():
    roots = [dict(catalogue_id=i//3) for i in range(6)]
    priorities = np.array([1., 2., 3., 2., 3., 1.])
    order, _ = exact_preset_epoch(roots, priorities, np.random.default_rng(5), 3,
                                 sampling="coverage_without_replacement")
    weights, _ = coverage_priority_weights(roots, priorities, order, batch_size=4)
    target = torch.tensor([[1., 1.2+i*.03] for i in order])
    mask = torch.ones_like(target, dtype=torch.bool)
    se = torch.full((6, 2, 2), .1)
    known = torch.ones_like(se, dtype=torch.bool)
    config = CalibrationConfig()
    cache = prepare_calibration(target, mask, se, known, config=config)
    fixed_weights = torch.tensor(weights/4, dtype=torch.float32)
    prediction = torch.tensor([[.06, .04]]*6, requires_grad=True)
    full = calibration_loss_prepared(prediction, target, mask, cache["pair_reliability"],
        cache["action_reliability"], config=config, root_weights=fixed_weights)
    full.backward()
    split_prediction = prediction.detach().clone().requires_grad_()
    split_loss = 0.
    for start, end in ((0, 4), (4, 6)):
        part = calibration_loss_prepared(split_prediction[start:end], target[start:end], mask[start:end],
            cache["pair_reliability"][start:end], cache["action_reliability"][start:end],
            config=config, root_weights=fixed_weights[start:end])
        part.backward()
        split_loss += float(part.detach())
    assert split_loss == pytest.approx(float(full.detach()), rel=1e-6)
    torch.testing.assert_close(split_prediction.grad, prediction.grad, atol=1e-9, rtol=1e-6)


def _fit_case(sampling, correction):
    roots = []
    for preset in range(2):
        for split, count in (("train", 3), ("val", 1)):
            for index in range(count):
                seed = preset*100+(20 if split == "val" else 0)+index
                q = np.array([1., 1.15+.03*preset+.01*index], np.float32)
                roots.append(dict(x=np.eye(2, dtype=np.float32), q=q,
                    pair_se=np.zeros((2, 2), np.float32),
                    family_id=f"{preset}_{split}_{index}", catalogue_id=preset,
                    preset_group=str(preset), teacher_id=f"M{preset+1:06d}", split=split,
                    source_environment_seed=seed, label_seed=seed+1000,
                    teacher_action_index=1, source_round=9))
    source = QNetwork(input_dim=2, hidden=(4,), seed=4, heuristic_init=False).freeze()
    config = dict(epochs=2, batch_roots=4, learning_rate=1e-3, patience=0, min_epochs=0,
        checkpoint_candidates=2, per_preset_teacher_bank=True, validation_unit="trajectory",
        checkpoint_metric="label_regret", calibration_require_all_presets=False,
        calibration_visits_per_preset=3, calibration_sampling=sampling,
        mistake_weight_strength=0., seed=11, training_round=9,
        calibration_coverage_priority_correction=correction)
    return fit_candidate_calibration(source, roots, config, device="cpu")


@pytest.mark.parametrize("sampling", ["replacement", "coverage_without_replacement"])
def test_flag_preserves_replacement_and_uniform_coverage_model_bits(sampling):
    # Includes a short last batch. Uniform priorities require unit visit
    # weights, while replacement must ignore the correction flag entirely.
    old, new = _fit_case(sampling, False), _fit_case(sampling, True)
    assert old.model.checksum() == new.model.checksum()
    assert [p["checksum"] for p in old.candidate_checkpoints] == [p["checksum"] for p in new.candidate_checkpoints]
    for a, b in zip(old.history, new.history):
        assert a["train"] == b["train"]
        assert a["val"] == b["val"]
    correction = new.history[-1]["epoch_quota"]["priority_correction"]
    if sampling == "replacement":
        assert not correction["applied"]
    else:
        assert correction["corrected_presets"] == 2
        assert correction["visit_weight_min"] == correction["visit_weight_max"] == 1.


def test_optional_config_retains_legacy_shape_and_requires_real_boolean():
    name = "calibration_coverage_priority_correction"
    assert name not in resolve({"workers": 1})["training"]
    for value in (True, False):
        assert resolve({"workers": 1, "training": {name: value}})["training"][name] is value
    for value in (0, 1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            resolve({"workers": 1, "training": {name: value}})


def test_invalid_visit_or_priority_input_is_not_silently_reweighted():
    roots = [dict(catalogue_id=0), dict(catalogue_id=1)]
    with pytest.raises(ValueError, match="same positive visit quota"):
        coverage_priority_weights(roots, [1., 1.], [0, 0, 1])
    with pytest.raises(ValueError, match="positive priority mass"):
        coverage_priority_weights(roots, [1., 0.], [0, 1])
    with pytest.raises(ValueError, match="integer visit order"):
        coverage_priority_weights(roots, [1., 1.], [.0, 1.])
