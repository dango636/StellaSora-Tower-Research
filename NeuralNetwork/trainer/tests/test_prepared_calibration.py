"""Prepared decision loss: cached evidence, original baseline, exact root mass."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.action_calibration import (CalibrationConfig, calibration_loss,
    calibration_loss_prepared, prepare_calibration)
import stella.fit_calibration as fitting
from stella.fit_torch import TorchQ, torch_teacher_predictions
from stella.network import QNetwork


@pytest.mark.parametrize("projection_z", [0., 1.5])
def test_cached_complete_root_slices_match_direct_loss_and_gradients(projection_z):
    """Repeated sampling, unknown pairs and poisoned padding retain semantics."""
    generator = torch.Generator().manual_seed(1701)
    batch, actions = 5, 4
    target = torch.rand((batch, actions), generator=generator)*20.
    reference = (target+torch.randn((batch, actions), generator=generator))/20.
    prediction = reference+torch.randn((batch, actions), generator=generator)*.02
    mask = torch.arange(actions)[None, :] < torch.tensor([4, 3, 2, 1, 0])[:, None]
    se = torch.rand((batch, actions, actions), generator=generator)*.4
    se = (se+se.transpose(1, 2))*.5
    known = torch.ones_like(se, dtype=torch.bool)
    known[0, 1, 3] = known[0, 3, 1] = False
    se[~known] = float("nan")
    target[~mask] = float("nan")
    reference[~mask] = float("inf")
    prediction[~mask] = float("nan")
    original_reference = reference.clone()
    config = CalibrationConfig(trust_projection_z=projection_z)
    cache = prepare_calibration(target, mask, se, known,
        reference_prediction=reference, config=config)
    indices = torch.tensor([4, 1, 0, 1, 3])
    weights = torch.tensor([.12, .23, .09, .17, .04])
    teacher = torch.tensor([-1, 1, 2, 0, -1])[indices]
    direct_prediction = prediction[indices].clone().requires_grad_()
    cached_prediction = prediction[indices].clone().requires_grad_()
    direct, direct_details = calibration_loss(direct_prediction, target[indices], mask[indices],
        se[indices], known[indices], teacher_actions=teacher, reference_prediction=reference[indices],
        config=config, root_weights=weights, return_details=True)
    cached, cached_details = calibration_loss_prepared(cached_prediction, target[indices], mask[indices],
        cache["pair_reliability"][indices], cache["action_reliability"][indices],
        teacher_actions=teacher, trust_reference=cache["trust_reference"][indices],
        pair_known=cache["pair_known"][indices], config=config, root_weights=weights,
        return_details=True)
    direct.backward()
    cached.backward()
    assert torch.equal(direct, cached)
    assert torch.equal(direct_prediction.grad, cached_prediction.grad)
    assert torch.count_nonzero(cached_prediction.grad[~mask[indices]]) == 0
    for name in direct_details:
        assert torch.equal(direct_details[name], cached_details[name])
    torch.testing.assert_close(reference, original_reference, equal_nan=True)
    assert not cache["trust_reference"].requires_grad


def test_projection_origin_is_not_changed_by_raw_score_rounding_tie():
    # Multiplication by 20 merges the first two adjacent FP32 values. Only
    # the true initial winner (action 1) has paired evidence to action 2.
    reference = torch.tensor([[1.6000001430511475, 1.600000262260437, 1.55]])
    assert reference.argmax(1).item() == 1
    assert (reference*20).argmax(1).item() == 0
    target = torch.tensor([[10., 10.2, 11.]])
    mask = torch.ones_like(target, dtype=torch.bool)
    known = torch.zeros((1, 3, 3), dtype=torch.bool)
    known[:, 1, 2] = known[:, 2, 1] = True
    se = torch.full((1, 3, 3), float("nan"))
    se[known] = .1
    cache = prepare_calibration(target, mask, se, known,
        reference_prediction=reference, config=CalibrationConfig(trust_projection_z=1.5))
    projected = cache["trust_reference"]
    assert projected.argmax(1).item() == 2
    assert projected[0, 2]-projected[0, 1] == pytest.approx(.65, abs=3e-6)
    assert projected[0, 0] == reference[0, 0]*20  # Unknown comparison stays unchanged.


def _roots():
    result = []
    for preset in range(2):
        for split, rounds in (("train", [9, 9, 8, 8]), ("val", [9])):
            for index, source_round in enumerate(rounds):
                seed = preset*1000 + (100 if split == "val" else 0) + index
                features = np.zeros((2, 4), np.float32)
                features[0, 0] = features[1, 1] = 1.
                features[:, 2] = .1*preset
                result.append(dict(x=features, q=np.array([10., 11.], np.float32),
                    pair_se=np.array([[0., .2], [.2, 0.]], np.float32),
                    family_id=f"p{preset}_{split}_{index}", preset_group=f"p{preset}",
                    catalogue_id=preset, split=split, teacher_id=f"M{preset+10:06d}",
                    source_environment_seed=seed, label_seed=10000+seed,
                    source_round=source_round, teacher_action_index=1,
                    behavior_source="student"))
    return result


def _source():
    model = QNetwork(input_dim=4, hidden=(8,), heuristic_init=False, seed=8)
    model.params["skip_W"][:2] = [.5, .475]  # Raw scores 10 and 9.5: wrong current action.
    model.metadata["label_best_epoch"] = 99  # A prior fit's metadata must not survive.
    return model


def test_projection_keeps_raw_student_baseline_and_cache_refresh_is_explicit(monkeypatch):
    calls = []
    original = fitting.prepare_calibration
    def counted(*args, **kwargs):
        calls.append(kwargs["config"])
        return original(*args, **kwargs)
    monkeypatch.setattr(fitting, "prepare_calibration", counted)
    roots = [root for root in _roots() if root["split"] == "val"]
    source = _source()
    model = TorchQ(source, "cpu")
    reference = list(torch_teacher_predictions(model, roots, torch.device("cpu")))
    settings = {"confidence_z": 1.5, "transfer_diagnostics": False}
    config = CalibrationConfig(trust_projection_z=1.5)
    data = fitting.CalibrationRoots(roots, 3, reference, settings, config)
    assert data.initial_action.tolist() == [0, 0]
    assert data.trust_reference.argmax(1).tolist() == [1, 1]
    baseline = data.reference.clone()
    first, choices = fitting.evaluate_calibration(model, data, settings, config)
    assert choices.tolist() == [0, 0]
    assert first["action_change_fraction"] == 0.
    assert first["paired_label_gain"] == 0.
    assert first["label_regret"] == pytest.approx(1.)
    assert first["fixed_reference_preparation"]["projected_choice_change_fraction"] == 1.
    assert first["fixed_reference_preparation"]["paired_supported_improvement_root_fraction"] == 1.
    fitting.evaluate_calibration(model, data, settings, config)
    assert len(calls) == 1
    old_reliability = data.pair_reliability.clone()
    revised = replace(config, noise_scale=.2)
    fitting.evaluate_calibration(model, data, settings, revised)
    assert len(calls) == 2
    assert torch.all(data.pair_reliability[:, 0, 1] > old_reliability[:, 0, 1])
    fitting.evaluate_calibration(model, data, settings, replace(revised, value_weight=.5))
    assert len(calls) == 2  # Loss coefficients do not invalidate fixed statistics.
    assert torch.equal(data.reference, baseline)
    assert data.initial_action.tolist() == [0, 0]
    # Refresh occurred inside an inference-mode evaluator. Its cached tensors
    # must still support saving constants in a later training backward pass.
    prediction = model(data.x.flatten(0, 1)).reshape_as(data.target)
    resumed_loss = calibration_loss_prepared(prediction, data.target, data.mask,
        data.pair_reliability, data.action_reliability,
        trust_reference=data.trust_reference, config=revised)
    resumed_loss.backward()
    assert torch.isfinite(model.weights["skip_W"].grad).all()


def test_projected_fit_prepares_only_two_splits_and_preserves_exact_preset_visits(monkeypatch):
    preparations = []
    original = fitting.prepare_calibration
    def counted(*args, **kwargs):
        preparations.append(args[0].shape)
        return original(*args, **kwargs)
    monkeypatch.setattr(fitting, "prepare_calibration", counted)
    source = _source().freeze()
    before = source.checksum()
    config = dict(epochs=2, batch_roots=4, learning_rate=1e-3, l2=0., patience=0,
        min_epochs=0, checkpoint_candidates=2, checkpoint_metric="label_regret",
        per_preset_teacher_bank=True, validation_unit="trajectory", seed=882,
        calibration_require_all_presets=False, calibration_visits_per_preset=4,
        calibration_sampling="coverage_without_replacement", calibration_current_fraction=.5,
        calibration={"trust_projection_z": 1.5}, training_round=9,
        initial_student_model_id="M000008", gpu_callback_interval_seconds=0.)
    result = fitting.fit_candidate_calibration(source, _roots(), config, device="cpu")
    assert len(preparations) == 2  # No per-epoch or per-step projection/preparation.
    assert source.checksum() == before
    assert result.model.checksum() != before
    assert result.optimizer_steps == 4
    assert result.model.metadata["label_best_epoch"] == result.best_epoch
    assert result.model.metadata["initial_student_model_id"] == "M000008"
    assert result.model.metadata["initial_parameter_sha256"] == before
    assert all(point["model"].metadata["label_best_epoch"] == result.best_epoch
               for point in result.candidate_checkpoints)
    for row in result.history[1:]:
        quota = row["epoch_quota"]
        assert quota["root_visits"] == quota["unique_roots_visited"] == 8
        assert quota["duplicate_visits"] == 0
        assert quota["current_root_visits"] == quota["unique_current_roots_visited"] == 4
        assert quota["presets_with_current_visits"] == 2
        assert quota["current_fraction_min"] == quota["current_fraction_max"] == .5


@pytest.mark.parametrize("size", [0, 1, 3, 6, 7, 11])
def test_coverage_full_and_partial_passes_respect_zero_and_extreme_priorities(size):
    local = np.arange(4, dtype=np.int64)
    probabilities = np.array([0., 1e-300, 1e300, 1.])
    order = fitting._weighted_coverage(local, probabilities, size, np.random.default_rng(52))
    repeated = fitting._weighted_coverage(local, probabilities, size, np.random.default_rng(52))
    np.testing.assert_array_equal(order, repeated)
    assert len(order) == size
    assert 0 not in order
    counts = Counter(order)
    full, partial = divmod(size, 3)
    assert all(counts[index] in (full, full+1) for index in (1, 2, 3))
    assert sum(counts[index] > full for index in (1, 2, 3)) == partial
    if partial:
        assert counts[2] == full+1


def test_partial_coverage_retains_weighted_draw_distribution():
    generator = np.random.default_rng(940)
    local, weights = np.arange(3), np.array([1., 2., 7.])
    draws = [fitting._weighted_coverage(local, weights, 1, generator)[0] for _ in range(6000)]
    proportions = np.bincount(draws, minlength=3)/len(draws)
    np.testing.assert_allclose(proportions, weights/weights.sum(), atol=.02, rtol=0.)


def test_empty_eligible_coverage_only_allows_zero_visits():
    local, weights = np.arange(3), np.zeros(3)
    assert len(fitting._weighted_coverage(local, weights, 0, np.random.default_rng(1))) == 0
    with pytest.raises(ValueError, match="positive mass"):
        fitting._weighted_coverage(local, weights, 1, np.random.default_rng(1))
