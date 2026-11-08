"""Learning through small action margins without changing candidate selection."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.action_calibration import CalibrationConfig, calibration_loss, prepare_calibration
from stella.config import resolve
import stella.fit_calibration as fitting
from stella.network import QNetwork


def _roots():
    result = []
    for preset in range(2):
        for split in ("train", "val"):
            seed = preset*10 + (1 if split == "train" else 2)
            result.append(dict(x=np.eye(2, dtype=np.float32), q=np.array([10., 10.04], np.float32),
                pair_se=np.array([[0., .2], [.2, 0.]], np.float32),
                family_id=f"p{preset}_{split}", preset_group=f"p{preset}", catalogue_id=preset,
                split=split, teacher_id=f"M{preset+10:06d}",
                source_environment_seed=seed, label_seed=10000+seed,
                teacher_action_index=1, behavior_source="student"))
    return result


def _source():
    model = QNetwork(input_dim=2, hidden=(4,), heuristic_init=False, seed=8)
    model.params["skip_W"][:] = [.5, .495]
    return model.freeze()


def _settings(**changes):
    result = dict(epochs=8, batch_roots=2, learning_rate=1e-5, l2=0., patience=2,
        min_epochs=0, checkpoint_candidates=6, checkpoint_metric="label_regret",
        per_preset_teacher_bank=True, validation_unit="trajectory", seed=1882,
        calibration_require_all_presets=False, calibration_visits_per_preset=1,
        calibration_sampling="coverage_without_replacement",
        calibration={"uncertainty_trust_weight": 0., "reliability_floor": .25},
        gpu_callback_interval_seconds=0.)
    result.update(changes)
    return result


def test_known_pair_floor_keeps_unknown_padding_and_self_comparisons_zero():
    target = torch.tensor([[10., 10.04, 10.02, float("nan")]])
    mask = torch.tensor([[True, True, True, False]])
    known = torch.ones((1, 4, 4), dtype=torch.bool)
    known[:, 0, 2] = known[:, 2, 0] = False
    se = torch.full((1, 4, 4), .2)
    se[~known] = float("nan")
    base = prepare_calibration(target, mask, se, known, config=CalibrationConfig())
    revised = prepare_calibration(target, mask, se, known,
        config=CalibrationConfig(reliability_floor=.25))
    r, evidence = revised["pair_reliability"], revised["pair_known"]
    assert torch.all(r[~evidence] == 0.)
    torch.testing.assert_close(r[evidence], .25+.75*base["pair_reliability"][evidence])
    assert torch.all(r[evidence] >= .25)
    full = prepare_calibration(target, mask, se, known,
        config=CalibrationConfig(reliability_floor=1.))
    assert torch.equal(full["pair_reliability"], evidence.float())


def test_zero_floor_and_default_stopping_are_exact_opt_outs():
    source = _source()
    implicit = _settings(calibration={"uncertainty_trust_weight": 1.}, epochs=3)
    explicit = dict(implicit, calibration={"uncertainty_trust_weight": 1., "reliability_floor": 0.},
                    calibration_stopping_metric="checkpoint_metric")
    first = fitting.fit_candidate_calibration(source, _roots(), implicit, device="cpu")
    second = fitting.fit_candidate_calibration(source, _roots(), explicit, device="cpu")
    assert first.model.checksum() == second.model.checksum()
    assert first.best_epoch == second.best_epoch
    assert first.optimizer_steps == second.optimizer_steps
    assert [p["checksum"] for p in first.candidate_checkpoints] == [p["checksum"] for p in second.candidate_checkpoints]
    for a, b in zip(first.history, second.history):
        assert a["train"] == b["train"] and a["val"] == b["val"]
        assert a["stopping_stale_epochs"] == b["stopping_stale_epochs"]


def test_zero_trust_removes_old_gap_preference_with_known_weak_evidence():
    target = torch.tensor([[0., .04]])
    prediction = torch.tensor([[0., -.02/20.]], requires_grad=True)
    mask = torch.ones((1, 2), dtype=torch.bool)
    se = torch.full((1, 2, 2), .4)
    known = torch.ones((1, 2, 2), dtype=torch.bool)
    cfg = CalibrationConfig(reliability_floor=.25, uncertainty_trust_weight=0., value_weight=0.)
    gradients = []
    for old_gap in (-10., 0., 10.):
        p = prediction.detach().clone().requires_grad_()
        loss = calibration_loss(p, target, mask, se, known,
            reference_prediction=torch.tensor([[0., old_gap/20.]]), config=cfg)
        loss.backward()
        gradients.append(p.grad)
    assert all(torch.equal(gradients[0], grad) for grad in gradients[1:])
    assert gradients[0][0, 1] < 0.  # Gradient descent raises the better action.


def test_changing_floor_refreshes_fixed_cache_but_not_original_student():
    roots = _roots()[:1]
    config = CalibrationConfig()
    data = fitting.CalibrationRoots(roots, 2, [np.array([.5, .495], np.float32)], {}, config)
    original = data.reference.clone()
    initial = data.pair_reliability.clone()
    data.prepare_loss(replace(config, reliability_floor=.25))
    assert torch.all(data.pair_reliability[data.pair_known] > initial[data.pair_known])
    assert torch.equal(data.reference, original)
    diagnostic = data.preparation_diagnostics
    assert diagnostic["initial_anchor_reliability_mass_after_floor"] > diagnostic["initial_anchor_reliability_mass_before_floor"]


def test_reusing_dataset_with_trust_disabled_refreshes_diagnostic_without_rebuilding():
    from stella.fit_torch import TorchQ
    roots = _roots()[:1]
    config = CalibrationConfig()
    data = fitting.CalibrationRoots(roots, 2, [np.array([.5, .495], np.float32)], {}, config)
    model = TorchQ(_source(), "cpu")
    with torch.no_grad():
        model.weights["skip_W"][1] = .52
    before, before_choices = fitting.evaluate_calibration(model, data, {}, config)
    cached_pointers = (data.pair_reliability.data_ptr(), data.action_reliability.data_ptr(),
                       data.trust_reference.data_ptr())
    after, after_choices = fitting.evaluate_calibration(model, data, {},
        replace(config, uncertainty_trust_weight=0.))
    assert before["trust_loss"] > 0. and after["trust_loss"] == 0.
    assert before["fixed_reference_preparation"]["uncertainty_trust_active"] is True
    assert after["fixed_reference_preparation"]["uncertainty_trust_active"] is False
    assert cached_pointers == (data.pair_reliability.data_ptr(), data.action_reliability.data_ptr(),
                               data.trust_reference.data_ptr())
    assert np.array_equal(before_choices, after_choices)


def _controlled_validation(monkeypatch, advantage, regret):
    original = fitting.evaluate_calibration
    calls = 0
    def evaluate(model, data, settings, config):
        nonlocal calls
        metrics, choices = original(model, data, settings, config)
        if data.roots[0]["split"] == "val":
            metrics["advantage_rmse"] = float(advantage(calls))
            metrics["label_regret"] = float(regret(calls))
            calls += 1
        return metrics, choices
    monkeypatch.setattr(fitting, "evaluate_calibration", evaluate)


def test_smooth_patience_crosses_no_flip_plateau_without_changing_selection():
    result = fitting.fit_candidate_calibration(_source(), _roots(),
        _settings(calibration_stopping_metric="advantage_rmse"), device="cpu")
    assert result.history[-1]["epoch"] == 8
    assert result.best_epoch == 1  # Candidate selection is still label_regret.
    assert all(row["selection_metric"] == "label_regret" for row in result.history)
    assert all(row["val"]["action_change_fraction"] == 0. for row in result.history)
    assert all(row["stopping_stale_epochs"] == 0 for row in result.history)
    assert result.history[-1]["stopping_best_epoch"] == 8
    errors = [row["val"]["advantage_rmse"] for row in result.history]
    assert all(after < before for before, after in zip(errors, errors[1:]))


def test_legacy_patience_still_stops_discrete_plateau():
    result = fitting.fit_candidate_calibration(_source(), _roots(), _settings(), device="cpu")
    assert result.history[-1]["epoch"] == 3
    assert result.history[-1]["stopping_metric"] == "label_regret"
    assert result.history[-1]["stopping_stale_epochs"] == 2
    assert result.history[-1]["val"]["advantage_rmse"] < result.history[-2]["val"]["advantage_rmse"]


def test_smooth_patience_still_stops_validation_overfit(monkeypatch):
    curve = [1.2, 1., .8, .7, .75, .8]
    _controlled_validation(monkeypatch, lambda epoch: curve[epoch], lambda epoch: 1./(1+epoch))
    result = fitting.fit_candidate_calibration(_source(), _roots(),
        _settings(calibration_stopping_metric="advantage_rmse"), device="cpu")
    assert result.history[-1]["epoch"] == 5
    assert result.best_epoch == 5  # Independent discrete selection kept improving.
    assert result.history[-1]["stopping_best_epoch"] == 3
    assert result.history[-1]["stopping_best_value"] == .7
    assert result.history[-1]["stopping_stale_epochs"] == 2


def test_real_validation_error_stops_while_training_error_keeps_improving():
    # Identical features with independent noisy labels make a small, concrete
    # overfit case. No evaluator or metric is mocked in this test.
    roots = _roots()
    for root in roots:
        if root["split"] == "val":
            root["q"][1] = 9.96
    result = fitting.fit_candidate_calibration(_source(), roots,
        _settings(calibration_stopping_metric="advantage_rmse", epochs=20, learning_rate=.0005),
        device="cpu")
    assert result.history[-1]["epoch"] < 20
    assert result.history[-1]["stopping_stale_epochs"] == 2
    assert result.history[-1]["stopping_best_epoch"] < result.history[-1]["epoch"]
    assert result.history[-1]["train"]["advantage_rmse"] < result.history[-2]["train"]["advantage_rmse"]
    assert result.history[-1]["val"]["advantage_rmse"] > result.history[-2]["val"]["advantage_rmse"]


def test_hybrid_keeps_action_metric_progress_when_rmse_worsens(monkeypatch):
    resolved = resolve({"training": {"calibration_stopping_metric": "checkpoint_or_advantage_rmse"}})
    mode = resolved["training"]["calibration_stopping_metric"]
    curve = [1.2, 1., .8, .7, .75, .8, .85, .9, .95]
    _controlled_validation(monkeypatch, lambda epoch: curve[epoch], lambda epoch: 1./(1+epoch))
    result = fitting.fit_candidate_calibration(_source(), _roots(),
        _settings(calibration_stopping_metric=mode), device="cpu")
    last = result.history[-1]
    assert last["epoch"] == result.best_epoch == 8
    assert all(row["stopping_stale_epochs"] == 0 for row in result.history)
    assert last["stopping_best_epochs"] == {"label_regret": 8, "advantage_rmse": 3}
    assert last["stopping_best_values"] == {"label_regret": 1./9, "advantage_rmse": .7}
    assert last["stopping_improved_metrics"] == ["label_regret"]
    assert last["stopping_value"] is None and last["stopping_best_value"] is None
    assert last["stopping_best_epoch"] == 8
    assert all(row["selection_metric"] == "label_regret" for row in result.history)


def test_hybrid_keeps_real_rmse_progress_before_any_action_changes():
    result = fitting.fit_candidate_calibration(_source(), _roots(),
        _settings(calibration_stopping_metric="checkpoint_or_advantage_rmse"), device="cpu")
    last = result.history[-1]
    assert last["epoch"] == 8 and result.best_epoch == 1
    assert all(row["val"]["action_change_fraction"] == 0. for row in result.history)
    assert all(row["stopping_stale_epochs"] == 0 for row in result.history)
    errors = [row["val"]["advantage_rmse"] for row in result.history]
    assert all(after < before for before, after in zip(errors, errors[1:]))
    assert last["stopping_best_epochs"] == {"label_regret": 1, "advantage_rmse": 8}
    assert last["stopping_improved_metrics"] == ["advantage_rmse"]


def test_hybrid_stops_real_overfit_after_both_metrics_stop_improving():
    roots = _roots()
    for root in roots:
        if root["split"] == "val":
            root["q"][1] = 9.96
    result = fitting.fit_candidate_calibration(_source(), roots,
        _settings(calibration_stopping_metric="checkpoint_or_advantage_rmse",
                  epochs=20, learning_rate=.0005), device="cpu")
    last = result.history[-1]
    assert last["epoch"] == 4 and last["stopping_stale_epochs"] == 2
    assert last["stopping_best_epochs"] == {"label_regret": 1, "advantage_rmse": 2}
    assert last["stopping_improved_metrics"] == []
    assert last["train"]["advantage_rmse"] < result.history[-2]["train"]["advantage_rmse"]
    assert last["val"]["advantage_rmse"] > result.history[-2]["val"]["advantage_rmse"]
    assert result.model.metadata["stopping_best_values"] == last["stopping_best_values"]
    assert all(point["model"].metadata["stopping_best_epochs"] == last["stopping_best_epochs"]
               for point in result.candidate_checkpoints)


def test_hybrid_baseline_interrupt_serializes_undefined_bests_as_null():
    source = _source()
    result = fitting.fit_candidate_calibration(source, _roots(),
        _settings(calibration_stopping_metric="checkpoint_or_advantage_rmse"),
        callback=lambda row: False, device="cpu")
    undefined = {"label_regret": None, "advantage_rmse": None}
    assert result.interrupted and result.optimizer_steps == 0
    assert result.model.checksum() == source.checksum()
    assert not result.candidate_checkpoints
    assert result.history[0]["stopping_best_values"] == undefined
    assert result.history[0]["stopping_best_epochs"] == undefined
    assert result.model.metadata["stopping_best_values"] == undefined
    assert result.model.metadata["stopping_best_epochs"] == undefined
    json.dumps(result.summary(), allow_nan=False)
    json.dumps(result.model.metadata, allow_nan=False)


@pytest.mark.parametrize("floor", [-.01, 1.01, True, float("nan"), float("inf")])
def test_invalid_reliability_floor_is_rejected(floor):
    with pytest.raises(ValueError, match="reliability_floor"):
        CalibrationConfig(reliability_floor=floor)


def test_new_config_options_do_not_appear_in_legacy_resolved_config():
    legacy = resolve({"training": {"calibration": {"noise_scale": .1}}})
    assert "reliability_floor" not in legacy["training"]["calibration"]
    assert "calibration_stopping_metric" not in legacy["training"]
    assert "learner_initialization" not in legacy["training"]
    revised = resolve({"catalogue_mode": True, "training": {
        "calibration": {"reliability_floor": .25},
        "calibration_stopping_metric": "advantage_rmse", "learner_initialization": "last_candidate"}})
    assert revised["training"]["calibration"]["reliability_floor"] == .25
    assert revised["training"]["learner_initialization"] == "last_candidate"


@pytest.mark.parametrize("raw", [
    {"training": {"calibration_stopping_metric": "label_regret"}},
    {"training": {"learner_initialization": "unknown"}},
    {"training": {"learner_initialization": "last_candidate"}},
    {"catalogue_mode": True, "workflow": {"protocol": "paired_anchor_v2"},
     "training": {"learner_initialization": "last_candidate"}},
])
def test_invalid_long_horizon_config_is_rejected(raw):
    with pytest.raises(ValueError):
        resolve(raw)
