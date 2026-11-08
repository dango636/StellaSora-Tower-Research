"""Integration risks: frozen teacher, preset mass, exported model, padding."""
from collections import Counter
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.action_calibration import CalibrationConfig
from stella.backend import fit_candidate
from stella.fit import _settings, prepare_roots
from stella.fit_calibration import (CalibrationRoots, evaluate_calibration,
                                   exact_preset_epoch, fit_candidate_calibration)
from stella.fit_torch import TorchQ, torch_teacher_predictions
from stella.network import QNetwork

torch.set_num_threads(2)


def synthetic_roots():
    """Full 958-feature interface, unequal families and action counts.

    Return columns share a real common fluctuation; paired uncertainty is zero
    although marginal action variance is positive. Features are synthetic, so
    this fixture checks the trainer contract, not the game's transition kernel.
    """
    roots = []
    for preset in range(3):
        for split, families in (("train", 1+preset), ("val", 1)):
            for family in range(families):
                actions = [1, 2, 4][preset]
                x = np.zeros((actions, 958), dtype=np.float32)
                x[:, preset] = 1.
                x[:, 9] = .1*(family+1)
                x[np.arange(actions), 40+np.arange(actions)] = 1.
                q = np.array([1., 1.25, .75, 1.10][:actions], dtype=np.float32)
                ret = q[:, None] + np.array([-.25, .25, -.25, .25], dtype=np.float32)
                seed = preset*100 + family*10 + (1 if split == "train" else 5)
                roots.append(dict(x=x, q=q, action_returns=ret,
                    family_id=f"p{preset}_{split}_f{family}",
                    root_id=f"p{preset}_{split}_f{family}_root00",
                    teacher_id=f"M{preset+10:06d}", catalogue_id=preset,
                    # Deliberately not catalogue groups: the new fitter must
                    # derive equal preset masses from explicit catalogue IDs.
                    preset_group="old_shared_bucket", split=split, cap=90,
                    source_environment_seed=seed, source_exploration_seed=seed+1000,
                    label_seed=seed+10000, teacher_action_index=min(1, actions-1),
                    student_action_index=-1, behavior_source="teacher"))
    return roots


def settings(**extra):
    return dict(epochs=2, batch_roots=4, learning_rate=3e-3, l2=0.,
        patience=0, min_epochs=0, checkpoint_metric="label_regret",
        checkpoint_candidates=2, preserve_early_checkpoints=True,
        per_preset_teacher_bank=True, validation_unit="trajectory",
        calibration_require_all_presets=False, calibration_visits_per_preset=3,
        initial_student_model_id="M000000", training_round=73, seed=20261106,
        gpu_callback_interval_seconds=0., **extra)


def test_cpu_fit_updates_only_student_and_exports_actual_checkpoints(tmp_path):
    source = QNetwork(input_dim=958, hidden=(12, 8), seed=5, heuristic_init=False).freeze()
    before = source.checksum()
    roots = synthetic_roots()
    copied = copy.deepcopy(roots)
    result = fit_candidate_calibration(source, roots, settings(), device="cpu")
    assert source.checksum() == before
    assert all(not p.flags.writeable for p in source.params.values())
    assert result.model.checksum() != before
    assert result.best_epoch in (1, 2)
    assert result.optimizer_steps == 6  # 3 presets * 3 visits; batch 4 => 3 steps/epoch.
    assert result.initial_parameter_sha256 == before
    assert result.candidate_checkpoints
    assert all(p["epoch"] >= 1 and p["checksum"] != before for p in result.candidate_checkpoints)
    for row in result.history:
        for split in ("train", "val"):
            assert row[split]["presets"] == 3
            assert np.isfinite([row[split][k] for k in ("selection_loss", "q_rmse",
                "advantage_rmse", "label_regret", "paired_label_gain")]).all()
        if row["epoch"]:
            assert row["epoch_quota"]["equal_preset_epoch_quota"]
            assert row["epoch_quota"]["root_visits"] == 9
    # Initialization may refresh mutable fit copies, never archived caller rows.
    for original, snapshot in zip(roots, copied):
        assert original["student_action_index"] == snapshot["student_action_index"] == -1
        np.testing.assert_array_equal(original["action_returns"], snapshot["action_returns"])
        np.testing.assert_array_equal(original["x"], snapshot["x"])
    inputs = np.concatenate([root["x"] for root in roots])
    for checkpoint in result.candidate_checkpoints:
        model = checkpoint["model"]
        path = model.save(tmp_path/f"epoch_{checkpoint['epoch']}.npz")
        reloaded = QNetwork.load(path)
        assert reloaded.checksum() == checkpoint["checksum"]
        np.testing.assert_allclose(reloaded.predict(inputs), model.predict(inputs), atol=1e-6, rtol=1e-6)
        assert reloaded.metadata["training_backend"]["backend"] == "torch_cpu"
        assert "calibration" in reloaded.metadata["training_backend"]


def test_production_default_requires_all_presets_in_both_splits():
    config = settings()
    config.pop("calibration_require_all_presets")
    source = QNetwork(input_dim=958, hidden=(8,), heuristic_init=False)
    with pytest.raises(ValueError, match="4960 presets"):
        fit_candidate_calibration(source, synthetic_roots(), config, device="cpu")


def test_new_objective_dispatches_to_real_cpu_torch_fit():
    source = QNetwork(input_dim=958, hidden=(8,), heuristic_init=False)
    result = fit_candidate(source, synthetic_roots(),
        settings(backend="numpy", objective_version="action_gap_calibration_v3"))
    assert result.model.metadata["objective_version"] == "action_gap_calibration_v3"
    assert result.history[0]["backend"] == "torch_cpu"
    assert result.model.checksum() != source.checksum()


def test_exact_preset_quota_survives_unequal_families_and_extreme_priorities():
    roots = [r for r in synthetic_roots() if r["split"] == "train"]
    priorities = np.array([1e6 if r["catalogue_id"] == 2 else 1. for r in roots])
    order, report = exact_preset_epoch(roots, priorities, np.random.default_rng(2), 7)
    counts = Counter(roots[i]["catalogue_id"] for i in order)
    assert counts == {0: 7, 1: 7, 2: 7}
    assert report["root_visits"] == 21
    repeat, _ = exact_preset_epoch(roots, priorities, np.random.default_rng(2), 7)
    np.testing.assert_array_equal(order, repeat)


def test_masked_padding_cannot_change_evaluation_metrics():
    roots = synthetic_roots()
    # Same grouping the fitter uses before preparing roots.
    roots = [dict(r, preset_group=f"catalogue_{r['catalogue_id']:04d}") for r in roots]
    cfg = _settings(settings())
    train, _, _ = prepare_roots(roots, 958, cfg)
    source = QNetwork(input_dim=958, hidden=(8,), seed=3, heuristic_init=False)
    source.params["head_W"][:] = .01
    model = TorchQ(source, "cpu")
    reference = list(torch_teacher_predictions(model, train, torch.device("cpu")))
    data = CalibrationRoots(train, 5, reference, cfg)
    initial, _ = evaluate_calibration(model, data, cfg, CalibrationConfig())
    data.x[~data.mask] = 1e6
    data.target[~data.mask] = float("nan")
    data.reference[~data.mask] = float("nan")
    after, _ = evaluate_calibration(model, data, cfg, CalibrationConfig())
    for key in ("selection_loss", "q_rmse", "advantage_rmse", "label_regret", "paired_label_gain"):
        assert np.isfinite(after[key])
        assert after[key] == pytest.approx(initial[key], rel=1e-6, abs=1e-8)


def test_margin_diagnostics_exclude_single_actions_and_do_not_change_decisions():
    roots = [dict(r, preset_group=f"catalogue_{r['catalogue_id']:04d}")
             for r in synthetic_roots()]
    cfg = _settings(settings())
    train, _, _ = prepare_roots(roots, 958, cfg)
    source = QNetwork(input_dim=958, hidden=(8,), seed=3, heuristic_init=False)
    source.params["head_W"][:] = 0.
    source.params["head_b"][:] = 0.
    source.params["skip_W"][:] = 0.
    source.params["skip_W"][41] = 1e-6/20
    source.params["skip_W"][42] = .01/20
    source.params["skip_W"][43] = .03/20
    model = TorchQ(source, "cpu")
    reference = list(torch_teacher_predictions(model, train, torch.device("cpu")))
    data = CalibrationRoots(train, 4, reference, cfg)
    result, choices = evaluate_calibration(model, data, cfg, CalibrationConfig())
    assert result["decision_margin_eligible_mass"] == pytest.approx(2/3)
    assert result["numeric_near_tie_fraction"] == pytest.approx(.5)
    assert result["small_margin_fraction"] == pytest.approx(.5)
    assert result["numeric_near_tie_threshold_raw"] == 1e-5
    expected = [int(np.argmax(values)) for values in reference]
    np.testing.assert_array_equal(choices, expected)


def test_stopped_before_training_cannot_create_a_trained_checkpoint():
    class Stopped:
        @staticmethod
        def is_set():
            return True
    source = QNetwork(input_dim=958, hidden=(8,), heuristic_init=False)
    result = fit_candidate_calibration(source, synthetic_roots(), settings(),
                                       stop_event=Stopped(), device="cpu")
    assert result.interrupted
    assert result.best_epoch == 0
    assert result.optimizer_steps == 0
    assert not result.candidate_checkpoints
    assert not result.model.metadata["trained_candidate"]
