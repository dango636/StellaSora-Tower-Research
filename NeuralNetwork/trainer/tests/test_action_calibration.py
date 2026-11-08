"""Behavioral tests for decision-gap calibration, without game-rule changes."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.action_calibration import (CalibrationConfig, batch_pair_moments,
                                      calibration_loss, root_pair_moments,
                                      _project_reference_q)


def loss_for(raw_prediction, raw_target, *, se=0.0, known=True, config=None,
             mask=None, teachers=None):
    prediction = torch.tensor(raw_prediction, dtype=torch.float32)/20.0
    prediction.requires_grad_()
    target = torch.tensor(raw_target, dtype=torch.float32)
    valid = torch.ones_like(target, dtype=torch.bool) if mask is None else torch.tensor(mask, dtype=torch.bool)
    n, a = target.shape
    pair_se = torch.full((n, a, a), se, dtype=torch.float32)
    pair_known = torch.full((n, a, a), known, dtype=torch.bool)
    teacher = None if teachers is None else torch.tensor(teachers)
    settings = {"value_weight": 0.0, "shape_weight": 0.0, **(config or {})}
    loss, details = calibration_loss(prediction, target, valid, pair_se, pair_known,
        teacher_actions=teacher, config=settings, return_details=True)
    loss.backward()
    return loss.detach(), prediction.grad, details


def test_shared_rollout_noise_cancels_in_paired_uncertainty():
    values = np.array([[1, 21, 41, 61], [4, 24, 44, 64]], dtype=float)
    m = root_pair_moments({"q": values.mean(1), "action_returns": values})
    assert m["pair_se"][0, 1] == pytest.approx(0.0)
    assert m["pair_known"][0, 1]
    assert m["pair_mean"][0, 1] == pytest.approx(-3.0)


def test_one_draw_or_only_marginal_se_is_explicitly_unknown():
    for root in ({"q": [1., 2.], "returns": [[1.], [2.]]},
                 {"q": [1., 2.], "standard_error": [0., 0.]}):
        m = root_pair_moments(root)
        assert not m["pair_known"].any()
        assert np.isnan(m["pair_se"][0, 1])


def test_precomputed_pair_table_and_archive_alias_are_supported():
    pre = root_pair_moments({"q": [1., 2.], "pair_se": [[0., .2], [.2, 0.]],
                            "teacher_action_index": 1})
    assert pre["pair_known"][0, 1]
    assert pre["teacher_action"] == 1
    pack = batch_pair_moments([{"q": [1., 2.], "returns": [[0., 2.], [1., 3.]]},
                              {"q": [4.], "returns": [[4., 4.]]}], max_actions=3)
    assert pack["target"].shape == (2, 3)
    assert pack["mask"].sum(axis=1).tolist() == [2, 1]
    assert not pack["pair_known"][:, 2].any()


@pytest.mark.parametrize("root", [
    {"q": [1., 2.], "returns": [[0., 0.], [1., 1.]]},
    {"q": [1., 2.], "pair_se": [[0., .2], [.3, 0.]]},
    {"q": [1., 2.], "pair_se": [[0., -.1], [-.1, 0.]]},
    {"q": [1., 2.], "teacher_action_index": 2},
])
def test_inconsistent_moments_are_rejected(root):
    with pytest.raises(ValueError):
        root_pair_moments(root)


def test_wrong_current_decision_gets_corrective_gradient():
    _, gradient, _ = loss_for([[10., 8.]], [[8., 10.]])
    # Gradient descent lowers the currently chosen bad action and raises the good one.
    assert gradient[0, 0] > 0
    assert gradient[0, 1] < 0
    assert gradient.sum() == pytest.approx(0., abs=1e-8)


def test_oversized_gap_is_pulled_back_to_finite_target():
    _, gradient, _ = loss_for([[0., 4.]], [[0., 1.]])
    assert gradient[0, 1] > 0
    assert gradient[0, 0] < 0


def test_small_uncertain_gap_has_soft_nonzero_signal_without_hard_threshold():
    loss, gradient, details = loss_for([[1.01, 1.]], [[1., 1.02]], se=.2)
    assert loss > 0
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0] > 0
    assert details["evidence_mass"] == pytest.approx(.2, abs=1e-6)


def test_unknown_uncertainty_never_becomes_zero_se_confidence():
    loss, gradient, details = loss_for([[2., 1.]], [[1., 2.]], se=0., known=False)
    assert loss == 0
    assert torch.count_nonzero(gradient) == 0
    assert details["unknown_pair_prior_mass"] == pytest.approx(1.)


def test_all_noisy_pairs_are_not_renormalized_to_full_strength():
    clear, _, _ = loss_for([[2., 1., .5]], [[1., 2., 1.5]], se=0.)
    noisy, _, details = loss_for([[2., 1., .5]], [[1., 2., 1.5]], se=.3)
    assert noisy/clear == pytest.approx(.1, abs=1e-6)
    assert details["evidence_mass"] == pytest.approx(.1, abs=1e-6)


def test_common_shift_changes_no_decision_loss_or_gradient():
    a, ga, _ = loss_for([[2., 1., 0.], [3., 2., 1.]], [[1., 2., .5], [3.5, 1., 2.]],
                       config={"shape_weight": .05})
    b, gb, _ = loss_for([[52., 51., 50.], [-7., -8., -9.]], [[1., 2., .5], [3.5, 1., 2.]],
                       config={"shape_weight": .05})
    assert a == pytest.approx(float(b), rel=1e-5, abs=1e-8)
    assert torch.allclose(ga, gb, atol=2e-7, rtol=1e-5)


def test_value_anchor_only_has_equal_gradients_for_same_state_actions():
    _, gradient, _ = loss_for([[3., 2., 1.]], [[1., 2., .5]],
        config={"pair_weight": 0., "value_weight": .02})
    assert torch.allclose(gradient, gradient[:, :1].expand_as(gradient), atol=1e-8)


def test_padding_nan_and_extreme_values_do_not_affect_loss_or_gradients():
    a, ga, _ = loss_for([[2., 1.]], [[1., 2.]])
    b, gb, _ = loss_for([[2., 1., float('nan'), 1e20]], [[1., 2., float('nan'), -1e20]],
                       mask=[[True, True, False, False]])
    assert a == pytest.approx(float(b), abs=1e-8)
    assert torch.allclose(ga, gb[:, :2], atol=1e-8)
    assert torch.count_nonzero(gb[:, 2:]) == 0


def test_equal_targets_and_predictions_have_no_arbitrary_tie_margin():
    loss, gradient, details = loss_for([[2., 2., 2.]], [[2., 2., 2.]], teachers=[-1])
    assert details["anchor"].tolist() == [0]
    assert loss == 0
    assert torch.count_nonzero(gradient) == 0


def test_root_weights_remain_outer_masses_without_chunk_renormalization():
    p = torch.tensor([[.10, .05], [.10, .05]], requires_grad=True)
    q = torch.tensor([[1., 2.], [1., 2.]])
    mask = torch.ones_like(q, dtype=torch.bool)
    se = torch.zeros((2, 2, 2))
    known = torch.ones((2, 2, 2), dtype=torch.bool)
    full = calibration_loss(p, q, mask, se, known)
    weighted = calibration_loss(p, q, mask, se, known, root_weights=torch.tensor([.1, .2]))
    assert float(weighted.detach()) == pytest.approx(float(full.detach()*.3), rel=1e-6)


def test_unknown_evidence_trust_penalty_preserves_initial_gap_without_hard_actions():
    prediction = torch.tensor([[.10, .11]], requires_grad=True)
    reference = torch.tensor([[.10, .09]], requires_grad=True)
    target = torch.tensor([[1., 9.]])
    mask = torch.ones_like(target, dtype=torch.bool)
    loss, details = calibration_loss(prediction, target, mask,
        torch.full((1, 2, 2), float('nan')), torch.zeros((1, 2, 2), dtype=torch.bool),
        reference_prediction=reference,
        config={"value_weight": 0., "shape_weight": 0., "uncertainty_trust_weight": .25},
        return_details=True)
    loss.backward()
    assert details["pair_raw"] == 0
    assert details["trust_raw"] > 0
    assert prediction.grad[0, 0] < 0
    assert prediction.grad[0, 1] > 0
    assert reference.grad is None


def test_clear_evidence_does_not_get_uncertainty_trust_penalty():
    prediction = torch.tensor([[.10, .11]], requires_grad=True)
    reference = torch.tensor([[.10, .09]])
    target = torch.tensor([[1., 9.]])
    mask = torch.ones_like(target, dtype=torch.bool)
    _, details = calibration_loss(prediction, target, mask,
        torch.zeros((1, 2, 2)), torch.ones((1, 2, 2), dtype=torch.bool),
        reference_prediction=reference, return_details=True)
    assert details["trust_raw"] == 0


def _two_action_trust_at_gap(gap, projection_z, *, known=True, se=.1):
    prediction = torch.tensor([[0., gap]], dtype=torch.float32)/20.
    prediction.requires_grad_()
    target = torch.tensor([[0., .2]])
    reference = torch.tensor([[0., -.5]], requires_grad=True)/20.
    reference.retain_grad()
    loss, details = calibration_loss(prediction, target, torch.ones((1,2), dtype=torch.bool),
        torch.full((1,2,2), se), torch.full((1,2,2), known, dtype=torch.bool),
        reference_prediction=reference,
        config=dict(value_weight=0., shape_weight=0., pair_huber_delta=1.,
                    uncertainty_trust_weight=1., trust_projection_z=projection_z),
        return_details=True)
    loss.backward()
    assert reference.grad is None
    return loss.detach(), prediction.grad, details


def test_projected_reference_releases_analytically_locked_two_action_error():
    # r=.5, y=.2, d0=-.5, lambda=1. The ordinary quadratic optimum is -.15,
    # still wrong. Projecting d0 to y-1.96*.1=.004 gives optimum +.102.
    _, old_gradient, old_details = _two_action_trust_at_gap(-.15, 0.)
    _, projected_gradient, projected_details = _two_action_trust_at_gap(-.15, 1.96)
    _, optimum_gradient, optimum_details = _two_action_trust_at_gap(.102, 1.96)
    assert old_gradient.abs().max() < 2e-8
    assert projected_gradient[0, 1] < 0
    assert optimum_gradient.abs().max() < 2e-8
    assert old_details["anchor"].tolist() == [0]
    assert optimum_details["anchor"].tolist() == [1]
    # Only the trust reference changes; the mean difference label is untouched.
    assert torch.equal(old_details["pair_raw"], projected_details["pair_raw"])


def test_projection_makes_one_coherent_q_vector_from_fixed_initial_argmax():
    reference = torch.tensor([[0., -.5, -.1, -.3]], requires_grad=True)
    target = torch.tensor([[0., .2, -.5, 10.]])
    valid = torch.ones_like(target, dtype=torch.bool)
    se = torch.full((1,4,4), .1)
    known = torch.ones((1,4,4), dtype=torch.bool)
    known[:, 3, :] = False
    known[:, :, 3] = False
    known[:, torch.arange(4), torch.arange(4)] = False
    projected = _project_reference_q(reference, target, valid, se, known, 1.96)
    assert not projected.requires_grad
    assert torch.allclose(projected, torch.tensor([[0., .004, -.304, -.3]]), atol=2e-8)
    # The projected winner becomes action 1, but the origin stays the initial 0.
    # Differences from either current anchor come from exactly the same vector.
    gap_from_0 = projected-projected[:, :1]
    gap_from_1 = projected-projected[:, 1:2]
    assert torch.allclose(gap_from_1, gap_from_0-gap_from_0[:, 1:2], atol=1e-8)
    # No evidence to initial action 0 means action 3's reference Q stays exact.
    assert projected[0, 3] == reference.detach()[0, 3]


def test_zero_se_projection_and_inside_interval_behavior():
    reference = torch.tensor([[1., .5, .9]])
    target = torch.tensor([[0., .2, -.15]])
    valid = torch.ones_like(target, dtype=torch.bool)
    se = torch.full((1,3,3), .1)
    se[:, 0, 1] = se[:, 1, 0] = 0.
    known = ~torch.eye(3, dtype=torch.bool)[None]
    projected = _project_reference_q(reference, target, valid, se, known, 1.96)
    assert projected[0, 1] == pytest.approx(1.2)
    assert projected[0, 2] == reference[0, 2]  # Already inside the interval.
    old_loss, old_gradient, _ = _two_action_trust_at_gap(-.15, 0., se=0.)
    new_loss, new_gradient, _ = _two_action_trust_at_gap(-.15, 1.96, se=0.)
    assert torch.equal(old_loss, new_loss)
    assert torch.equal(old_gradient, new_gradient)


def test_projection_initial_tie_uses_first_legal_action_as_fixed_origin():
    reference = torch.tensor([[0., 0., -.5]])
    target = torch.tensor([[0., .2, -.1]])
    valid = torch.ones_like(target, dtype=torch.bool)
    known = torch.zeros((1,3,3), dtype=torch.bool)
    known[:, 0, 2] = known[:, 2, 0] = True
    projected = _project_reference_q(reference, target, valid,
        torch.full((1,3,3), .1), known, 1.96)
    assert torch.allclose(projected, torch.tensor([[0., 0., -.296]]), atol=2e-8)


def test_unknown_projection_preserves_old_reference_without_fabricating_evidence():
    old_loss, old_gradient, _ = _two_action_trust_at_gap(-.15, 0., known=False, se=float('nan'))
    new_loss, new_gradient, details = _two_action_trust_at_gap(-.15, 1.96, known=False, se=float('nan'))
    assert torch.equal(old_loss, new_loss)
    assert torch.equal(old_gradient, new_gradient)
    assert details["pair_raw"] == 0


def test_projected_reference_padding_has_no_contribution():
    def run(padded):
        p = [[0., -.15, float('nan')]] if padded else [[0., -.15]]
        q = [[0., .2, float('nan')]] if padded else [[0., .2]]
        r = [[0., -.5, float('nan')]] if padded else [[0., -.5]]
        prediction = (torch.tensor(p)/20.).requires_grad_()
        target, reference = torch.tensor(q), torch.tensor(r)/20.
        valid = torch.tensor([[True, True, False]]) if padded else torch.ones((1,2), dtype=torch.bool)
        a = prediction.shape[1]
        loss = calibration_loss(prediction, target, valid, torch.full((1,a,a),.1),
            torch.ones((1,a,a),dtype=torch.bool), reference_prediction=reference,
            config=dict(trust_projection_z=1.96, uncertainty_trust_weight=1.))
        loss.backward()
        return loss.detach(), prediction.grad
    a, ga = run(False)
    b, gb = run(True)
    assert torch.equal(a, b)
    assert torch.equal(ga, gb[:, :2])
    assert gb[0, 2] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("projection_z", [0., 1.96])
def test_cuda_matches_cpu_loss_and_gradient(projection_z):
    args = (torch.tensor([[.12, .10, -.03]], requires_grad=True),
            torch.tensor([[2., 2.5, -.2]]), torch.tensor([[True, True, True]]),
            torch.tensor([[[0., .1, .2], [.1, 0., .3], [.2, .3, 0.]]]),
            torch.ones((1, 3, 3), dtype=torch.bool))
    reference = torch.tensor([[.15, .08, -.03]])
    config = dict(uncertainty_trust_weight=1., trust_projection_z=projection_z)
    cpu = calibration_loss(*args, reference_prediction=reference, config=config)
    cpu.backward()
    gpu_args = tuple(x.detach().cuda().requires_grad_(i == 0) for i, x in enumerate(args))
    gpu = calibration_loss(*gpu_args, reference_prediction=reference.cuda(), config=config)
    gpu.backward()
    assert float(gpu.detach().cpu()) == pytest.approx(float(cpu.detach()), rel=2e-5)
    assert torch.allclose(gpu_args[0].grad.cpu(), args[0].grad, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("kwargs", [{"noise_scale": 0.}, {"value_weight": -1.},
                                    {"pair_huber_delta": float('nan')},
                                    {"trust_projection_z": -1.},
                                    {"trust_projection_z": float('inf')}])
def test_configuration_rejects_invalid_scales(kwargs):
    with pytest.raises(ValueError):
        CalibrationConfig(**kwargs)
