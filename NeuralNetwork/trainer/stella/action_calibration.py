"""Calibrate *decision-relevant action differences* under a frozen teacher.

The existing trainer already regresses centered Q values, which is equivalent
to an all-pairs squared-error objective when its confidence weights are equal.
This module changes the emphasis: it anchors comparisons at the current
student decision, emphasizes its closest predicted competitor and the fixed
teacher action, and keeps only a small common-value anchor. It does not change
the reward, the teacher rollout policy, the network format, or preset weights.

All targets, standard errors, and configuration scales below use ordinary
effective-score units. ``prediction`` alone is the network's normalized output.
The final loss is divided by ``target_scale ** 2`` to match the existing
trainer's output/loss scale. Label moments are computed in float64; training
arithmetic is float32 on the prediction tensor's device.

No uncertain sample winner is a hard target. Pair targets are the observed
mean differences, never lower confidence bounds. The loss has a finite target:
it also penalizes an action gap that has grown beyond its estimated value.
Unknown paired uncertainty gets zero evidence weight, not a fabricated zero SE.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import operator
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CalibrationConfig:
    """Conservative starting values, subject to paired controlled experiments."""

    target_scale: float = 20.0
    pair_weight: float = 1.0
    value_weight: float = 0.02
    shape_weight: float = 0.05
    pair_huber_delta: float = 0.25
    value_huber_delta: float = 2.0
    noise_scale: float = 0.10
    # Optional learning floor for comparisons with known paired uncertainty.
    # Zero retains the original arithmetic exactly. Unknown pairs stay zero.
    reliability_floor: float = 0.0
    competitor_boost: float = 2.0
    teacher_boost: float = 1.0
    label_best_boost: float = 0.0
    # Matches the tested main action-calibration profile. Weak trust remains an
    # explicit ablation rather than an accidental constructor fallback.
    uncertainty_trust_weight: float = 1.0
    # Zero preserves legacy behavior. Production may opt into one coherent
    # reference Q projection around the fixed initial student's argmax.
    # The interval is a soft regularizer, not a coverage guarantee.
    trust_projection_z: float = 0.0

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{f.name} must be a finite numeric value")
        for name in ("target_scale", "pair_huber_delta", "value_huber_delta", "noise_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("pair_weight", "value_weight", "shape_weight", "competitor_boost",
                     "teacher_boost", "label_best_boost", "uncertainty_trust_weight",
                     "trust_projection_z"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("reliability_floor must be in [0,1]")


def _config(config):
    if config is None:
        return CalibrationConfig()
    if isinstance(config, CalibrationConfig):
        return config
    if isinstance(config, Mapping):
        return CalibrationConfig(**dict(config))
    raise TypeError("config must be CalibrationConfig, a mapping, or None")


def _integer(value, name, *, minimum=0):
    """Validate host metadata without silently truncating action indices."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be an integer in [{minimum}, int64 max]")
    return value


def root_pair_moments(root):
    """Return raw Q and paired Monte Carlo SE without assuming independence.

    ``root['action_returns']`` must contain complete, aligned rollout columns
    for all legal actions. Its column pairing is the sampler's responsibility.
    A covariance calculation only measures the pairing supplied here; it cannot
    repair misaligned seeds. ``root['returns']`` is accepted for archive users.

    If only an explicit ``pair_se`` table is available, it is accepted together
    with optional ``pair_known``. Marginal per-action standard errors are not
    sufficient to reconstruct paired SE and are deliberately not substituted.
    """
    q = np.asarray(root["q"], dtype=np.float64)
    if q.ndim != 1 or len(q) == 0 or not np.isfinite(q).all():
        raise ValueError("q must contain finite values for a nonempty action set")
    count = len(q)
    se = np.full((count, count), np.nan, dtype=np.float64)
    known = np.zeros((count, count), dtype=np.bool_)
    sample_count = 0
    returns = root.get("action_returns", root.get("returns"))
    if returns is not None:
        values = np.asarray(returns, dtype=np.float64)
        if (values.ndim != 2 or values.shape[0] != count or values.shape[1] < 1
                or not np.isfinite(values).all()):
            raise ValueError("returns must be finite [actions, aligned rollouts]")
        sample_count = int(values.shape[1])
        means = values.mean(axis=1)
        if not np.allclose(q, means, rtol=1e-6, atol=2e-5):
            raise ValueError("q does not match the supplied action-return means")
        if sample_count >= 2:
            centered = values - means[:, None]
            covariance_of_mean = (centered @ centered.T)/(sample_count*(sample_count-1))
            variances = np.diag(covariance_of_mean)
            variance_of_difference = (variances[:, None] + variances[None, :]
                                      - 2.0*covariance_of_mean)
            se = np.sqrt(np.maximum(0.0, variance_of_difference))
            known[:] = True
    elif "pair_se" in root:
        supplied = np.asarray(root["pair_se"], dtype=np.float64)
        if supplied.shape != (count, count):
            raise ValueError("pair_se must have shape [actions, actions]")
        if np.any(np.isfinite(supplied) & (supplied < 0)):
            raise ValueError("pair_se cannot be negative")
        explicit = root.get("pair_known")
        if explicit is None:
            known = np.isfinite(supplied)
        else:
            known = np.asarray(explicit, dtype=np.bool_)
            if known.shape != supplied.shape:
                raise ValueError("pair_known must have the same shape as pair_se")
            if np.any(known & ~np.isfinite(supplied)):
                raise ValueError("Known paired SE must be finite")
        if not np.array_equal(known, known.T):
            raise ValueError("pair_known must be symmetric")
        if not np.allclose(supplied[known], supplied.T[known], atol=1e-8, rtol=1e-6):
            raise ValueError("pair_se must be symmetric for known pairs")
        se = np.where(known, supplied, np.nan)
        sample_count = _integer(root.get("sample_count", 0), "sample_count")
    # Self-comparisons contain no decision evidence and are never trained.
    np.fill_diagonal(se, 0.0)
    np.fill_diagonal(known, False)
    teacher = root.get("teacher_action_index", -1)
    teacher = -1 if teacher is None else _integer(teacher, "teacher_action_index", minimum=-1)
    if teacher < -1 or teacher >= count:
        raise ValueError("teacher_action_index must be -1 or a legal action index")
    return {
        "target": q.astype(np.float32),
        "pair_mean": (q[:, None]-q[None, :]).astype(np.float32),
        "pair_se": se.astype(np.float32),
        "pair_known": known,
        "teacher_action": teacher,
        "sample_count": sample_count,
    }


def batch_pair_moments(roots, max_actions=None):
    """Build padded NumPy targets once, outside the optimizer loop.

    The tensors returned here can be moved with ``torch.as_tensor(...,
    device=device)``. No Torch import is needed for this preprocessing helper.
    ``pair_mean`` is provided for diagnostics; the loss obtains the same
    difference from raw ``target`` after choosing its dynamic anchor.
    """
    roots = list(roots)
    if not roots:
        raise ValueError("At least one root is required")
    moments = [root_pair_moments(root) for root in roots]
    required = max(len(m["target"]) for m in moments)
    width = required if max_actions is None else _integer(max_actions, "max_actions", minimum=1)
    if width < required:
        raise ValueError("max_actions is smaller than a legal action set")
    count = len(roots)
    result = {
        "target": np.zeros((count, width), dtype=np.float32),
        "mask": np.zeros((count, width), dtype=np.bool_),
        "pair_mean": np.zeros((count, width, width), dtype=np.float32),
        "pair_se": np.full((count, width, width), np.nan, dtype=np.float32),
        "pair_known": np.zeros((count, width, width), dtype=np.bool_),
        "teacher_actions": np.full(count, -1, dtype=np.int64),
        "sample_count": np.zeros(count, dtype=np.int64),
    }
    for i, m in enumerate(moments):
        n = len(m["target"])
        result["target"][i, :n] = m["target"]
        result["mask"][i, :n] = True
        for name in ("pair_mean", "pair_se", "pair_known"):
            result[name][i, :n, :n] = m[name]
        result["teacher_actions"][i] = m["teacher_action"]
        result["sample_count"][i] = m["sample_count"]
    return result


def _project_reference_q(reference, observed, valid, safe_se, evidence, z, *, origin=None):
    """Project raw reference Q around its own fixed argmax, not the student.

    Internal helper; inputs are already padding-clean and pair uncertainty is
    already validated by the loss. A known origin/action comparison bounds the
    reference *gap*, while an unknown origin/action comparison keeps that
    action's reference Q unchanged. All current decision anchors subsequently
    use differences from this one coherent Q vector. The interval is a soft
    regularization device, not a simultaneous or finite-sample confidence set.
    """
    import torch

    reference = reference.detach()
    observed = observed.detach()
    if origin is None:
        origin = reference.masked_fill(~valid, -torch.inf).argmax(1)
    origin_column = origin[:, None]
    origin_row = origin[:, None, None].expand(-1, 1, reference.shape[1])
    known_origin = evidence.gather(1, origin_row).squeeze(1)
    origin_se = safe_se.detach().gather(1, origin_row).squeeze(1)
    reference_origin = reference.gather(1, origin_column)
    reference_gap = reference-reference_origin
    mean_gap = observed-observed.gather(1, origin_column)
    radius = z*origin_se
    projected_gap = torch.maximum(mean_gap-radius,
                                  torch.minimum(reference_gap, mean_gap+radius))
    return torch.where(known_origin, reference_origin+projected_gap, reference)


def prepare_calibration(target, mask, pair_se, pair_known, *,
                        reference_prediction=None, config=None):
    """Prepare fixed label statistics once, outside the optimizer loop.

    The returned tensors depend only on the labels, their paired uncertainty,
    the fixed initial student and fixed calibration settings. They do not
    depend on the current candidate. ``trust_reference`` is in ordinary score
    units; the supplied ``reference_prediction`` remains normalized and is
    never modified. Callers must retain that original reference for baseline
    decisions and transfer diagnostics.

    This helper also backs the original one-call loss API. It uses the same
    FP32 arithmetic as that API, including explicit padding/unknown handling.
    Dataset callers can gather these tensors with their complete-root batches
    and avoid rebuilding any [B,A,A] statistics or projection during a step.
    """
    import torch

    cfg = _config(config)
    if target.ndim != 2 or mask.shape != target.shape:
        raise ValueError("target and mask must have matching [B,A] shapes")
    batch, actions = target.shape
    if batch == 0 or actions == 0:
        raise ValueError("A nonempty target tensor is required")
    if pair_se.shape != (batch, actions, actions) or pair_known.shape != pair_se.shape:
        raise ValueError("pair_se and pair_known must have shape [B,A,A]")
    if reference_prediction is not None and reference_prediction.shape != target.shape:
        raise ValueError("reference_prediction must have the same shape as target")

    valid = mask.bool()
    counts = valid.sum(1).to(dtype=torch.float32)
    evidence = pair_known.bool() & torch.isfinite(pair_se) & (pair_se >= 0)
    evidence = evidence & valid[:, :, None] & valid[:, None, :]
    evidence = evidence & ~torch.eye(actions, device=target.device, dtype=torch.bool)[None, :, :]
    safe_se = torch.where(evidence, pair_se.float(), 0.0)
    variance_scale = cfg.noise_scale*cfg.noise_scale
    reliability = (variance_scale/(variance_scale+safe_se.square()))*evidence.float()
    if cfg.reliability_floor:
        # Mix a nonzero learning mass into KNOWN pairs only. This prevents
        # weak per-root evidence from suppressing a signal shared by many
        # independent roots. It does not change the observed mean gap target.
        reliability = ((1.0-cfg.reliability_floor)*reliability
                       + cfg.reliability_floor*evidence.float())
    action_reliability = reliability.sum(2)/(counts-1).clamp_min(1.0)[:, None]

    reference = None
    if reference_prediction is not None:
        reference = torch.where(valid, reference_prediction.detach().float(), 0.0)*cfg.target_scale
        if cfg.trust_projection_z > 0.0:
            # Use the exact same unscaled FP32 argmax as initial_action and
            # transfer diagnostics. Scaling adjacent floats can round them to
            # a tie, which must not silently choose a different paired row.
            origin = reference_prediction.detach().float().masked_fill(~valid, -torch.inf).argmax(1)
            observed = torch.where(valid, target.float(), 0.0)
            reference = _project_reference_q(reference, observed, valid, safe_se,
                                             evidence, cfg.trust_projection_z, origin=origin)
    return {"pair_reliability": reliability,
            "action_reliability": action_reliability,
            "pair_known": evidence,
            "trust_reference": reference}


def calibration_loss(prediction, target, mask, pair_se, pair_known, *,
                     teacher_actions=None, reference_prediction=None, config=None, root_weights=None,
                     return_details=False):
    """Return a differentiable, normalized-unit scalar loss.

    Args:
        prediction: [B,A] normalized Q outputs, as produced by ``TorchQ``.
        target: [B,A] raw ordinary-score Q labels (NOT divided by 20).
        mask: [B,A] valid action mask; padding has exactly zero contribution.
        pair_se, pair_known: [B,A,A] paired raw-score SE and explicit evidence.
        teacher_actions: [B] frozen teacher action indices, with -1 for unknown.
        reference_prediction: Optional [B,A] normalized predictions of the
            fixed initial student. Where paired evidence is weak, a soft gap
            penalty limits drift from this baseline. It is never a hard action
            switch or a new teacher; no trust term is active without this input.
            If experimental trust_projection_z is positive, known comparisons
            to its fixed argmax first project this reference into mean +/- zSE
            gap intervals. The observed main gap target remains unchanged.
        root_weights: Optional outer masses; used as a weighted sum, without
            renormalizing. This permits chunks of an already normalized corpus.

    The current prediction argmax is a stop-gradient anchor. All other legal
    actions receive a uniform base prior; the closest predicted competitor and
    the fixed teacher action receive additional configurable attention. No
    sampled label winner receives extra weight by default. Prior weights are
    normalized *before* continuous uncertainty attenuation. They are never
    renormalized afterward. An optional known-pair reliability floor preserves
    learning mass for weak evidence; unknown pairs retain zero evidence mass.

    The preprocessing helpers validate data. This hot path avoids Python tests
    of CUDA tensor values, host reads, and scalar synchronization. Empty padded
    rows are harmless and contribute zero, although helpers never create them.
    """
    cfg = _config(config)
    prepared = prepare_calibration(target, mask, pair_se, pair_known,
        reference_prediction=reference_prediction, config=cfg)
    return calibration_loss_prepared(prediction, target, mask,
        prepared["pair_reliability"], prepared["action_reliability"],
        teacher_actions=teacher_actions, trust_reference=prepared["trust_reference"],
        pair_known=prepared["pair_known"], config=cfg, root_weights=root_weights,
        return_details=return_details)


def calibration_loss_prepared(prediction, target, mask, pair_reliability,
                              action_reliability, *, teacher_actions=None,
                              trust_reference=None, pair_known=None, config=None,
                              root_weights=None, return_details=False):
    """Decision loss using cached statistics from :func:`prepare_calibration`.

    The current candidate still supplies the dynamic argmax and closest
    competitor on every call. Only fixed computations are cached. In
    particular this path never recomputes the reference projection; callers
    must regenerate the cache when target_scale, noise_scale, reliability_floor or
    trust_projection_z changes. ``pair_known`` is needed only for requested
    diagnostics, so normal training does not transfer/gather its [B,A,A] mask.
    No tensor values are read on the host and all action sets remain complete.
    """
    import torch

    cfg = _config(config)
    if prediction.ndim != 2 or target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, target and mask must have matching [B,A] shapes")
    batch, actions = prediction.shape
    if batch == 0 or actions == 0:
        raise ValueError("A nonempty prediction tensor is required")
    if pair_reliability.shape != (batch, actions, actions):
        raise ValueError("pair_reliability must have shape [B,A,A]")
    if action_reliability.shape != prediction.shape:
        raise ValueError("action_reliability must have shape [B,A]")
    if teacher_actions is not None and teacher_actions.shape != (batch,):
        raise ValueError("teacher_actions must have shape [B]")
    if trust_reference is not None and trust_reference.shape != prediction.shape:
        raise ValueError("trust_reference must have the same shape as prediction")
    if root_weights is not None and root_weights.shape != (batch,):
        raise ValueError("root_weights must have shape [B]")
    if return_details and (pair_known is None or pair_known.shape != pair_reliability.shape):
        raise ValueError("return_details requires prepared pair_known with shape [B,A,A]")

    valid = mask.bool()
    # Clean padding before arithmetic, including padding containing NaN/Inf.
    predicted = torch.where(valid, prediction.float(), 0.0)*cfg.target_scale
    observed = torch.where(valid, target.float(), 0.0)
    counts = valid.sum(1).to(dtype=torch.float32)
    denominator = counts.clamp_min(1.0)
    anchor = predicted.detach().masked_fill(~valid, -torch.inf).argmax(1)
    columns = torch.arange(actions, device=prediction.device)[None, :]
    eligible = valid & (columns != anchor[:, None])
    competitor = predicted.detach().masked_fill(~eligible, -torch.inf).argmax(1)

    prior = eligible.to(dtype=torch.float32)
    prior = prior*(1.0 + cfg.competitor_boost*(columns == competitor[:, None]).float())
    if teacher_actions is not None and cfg.teacher_boost:
        prior = prior + cfg.teacher_boost*(eligible & (columns == teacher_actions[:, None])).float()
    if cfg.label_best_boost:
        label_best = observed.detach().masked_fill(~valid, -torch.inf).argmax(1)
        prior = prior + cfg.label_best_boost*(eligible & (columns == label_best[:, None])).float()
    prior = prior/prior.sum(1, keepdim=True).clamp_min(1.0)

    anchor_row = anchor[:, None, None].expand(-1, 1, actions)
    decision_reliability = pair_reliability.gather(1, anchor_row).squeeze(1)
    effective_weights = prior*decision_reliability

    predicted_gap = predicted - predicted.gather(1, anchor[:, None])
    observed_gap = observed - observed.gather(1, anchor[:, None])
    residual = predicted_gap-observed_gap

    def huber(error, delta):
        absolute = error.abs()
        return torch.where(absolute <= delta, 0.5*error.square(),
                           delta*(absolute-0.5*delta))

    pair_root = (effective_weights*huber(residual, cfg.pair_huber_delta)).sum(1)
    mean_error = ((predicted-observed)*valid).sum(1)/denominator
    value_root = huber(mean_error, cfg.value_huber_delta)*(counts > 0).float()
    centered_error = predicted-observed-mean_error[:, None]
    shape_root = (huber(centered_error, cfg.pair_huber_delta)*action_reliability*valid).sum(1)/denominator
    trust_root = torch.zeros_like(pair_root)
    if trust_reference is not None and cfg.uncertainty_trust_weight:
        reference = torch.where(valid, trust_reference.detach().float(), 0.0)
        reference_gap = reference-reference.gather(1, anchor[:, None])
        trust_root = (prior*(1.0-decision_reliability)*
                      huber(predicted_gap-reference_gap, cfg.pair_huber_delta)).sum(1)
    raw_root = (cfg.pair_weight*pair_root + cfg.value_weight*value_root + cfg.shape_weight*shape_root
                + cfg.uncertainty_trust_weight*trust_root)

    def reduce(values):
        return values.mean() if root_weights is None else (values*root_weights.float()).sum()

    raw = reduce(raw_root)
    result = raw/(cfg.target_scale*cfg.target_scale)
    if not return_details:
        return result
    known_selected = pair_known.gather(1, anchor_row).squeeze(1)
    details = {
        "total_raw": raw.detach(),
        "pair_raw": reduce(cfg.pair_weight*pair_root).detach(),
        "value_raw": reduce(cfg.value_weight*value_root).detach(),
        "shape_raw": reduce(cfg.shape_weight*shape_root).detach(),
        "trust_raw": reduce(cfg.uncertainty_trust_weight*trust_root).detach(),
        "evidence_mass": reduce(effective_weights.sum(1)).detach(),
        "known_pair_prior_mass": reduce((prior*known_selected.float()).sum(1)).detach(),
        "unknown_pair_prior_mass": reduce((prior*~known_selected).sum(1)).detach(),
        "anchor": anchor.detach(),
    }
    return result, details
