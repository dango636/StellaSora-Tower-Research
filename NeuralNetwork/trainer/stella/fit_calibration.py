"""Opt-in fixed-teacher action-gap training, with exact preset epoch quotas.

This is a drop-in FitResult producer for the existing catalogue engine. It
keeps the QNetwork format, full legal-action labels and teacher identities.
The ordinary fitter remains available; this module is used only for the
explicit objective_version ``action_gap_calibration_v3``.
"""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict
import math
import time

import numpy as np
import torch

from .action_calibration import (CalibrationConfig, batch_pair_moments,
                                 prepare_calibration, calibration_loss_prepared)
from .features import TARGET_SCALE
from .fit import FitResult, CandidatePool, _settings, _stopped, balanced_weights, prepare_roots
from .fit_diagnostics import initialize_student_diagnostics, TransferDiagnostics
from .fit_torch import (TorchQ, TORCH_DEFAULTS, training_precision, fit_runtime_options,
                        place_root_data, torch_teacher_predictions)


OBJECTIVE = "action_gap_calibration_v3"


def _preset_identity(root):
    # dict.get's fallback is evaluated eagerly, even when catalogue_id exists.
    # Accept explicit catalogue identity without requiring an unrelated label.
    return root["catalogue_id"] if "catalogue_id" in root else root["preset_group"]


def _weighted_coverage(local, probabilities, size, generator):
    """Complete coverage passes plus a weighted partial pass, before shuffle.

    Exponential keys implement the weighted draw without replacement in the
    partial pass. Priorities choose its roots; a full pass
    covers every root before the final global shuffle. Positive base masses in the
    normal fitter make every root eligible; an explicit zero remains excluded.
    """
    if size == 0:
        return np.empty(0, dtype=np.int64)
    local = local[probabilities[local] > 0]
    if not len(local):
        raise ValueError("Every sampled stratum must retain positive mass")
    complete, remainder = divmod(size, len(local))
    # A complete pass contains every eligible root exactly once regardless of
    # its race order. exact_preset_epoch shuffles all visits after assembling
    # strata, so sorting complete passes changes no sampled multiset or final
    # ordering distribution. Only a partial pass needs weighted race keys.
    # This revision deliberately consumes fewer RNG draws; the same new seed
    # is reproducible, but is not a bitwise replay of the old coverage mode.
    full = local if complete == 1 else np.tile(local, complete)
    if not remainder:
        return full
    mass = probabilities[local]
    race = generator.exponential(size=len(local))
    keys = np.log(np.maximum(race, np.finfo(np.float64).tiny))-np.log(mass)
    partial = local[np.argsort(keys, kind="stable")[:remainder]]
    return np.concatenate((full, partial)) if complete else partial


def _systematic_resampling(local, probabilities, size, generator):
    """Retain weighted exposure in expectation with a bounded count error.

    A fresh random order prevents catalogue/root position from repeatedly
    coupling the same roots. One random offset places equally spaced points
    on the cumulative probability interval. Each count differs from size*p
    by less than one (up to floating point boundary ties), and all counts sum
    exactly to size. This is a sampling property, not a convergence guarantee.
    """
    if size == 0:
        return np.empty(0, dtype=np.int64)
    local = local[probabilities[local] > 0]
    if not len(local):
        raise ValueError("Every sampled stratum must retain positive mass")
    local = local[generator.permutation(len(local))]
    mass = probabilities[local]/probabilities[local].max()
    cumulative = np.cumsum(mass/mass.sum())
    cumulative[-1] = 1.0
    positions = (np.arange(size, dtype=np.float64)+generator.random())/size
    positions = np.minimum(positions, np.nextafter(1.0, 0.0))
    return local[np.searchsorted(cumulative, positions, side="right")]


def exact_preset_epoch(roots, priorities, generator, visits_per_preset=None, *,
                       sampling="replacement", current_round=None,
                       current_fraction=0.5, require_current=False):
    """Visit exactly k roots per preset under an explicit sampling policy.

    ``coverage_without_replacement`` covers distinct roots before repeating
    within each stratum. Explicit ``source_round`` permits a current/history split: at
    least ceil(k*current_fraction) visits use current labels, and all distinct
    current roots are included when k allows in this coverage mode. No
    filename/teacher heuristic supplies missing provenance. Production may
    require current labels for every preset. If provenance is unavailable and
    not required, this mode uses weighted coverage without a current quota.

    ``stratified_resampling`` retains within-stratum priority frequencies in
    expectation, using randomized systematic resampling instead of independent
    replacement draws; it does not guarantee a distinct-root pass. The same
    explicit current/history quota applies when provenance is available.
    ``replacement`` retains the former sampling for
    controlled comparisons and unversioned legacy callers. It is the fallback
    and does not enforce the new current-round quota. Production profiles select
    their sampling method explicitly.
    Preset mass and total optimizer visits remain identical in all modes.
    """
    if sampling not in ("coverage_without_replacement", "stratified_resampling", "replacement"):
        raise ValueError("Unknown calibration sampling mode")
    if (isinstance(current_fraction, bool) or not isinstance(current_fraction, (int, float))
            or not math.isfinite(current_fraction) or not 0 <= current_fraction <= 1):
        raise ValueError("current_fraction must be finite in [0,1]")
    if current_round is not None and (type(current_round) is not int or current_round < 1):
        raise ValueError("current_round must be a positive integer")
    if type(require_current) is not bool:
        raise ValueError("require_current must be boolean")
    groups = defaultdict(list)
    for i, root in enumerate(roots):
        key = _preset_identity(root)
        groups[key].append(i)
    if not groups:
        raise ValueError("Cannot schedule an empty training corpus")
    probabilities = np.asarray(priorities, dtype=np.float64)
    if (probabilities.shape != (len(roots),) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()):
        raise ValueError("Invalid within-preset sampling priorities")
    k = math.ceil(len(roots)/len(groups)) if visits_per_preset is None else visits_per_preset
    if type(k) is not int or k < 1:
        raise ValueError("visits_per_preset must be a positive integer")
    indices, current_ratios = [], []
    current_visits = current_unique = current_presets = 0
    missing_current = 0
    all_provenance = all(type(root.get("source_round")) is int and
                         root["source_round"] >= 1 for root in roots)
    can_stratify = current_round is not None and all_provenance
    if can_stratify and any(root["source_round"] > current_round for root in roots):
        raise ValueError("Training roots cannot come from a future round")
    if require_current and sampling != "replacement" and not can_stratify:
        raise ValueError("Current-label quota requires explicit source_round for every root")
    is_current = (np.fromiter((root["source_round"] == current_round for root in roots),
                              dtype=np.bool_, count=len(roots)) if can_stratify else None)
    draw = _systematic_resampling if sampling == "stratified_resampling" else _weighted_coverage
    for key in sorted(groups, key=str):
        local = np.asarray(groups[key], dtype=np.int64)
        mass = probabilities[local]
        if mass.sum() <= 0:
            raise ValueError("Every preset must retain positive sampling mass")
        if sampling == "replacement":
            selected = generator.choice(local, size=k, replace=True, p=mass/mass.sum())
        elif can_stratify:
            current_mask = is_current[local]
            fresh = local[current_mask & (mass > 0)]
            historical = local[~current_mask & (mass > 0)]
            if not len(fresh):
                missing_current += 1
                if require_current:
                    raise ValueError(f"Preset {key!r} has no current training roots")
                selected = draw(local, probabilities, k, generator)
            elif not len(historical):
                selected = draw(fresh, probabilities, k, generator)
            else:
                quota = max(min(k, len(fresh)), math.ceil(k*current_fraction))
                selected = np.concatenate([
                    draw(fresh, probabilities, quota, generator),
                    draw(historical, probabilities, k-quota, generator)])
        else:
            selected = draw(local, probabilities, k, generator)
        if can_stratify:
            selected_current_count = int(np.count_nonzero(is_current[selected]))
            current_visits += selected_current_count
            current_presets += int(selected_current_count > 0)
            current_ratios.append(selected_current_count/k)
        indices.append(selected)
    order = np.concatenate(indices)
    generator.shuffle(order)
    # Roots belong to one preset only. One global unique pass gives the exact
    # former sum of per-preset uniques without thousands of small allocations.
    if can_stratify:
        current_unique = len(np.unique(order[is_current[order]]))
    return order, {"presets": len(groups), "visits_per_preset": k,
                   "root_visits": len(order), "equal_preset_epoch_quota": True,
                   "sampling_method": sampling,
                   "unique_roots_visited": int(len(np.unique(order))),
                   "duplicate_visits": int(len(order)-len(np.unique(order))),
                   "current_round": current_round,
                   "source_round_provenance_complete": all_provenance,
                   "current_quota_enforced": can_stratify and missing_current == 0 and sampling != "replacement",
                   "current_fraction_floor": current_fraction,
                   "current_root_visits": current_visits if can_stratify else None,
                   "unique_current_roots_visited": current_unique if can_stratify else None,
                   "presets_with_current_visits": current_presets if can_stratify else None,
                   "presets_missing_current": missing_current if can_stratify else None,
                   "current_fraction_min": min(current_ratios) if current_ratios else None,
                   "current_fraction_max": max(current_ratios) if current_ratios else None}


def _preset_indices(roots):
    groups = {}
    return np.fromiter((groups.setdefault(_preset_identity(root), len(groups)) for root in roots),
                       dtype=np.int64, count=len(roots))


def coverage_priority_weights(roots, priorities, order, *, preset_indices=None, batch_size=None):
    """Preserve each fully covered preset's former priority-weighted objective.

    For k scheduled visits, normalized within-preset priority p_i and actual
    visit count c_i, each visit to root i receives k*p_i/c_i. At a fixed model,
    summing the epoch loss then matches the expectation of k replacement draws
    from p_i. This is not an assertion about Adam trajectories or conditional
    unbiasedness of individual minibatch steps.

    A preset missing any positive-priority root instead uses uniform visit
    weights and is explicitly reported as a fallback. Every preset still has
    exactly its original k loss mass. No clipping or batch renormalization is
    applied. The complete vector is prepared on CPU once per epoch; the fitter
    divides by its fixed batch size and transfers it once for cheap slices.
    """
    count = len(roots)
    mass = np.asarray(priorities, dtype=np.float64)
    indices = np.asarray(order)
    if (not count or mass.shape != (count,) or not np.isfinite(mass).all()
            or (mass < 0).any()):
        raise ValueError("Invalid coverage correction priorities")
    if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
            or indices.min() < 0 or indices.max() >= count):
        raise ValueError("Coverage correction requires a nonempty valid integer visit order")
    groups = _preset_indices(roots) if preset_indices is None else np.asarray(preset_indices)
    if (groups.shape != (count,) or not np.issubdtype(groups.dtype, np.integer)
            or groups.min() < 0 or groups.max() >= count):
        raise ValueError("Invalid dense preset indices for coverage correction")
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    group_count = int(groups.max())+1
    active = np.bincount(groups, minlength=group_count) > 0
    selected_groups = groups[indices]
    visits = np.bincount(indices, minlength=count)
    preset_visits = np.bincount(selected_groups, minlength=group_count).astype(np.float64)
    if (preset_visits[active] <= 0).any() or not np.all(preset_visits[active] == preset_visits[active][0]):
        raise ValueError("Coverage correction requires the same positive visit quota for every preset")
    # Scaling within each preset avoids overflow for extreme finite priorities.
    maximum = np.zeros(group_count, dtype=np.float64)
    np.maximum.at(maximum, groups, mass)
    if (maximum[active] <= 0).any():
        raise ValueError("Every preset must retain positive priority mass")
    scaled = mass/maximum[groups]
    totals = np.bincount(groups, weights=scaled, minlength=group_count)
    probabilities = scaled/totals[groups]
    missed = (mass > 0) & (visits == 0)
    complete = active & (np.bincount(groups[missed], minlength=group_count) == 0)
    root_weights = np.ones(count, dtype=np.float64)
    corrected = complete[groups] & (visits > 0)
    root_weights[corrected] = (preset_visits[groups[corrected]]*probabilities[corrected]
                               /visits[corrected])
    weights = root_weights[indices]
    actual_mass = np.bincount(selected_groups, weights=weights, minlength=group_count)
    fp32_mass = np.bincount(selected_groups, weights=weights.astype(np.float32), minlength=group_count)
    total_weight = float(weights.sum())
    squared_weight = float(weights@weights)
    batch_mass = None
    if batch_size is not None:
        batch_mass = np.add.reduceat(weights, np.arange(0, len(weights), batch_size))/batch_size
    report = {
        "requested": True,
        "applied": bool(complete.any()),
        "corrected_presets": int(complete.sum()),
        "fallback_presets": int((active & ~complete).sum()),
        "fallback_reason": "uniform visits for presets missing any positive-priority root",
        "positive_roots": int((mass > 0).sum()),
        "covered_positive_roots": int(((mass > 0) & (visits > 0)).sum()),
        "missed_positive_roots": int(missed.sum()),
        "max_abs_preset_mass_error": float(np.max(np.abs(actual_mass[active]-preset_visits[active]))),
        "max_abs_fp32_preset_mass_error": float(np.max(np.abs(fp32_mass[active]-preset_visits[active]))),
        "visit_weight_min": float(weights.min()),
        "visit_weight_max": float(weights.max()),
        "visit_weight_p99": float(np.quantile(weights, .99)),
        "visit_weight_sum": total_weight,
        "visit_weight_square_sum": squared_weight,
        "weight_effective_visits": total_weight*total_weight/squared_weight,
        "batch_loss_mass_min": None if batch_mass is None else float(batch_mass.min()),
        "batch_loss_mass_max": None if batch_mass is None else float(batch_mass.max()),
        "semantics": "fixed-model epoch objective equals replacement priority expectation only for fully covered presets; fallback has uniform visit loss; weight effective visits is a mass diagnostic, not independent samples",
    }
    return weights, report


class CalibrationRoots:
    """Complete-action roots, paired uncertainty, fixed initial predictions."""

    def __init__(self, roots, width, reference, settings, loss_config=None):
        moment = batch_pair_moments(roots, max_actions=width)
        self.roots = roots
        self.root_count = len(roots)
        self.family_count = len({r["family_id"] for r in roots})
        self.preset_count = len({_preset_identity(r) for r in roots})
        self.preset_indices_np = _preset_indices(roots)
        features = np.zeros((len(roots), width, roots[0]["x"].shape[1]), dtype=np.float32)
        baseline = np.zeros((len(roots), width), dtype=np.float32)
        for i, (root, values) in enumerate(zip(roots, reference)):
            n = len(root["q"])
            features[i,:n] = root["x"]
            baseline[i,:n] = values
        self.x = torch.from_numpy(features)
        self.target = torch.from_numpy(moment["target"])
        self.mask = torch.from_numpy(moment["mask"])
        self.pair_se = torch.from_numpy(moment["pair_se"])
        self.pair_known = torch.from_numpy(moment["pair_known"])
        # Small host tables support current-student priority refresh without
        # rescanning rollout matrices or reading GPU tensors per root.
        self.target_np = moment["target"]
        self.pair_se_np = moment["pair_se"]
        self.pair_known_np = moment["pair_known"]
        self.mask_np = moment["mask"]
        self.teacher_actions = torch.from_numpy(moment["teacher_actions"])
        self.reference = torch.from_numpy(baseline)
        self.weights_np = balanced_weights(roots)
        self.weights = torch.tensor(self.weights_np, dtype=torch.float32)
        self.initial_action = self.reference.masked_fill(~self.mask, -torch.inf).argmax(1)
        self.best_action = self.target.masked_fill(~self.mask, -torch.inf).argmax(1)
        self.diagnostics = TransferDiagnostics(roots, self.weights_np, settings)
        self.preparation_confidence_z = float(settings.get("confidence_z", 1.5))
        self.preparation_key = None
        self.prepare_loss(loss_config or CalibrationConfig(**dict(settings.get("calibration", {}))))

    def prepare_loss(self, config):
        """Cache fixed tensors, retaining the original student as the baseline.

        The key only includes scales used by preparation. Hyperparameters that
        combine loss components can vary without rebuilding the fixed labels.
        The normal fitter calls this once while all source tensors are on CPU.
        A direct evaluate_calibration caller may deliberately change a scale;
        that triggers one refresh rather than silently reusing a stale cache.
        """
        key = (config.target_scale, config.noise_scale, config.trust_projection_z,
               config.reliability_floor)
        if key == self.preparation_key:
            # Component weights do not invalidate fixed tensors, but a direct
            # evaluator may toggle trust while reusing the prepared dataset.
            self.preparation_diagnostics["uncertainty_trust_active"] = bool(config.uncertainty_trust_weight)
            return
        # An explicit scale refresh can be requested by the inference-mode
        # evaluator. Cache ordinary, grad-free tensors so a subsequent training
        # step can safely save them for prediction gradients.
        with torch.inference_mode(False), torch.no_grad():
            prepared = prepare_calibration(self.target, self.mask, self.pair_se, self.pair_known,
                reference_prediction=self.reference, config=config)
        self.pair_reliability = prepared["pair_reliability"]
        self.action_reliability = prepared["action_reliability"]
        self.pair_known = prepared["pair_known"]
        self.trust_reference = prepared["trust_reference"]
        self.preparation_key = key
        self.preparation_diagnostics = self._preparation_diagnostics(config)
        self.bytes = sum(v.numel()*v.element_size() for v in vars(self).values()
                         if isinstance(v, torch.Tensor))

    def _preparation_diagnostics(self, config):
        """One CPU aggregation per preparation; never part of an update step."""
        original = self.reference.detach().cpu().numpy()*np.float32(config.target_scale)
        projected = self.trust_reference.detach().cpu().numpy()
        mask = self.mask_np
        rows = np.arange(self.root_count)
        # Keep the exact pre-projection baseline used by all transfer metrics.
        origin = self.initial_action.detach().cpu().numpy()
        chosen = np.where(mask, projected, -np.inf).argmax(1)
        delta = np.where(mask, np.abs(projected-original), 0.)
        # Do not count a raw-score FP32 subtract/add roundoff as a projection.
        changed = delta > 1e-5
        counts = mask.sum(1)
        known = self.pair_known_np[rows, origin]
        se = np.where(known, self.pair_se_np[rows, origin], 0.)
        threshold = np.where(known, self.preparation_confidence_z*se, np.inf)
        gains = self.target_np-self.target_np[rows, origin, None]
        supported = (known & mask & (gains > threshold)).any(1)
        mass = self.weights_np
        reliability = (config.noise_scale**2/(config.noise_scale**2+se**2))*known
        floored = ((1.0-config.reliability_floor)*reliability
                   + config.reliability_floor*known)
        comparison_counts = np.maximum(1, counts-1)
        return {
            "version": "cached_fixed_pair_statistics_v1",
            "trust_projection_z": float(config.trust_projection_z),
            "reliability_floor": float(config.reliability_floor),
            "reliability_floor_semantics": "known pairs: floor + (1-floor)*reliability; unknown pairs: zero; observed mean gap unchanged",
            "initial_anchor_reliability_mass_before_floor": float(mass@(reliability.sum(1)/comparison_counts)),
            "initial_anchor_reliability_mass_after_floor": float(mass@(floored.sum(1)/comparison_counts)),
            "uncertainty_trust_active": bool(config.uncertainty_trust_weight),
            "origin": "unmodified_initial_student_argmax",
            "projected_root_fraction": float(mass@changed.any(1)),
            "projected_action_fraction": float(mass@(changed.sum(1)/np.maximum(1, counts-1))),
            "projected_choice_change_fraction": float(mass@(chosen != origin)),
            "mean_absolute_gap_adjustment": float(mass@(delta.sum(1)/np.maximum(1, counts-1))),
            "paired_supported_improvement_root_fraction": float(mass@supported),
            "paired_support_z": self.preparation_confidence_z,
            "change_threshold_raw": 1e-5,
            "semantics": "fixed-reference diagnostics; original preset masses; paired teacher-continuation label evidence, not whole-game gain",
        }

    def to(self, device):
        for key, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self

    def metric_view(self, device):
        if self.target.device == device:
            return self
        result = copy.copy(self)
        for key, value in vars(self).items():
            if key != "x" and isinstance(value, torch.Tensor):
                setattr(result,key,value.to(device))
        return result

    def batch(self, indices, device, *, prepared=False):
        values = []
        names = (("x", "target", "mask", "pair_reliability", "action_reliability",
                  "teacher_actions", "trust_reference") if prepared else
                 ("x", "target", "mask", "pair_se", "pair_known", "teacher_actions", "reference"))
        for name in names:
            source = getattr(self,name)
            # Indices are always placed on the same device as their source.
            # No indexed/unindexed CUDA equality test and no per-step readback.
            pinned = source.device.type == "cpu" and device.type == "cuda"
            if pinned:
                selected = torch.empty((len(indices), *source.shape[1:]), dtype=source.dtype,
                                       device="cpu", pin_memory=True)
                torch.index_select(source,0,indices,out=selected)
            else:
                selected = source.index_select(0,indices)
            values.append(selected.to(device,non_blocking=pinned))
        return values


@torch.inference_mode()
def evaluate_calibration(model, data, settings, loss_config):
    """Preset-macro metrics; empirical label best is explicitly not Q*."""
    device = next(model.parameters()).device
    data.prepare_loss(loss_config)
    features = data.x
    local = data.metric_view(device)
    model.eval()
    with training_precision(device,tf32=False):
        flat = features.flatten(0,1)
        prediction = torch.cat([model(flat[i:i+4096].to(device),amp=False)
                                for i in range(0,len(flat),4096)]).reshape_as(local.target)
        _, details = calibration_loss_prepared(prediction,local.target,local.mask,
            local.pair_reliability,local.action_reliability,
            teacher_actions=local.teacher_actions,trust_reference=local.trust_reference,
            pair_known=local.pair_known,
            config=loss_config,root_weights=local.weights,return_details=True)
        p = prediction*TARGET_SCALE
        mask = local.mask
        n = mask.sum(1).clamp_min(1)
        error = torch.where(mask,p-local.target,0.0)
        mean = error.sum(1)/n
        q_rmse = (local.weights*(error.square().sum(1)/n)).sum().sqrt()
        advantage_rmse = (local.weights*((error-mean[:,None]).square()*mask).sum(1)/n).sum().sqrt()
        choice = p.masked_fill(~mask,-torch.inf).argmax(1)
        best = local.target.gather(1,local.best_action[:,None]).squeeze(1)
        selected = local.target.gather(1,choice[:,None]).squeeze(1)
        initial = local.target.gather(1,local.initial_action[:,None]).squeeze(1)
        regret = (local.weights*(best-selected)).sum()
        agreement = (local.weights*(choice==local.best_action).float()).sum()
        gain = (local.weights*(selected-initial)).sum()
        change = (local.weights*(choice!=local.initial_action).float()).sum()
        # Monitor, but never alter, decisions that may change under FP32
        # backend rounding. The denominator contains only roots with at least
        # two legal actions and keeps their original preset-macro masses.
        multiple = n >= 2
        eligible_mass = (local.weights*multiple).sum()
        if mask.shape[1] >= 2:
            top = p.masked_fill(~mask, -torch.inf).topk(2, dim=1).values
            decision_margin = top[:, 0]-top[:, 1]
            near_tie = (local.weights*(multiple & (decision_margin <= 1e-5))).sum()/eligible_mass.clamp_min(1e-30)
            small_margin = (local.weights*(multiple & (decision_margin <= 1e-4))).sum()/eligible_mass.clamp_min(1e-30)
        else:
            near_tie = small_margin = eligible_mass*0.0
        scalar_names = ["selection_loss","pair_loss","q_anchor_loss","shape_loss","trust_loss",
                        "evidence_mass","unknown_pair_prior_mass","q_rmse","advantage_rmse",
                        "label_regret","label_action_agreement","paired_label_gain","action_change_fraction",
                        "numeric_near_tie_fraction","small_margin_fraction","decision_margin_eligible_mass"]
        values = torch.stack([details[k] for k in ("total_raw","pair_raw","value_raw","shape_raw",
            "trust_raw","evidence_mass","unknown_pair_prior_mass")]+[q_rmse,advantage_rmse,regret,agreement,gain,change,
            near_tie,small_margin,eligible_mass]).cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite action-calibration metrics")
    result = dict(zip(scalar_names,map(float,values)))
    result.update(objective_version=OBJECTIVE,roots=data.root_count,families=data.family_count,
                  fixed_reference_preparation=dict(data.preparation_diagnostics),
                  presets=data.preset_count, metric_units="ordinary effective score; losses score squared; fractions dimensionless",
                  numeric_near_tie_threshold_raw=1e-5, small_margin_threshold_raw=1e-4,
                  decision_margin_semantics="conditional original preset weights on roots with >=2 legal actions; FP32 risk indicator, not an error bound; zero when eligible mass is zero",
                  label_regret_semantics="empirical max of teacher-continuation MC means, not optimal regret",
                  paired_gain_semantics="same root/label baseline difference, not whole-game improvement")
    choices = choice.cpu().numpy()
    result.update(data.diagnostics.evaluate(choices))
    # The CPU vector also refreshes within-preset priorities; it never leaves
    # this process in serialized history.
    return result, choices


def _priorities(data, choices, settings):
    """Recompute current-student difficulty without moving mass across presets."""
    rows = np.arange(len(data.roots))
    best = np.where(data.mask_np,data.target_np,-np.inf).argmax(1)
    gaps = data.target_np[rows,best]-data.target_np[rows,choices]
    known = data.pair_known_np[rows,best,choices]
    se = np.where(known,data.pair_se_np[rows,best,choices],np.inf)
    effective = np.maximum(0.,gaps-se)
    values = np.minimum(settings["mistake_weight_max"],1.+settings["mistake_weight_strength"]*
                        effective/(settings["mistake_weight_scale"]+effective))
    return data.weights_np*values


def fit_candidate_calibration(teacher_model, roots, config=None, callback=None,stop_event=None,
                              *,device="cuda",runtime=None):
    """Train a candidate; a frozen per-preset teacher only generates labels."""
    started = time.monotonic()
    settings = _settings(config)
    settings.update(objective_version=OBJECTIVE,refresh_student_actions=True,transfer_diagnostics=True)
    options = {**TORCH_DEFAULTS,**(config or {})}
    loss_config = CalibrationConfig(**dict(options.get("calibration",{})))
    if loss_config.target_scale != TARGET_SCALE:
        raise ValueError("Calibration target scale must match the portable model")
    require_all = options.get("calibration_require_all_presets",True)
    if type(require_all) is not bool:
        raise ValueError("calibration_require_all_presets must be boolean")
    for key in ("gpu_batch_roots","gpu_min_steps_per_epoch","gpu_memory_limit_mb"):
        if type(options[key]) is not int or options[key]<1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("gpu_amp","gpu_tf32"):
        if type(options[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    interval = float(options["gpu_callback_interval_seconds"])
    if not math.isfinite(interval) or interval<0:
        raise ValueError("GPU callback interval must be finite and nonnegative")
    visits = options.get("calibration_visits_per_preset",0)
    if type(visits) is not int or visits<0:
        raise ValueError("calibration_visits_per_preset must be zero (auto) or positive integer")
    sampling = options.get("calibration_sampling", "replacement")
    stopping_option = options.get("calibration_stopping_metric", "checkpoint_metric")
    if stopping_option not in ("checkpoint_metric", "advantage_rmse", "checkpoint_or_advantage_rmse"):
        raise ValueError("calibration_stopping_metric must be checkpoint_metric, advantage_rmse or checkpoint_or_advantage_rmse")
    # Candidate ranking remains on checkpoint_metric. A smooth validation
    # metric can keep patience alive while action gaps improve before argmax
    # changes. The hybrid also preserves progress on the candidate metric if
    # the smooth proxy worsens. Both use already evaluated host scalars.
    hybrid_stopping = stopping_option == "checkpoint_or_advantage_rmse"
    stopping_metric = (settings["checkpoint_metric"] if stopping_option == "checkpoint_metric"
                       else stopping_option)
    stopping_metrics = (tuple(dict.fromkeys((settings["checkpoint_metric"], "advantage_rmse")))
                        if hybrid_stopping else (stopping_metric,))
    stopping_bests = {name: math.inf for name in stopping_metrics}
    stopping_best_epochs = {name: None for name in stopping_metrics}
    priority_correction = options.get("calibration_coverage_priority_correction", False)
    if type(priority_correction) is not bool:
        raise ValueError("calibration_coverage_priority_correction must be boolean")
    current_fraction = options.get("calibration_current_fraction", 0.5)
    training_round = options.get("training_round")
    if sampling not in ("coverage_without_replacement", "stratified_resampling", "replacement"):
        raise ValueError("Invalid calibration_sampling")
    if (isinstance(current_fraction, bool) or not isinstance(current_fraction, (int, float))
            or not math.isfinite(current_fraction) or not 0 <= current_fraction <= 1):
        raise ValueError("calibration_current_fraction must be finite in [0,1]")
    device = torch.device(device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Action calibration requested CUDA but it is unavailable")
    if device.type=="cuda" and device.index is None:
        device = torch.device("cuda",torch.cuda.current_device())
    if device.type not in ("cpu","cuda"):
        raise ValueError("Only CPU and CUDA training are supported")
    before = teacher_model.checksum()
    roots = [dict(root,preset_group=f"catalogue_{root['catalogue_id']:04d}")
             if "catalogue_id" in root else dict(root) for root in roots]
    train,val,teacher_id = prepare_roots(roots,teacher_model.input_dim,settings)
    if require_all:
        from .preset_catalogue import CATALOGUE_SIZE
        wanted = set(range(CATALOGUE_SIZE))
        if ({r.get("catalogue_id") for r in train}!=wanted or
                {r.get("catalogue_id") for r in val}!=wanted):
            raise ValueError("Production calibration requires all 4960 presets in BOTH independent splits")
    model = TorchQ(teacher_model,device)
    all_roots = train+val
    reference = list(torch_teacher_predictions(model,all_roots,device))
    initial_iterator = lambda selected: (reference[i] for i in range(len(reference)))
    initialize_student_diagnostics(all_roots,initial_iterator,before,settings)
    width = max(len(r["q"]) for r in all_roots)
    train_data = CalibrationRoots(train,width,reference[:len(train)],settings,loss_config)
    val_data = CalibrationRoots(val,width,reference[len(train):],settings,loss_config)
    runtime = fit_runtime_options(options,runtime)
    placement,budget,reason = place_root_data(train_data,val_data,device,runtime)
    amp = bool(device.type=="cuda" and options["gpu_amp"] and torch.cuda.is_bf16_supported())
    tf32 = bool(device.type=="cuda" and options["gpu_tf32"])
    batch_size = options["gpu_batch_roots"] if device.type=="cuda" else settings["batch_roots"]
    epoch_visits = visits or math.ceil(len(train)/train_data.preset_count)
    batches = math.ceil(epoch_visits*train_data.preset_count/batch_size)
    groups = [
        {"params":[p for k,p in model.weights.items() if not k.startswith("b") and not k.endswith("_b")],
         "weight_decay":settings["l2"]},
        {"params":[p for k,p in model.weights.items() if k.startswith("b") or k.endswith("_b")],
         "weight_decay":0.0}]
    optimizer = torch.optim.AdamW(groups,lr=settings["learning_rate"],eps=1e-8,
                                  fused=(device.type=="cuda"))
    generator = np.random.default_rng(settings["seed"])
    information = dict(backend="torch_cuda" if device.type=="cuda" else "torch_cpu",
        device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
        torch_version=torch.__version__,precision="BF16 hidden / FP32 head and loss" if amp else "FP32",
        tf32_training=tf32,data_placement=placement,data_placement_reason=reason,
        resident_data_mib=(train_data.bytes+val_data.bytes)/1024**2,data_budget_mib=budget/1024**2,
        effective_batch_roots=batch_size,steps_per_epoch=batches,
        roots_sampled_per_epoch=epoch_visits*train_data.preset_count,visits_per_preset=epoch_visits,
        sampling=sampling,
        calibration_coverage_priority_correction=priority_correction,
        calibration_stopping_metric=stopping_option,
        stopping_rule="reset patience if any monitored metric improves its own best; epoch0 is baseline only",
        current_round_fraction_floor_requested=current_fraction,
        current_round_quota_requested=(sampling != "replacement" and training_round is not None),
        sampling_preset_mass="exact equal preset quota; priorities operate only within each preset",
        train_presets=train_data.preset_count,val_presets=val_data.preset_count,
        all_presets_required=require_all,calibration=asdict(loss_config),
        fixed_loss_preparation="cached_paired_reliability_shape_and_coherent_trust_reference_v1")
    train_metrics,choices = evaluate_calibration(model,train_data,settings,loss_config)
    val_metrics,_ = evaluate_calibration(model,val_data,settings,loss_config)
    history = [dict(kind="epoch",epoch=0,eligible_candidate=False,train=train_metrics,val=val_metrics,
                    selection_metric=settings["checkpoint_metric"],
                    selection_score=val_metrics[settings["checkpoint_metric"]],
                    stopping_metric=stopping_metric,
                    stopping_value=None if hybrid_stopping else val_metrics[stopping_metric],
                    stopping_best_value=None,stopping_best_epoch=None,stopping_stale_epochs=0,
                    stopping_values={name: val_metrics[name] for name in stopping_metrics},
                    stopping_best_values={name: None for name in stopping_metrics},
                    stopping_best_epochs=dict(stopping_best_epochs),stopping_improved_metrics=[],
                    elapsed_seconds=time.monotonic()-started,**information)]
    interrupted = _stopped(stop_event)
    if callback is not None and callback(dict(history[0])) is False:
        interrupted = True
    pool = CandidatePool(settings["checkpoint_candidates"],before,settings["preserve_early_checkpoints"],
                         settings["prefer_first_four_checkpoints"])
    best,best_epoch,best_score,stale,steps = None,0,math.inf,0,0
    stopping_best_epoch = 0
    last_report = time.monotonic()
    for epoch in range(1,settings["epochs"]+1):
        if interrupted or _stopped(stop_event):
            interrupted=True
            break
        priorities = _priorities(train_data,choices,settings)
        order,quota = exact_preset_epoch(train,priorities,generator,epoch_visits,
            sampling=sampling, current_round=training_round, current_fraction=current_fraction,
            require_current=bool(require_all and training_round is not None and current_fraction > 0))
        visit_loss_weights = None
        if priority_correction and sampling == "coverage_without_replacement":
            weights, correction_report = coverage_priority_weights(train,priorities,order,
                preset_indices=train_data.preset_indices_np,batch_size=batch_size)
            quota["priority_correction"] = correction_report
            # One small CPU computation/transfer per epoch; aligned slices need
            # no extra per-step gather, host synchronization or normalization.
            visit_loss_weights = torch.from_numpy(np.asarray(weights/batch_size,dtype=np.float32)).to(device)
        elif priority_correction:
            quota["priority_correction"] = {"requested": True,"applied": False,
                                             "reason": "sampling mode is not coverage_without_replacement"}
        order = torch.from_numpy(order).to(train_data.x.device)
        model.train()
        for batch_index,start in enumerate(range(0,len(order),batch_size)):
            if _stopped(stop_event):
                interrupted=True
                break
            x,target,mask,reliability,action_reliability,teacher,ref = train_data.batch(
                order[start:start+batch_size],device,prepared=True)
            optimizer.zero_grad(set_to_none=True)
            progress = steps/max(1,batches*settings["epochs"]-1)
            rate = settings["learning_rate"]*(.25+.75*.5*(1.+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:
                group["lr"] = rate
            with training_precision(device,tf32=tf32):
                prediction = model(x.flatten(0,1),amp=amp).reshape_as(target)
                loss = calibration_loss_prepared(prediction,target,mask,reliability,action_reliability,
                    teacher_actions=teacher,trust_reference=ref,config=loss_config,
                    root_weights=(None if visit_loss_weights is None else
                                  visit_loss_weights[start:start+len(target)]))
                # A short final batch must not give each of its roots a larger
                # loss coefficient than the other exactly scheduled visits.
                if visit_loss_weights is None:
                    loss = loss*(len(target)/batch_size)
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip"],
                                                 error_if_nonfinite=False,foreach=True)
            optimizer.step()
            steps+=1
            if time.monotonic()-last_report>=interval or batch_index+1==batches:
                scalars = torch.stack((loss.detach()*TARGET_SCALE**2,norm.detach())).cpu().numpy()
                if not np.isfinite(scalars).all():
                    raise FloatingPointError("Nonfinite calibration update")
                if callback is not None and callback(dict(kind="train_step",epoch=epoch,epochs=settings["epochs"],batch=batch_index+1,
                    batches=batches,step=steps,loss=float(scalars[0]),gradient_norm=float(scalars[1]),
                    learning_rate=rate,elapsed_seconds=time.monotonic()-started,**information)) is False:
                    interrupted=True
                    break
                last_report=time.monotonic()
        if interrupted:
            break
        train_metrics,choices = evaluate_calibration(model,train_data,settings,loss_config)
        val_metrics,_ = evaluate_calibration(model,val_data,settings,loss_config)
        row = dict(kind="epoch",epoch=epoch,epochs=settings["epochs"],eligible_candidate=True,
            train=train_metrics,val=val_metrics,selection_metric=settings["checkpoint_metric"],
            selection_score=val_metrics[settings["checkpoint_metric"]],optimizer_steps=steps,
            elapsed_seconds=time.monotonic()-started,epoch_quota=quota,**information)
        stopping_values = {name: val_metrics[name] for name in stopping_metrics}
        improved_metrics = []
        for name,value in stopping_values.items():
            if value < stopping_bests[name]:
                stopping_bests[name],stopping_best_epochs[name] = value,epoch
                improved_metrics.append(name)
        if improved_metrics:
            stopping_best_epoch,stale = epoch,0
        else:
            stale += 1
        # There is no single numerical hybrid score: log both independent
        # bests, and the last epoch improving either one, without mixing units.
        row.update(stopping_metric=stopping_metric,
                   stopping_value=None if hybrid_stopping else stopping_values[stopping_metric],
                   stopping_best_value=None if hybrid_stopping else stopping_bests[stopping_metric],
                   stopping_best_epoch=stopping_best_epoch,stopping_stale_epochs=stale,
                   stopping_values=stopping_values,stopping_best_values=dict(stopping_bests),
                   stopping_best_epochs=dict(stopping_best_epochs),
                   stopping_improved_metrics=improved_metrics)
        history.append(row)
        exported = model.export(teacher_model)
        pool.add(exported,row)
        if row["selection_score"]<best_score:
            best,best_epoch,best_score=exported,epoch,row["selection_score"]
        if callback is not None and callback(dict(row)) is False:
            interrupted=True
            break
        if settings["patience"] and stale>=settings["patience"] and epoch>=settings["min_epochs"]:
            break
        if settings["max_seconds"] and time.monotonic()-started>=settings["max_seconds"]:
            break
    if teacher_model.checksum()!=before:
        raise AssertionError("Action calibration mutated the frozen initial model")
    result = best if best is not None else model.export(teacher_model)
    finite_stopping_bests = {name: None if math.isinf(value) else value
                            for name,value in stopping_bests.items()}
    stopping_state = dict(stopping_metric=stopping_metric,stopping_best_epoch=stopping_best_epoch,
        stopping_best_value=None if hybrid_stopping else finite_stopping_bests[stopping_metric],
        stopping_stale_epochs=stale,stopping_best_values=finite_stopping_bests,
        stopping_best_epochs=dict(stopping_best_epochs))
    result.metadata.update(trained_against_teacher=teacher_id,best_epoch=best_epoch,label_best_epoch=best_epoch,
        objective_version=OBJECTIVE,optimizer_steps=steps,trained_candidate=best_epoch>0,
        training_interrupted=interrupted,checkpoint_metric=settings["checkpoint_metric"],
        training_backend=information,initial_parameter_sha256=before,
        initial_student_model_id=settings.get("initial_student_model_id"),**stopping_state)
    checkpoints=pool.selected()
    for point in checkpoints:
        point["model"].metadata.update(trained_against_teacher=teacher_id,best_epoch=point["epoch"],
            label_best_epoch=best_epoch,
            objective_version=OBJECTIVE,optimizer_steps=point["optimizer_steps"],trained_candidate=True,
            initial_parameter_sha256=before,initial_student_model_id=settings.get("initial_student_model_id"),
            checkpoint_metric=settings["checkpoint_metric"],
            training_backend=information,training_interrupted=interrupted,**stopping_state)
    return FitResult(result,best_epoch,history,train_data.family_count,val_data.family_count,
        interrupted,len(train),len(val),teacher_id,time.monotonic()-started,steps,
        settings["checkpoint_metric"],checkpoints,initial_parameter_sha256=before)
