"""Pilot-guided *planning* of fresh full-action labels inside one preset/split.

This is an optional CPU planning utility, not a change to the GPU kernels or to
the default labeling pipeline. It keeps every supplied root and every legal
action. Each root receives one common final replication count for all actions.
Teachers, preset weights, and action identities are preserved.

Statistical contract
--------------------
Pilot returns may determine the final sample count, but are NEVER appended to
the final label estimate. The result gives a new domain-separated label_seed
and a fixed final_n for each root. The caller must run exactly that fresh block,
then average only its returns. Conditional on the pilot, this fixed-block mean
does not have the bias of a mean stopped after inspecting its own precision or
winner. An adaptively stopped pilot mean is not an unbiased final label.
This concerns each action's mean: selecting the largest of noisy final means
still has winner-selection optimism and needs separate calibration/evaluation.

Budget contract
---------------
``final_branch_budget`` counts *new final* simulator branches, not the already
spent pilot branches. The latter and the combined cost are reported separately.
For a cap including the pilot, subtract its known cost before calling. Counts
are a compute proxy; one early-state branch can take longer than a late branch.

Allocation is a heuristic, not an optimal-design theorem or a confidence
guarantee. Empirical paired SD and a normal approximation estimate the marginal
reduction in sign-error regret. Uncertain positive, potentially material gaps
get priority; deterministic/resolved comparisons do not. Small positive pilot
gaps have a bounded extra quota and exact zero/negative gaps do not consume an
unbounded precision budget. The common minimum fresh-label coverage protects
against pilot mistakes. Real gains must be checked with independent evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .preset_catalogue import CATALOGUE_SIZE
from .presets import canonical_seed


BUDGET_PROTOCOL = "paired_pilot_fresh_final_budget_v1"
FINAL_SEED_DOMAIN = "paired_label_budget_fresh_final_v1"
BRANCH_SEED_DOMAIN = "complete_teacher_branch"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _tail(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass
class _Root:
    source: Mapping
    actions: int
    pilot_n: int
    pilot_label_seed: int
    pilot_seed_offset: int
    means: np.ndarray
    sd: np.ndarray
    se: np.ndarray
    plausible: np.ndarray
    material_probability: np.ndarray
    final_n: int = 0
    maximum_n: int = 0
    desired: np.ndarray | None = None


def plan_paired_label_budget(
    roots, final_branch_budget, *, seed, round_number,
    min_rollouts=64, max_rollouts=512, min_material_gain=0.05,
    confidence_z=1.96, allocation_block=16, near_tie_extra_rollouts=32,
    near_tie_weight=0.1,
):
    """Plan fresh labels for one frozen ``catalogue_id`` and ``split``.

    Each root requires: ``root_id``, ``catalogue_id``, ``split``, ``teacher_id``,
    ``action_returns[K,Npilot]``, ``student_action_index``,
    ``legal_action_count=K``, unique ``action_keys[K]``, and ``label_seed`` (the
    pilot's seed base). ``pilot_seed_offset`` defaults to zero. Pilot seeds use
    the existing ``complete_teacher_branch`` convention; arbitrary pilot RNG
    schemes must be adapted explicitly before using this helper.

    A small final budget first reduces the common minimum replication count to
    ``floor(budget/sum(K))``. It never drops a root or an action. If even two
    fresh samples per action cannot fit, it raises rather than silently violating
    coverage. Extra replications are full-action bundles and obey max_rollouts.

    The caller should give every preset the same predeclared budget policy and
    keep equal-preset loss weights. This function rejects mixed presets, splits,
    or teachers instead of spending one preset's quota on another.
    """
    budget = _integer(final_branch_budget, "final_branch_budget")
    _integer(seed, "seed")
    _integer(round_number, "round_number", 1)
    minimum = _integer(min_rollouts, "min_rollouts", 2)
    maximum = _integer(max_rollouts, "max_rollouts", 2)
    block = _integer(allocation_block, "allocation_block", 1)
    tie_extra = _integer(near_tie_extra_rollouts, "near_tie_extra_rollouts")
    gain = _positive_float(min_material_gain, "min_material_gain")
    z = _positive_float(confidence_z, "confidence_z")
    if isinstance(near_tie_weight, bool) or not isinstance(near_tie_weight, (int, float)) or not 0 <= near_tie_weight <= 1:
        raise ValueError("near_tie_weight must be in [0,1]")
    if maximum < minimum:
        raise ValueError("max_rollouts cannot be smaller than min_rollouts")
    if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)) or not roots:
        raise ValueError("A nonempty sequence of full-action roots is required")

    prepared, root_ids, identities, pilot_seeds = [], set(), set(), set()
    teacher_checksums = set()
    for source in roots:
        if not isinstance(source, Mapping):
            raise ValueError("Every root must be a mapping")
        required = {"root_id", "catalogue_id", "split", "teacher_id", "action_returns",
                    "student_action_index", "legal_action_count", "action_keys", "label_seed"}
        if required - set(source):
            raise ValueError(f"Missing pilot fields: {sorted(required-set(source))}")
        root_id = source["root_id"]
        if not isinstance(root_id, str) or not root_id or root_id in root_ids:
            raise ValueError("Pilot root IDs must be unique nonempty strings")
        root_ids.add(root_id)
        preset = _integer(source["catalogue_id"], "catalogue_id")
        if preset >= CATALOGUE_SIZE or source["split"] not in ("train", "val"):
            raise ValueError("A known catalogue preset and train/val split are required")
        if not isinstance(source["teacher_id"], str) or not source["teacher_id"]:
            raise ValueError("Frozen teacher_id must be nonempty")
        identities.add((preset, source["split"], source["teacher_id"]))
        if source.get("teacher_sha256") is not None:
            checksum = source["teacher_sha256"]
            if not isinstance(checksum, str) or not checksum:
                raise ValueError("teacher_sha256 must be a nonempty string when supplied")
            teacher_checksums.add(checksum)
        if len(identities) != 1 or len(teacher_checksums) > 1:
            raise ValueError("One planning call must contain one preset, split, and frozen teacher")
        count = _integer(source["legal_action_count"], "legal_action_count", 2)
        keys = source["action_keys"]
        if (not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)) or len(keys) != count
                or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != count):
            raise ValueError("action_keys must describe every legal action exactly once")
        current = _integer(source["student_action_index"], "student_action_index")
        if current >= count:
            raise ValueError("Student action is outside the legal action set")
        raw_values = np.asarray(source["action_returns"])
        if raw_values.dtype.kind not in "fiu":
            raise ValueError("Pilot returns must be real numeric arrays")
        values = raw_values.astype(np.float64, copy=False)
        if (values.ndim != 2 or values.shape[0] != count or values.shape[1] < 2
                or not np.isfinite(values).all() or (values < 0).any()):
            raise ValueError("Pilot returns must be finite nonnegative [all actions, >=2 paired seeds]")
        if "cap" in source and "root_score" in source:
            remaining = float(source["cap"]) - float(source["root_score"])
            if not math.isfinite(remaining) or remaining < 0 or (values > remaining).any():
                raise ValueError("Pilot returns exceed the remaining possible score")
        label_seed = _integer(source["label_seed"], "label_seed")
        offset = _integer(source.get("pilot_seed_offset", 0), "pilot_seed_offset")
        n = values.shape[1]
        actual_pilot_seeds = {canonical_seed(BRANCH_SEED_DOMAIN, label_seed, k)
                              for k in range(offset, offset+n)}
        if len(actual_pilot_seeds) != n or pilot_seeds & actual_pilot_seeds:
            raise ValueError("Duplicate pilot seed blocks; fix provenance before planning")
        pilot_seeds.update(actual_pilot_seeds)
        differences = values-values[current]
        means = differences.mean(axis=1)
        sd = differences.std(axis=1, ddof=1)
        if not np.isfinite(means).all() or not np.isfinite(sd).all():
            raise ValueError("Pilot gap statistics overflowed")
        se = sd/math.sqrt(n)
        plausible = (means > 0) & (means + z*se > gain) & (sd > 0)
        plausible[current] = False
        probability = np.zeros(count, dtype=np.float64)
        for action in np.flatnonzero(plausible):
            probability[action] = _tail((gain-means[action])/se[action])
        prepared.append(_Root(source, count, n, label_seed, offset, means, sd, se,
                              plausible, probability))

    sum_actions = sum(root.actions for root in prepared)
    floor = min(minimum, budget//sum_actions)
    if floor < 2:
        raise ValueError(f"Budget cannot preserve all roots/actions: need at least {2*sum_actions} fresh branches")
    for root in prepared:
        root.final_n = floor
        desired = np.full(root.actions, floor, dtype=np.int64)
        for action in np.flatnonzero(root.plausible):
            # An indifference scale in the denominator prevents the precision
            # target from diverging as a pilot gap approaches zero. A further
            # near-tie cap limits low-value precision even with a huge budget.
            precision_n = math.ceil((z*root.sd[action]/max(root.means[action], gain))**2)
            cap = maximum if root.means[action] > gain else min(maximum, floor+tie_extra)
            desired[action] = max(floor, min(cap, precision_n))
        root.desired = desired
        root.maximum_n = int(desired.max())

    def marginal(root, amount):
        best = 0.0
        before, after = root.final_n, root.final_n+amount
        for action in np.flatnonzero(root.plausible & (root.desired > before)):
            mu, sd = float(root.means[action]), float(root.sd[action])
            effective_gain = max(mu-gain, 0.0) + float(near_tie_weight)*min(mu, gain)
            reduction = _tail(mu*math.sqrt(before)/sd)-_tail(mu*math.sqrt(min(after, int(root.desired[action])))/sd)
            value = float(root.material_probability[action])*effective_gain*max(0.0, reduction)
            # One material competitor is enough to justify a root. Summing all
            # comparisons would automatically favor states with more actions.
            best = max(best, value)
        return best/(root.actions*amount)

    spent = floor*sum_actions
    increments = 0
    while True:
        best = None
        for root in prepared:
            amount = min(block, root.maximum_n-root.final_n, (budget-spent)//root.actions)
            if amount <= 0:
                continue
            priority = marginal(root, amount)
            if priority <= 0:
                continue
            candidate = (priority, root.source["root_id"], root, amount)
            if best is None or priority > best[0] or (priority == best[0] and candidate[1] < best[1]):
                best = candidate
        if best is None:
            break
        _, _, root, amount = best
        root.final_n += amount
        spent += root.actions*amount
        increments += 1

    preset, split, teacher = next(iter(identities))
    allocations, final_seeds = [], set()
    for root in prepared:
        source = root.source
        final_label_seed = canonical_seed(FINAL_SEED_DOMAIN, seed, round_number,
                                          preset, split, source["root_id"])
        local = {canonical_seed(BRANCH_SEED_DOMAIN, final_label_seed, k) for k in range(root.final_n)}
        if len(local) != root.final_n or local & pilot_seeds or local & final_seeds:
            raise ValueError("Pilot/final seed collision; fresh-label independence is not established")
        final_seeds.update(local)
        allocations.append({
            "root_id": source["root_id"], "catalogue_id": preset, "split": split,
            "teacher_id": teacher, "action_keys": list(source["action_keys"]),
            "legal_action_count": root.actions, "student_action_index": source["student_action_index"],
            "pilot_n": root.pilot_n, "pilot_label_seed": root.pilot_label_seed,
            "pilot_seed_offset": root.pilot_seed_offset,
            "final_n": root.final_n, "rollouts_per_action": root.final_n,
            "label_seed": final_label_seed, "rollout_seed_offset": 0,
            "branch_seed_domain": BRANCH_SEED_DOMAIN,
            "final_branches": root.actions*root.final_n,
            "all_legal_actions_share_final_n": True, "reuse_pilot_returns": False,
            "paired_gap_mean": root.means.tolist(), "paired_gap_pilot_se": root.se.tolist(),
            "plausible_positive_contenders": np.flatnonzero(root.plausible).tolist(),
            "pilot_planned_maximum_n": root.maximum_n,
            **({"teacher_sha256": source["teacher_sha256"]} if "teacher_sha256" in source else {}),
        })
    if (spent > budget or len(allocations) != len(roots)
            or any(not floor <= row["final_n"] <= maximum for row in allocations)):
        raise AssertionError("Planner violated branch budget, root coverage, or replication bounds")
    pilot_cost = sum(root.actions*root.pilot_n for root in prepared)
    return {
        "protocol": BUDGET_PROTOCOL, "catalogue_id": preset, "split": split, "teacher_id": teacher,
        "allocations": allocations,
        "summary": {
            "roots": len(roots), "roots_dropped": 0, "legal_actions": sum_actions,
            "teacher_changed": False, "preset_weight_changed": False,
            "requested_min_rollouts": minimum, "effective_min_rollouts": floor,
            "max_rollouts": maximum, "final_branch_budget": budget,
            "final_branches": spent, "unused_final_branch_budget": budget-spent,
            "pilot_branches_already_spent": pilot_cost,
            "combined_pilot_and_final_branches": pilot_cost+spent,
            "allocation_increments": increments,
            "final_seed_domain": FINAL_SEED_DOMAIN, "fresh_seed_blocks_verified": True,
            "label_estimator": "mean of fresh final block only; pilot excluded",
            "priority_model": "pilot paired-SD normal sign-error regret heuristic",
            "min_material_gain": gain, "confidence_z": z,
            "near_tie_extra_rollouts": tie_extra, "near_tie_weight": float(near_tie_weight),
            "unused_budget_policy": "keep unused when comparisons are resolved, immaterial, capped, or a full-action bundle does not fit",
        },
    }
