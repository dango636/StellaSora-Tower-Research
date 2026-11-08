"""Train an unconstrained candidate from complete root action-return labels.

For each root, labels are Q^pi(s,a)=E[terminal_score-current_score] after taking
action a and then following the round's frozen teacher pi. Each root contains
all modeled legal actions. A full family stays on one side of development
validation. Only actual trained epochs can become a candidate; epoch zero is
reported diagnostically and never suppresses the independent game evaluation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Callable

import numpy as np

from .features import TARGET_SCALE
from .fit_diagnostics import initialize_student_diagnostics, TransferDiagnostics
from .network import AdamW, QNetwork
from .indexing import action_index

DEFAULTS = {"epochs": 12, "batch_roots": 32, "initial_student_model_id": None, "training_round": None, "learning_rate": 8e-4,
            "advantage_weight": .75, "q_anchor_weight": None,
            "objective_version": "q_advantage_ranking_v1", "refresh_student_actions": False, "transfer_diagnostics": False, "l2": 1e-5, "seed": 0,
            "huber_delta": 2.0, "gradient_clip": 5.0,
            "validation_fraction": .2, "patience": 4, "min_epochs": 4,
            "max_seconds": 0.0, "checkpoint_metric": "advantage_rmse",
            "mistake_weight_strength": 0.0, "mistake_weight_scale": .25,
            "mistake_weight_max": 3.0,
            "ranking_weight": 0.0, "ranking_min_gap": .03,
            "ranking_confidence_z": 1.5, "ranking_top_k": 3,
            "ranking_temperature": .1, "ranking_critical_boost": 2.0,
            "checkpoint_candidates": 1, "preserve_early_checkpoints": False,
            "prefer_first_four_checkpoints": False,
            "advantage_confidence_weighting": False, "confidence_floor": .25,
            "confidence_z": 1.5, "validation_unit": "preset"}


@dataclass
class FitResult:
    model: QNetwork
    best_epoch: int
    history: list[dict]
    train_families: int
    val_families: int
    interrupted: bool
    train_roots: int = 0
    val_roots: int = 0
    teacher_id: str = ""
    elapsed_seconds: float = 0.0
    optimizer_steps: int = 0
    selection_metric: str = "advantage_rmse"

    candidate_checkpoints: list[dict] = field(default_factory=list)
    initial_parameter_sha256: str | None = None

    @property
    def candidate(self):
        return self.model

    def summary(self) -> dict:
        result = {key: getattr(self, key) for key in (
            "best_epoch", "history", "train_families", "val_families", "interrupted",
            "train_roots", "val_roots", "teacher_id", "elapsed_seconds", "optimizer_steps",
            "selection_metric", "initial_parameter_sha256")}
        result["candidate_checkpoints"] = [{k: v for k, v in point.items() if k != "model"}
                                           for point in self.candidate_checkpoints]
        return result


def _settings(config):
    settings = {**DEFAULTS, **(config or {})}
    # These aliases make the training unit explicit in external configurations.
    if "weight_decay" in settings:
        settings["l2"] = settings["weight_decay"]
    if "batch_size" in settings:
        settings["batch_roots"] = settings["batch_size"]
    for name in ("epochs", "batch_roots", "patience", "min_epochs", "seed"):
        settings[name] = int(settings[name])
    if min(settings["epochs"], settings["batch_roots"]) < 1 or min(settings["patience"], settings["min_epochs"]) < 0:
        raise ValueError("Invalid training epoch or batch counts")
    for name in ("learning_rate", "advantage_weight", "l2", "huber_delta", "gradient_clip", "validation_fraction", "max_seconds"):
        settings[name] = float(settings[name])
        if not math.isfinite(settings[name]):
            raise ValueError("Training settings must be finite")
    if (settings["learning_rate"] <= 0 or settings["l2"] < 0 or settings["huber_delta"] <= 0
        or settings["gradient_clip"] <= 0 or not 0 <= settings["advantage_weight"] <= 1
        or not 0 < settings["validation_fraction"] < 1 or settings["max_seconds"] < 0):
        raise ValueError("Invalid training settings")
    if settings["checkpoint_metric"] not in ("advantage_rmse", "label_regret", "selection_loss"):
        raise ValueError("checkpoint_metric must be advantage_rmse, label_regret or selection_loss")
    if settings["validation_unit"] not in ("preset", "trajectory"):
        raise ValueError("validation_unit must be preset or trajectory")
    for name in ("mistake_weight_strength", "mistake_weight_scale", "mistake_weight_max"):
        settings[name] = float(settings[name])
    if (settings["mistake_weight_strength"] < 0 or settings["mistake_weight_scale"] <= 0
        or settings["mistake_weight_max"] < 1):
        raise ValueError("Invalid mistake sampling settings")
    for name in ("ranking_weight", "ranking_min_gap", "ranking_confidence_z",
                 "ranking_temperature", "ranking_critical_boost"):
        settings[name] = float(settings[name])
    settings["ranking_top_k"] = int(settings["ranking_top_k"])
    settings["checkpoint_candidates"] = min(6, max(1, int(settings["checkpoint_candidates"])))
    if (min(settings["ranking_weight"], settings["ranking_min_gap"], settings["ranking_confidence_z"]) < 0
        or settings["ranking_temperature"] <= 0 or settings["ranking_top_k"] < 1
        or settings["ranking_critical_boost"] < 1):
        raise ValueError("Invalid confident ranking settings")
    settings["confidence_floor"] = float(settings["confidence_floor"])
    settings["confidence_z"] = float(settings["confidence_z"])
    if not 0 <= settings["confidence_floor"] <= 1 or not math.isfinite(settings["confidence_z"]) or settings["confidence_z"] < 0:
        raise ValueError("Invalid advantage confidence settings")
    if settings["q_anchor_weight"] is not None:
        settings["q_anchor_weight"] = float(settings["q_anchor_weight"])
        if not math.isfinite(settings["q_anchor_weight"]) or settings["q_anchor_weight"] < 0:
            raise ValueError("q_anchor_weight must be finite and nonnegative")
    if not isinstance(settings["objective_version"], str) or not settings["objective_version"]:
        raise ValueError("objective_version must be a nonempty string")
    for name in ("refresh_student_actions", "transfer_diagnostics"):
        if type(settings[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if settings["initial_student_model_id"] is not None and (
            not isinstance(settings["initial_student_model_id"], str) or not settings["initial_student_model_id"]):
        raise ValueError("initial_student_model_id must be a nonempty string or None")
    if settings["training_round"] is not None and (
            type(settings["training_round"]) is not int or settings["training_round"] < 1):
        raise ValueError("training_round must be a positive integer or None")
    return settings


def prepare_roots(roots, input_dim, settings) -> tuple[list[dict], list[dict], str]:
    """Validate labels and split whole source trajectories, not action rows."""
    normalized = []
    teachers, family_group = set(), {}
    family_teachers, preset_teachers, preset_splits = {}, {}, {}
    for root in roots:
        x = np.asarray(root["x"], dtype=np.float32)
        q = np.asarray(root["q"], dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != input_dim or q.shape != (len(x),) or not len(x):
            raise ValueError("Each root needs x[actions,features] and q[actions]")
        if not np.isfinite(x).all() or not np.isfinite(q).all() or (q < -1e-5).any():
            raise ValueError("Root labels and features must be finite and returns nonnegative")
        if "cap" in root and ((q > float(root["cap"])+1e-5).any() or not 72 <= float(root["cap"]) <= 144):
            raise ValueError("Out-of-scope preset or physically impossible root return")
        # Validate on the host before any GPU gather can see these indices.
        # -1/None remains an unknown decision, never Python's last action.
        for name in ("student_action_index", "teacher_action_index"):
            action_index(root.get(name), len(q), name)
        family = str(root["family_id"])
        teacher = str(root["teacher_id"])
        group = str(root.get("preset_group", "all"))
        if not family or not teacher:
            raise ValueError("Every root requires a source family and frozen teacher identity")
        teachers.add(teacher)
        if settings.get("per_preset_teacher_bank"):
            if family_teachers.setdefault(family, teacher) != teacher:
                raise ValueError("A source family cannot change its frozen teacher")
            preset_id = root["catalogue_id"]
            if preset_teachers.setdefault(preset_id, teacher) != teacher:
                raise ValueError("A catalogue preset cannot mix frozen teachers in one fit")
            if (settings["validation_unit"] == "preset" and
                    preset_splits.setdefault(preset_id, root.get("split")) != root.get("split")):
                raise ValueError("A catalogue preset leaks across training and development")
        if family in family_group and family_group[family] != group:
            raise ValueError("A source family cannot change preset group")
        family_group[family] = group
        split = root.get("split")
        if split in (0, "training"):
            split = "train"
        elif split in (1, "validation", "development", "dev"):
            split = "val"
        if split not in (None, "train", "val"):
            raise ValueError("Training roots must not contain a final evaluation/test split")
        normalized.append({**{key: root[key] for key in (
                               "decision_regret", "decision_regret_se", "hard_example",
                               "hard_kind", "training_priority", "hard_sample_slot",
                               "action_returns", "action_keys", "standard_error", "pair_se",
                               "student_action_index", "teacher_action_index",
                               "teacher_student_disagreement", "catalogue_id", "behavior_source",
                               "behavior_model_id", "behavior_sha256", "student_model_id",
                               "student_sha256", "teacher_sha256", "student_parameter_sha256",
                               "student_action_source_sha256", "decision_regret_source_sha256",
                               "root_id", "behavior_parameter_sha256", "teacher_parameter_sha256",
                               "initial_student_model_id", "source_environment_seed",
                               "source_exploration_seed", "label_seed", "coverage_protocol",
                               "root_selection_role", "pair_known", "source_round") if key in root},
                           "x": x, "q": q, "family_id": family, "teacher_id": teacher,
                           "preset_group": group, "split": split})
    if len(teachers) != 1 and not settings.get("per_preset_teacher_bank"):
        raise ValueError("A value-iteration fit may contain only one frozen teacher's returns")
    families = set(family_group)
    if len(families) < 2:
        raise ValueError("At least two independent source families are needed")
    if any(root["split"] is not None for root in normalized):
        if any(root["split"] is None for root in normalized):
            raise ValueError("Supply split for every root, or for none")
        assignment = {}
        for root in normalized:
            old = assignment.setdefault(root["family_id"], root["split"])
            if old != root["split"]:
                raise ValueError("A family leaks across training and development validation")
    else:
        assignment = {}
        by_group = defaultdict(list)
        for family, group in family_group.items():
            by_group[group].append(family)
        for group, local in sorted(by_group.items()):
            local.sort(key=lambda family: hashlib.sha256(f"split:{settings['seed']}:{family}".encode()).digest())
            count = max(1, round(len(local)*settings["validation_fraction"])) if len(local) >= 2 else 0
            count = min(count, len(local)-1)
            assignment.update({family: ("val" if i < count else "train") for i, family in enumerate(local)})
        if "val" not in assignment.values():
            assignment[sorted(families)[0]] = "val"
        for root in normalized:
            root["split"] = assignment[root["family_id"]]
    train = [root for root in normalized if root["split"] == "train"]
    val = [root for root in normalized if root["split"] == "val"]
    if not train or not val:
        raise ValueError("Independent training and development families are both required")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise AssertionError("Family leakage")
    if settings["validation_unit"] == "trajectory":
        train_ids = {r.get("catalogue_id") for r in train}
        val_ids = {r.get("catalogue_id") for r in val}
        overlapping = (train_ids & val_ids) - {None}
        source_splits, label_splits = {}, {}
        for root in normalized:
            if root.get("catalogue_id") not in overlapping:
                continue
            if "source_environment_seed" not in root or "label_seed" not in root:
                raise ValueError("Shared-preset validation requires source and label seed provenance")
            key = (root["catalogue_id"], root["source_environment_seed"])
            if source_splits.setdefault(key, root["split"]) != root["split"]:
                raise ValueError("One source environment stream crosses train/validation")
            label_key = (root["catalogue_id"], root["label_seed"])
            if label_splits.setdefault(label_key, root["split"]) != root["split"]:
                raise ValueError("One rollout-label stream crosses train/validation")
    teacher_identity = ("bank:" + hashlib.sha256(",".join(sorted(teachers)).encode()).hexdigest()[:16]
                        if settings.get("per_preset_teacher_bank") else next(iter(teachers)))
    return train, val, teacher_identity


def balanced_weights(roots) -> np.ndarray:
    """Equal stratum, equal family within stratum, equal root within family."""
    grouped = defaultdict(lambda: defaultdict(list))
    for i, root in enumerate(roots):
        grouped[root["preset_group"]][root["family_id"]].append(i)
    weights = np.zeros(len(roots), np.float64)
    for families in grouped.values():
        for indices in families.values():
            weights[indices] = 1.0 / (len(grouped)*len(families)*len(indices))
    return weights / weights.sum()


def mistake_sampling_weights(roots, settings, predict=None) -> np.ndarray:
    """Static frozen-teacher mistakes; retain each preset group's original mass.

    Paired standard error discounts noisy regret by one SE. A bounded smooth
    multiplier changes sampling only; the loss and development measure stay
    unchanged. ``predict`` returns one FP32 action vector per supplied root.
    """
    base = balanced_weights(roots)
    if settings["mistake_weight_strength"] <= 0:
        return base
    missing = [root for root in roots if "decision_regret" not in root or
               ("decision_regret_se" not in root and "action_returns" in root)]
    if missing:
        # Transfer initialization already evaluated these exact frozen-student
        # actions. Reuse only the baseline refreshed in this fit; do not run a
        # second network pass solely to fill legacy priority metadata.
        cached = {id(root): root["_student_baseline"]["action"] for root in missing
                  if settings["transfer_diagnostics"] and root.get("_student_baseline") is not None}
        unpredicted = [root for root in missing if id(root) not in cached]
        predictions = iter(predict(unpredicted)) if unpredicted else iter(())
        for root in missing:
            chosen = cached[id(root)] if id(root) in cached else int(np.argmax(next(predictions)))
            root.setdefault("decision_regret", float(root["q"].max()-root["q"][chosen]))
            if "decision_regret_se" not in root and "action_returns" in root:
                returns = np.asarray(root["action_returns"])
                paired_difference = returns[int(np.argmax(root["q"]))]-returns[chosen]
                root["decision_regret_se"] = (float(np.std(paired_difference, ddof=1) /
                    np.sqrt(len(paired_difference))) if len(paired_difference) > 1 else 0.0)
    priorities = []
    for root in roots:
        regret = max(0.0, float(root["decision_regret"]))
        effective = max(0.0, regret-float(root.get("decision_regret_se", 0.0)))
        priority = min(settings["mistake_weight_max"], 1.0 +
                       settings["mistake_weight_strength"]*effective /
                       (settings["mistake_weight_scale"]+effective))
        targeted = bool(root.get("hard_sample_slot", False))
        root.update(hard_example=targeted or effective > 0, hard_kind=(
            root.get("hard_kind", "targeted_sample") if targeted else "teacher_label_mistake" if effective > 0 else "none"),
            training_priority=priority)
        priorities.append(priority)
    result = base*np.asarray(priorities)
    # One pass instead of scanning the entire root list once per preset.
    # Integer indexing preserves root order and therefore the exact reduction
    # order of the former boolean mask (important for deterministic fitting).
    grouped_indices = defaultdict(list)
    for index, root in enumerate(roots):
        grouped_indices[root["preset_group"]].append(index)
    for indices in grouped_indices.values():
        result[indices] *= base[indices].sum()/result[indices].sum()
    return result


def ranking_pairs(root, settings):
    """Confident label-best versus plausible rivals, using paired rollout SE.

    Unknown uncertainty never creates a hard ordering. The squared smooth
    hinge stops pushing an already adequate action gap; its gain weights
    and selective top competitors differ from all-pairs centered regression.
    """
    if settings["ranking_weight"] <= 0 or len(root["q"]) < 2:
        return np.empty((0, 4), np.float32)
    q = root["q"]
    winner = int(np.argmax(q))
    keys = root.get("action_keys", [""]*len(q))
    def expensive(key):
        return ("BUY" in key and key.endswith("_0160")) or ("ENHANCE" in key and key.endswith("_0180"))
    def skip(key):
        return "RESTOCK" in key or "LEAVE" in key
    def critical(j):
        return (expensive(keys[winner]) and skip(keys[j])) or (skip(keys[winner]) and expensive(keys[j]))
    rivals = set(int(j) for j in np.argsort(-q)[:settings["ranking_top_k"]+1] if j != winner)
    rivals.update(j for j in range(len(q)) if j != winner and critical(j))
    # Include the student's actual decision even when it is outside label top-k.
    # Rollout confidence below decides whether this pair receives any gradient.
    student_action = action_index(root.get("student_action_index"), len(q), "student_action_index")
    if student_action is not None and student_action != winner:
        rivals.add(student_action)
    returns = root.get("action_returns")
    pairs = []
    for j in sorted(rivals):
        gap = float(q[winner]-q[j])
        if returns is not None and np.shape(returns)[1] > 1:
            delta = np.asarray(returns)[winner]-np.asarray(returns)[j]
            se = float(np.std(delta, ddof=1)/np.sqrt(len(delta)))
        elif "pair_se" in root:
            se = float(np.asarray(root["pair_se"])[winner, j])
        elif "standard_error" in root:
            se = float(np.hypot(root["standard_error"][winner], root["standard_error"][j]))
        else:
            continue
        # Unknown/nonfinite uncertainty cannot establish a hard ordering.
        if not math.isfinite(se) or se < 0:
            continue
        confident_gap = gap-settings["ranking_confidence_z"]*se
        if confident_gap < settings["ranking_min_gap"] or confident_gap <= 0:
            continue
        # Bound per-pair gain while retaining absolute confidence magnitude.
        weight = min(3.0, confident_gap) * (settings["ranking_critical_boost"] if critical(j) else 1.0)
        pairs.append((winner, j, weight, min(confident_gap, 1.0)/TARGET_SCALE))
    return np.asarray(pairs, np.float32).reshape(-1, 4)


def ranking_loss_and_gradient(predictions, roots, root_weights, settings):
    gradient = np.zeros_like(predictions)
    total, cursor = 0.0, 0
    temperature = settings["ranking_temperature"]/TARGET_SCALE
    for root, root_weight in zip(roots, root_weights):
        pairs = root.get("ranking_pairs")
        if pairs is None:
            pairs = ranking_pairs(root, settings)
        for winner, rival, weight, margin in pairs:
            i, j = cursor+int(winner), cursor+int(rival)
            z = (float(margin)-float(predictions[i]-predictions[j]))/temperature
            hinge = temperature*np.logaddexp(0.0, z)
            factor = settings["ranking_weight"]*float(root_weight)*float(weight)/max(1, len(pairs))
            total += factor*hinge*hinge
            derivative = 2*factor*hinge*np.exp(-np.logaddexp(0.0, -z))
            gradient[i] -= derivative
            gradient[j] += derivative
        cursor += len(root["q"])
    return float(total), gradient


def advantage_confidence(root, settings):
    """Paired best-action uncertainty softens near ties, without rescaling mass."""
    q = np.asarray(root["q"])
    weights = np.ones(len(q), np.float32)
    if not settings["advantage_confidence_weighting"] or len(q) < 2 or "action_returns" not in root:
        return weights
    returns = np.asarray(root["action_returns"], dtype=np.float64)
    if returns.shape[1] < 2:
        return weights
    winner = int(np.argmax(q))
    gaps = np.maximum(0., q[winner]-q).astype(np.float64)
    se = np.std(returns[winner]-returns, axis=1, ddof=1)/np.sqrt(returns.shape[1])
    floor = settings["confidence_floor"]
    weights[:] = floor+(1-floor)*gaps/(gaps+settings["confidence_z"]*se+1e-6)
    runner = max((j for j in range(len(q)) if j != winner), key=lambda j: q[j])
    weights[winner] = weights[runner]
    return weights


def confidence_diagnostics(roots, settings, *, local=None, mass=None):
    if not settings["advantage_confidence_weighting"]:
        return {}
    mass = balanced_weights(roots) if mass is None else mass
    local = [advantage_confidence(root, settings) for root in roots] if local is None else local
    return {"advantage_confidence_mean": float(sum(w*a.mean() for w, a in zip(mass, local))),
            "advantage_downweighted_fraction": float(sum(w*np.mean(a < .999) for w, a in zip(mass, local)))}


class CandidatePool:
    """Keep the original metric best plus diverse actual trained epochs."""
    def __init__(self, count, teacher_checksum, preserve_early_checkpoints=False,
                 prefer_first_four_checkpoints=False):
        self.count, self.teacher_checksum, self.points = count, teacher_checksum, []
        self.preserve_early_checkpoints = preserve_early_checkpoints
        self.prefer_first_four_checkpoints = prefer_first_four_checkpoints

    def add(self, model, metrics):
        checksum = model.checksum()
        if checksum == self.teacher_checksum or any(p["checksum"] == checksum for p in self.points):
            return
        self.points.append(dict(epoch=metrics["epoch"], model=model, val=dict(metrics["val"]),
                                optimizer_steps=metrics["optimizer_steps"], checksum=checksum,
                                selection_score=metrics["selection_score"]))
        # A small pool permits re-binning over actual completed epochs after
        # early stopping, without retaining every full network snapshot.
        # Six-point development keeps CPU snapshots until early stopping so
        # later selection can cover the actual training span, not only loss minima.
        if self.count <= 3 and len(self.points) > 9:
            best = min(self.points, key=lambda p: p["selection_score"])
            others = sorted((p for p in self.points if p is not best),
                            key=lambda p: (p["val"]["label_regret"], -abs(p["epoch"]-best["epoch"])))
            reserved = [p for p in others if p["epoch"] in (1, 2, 4)] if self.preserve_early_checkpoints else []
            self.points = [best]+reserved+[p for p in others if p not in reserved][:8-len(reserved)]

    def selected(self):
        if not self.points:
            return []
        chosen = [min(self.points, key=lambda p: p["selection_score"])]
        if self.preserve_early_checkpoints:
            early_epochs = ((1, 2, 3, 4) if self.prefer_first_four_checkpoints else
                            (1, 2) if self.count > 3 else (1, 2, 4))
            for epoch in early_epochs:
                early = next((p for p in self.points if p["epoch"] == epoch), None)
                if early is not None and early not in chosen and len(chosen) < self.count:
                    chosen.append(early)
        if self.prefer_first_four_checkpoints and len(chosen) < self.count:
            late = max(self.points, key=lambda p: p["epoch"])
            if late not in chosen:
                chosen.append(late)
        while len(chosen) < self.count:
            remaining = [p for p in self.points if p not in chosen]
            if not remaining:
                break
            if self.count > 3:
                chosen.append(max(remaining, key=lambda p: (
                    min(abs(p["epoch"]-c["epoch"]) for c in chosen), -p["selection_score"])))
                continue
            separated = [p for p in remaining if min(abs(p["epoch"]-c["epoch"]) for c in chosen) >= 2]
            chosen.append(min(separated or remaining, key=lambda p: (p["val"]["label_regret"],
                -min(abs(p["epoch"]-c["epoch"]) for c in chosen))))
        return [{k: v for k, v in p.items() if k != "selection_score"} for p in sorted(chosen, key=lambda p: p["epoch"])]


def numpy_teacher_predictions(model, roots):
    for start in range(0, len(roots), 64):
        batch = roots[start:start+64]
        values = model.forward_normalized(np.concatenate([root["x"] for root in batch]))
        cursor = 0
        for root in batch:
            count = len(root["q"])
            yield values[cursor:cursor+count]
            cursor += count


def loss_and_gradient(predictions, targets, lengths, root_weights, *, advantage_weight=.75,
                      huber_delta=2.0 / TARGET_SCALE, confidence_weights=None, q_anchor_weight=None):
    """Normalized-unit objective and analytic derivative, grouped by root.

    Absolute Q uses Huber loss. Centered advantages use optionally weighted
    squared residuals. Differentiating through their unweighted root mean
    requires centering the weighted derivative, not just the residual.
    """
    anchor_weight = 1-advantage_weight if q_anchor_weight is None else q_anchor_weight
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    derivative = np.empty_like(predictions)
    cursor, total = 0, 0.0
    for count, weight in zip(lengths, root_weights):
        sl = slice(cursor, cursor+count)
        error = predictions[sl]-targets[sl]
        absolute = np.abs(error)
        huber = np.where(absolute <= huber_delta, .5*error*error,
                         huber_delta*(absolute-.5*huber_delta))
        anchor_gradient = np.clip(error, -huber_delta, huber_delta)
        advantage_error = error-error.mean()
        advantage_square = advantage_error*advantage_error
        advantage_gradient = 2.0*advantage_error
        if confidence_weights is not None:
            advantage_square = confidence_weights[sl]*advantage_square
            advantage_gradient *= confidence_weights[sl]
        advantage_gradient -= advantage_gradient.mean()
        local_loss = (anchor_weight*huber.mean() +
                      advantage_weight*np.mean(advantage_square))
        total += float(weight)*float(local_loss)
        derivative[sl] = float(weight)/count*(anchor_weight*anchor_gradient +
                                             advantage_weight*advantage_gradient)
        cursor += count
    if cursor != len(predictions):
        raise ValueError("Root lengths do not cover the prediction batch")
    return total, derivative


def evaluate_fit(model, roots, settings=None, *, diagnostics=None) -> dict:
    settings = _settings(settings)
    weights = balanced_weights(roots)
    loss = q_squared = advantage_squared = regret = agreement = 0.0
    hard_mass = hard_regret = hard_agreement = 0.0
    targeted_mass = targeted_regret = 0.0
    cursor = 0
    chosen_actions = []
    components = {"q_anchor_loss": 0., "advantage_loss": 0., "ranking_loss": 0.}
    # Keep memory independent of total replay size, while batching BLAS calls.
    while cursor < len(roots):
        selected = roots[cursor:cursor+64]
        lengths = [len(root["q"]) for root in selected]
        x = np.concatenate([root["x"] for root in selected])
        y = np.concatenate([root["q"] for root in selected]) / TARGET_SCALE
        p = model.forward_normalized(x)
        local_weights = weights[cursor:cursor+len(selected)]
        local_loss, _ = loss_and_gradient(p, y, lengths, local_weights,
            advantage_weight=settings["advantage_weight"], huber_delta=settings["huber_delta"]/TARGET_SCALE,
            q_anchor_weight=settings["q_anchor_weight"],
            confidence_weights=(np.concatenate([advantage_confidence(r, settings) for r in selected])
                                if settings["advantage_confidence_weighting"] else None))
        rank_loss, _ = ranking_loss_and_gradient(p, selected, local_weights, settings)
        loss += (local_loss+rank_loss) * TARGET_SCALE**2
        anchor, _ = loss_and_gradient(p, y, lengths, local_weights,
            advantage_weight=0., q_anchor_weight=(1-settings["advantage_weight"] if settings["q_anchor_weight"] is None else settings["q_anchor_weight"]),
            huber_delta=settings["huber_delta"]/TARGET_SCALE)
        components["q_anchor_loss"] += anchor*TARGET_SCALE**2
        components["advantage_loss"] += (local_loss-anchor)*TARGET_SCALE**2
        components["ranking_loss"] += rank_loss*TARGET_SCALE**2
        start = 0
        for root, count, weight in zip(selected, lengths, local_weights):
            predicted = p[start:start+count] * TARGET_SCALE
            target = root["q"]
            error = predicted-target
            q_squared += weight*float(np.mean(error*error))
            advantage_squared += weight*float(np.mean((error-error.mean())**2))
            chosen = int(np.argmax(predicted))
            chosen_actions.append(chosen)
            regret += weight*float(target.max()-target[chosen])
            agreement += weight*float(chosen == int(np.argmax(target)))
            if root.get("hard_example", False):
                hard_mass += weight
                hard_regret += weight*float(target.max()-target[chosen])
                hard_agreement += weight*float(chosen == int(np.argmax(target)))
            if root.get("hard_sample_slot", False):
                targeted_mass += weight
                targeted_regret += weight*float(target.max()-target[chosen])
            start += count
        cursor += len(selected)
    return {**components, "objective_version": settings["objective_version"],
            **(diagnostics or TransferDiagnostics(roots, weights, settings)).evaluate(chosen_actions),
            **confidence_diagnostics(roots, settings), "hard_roots": sum(bool(r.get("hard_example")) for r in roots),
            "targeted_roots": sum(bool(r.get("hard_sample_slot")) for r in roots),
            "targeted_label_regret": float(targeted_regret/targeted_mass) if targeted_mass else None,
            "hard_label_regret": float(hard_regret/hard_mass) if hard_mass else None,
            "hard_label_action_agreement": float(hard_agreement/hard_mass) if hard_mass else None,
            "selection_loss": float(loss), "q_rmse": math.sqrt(float(q_squared)),
            "advantage_rmse": math.sqrt(float(advantage_squared)),
            "label_regret": float(regret), "label_action_agreement": float(agreement),
            "roots": len(roots), "families": len({root["family_id"] for root in roots})}


def _stopped(stop_event) -> bool:
    if stop_event is None:
        return False
    if hasattr(stop_event, "is_set"):
        return bool(stop_event.is_set())
    return bool(stop_event())


def fit_candidate(teacher_model: QNetwork, roots, config=None,
                  callback: Callable[[dict], object] | None = None,
                  stop_event=None) -> FitResult:
    settings = _settings(config)
    started = time.monotonic()
    before = teacher_model.checksum()
    train, val, teacher_id = prepare_roots(roots, teacher_model.input_dim, settings)
    capture_identity = settings["objective_version"] != DEFAULTS["objective_version"] or settings["refresh_student_actions"] or settings["transfer_diagnostics"]
    candidate = teacher_model.clone(frozen=False)
    optimizer = AdamW(candidate, settings["learning_rate"], settings["l2"], settings["gradient_clip"])
    predict = lambda selected: numpy_teacher_predictions(teacher_model, selected)
    if settings["refresh_student_actions"] or settings["transfer_diagnostics"]:
        initialize_student_diagnostics(train+val, predict, teacher_model.checksum(), settings)
    probabilities = mistake_sampling_weights(train, settings, predict)
    mistake_sampling_weights(val, settings, predict)
    # Labels and paired uncertainty are immutable during fitting. Compute
    # these once, rather than recomputing per-action return statistics for
    # every replacement-sampled occurrence in every epoch.
    normalized_targets = {id(root): root["q"] / TARGET_SCALE for root in train}
    confidence_cache = ({id(root): advantage_confidence(root, settings) for root in train}
                        if settings["advantage_confidence_weighting"] else None)
    for root in train+val:
        root["ranking_pairs"] = ranking_pairs(root, settings)
    pool = CandidatePool(settings["checkpoint_candidates"], before, settings["preserve_early_checkpoints"],
                         settings["prefer_first_four_checkpoints"])
    train_diagnostics = TransferDiagnostics(train, balanced_weights(train), settings)
    val_diagnostics = TransferDiagnostics(val, balanced_weights(val), settings)
    rng = np.random.default_rng(settings["seed"])
    batch_count = max(1, math.ceil(len(train)/settings["batch_roots"]))
    total_steps = batch_count*settings["epochs"]
    history = [{"epoch": 0, "kind": "epoch", "train": evaluate_fit(candidate, train, settings, diagnostics=train_diagnostics),
                "val": evaluate_fit(candidate, val, settings, diagnostics=val_diagnostics), "elapsed_seconds": time.monotonic()-started,
                "eligible_candidate": False, "selection_metric": settings["checkpoint_metric"]}]
    history[0]["selection_score"] = history[0]["val"][settings["checkpoint_metric"]]
    best, best_epoch, best_loss = None, 0, math.inf
    stale, interrupted = 0, _stopped(stop_event)
    if callback is not None and callback(dict(history[-1])) is False:
        interrupted = True
    for epoch in range(1, settings["epochs"]+1):
        if interrupted or _stopped(stop_event):
            interrupted = True
            break
        epoch_losses = []
        order = rng.choice(len(train), size=len(train), replace=True, p=probabilities)
        for batch_index, start in enumerate(range(0, len(order), settings["batch_roots"]), 1):
            if _stopped(stop_event):
                interrupted = True
                break
            selected = [train[int(i)] for i in order[start:start+settings["batch_roots"]]]
            lengths = [len(root["q"]) for root in selected]
            x = np.concatenate([root["x"] for root in selected])
            target = np.concatenate([normalized_targets[id(root)] for root in selected])
            prediction, cached = candidate.forward_normalized(x, cache=True)
            loss, gradient = loss_and_gradient(prediction, target, lengths,
                np.full(len(selected), 1.0/len(selected)), advantage_weight=settings["advantage_weight"],
                huber_delta=settings["huber_delta"]/TARGET_SCALE,
                q_anchor_weight=settings["q_anchor_weight"],
                confidence_weights=(np.concatenate([confidence_cache[id(r)] for r in selected])
                                    if settings["advantage_confidence_weighting"] else None))
            rank_loss, rank_gradient = ranking_loss_and_gradient(prediction, selected,
                np.full(len(selected), 1.0/len(selected)), settings)
            loss += rank_loss
            gradient += rank_gradient
            progress = optimizer.steps / max(1, total_steps-1)
            learning_rate = settings["learning_rate"] * (.25 + .75*.5*(1+math.cos(math.pi*progress)))
            gradient_norm = optimizer.step(candidate.backward(cached, gradient), learning_rate)
            epoch_losses.append(loss*TARGET_SCALE**2)
            if callback is not None:
                message = {"kind": "train_step", "epoch": epoch, "epochs": settings["epochs"],
                           "batch": batch_index, "batches": batch_count, "step": optimizer.steps,
                           "steps_total": total_steps, "loss": float(epoch_losses[-1]),
                           "learning_rate": learning_rate, "gradient_norm": gradient_norm,
                           "elapsed_seconds": time.monotonic()-started}
                if callback(message) is False:
                    interrupted = True
                    break
        if interrupted:
            break
        metrics = {"kind": "epoch", "epoch": epoch, "epochs": settings["epochs"],
                   "train": evaluate_fit(candidate, train, settings, diagnostics=train_diagnostics),
                   "val": evaluate_fit(candidate, val, settings, diagnostics=val_diagnostics), "eligible_candidate": True,
                   "elapsed_seconds": time.monotonic()-started,
                   "mean_batch_loss": float(np.mean(epoch_losses)), "optimizer_steps": optimizer.steps,
                   "selection_metric": settings["checkpoint_metric"]}
        metrics["selection_score"] = metrics["val"][settings["checkpoint_metric"]]
        history.append(metrics)
        pool.add(candidate.clone(), metrics)
        # Absolute-Q calibration remains an auxiliary training objective. It
        # must not hide worsening action differences when choosing an epoch.
        if metrics["selection_score"] < best_loss:
            best, best_epoch, best_loss = candidate.clone(), epoch, metrics["selection_score"]
            stale = 0
        else:
            stale += 1
        if callback is not None and callback(dict(metrics)) is False:
            interrupted = True
            break
        if settings["patience"] and stale >= settings["patience"] and epoch >= settings["min_epochs"]:
            break
        if settings["max_seconds"] and time.monotonic()-started >= settings["max_seconds"]:
            break
    if teacher_model.checksum() != before:
        raise AssertionError("Training mutated its frozen teacher")
    result_model = best if best is not None else candidate
    result_model.metadata.update({"trained_against_teacher": teacher_id, "best_epoch": best_epoch,
                                  "objective_version": settings["objective_version"],
                                  "q_anchor_weight": settings["q_anchor_weight"],
                                  "optimizer_steps": optimizer.steps, "trained_candidate": best_epoch > 0,
                                  "training_interrupted": interrupted,
                                  "checkpoint_metric": settings["checkpoint_metric"]})
    checkpoints = pool.selected()
    for point in checkpoints:
        point["model"].metadata.update({"trained_against_teacher": teacher_id,
            "best_epoch": point["epoch"], "optimizer_steps": point["optimizer_steps"],
            "objective_version": settings["objective_version"], "q_anchor_weight": settings["q_anchor_weight"],
            "trained_candidate": True, "checkpoint_metric": settings["checkpoint_metric"]})
    return FitResult(result_model, best_epoch, history,
        len({r["family_id"] for r in train}), len({r["family_id"] for r in val}),
        interrupted, len(train), len(val), teacher_id, time.monotonic()-started, optimizer.steps,
        settings["checkpoint_metric"], checkpoints,
        initial_parameter_sha256=before if capture_identity else None)
