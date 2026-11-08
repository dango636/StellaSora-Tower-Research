"""Frozen-initial-student diagnostics; paired rollouts are evidence, not truth.

The only network pass is at fit initialization. Epoch evaluation reuses argmax
indices already computed for metrics and cached paired SE vectors.
"""
from __future__ import annotations

import re

import numpy as np

from .indexing import action_index


def _paired_se(root, reference):
    count = len(root["q"])
    returns = root.get("action_returns")
    if returns is not None:
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim == 2 and values.shape[0] == count and values.shape[1] > 1:
            return np.std(values-values[reference], axis=1, ddof=1)/np.sqrt(values.shape[1])
    if "pair_se" in root:
        return np.asarray(root["pair_se"], dtype=np.float64)[reference].copy()
    # Unpaired SE is not evidence of paired superiority.
    return np.full(count, np.nan)


def initialize_student_diagnostics(roots, predict, checksum, settings):
    """Refresh mutable fit copies only, leaving archived labels untouched."""
    for root, predictions in zip(roots, predict(roots)):
        action = int(np.argmax(predictions))
        q = root["q"]
        best = int(np.argmax(q))
        se = _paired_se(root, action)
        teacher = action_index(root.get("teacher_action_index"), len(q), "teacher_action_index")
        if settings.get("refresh_student_actions"):
            root.update(student_action_index=action,
                        student_action_source_sha256=checksum,
                        decision_regret=float(q[best]-q[action]),
                        decision_regret_se=float(se[best]) if np.isfinite(se[best]) else float(q[best]-q[action]),
                        decision_regret_source_sha256=checksum,
                        student_parameter_sha256=checksum)
            # Unknown paired uncertainty cancels effective regret for refreshed
            # priority, rather than fabricating confidence from a zero SE.
            root["decision_regret_paired_se_available"] = bool(np.isfinite(se[best]))
            if teacher is not None:
                root["teacher_student_disagreement"] = int(teacher) != action
        if settings.get("transfer_diagnostics"):
            teacher_gain = None if teacher is None else float(q[int(teacher)]-q[action])
            teacher_se = np.nan if teacher is None else se[int(teacher)]
            root["_student_baseline"] = dict(action=action, checksum=checksum, paired_se=se,
                regret=float(q[best]-q[action]),
                disagreement=teacher is not None and int(teacher) != action,
                teacher_supported=(teacher_gain is not None and np.isfinite(teacher_se)
                                   and teacher_gain > settings["confidence_z"]*teacher_se),
                paired_available=bool(np.isfinite(se).all()))


class TransferDiagnostics:
    """Compact cached root arrays, vectorized epoch aggregation without rollouts."""
    def __init__(self, roots, weights, settings):
        self.enabled = bool(settings.get("transfer_diagnostics"))
        if not self.enabled:
            return
        self.weights = np.asarray(weights, dtype=np.float64)
        self.settings = settings
        baselines = [root.get("_student_baseline") for root in roots]
        self.valid = np.array([base is not None for base in baselines])
        self.initial = np.array([base["action"] if base else 0 for base in baselines])
        self.offsets = np.cumsum([0]+[len(root["q"]) for root in roots[:-1]])
        self.q = np.concatenate([root["q"] for root in roots])
        self.se = np.concatenate([base["paired_se"] if base else np.full(len(root["q"]), np.nan)
                                  for root, base in zip(roots, baselines)])
        self.initial_q = self.q[self.offsets+self.initial]
        self.best_q = np.array([root["q"].max() for root in roots])
        source = [root.get("behavior_source") for root in roots]
        student = np.array([value in ("student", "best_candidate", "last_candidate") for value in source])
        teacher = np.array([value == "teacher" for value in source])
        initial_id = settings.get("initial_student_model_id")
        training_round = settings.get("training_round")
        teacher_ids = [root.get("teacher_id") for root in roots]
        known_identity = np.array([bool(value and initial_id) for value in teacher_ids])
        same_identity = np.array([bool(initial_id and value == initial_id) for value in teacher_ids])
        # Model IDs preserve creation round before optional branch/epoch suffix.
        rounds = []
        for value in teacher_ids:
            match = re.fullmatch(r"M(\d+)(?:_(?:B|E)[A-Za-z0-9_]+)?", str(value or ""))
            rounds.append(int(match.group(1)) if match else None)
        previous = np.array([bool(training_round is not None and value is not None and value == training_round-1) for value in rounds])
        older = np.array([bool(training_round is not None and value is not None and value < training_round-1) for value in rounds])
        unknown_age = np.array([training_round is None or value is None for value in rounds])
        self.groups = dict(same_teacher_as_initial=self.valid & known_identity & same_identity,
            different_teacher_from_initial=self.valid & known_identity & ~same_identity,
            teacher_from_previous_round=self.valid & previous, older_teacher=self.valid & older,
            unknown_teacher=self.valid & (~known_identity | unknown_age), all=self.valid, student_trajectory=self.valid & student,
            teacher_trajectory=self.valid & teacher, unknown_trajectory=self.valid & ~(student | teacher),
            disagreement=np.array([bool(base and base["disagreement"]) for base in baselines]),
            teacher_advantage_paired_supported=np.array([bool(base and base["teacher_supported"]) for base in baselines]))

    def evaluate(self, choices):
        if not self.enabled:
            return {}
        choices = np.asarray(choices, dtype=np.int64)
        selected = self.q[self.offsets+choices]
        gain = selected-self.initial_q
        se = self.se[self.offsets+choices]
        paired = np.isfinite(se)
        threshold = self.settings["confidence_z"]*np.where(paired, se, 0.)
        values = dict(changed=choices != self.initial, paired_mass=paired, label_gain=gain,
            supported_gain=np.where(paired, np.maximum(0., gain-threshold), 0.),
            supported_loss=np.where(paired, np.maximum(0., -gain-threshold), 0.),
            baseline_regret=self.best_q-self.initial_q, current_regret=self.best_q-selected)
        result = {}
        for name, mask in self.groups.items():
            weights = self.weights*mask
            mass = float(weights.sum())
            result[name] = dict(roots=int(mask.sum()), mass=mass,
                **{key: float(np.sum(weights*value)/mass) if mass else None for key,value in values.items()})
        return {"initial_student_action_transfer": result,
                "transfer_uncertainty_z": self.settings["confidence_z"],
                "initial_student_model_id": self.settings.get("initial_student_model_id"),
                "training_round": self.settings.get("training_round"),
                "teacher_group_semantics": "source teacher identity and creation round only, not teacher quality; conditional original preset weights",
                "transfer_semantics": "paired rollout label evidence under frozen preset teacher, not causal policy gain"}


def action_transfer_diagnostics(roots, choices, weights, settings):
    return TransferDiagnostics(roots, weights, settings).evaluate(choices)
