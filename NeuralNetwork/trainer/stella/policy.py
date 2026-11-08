"""Unrestricted greedy public Q policy, with batched legal-action inference."""
from __future__ import annotations

import numpy as np
from .environment import PublicState, legal_actions
from .features import FEATURE_DIM, encode_actions
from .network import QNetwork


class QPolicy:
    def __init__(self, model, model_id="model"):
        self.model = model
        self.model_id = str(model_id)

    @classmethod
    def load(cls, path, model_id="model"):
        return cls(QNetwork.load(path).clone(frozen=True), model_id)

    def _predict_actions(self, x, sizes):
        grouped = getattr(self.model, "predict_actions", None)
        return grouped(x, sizes) if grouped is not None else self.model.predict(x)

    def values(self, state, actions=None, preset=None):
        if not isinstance(state, PublicState):
            raise TypeError("Only public states may enter a policy")
        actions = legal_actions(state) if actions is None else tuple(actions)
        q = self._predict_actions(encode_actions(state, actions, preset), [len(actions)])
        if not np.isfinite(q).all():
            raise FloatingPointError("Network produced nonfinite action values")
        return q

    def choose(self, state, actions=None, preset=None):
        actions = legal_actions(state) if actions is None else tuple(actions)
        if len(actions) == 1:
            return actions[0]
        # No min-gain, incumbent lock, confidence test, or hard output clipping.
        return actions[int(np.argmax(self.values(state, actions, preset)))]

    def choose_batch(self, states, action_lists=None, presets=None):
        action_lists = ([legal_actions(s) for s in states] if action_lists is None else action_lists)
        presets = [None] * len(states) if presets is None else presets
        if not len(states) == len(action_lists) == len(presets):
            raise ValueError("Mismatched public inference batch")
        result, pending = [None] * len(states), []
        for i, (state, actions, preset) in enumerate(zip(states, action_lists, presets)):
            if not isinstance(state, PublicState) or not actions:
                raise ValueError("Invalid public decision batch")
            if len(actions) == 1:
                result[i] = actions[0]
            else:
                pending.append((i, actions))
        if pending:
            sizes = [len(actions) for _, actions in pending]
            matrix = np.zeros((sum(sizes), FEATURE_DIM), dtype=np.float32)
            offset = 0
            for i, actions in pending:
                encode_actions(states[i], actions, presets[i], out=matrix[offset:offset+len(actions)])
                offset += len(actions)
            values = self._predict_actions(matrix, sizes)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite policy prediction")
            offset = 0
            for i, actions in pending:
                result[i] = actions[int(np.argmax(values[offset:offset+len(actions)]))]
                offset += len(actions)
        return result
