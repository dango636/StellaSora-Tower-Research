"""Small independently designed NumPy residual action-value neural network.

Default architecture: complete public/action input -> 128 SiLU -> 96 SiLU -> 1,
plus a trainable input-to-output linear skip. All parameters remain trainable.
The skip initially reproduces a public heuristic Q prior; no recovered old
weights, hard action guard, frozen baseline branch, or dropout is involved.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .features import FEATURE_DIM, FEATURE_VERSION, HEURISTIC_Q_INDEX, TARGET_SCALE

MODEL_VERSION = "numpy_residual_direct_q_v1"


def _silu(x):
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    return x * sigmoid, sigmoid


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class QNetwork:
    """Direct Q^teacher(s,a) in ordinary effective level units.

    ``predict`` deliberately returns the raw estimate. Clipping every action to
    an identical physical bound can destroy action differences; the learner
    receives real bounded labels and diagnostics report prediction errors.
    """

    def __init__(self, input_dim: int = FEATURE_DIM, hidden=(128, 96), seed: int = 0,
                 *, heuristic_init: bool = True):
        self.input_dim = int(input_dim)
        self.hidden = tuple(int(width) for width in hidden)
        if self.input_dim < 1 or not self.hidden or any(width < 1 for width in self.hidden):
            raise ValueError("Positive neural network dimensions required")
        self.frozen = False
        self.metadata: dict[str, Any] = {}
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        previous = self.input_dim
        for i, width in enumerate(self.hidden):
            self.params[f"W{i}"] = (rng.standard_normal((previous, width)) * np.sqrt(2.0/previous)).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(width, np.float32)
            previous = width
        # Zero head makes model 0 reproduce the documented heuristic exactly.
        self.params["head_W"] = np.zeros(previous, np.float32)
        self.params["head_b"] = np.zeros(1, np.float32)
        self.params["skip_W"] = np.zeros(self.input_dim, np.float32)
        if heuristic_init and self.input_dim == FEATURE_DIM:
            self.params["skip_W"][HEURISTIC_Q_INDEX] = 1.0

    @property
    def parameter_count(self) -> int:
        return sum(array.size for array in self.params.values())

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return self.params

    def _inputs(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected [batch, {self.input_dim}] public/action features")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite neural input")
        return x

    def forward_normalized(self, x, *, cache: bool = False):
        x = self._inputs(x)
        h = x
        layers = []
        for i in range(len(self.hidden)):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            nxt, sigmoid = _silu(z)
            if cache:
                layers.append((h, z, sigmoid))
            h = nxt
        y = h @ self.params["head_W"] + self.params["head_b"][0] + x @ self.params["skip_W"]
        if cache:
            return y, (x, layers, h)
        return y

    def predict(self, x) -> np.ndarray:
        result = self.forward_normalized(x) * TARGET_SCALE
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite neural action value")
        return result

    predict_batch = predict

    def backward(self, cache, output_gradient) -> dict[str, np.ndarray]:
        """Gradient of a scalar loss w.r.t. NORMALIZED network predictions."""
        if self.frozen:
            raise RuntimeError("A frozen teacher cannot be trained")
        x, layers, last = cache
        dy = np.asarray(output_gradient, dtype=np.float32).reshape(-1)
        if len(dy) != len(x):
            raise ValueError("Output gradient batch mismatch")
        gradients = {"head_W": last.T @ dy,
                     "head_b": np.array((dy.sum(),), dtype=np.float32),
                     "skip_W": x.T @ dy}
        dh = dy[:, None] * self.params["head_W"][None, :]
        for i in reversed(range(len(self.hidden))):
            inputs, z, sigmoid = layers[i]
            dz = dh * (sigmoid + z * sigmoid * (1.0-sigmoid))
            gradients[f"W{i}"] = inputs.T @ dz
            gradients[f"b{i}"] = dz.sum(0)
            dh = dz @ self.params[f"W{i}"].T
        return gradients

    def clone(self, *, frozen: bool = False) -> "QNetwork":
        # A checkpoint clone only copies weights. Initializing and immediately
        # discarding fresh random matrices is wasteful for every epoch snapshot.
        # Preserve the constructor contract of any third-party subclass.
        if type(self) is QNetwork:
            obj = object.__new__(QNetwork)
            obj.input_dim, obj.hidden, obj.frozen = self.input_dim, self.hidden, False
        else:
            obj = self.__class__(self.input_dim, self.hidden, heuristic_init=False)
        obj.params = {key: value.copy() for key, value in self.params.items()}
        obj.metadata = json.loads(json.dumps(self.metadata, ensure_ascii=False))
        if frozen:
            obj.freeze()
        return obj

    copy = clone

    def freeze(self) -> "QNetwork":
        self.frozen = True
        for array in self.params.values():
            array.flags.writeable = False
        return self

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for key in sorted(self.params):
            array = self.params[key]
            digest.update(key.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.astype("<f4", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path, metadata: dict | None = None) -> Path:
        """Atomically save a safe NPZ plus readable JSON metadata, no pickle."""
        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.metadata)
        meta.update(metadata or {})
        meta.update({"model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
                     "input_dim": self.input_dim, "hidden": list(self.hidden),
                     "target_scale": TARGET_SCALE, "parameter_count": self.parameter_count,
                     "parameter_sha256": self.checksum(),
                     "value_semantics": "expected terminal ordinary score minus current score after action, then frozen teacher"})
        meta.setdefault("initialization", "trainable public heuristic skip, zero neural head; no recovered historical weights")
        payload = {**self.params, "__metadata__": np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False))}
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, **payload)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        _write_json(path.with_suffix(".json"), meta)
        self.metadata = meta
        return path

    @classmethod
    def load(cls, path, *, frozen: bool = False) -> "QNetwork":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__metadata__"].item()))
            if meta.get("model_version") != MODEL_VERSION or meta.get("feature_version") != FEATURE_VERSION:
                raise ValueError("Unsupported model/feature version")
            if meta.get("target_scale") != TARGET_SCALE:
                raise ValueError("Checkpoint target units disagree")
            obj = cls(meta["input_dim"], meta["hidden"], heuristic_init=False)
            if set(data.files) != {*obj.params, "__metadata__"}:
                raise ValueError("Checkpoint parameter names disagree")
            for key, expected in obj.params.items():
                value = np.asarray(data[key], dtype=np.float32)
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid checkpoint parameter: {key}")
                obj.params[key] = value.copy()
        obj.metadata = meta
        if obj.checksum() != meta.get("parameter_sha256"):
            raise ValueError("Checkpoint parameter checksum mismatch")
        if frozen:
            obj.freeze()
        return obj


ValueNetwork = QNetwork


class AdamW:
    """AdamW with bias correction, global gradient clipping and finite checks."""

    def __init__(self, model: QNetwork, learning_rate=8e-4, weight_decay=1e-5,
                 clip_norm=5.0):
        if model.frozen:
            raise RuntimeError("Optimizer must own a mutable candidate, never the teacher")
        if learning_rate <= 0 or weight_decay < 0 or clip_norm <= 0:
            raise ValueError("Invalid optimizer parameters")
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.clip_norm = float(clip_norm)
        self.steps = 0
        self.m = {key: np.zeros_like(value) for key, value in model.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in model.params.items()}

    def step(self, gradients, learning_rate=None) -> float:
        if self.model.frozen:
            raise RuntimeError("Cannot mutate a frozen teacher")
        norm = np.sqrt(sum(float(np.square(g, dtype=np.float64).sum()) for g in gradients.values()))
        if not np.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient")
        multiplier = min(1.0, self.clip_norm / max(norm, 1e-12))
        self.steps += 1
        beta1, beta2 = .9, .999
        rate = self.learning_rate if learning_rate is None else float(learning_rate)
        correction1, correction2 = 1-beta1**self.steps, 1-beta2**self.steps
        for key, parameter in self.model.params.items():
            gradient = gradients[key] * multiplier
            self.m[key] *= beta1; self.m[key] += (1-beta1)*gradient
            self.v[key] *= beta2; self.v[key] += (1-beta2)*gradient*gradient
            update = (self.m[key]/correction1) / (np.sqrt(self.v[key]/correction2) + 1e-8)
            # Biases do not receive decoupled weight decay.
            if not key.startswith("b") and not key.endswith("_b"):
                parameter *= 1-rate*self.weight_decay
            parameter -= rate*update
            if not np.isfinite(parameter).all():
                raise FloatingPointError("Nonfinite candidate parameter")
        return float(norm)


def save_model(model: QNetwork, path, metadata=None):
    return model.save(path, metadata)


def load_model(path, *, frozen=False):
    return QNetwork.load(path, frozen=frozen)
