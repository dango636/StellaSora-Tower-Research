"""Real optional CUDA training for the same portable NumPy QNetwork.

Complete-root batches preserve the existing objective and sampling rules.
Data can remain on the selected device or use a host-resident fallback when
its measured tensor size exceeds the runtime memory budget. FP32 is the
default; runtime placement never changes the configured precision or batch.
Game evaluation and label rollouts use the separate shared FP32 service.

This module imports torch only when explicitly loaded by stella.backend. The
normal offline NumPy runtime does not depend on PyTorch or a CUDA installation.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
import copy
import time

import numpy as np
import torch
from torch import nn

from .features import TARGET_SCALE
from .fit import (FitResult, _settings, _stopped, balanced_weights, prepare_roots,
                  mistake_sampling_weights, ranking_pairs, CandidatePool,
                  advantage_confidence, confidence_diagnostics)
from .network import QNetwork
from .indexing import require_integer
from .fit_diagnostics import initialize_student_diagnostics, TransferDiagnostics

TORCH_DEFAULTS = {
    "gpu_batch_roots": 128,
    "gpu_min_steps_per_epoch": 8,
    "gpu_amp": False,
    "gpu_tf32": False,
    "gpu_memory_limit_mb": 1024,
    "gpu_callback_interval_seconds": .25,
}


class TorchQ(nn.Module):
    """Exactly the existing W0/b0/.../head/skip weight layout, no transpose."""

    def __init__(self, source: QNetwork, device="cpu"):
        super().__init__()
        self.input_dim, self.hidden = source.input_dim, source.hidden
        self.weights = nn.ParameterDict({key: nn.Parameter(torch.tensor(
            value, dtype=torch.float32, device=device)) for key, value in source.params.items()})

    def forward(self, x, *, amp=False):
        h = x
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                            enabled=bool(amp and x.device.type == "cuda")):
            for i in range(len(self.hidden)):
                z = h @ self.weights[f"W{i}"] + self.weights[f"b{i}"]
                h = z * torch.sigmoid(torch.clamp(z, -50.0, 50.0))
        # The small differences between actions must survive addition to a
        # much larger common value. Keep the head and skip in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = (h.float() @ self.weights["head_W"] + self.weights["head_b"][0]
                 + x.float() @ self.weights["skip_W"])
        return y

    def export(self, source: QNetwork, state=None) -> QNetwork:
        result = source.clone(frozen=False)
        values = self.weights if state is None else state
        for key, parameter in values.items():
            value = parameter.detach().to(device="cpu", dtype=torch.float32).numpy()
            if value.shape != result.params[key].shape or not np.isfinite(value).all():
                raise FloatingPointError(f"Invalid torch-exported parameter: {key}")
            np.copyto(result.params[key], value)
        return result


class ResidentRoots:
    """Padded complete-root tensor batch; padding never contributes to loss."""

    def __init__(self, roots, device, *, max_actions=None, settings=None):
        count = len(roots)
        if not count:
            raise ValueError("ResidentRoots requires a nonempty root set")
        required_actions = max(len(root["q"]) for root in roots)
        action_count = (required_actions if max_actions is None else
                        require_integer(max_actions, "max_actions", minimum=required_actions))
        input_dim = roots[0]["x"].shape[1]
        x = np.zeros((count, action_count, input_dim), dtype=np.float32)
        y = np.zeros((count, action_count), dtype=np.float32)
        mask = np.zeros((count, action_count), dtype=np.float32)
        for i, root in enumerate(roots):
            n = len(root["q"])
            x[i, :n], y[i, :n], mask[i, :n] = root["x"], root["q"] / TARGET_SCALE, 1
        settings = _settings(settings)
        base_weights = balanced_weights(roots)
        self.confidence = None
        local_confidence = None
        if settings["advantage_confidence_weighting"]:
            confidence = np.zeros_like(mask)
            local_confidence = [advantage_confidence(root, settings) for root in roots]
            for i, local in enumerate(local_confidence):
                confidence[i, :len(local)] = local
            self.confidence = torch.from_numpy(confidence).to(device)
        self.confidence_diagnostics = confidence_diagnostics(
            roots, settings, local=local_confidence, mass=base_weights)
        pairs = [ranking_pairs(root, settings) for root in roots]
        width = max(1, max(map(len, pairs)))
        pair_data = np.zeros((count, width, 4), np.float32)
        for i, local in enumerate(pairs):
            if len(local):
                indices = np.asarray(local[:, :2])
                if (not np.isfinite(indices).all() or
                        not np.equal(indices, np.floor(indices)).all() or
                        (indices < 0).any() or (indices >= len(roots[i]["q"])).any()):
                    raise ValueError("Ranking pairs contain an illegal action index")
                pair_data[i, :len(local)] = local
                pair_data[i, :len(local), 2] /= len(local)
        self.pairs = torch.from_numpy(pair_data).to(device)
        # Integer gather indices are invariant over all epochs. Converting the
        # padded float pair table twice per optimizer step launches extra kernels.
        self.pair_indices = torch.from_numpy(
            pair_data[:, :, :2].astype(np.int64)).to(device)
        self.x = torch.from_numpy(x).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.mask = torch.from_numpy(mask).to(device)
        self.lengths = self.mask.sum(1).clamp_min(1.0)
        self.invalid_actions = self.mask == 0
        self.best_value, self.best_action = self.y.masked_fill(self.invalid_actions, -torch.inf).max(1)
        self.weights = torch.tensor(base_weights, dtype=torch.float32, device=device)
        hard = np.array([bool(root.get("hard_example")) for root in roots])
        hard_weights = base_weights*hard
        if hard_weights.sum():
            hard_weights /= hard_weights.sum()
        self.hard_weights = torch.tensor(hard_weights, dtype=torch.float32, device=device)
        self.hard_count = int(hard.sum())
        targeted = np.array([bool(root.get("hard_sample_slot")) for root in roots])
        targeted_weights = base_weights*targeted
        if targeted_weights.sum():
            targeted_weights /= targeted_weights.sum()
        self.targeted_weights = torch.tensor(targeted_weights, dtype=torch.float32, device=device)
        self.targeted_count = int(targeted.sum())
        self.transfer_diagnostics = TransferDiagnostics(roots, base_weights, settings)
        self.root_count = count
        self.family_count = len({root["family_id"] for root in roots})
        self.action_count = action_count
        self.bytes = sum(t.numel()*t.element_size() for t in (
            self.x, self.y, self.mask, self.weights, self.hard_weights,
            self.targeted_weights, self.pairs, self.pair_indices,
            self.lengths, self.invalid_actions, self.best_value, self.best_action))
        if self.confidence is not None:
            self.bytes += confidence.nbytes

    def _check_index_layout(self, indices):
        # Metadata-only checks: no host read or CUDA synchronization. The values
        # come from the bounded epoch sampler; root/pair indices were validated
        # on the host at corpus construction.
        if (indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64)
                or indices.device != self.x.device):
            raise ValueError("Root indices must be a 1-D integer tensor on the data device")

    def batch(self, indices):
        self._check_index_layout(indices)
        return (self.x.index_select(0, indices), self.y.index_select(0, indices),
                self.mask.index_select(0, indices))

    def to(self, device):
        """Move immutable tensors together; keep host diagnostics shared."""
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def metric_view(self, device):
        """Only small metric tables move; the padded feature corpus stays put."""
        if self.x.device == device:
            return self
        result = copy.copy(self)
        for name, value in vars(self).items():
            if name != "x" and isinstance(value, torch.Tensor):
                setattr(result, name, value.to(device))
        return result

    def training_batch(self, indices, device):
        """Gather host batches directly into bounded pinned transfer buffers.

        Ordinary index_select outputs are pageable even for a pinned source.
        The explicit pinned output avoids a second host copy and permits H2D
        transfer without a host wait. PyTorch's pinned allocator records the
        copy event before recycling these temporary buffers. No custom stream
        or shared mutable staging buffer is needed.
        """
        self._check_index_layout(indices)
        names = ("x", "y", "mask", "pairs", "pair_indices", "lengths", "confidence")
        values = []
        for name in names:
            source = getattr(self, name)
            if source is None:
                values.append(None)
                continue
            pinned = source.device.type == "cpu" and device.type == "cuda"
            if pinned:
                selected = torch.empty((len(indices), *source.shape[1:]), dtype=source.dtype,
                                       device="cpu", pin_memory=True)
                torch.index_select(source, 0, indices, out=selected)
            else:
                selected = source.index_select(0, indices)
            values.append(selected.to(device, non_blocking=pinned))
        return tuple(values)


def fit_runtime_options(options, runtime=None):
    """Placement is an operational choice, separate from frozen learning config."""
    runtime = dict(runtime or {})
    residency = runtime.get("data_residency", os.environ.get("STELLA_FIT_DATA_RESIDENCY", "device"))
    if residency not in ("auto", "device", "host"):
        raise ValueError("data_residency must be auto, device, or host")
    def megabytes(key, environment, default):
        value = runtime.get(key, os.environ.get(environment, default))
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a positive integer")
        try:
            parsed = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"{key} must be a positive integer") from None
        if parsed < 1 or str(parsed) != str(value):
            raise ValueError(f"{key} must be a positive integer")
        return parsed
    return dict(data_residency=residency,
        data_budget_mb=megabytes("data_budget_mb", "STELLA_FIT_DATA_BUDGET_MB", options["gpu_memory_limit_mb"]),
        memory_reserve_mb=megabytes("memory_reserve_mb", "STELLA_FIT_MEMORY_RESERVE_MB", 1024))


def place_root_data(train_data, val_data, device, runtime):
    """Choose data placement from exact storage bytes, with recoverable OOM.

    Auto leaves both datasets on the host when the data cap or available-memory
    headroom would be exceeded. It never shrinks a batch or changes optimizer
    steps. Activation/optimizer allocations still need the reserved headroom;
    an OOM in actual fitting is reported, never retried after a partial update.
    """
    size = train_data.bytes+val_data.bytes
    requested = runtime["data_residency"]
    budget = runtime["data_budget_mb"]*1024**2
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(budget, max(0, free-runtime["memory_reserve_mb"]*1024**2))
    reason = "requested_host" if requested == "host" else "within_budget"
    if requested == "host" or (requested == "auto" and size > budget):
        return "host", budget, reason if requested == "host" else "data_budget_exceeded"
    if device.type == "cuda" and requested == "device" and size > budget:
        raise torch.cuda.OutOfMemoryError(
            f"Resident tensors need {size/1024**2:.1f} MiB; runtime budget {budget/1024**2:.1f} MiB")
    try:
        train_data.to(device)
        val_data.to(device)
    except torch.cuda.OutOfMemoryError:
        train_data.to("cpu")
        val_data.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if requested != "auto":
            raise
        return "host", budget, "data_allocation_oom"
    return "device", budget, reason


def root_loss(prediction, target, mask, root_weights=None, *, advantage_weight=.75,
              huber_delta=2.0 / TARGET_SCALE, confidence_weights=None, lengths=None, q_anchor_weight=None):
    """Vectorized equivalent of fit.loss_and_gradient, entirely on device."""
    anchor_weight = 1-advantage_weight if q_anchor_weight is None else q_anchor_weight
    error = prediction.float()-target.float()
    mask = mask.float()
    lengths = mask.sum(1).clamp_min(1.0) if lengths is None else lengths
    mean_error = (error*mask).sum(1) / lengths
    centered = error-mean_error[:, None]
    absolute = error.abs()
    huber = torch.where(absolute <= huber_delta, .5*error.square(),
                        huber_delta*(absolute-.5*huber_delta))
    advantage_square = centered.square()
    if confidence_weights is not None:
        advantage_square = advantage_square*confidence_weights
    per_root = (anchor_weight*(huber*mask).sum(1)/lengths
                + advantage_weight*(advantage_square*mask).sum(1)/lengths)
    return per_root.mean() if root_weights is None else (per_root*root_weights.float()).sum()


def ranking_root_loss(prediction, pairs, settings, root_weights=None, *, pair_indices=None):
    """Same confidence/gain-weighted squared smooth hinge as NumPy."""
    indices = (pairs[:, :, 0].long(), pairs[:, :, 1].long()) if pair_indices is None else pair_indices
    winner = prediction.gather(1, indices[0])
    rival = prediction.gather(1, indices[1])
    temperature = settings["ranking_temperature"]/TARGET_SCALE
    hinge = temperature*torch.nn.functional.softplus((pairs[:, :, 3]-(winner-rival))/temperature)
    per_root = settings["ranking_weight"]*(pairs[:, :, 2]*hinge.square()).sum(1)
    return per_root.mean() if root_weights is None else (per_root*root_weights.float()).sum()


@contextmanager
def training_precision(device, *, tf32):
    """Confine TF32 to training forward/backward, restore global torch state."""
    if device.type != "cuda":
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.inference_mode()
def evaluate_torch(model: TorchQ, data: ResidentRoots, settings=None) -> dict:
    """Development metrics remain on the device, computed in FP32 without AMP."""
    settings = _settings(settings)
    model.eval()
    device = next(model.parameters()).device
    features = data.x
    data = data.metric_view(device)
    with training_precision(device, tf32=False):
        # Keep large-network development activations bounded even when the
        # replay pool contains tens of thousands of complete roots.
        flat = features.flatten(0, 1)
        prediction = torch.cat([model(flat[start:start+4096].to(device), amp=False)
                                for start in range(0, len(flat), 4096)]).reshape_as(data.y)
        loss = root_loss(prediction, data.y, data.mask, data.weights,
                         advantage_weight=settings["advantage_weight"],
                         huber_delta=settings["huber_delta"]/TARGET_SCALE,
                         q_anchor_weight=settings["q_anchor_weight"],
                         confidence_weights=data.confidence, lengths=data.lengths)
        anchor = root_loss(prediction, data.y, data.mask, data.weights, advantage_weight=0.,
            q_anchor_weight=(1-settings["advantage_weight"] if settings["q_anchor_weight"] is None else settings["q_anchor_weight"]),
            huber_delta=settings["huber_delta"]/TARGET_SCALE, lengths=data.lengths)
        rank = ranking_root_loss(prediction, data.pairs, settings, data.weights,
            pair_indices=(data.pair_indices[:, :, 0], data.pair_indices[:, :, 1]))
        advantage = loss-anchor
        loss = loss + rank
        error = (prediction-data.y)*TARGET_SCALE
        lengths = data.lengths
        mean_error = (error*data.mask).sum(1)/lengths
        q_square = (data.weights*(error.square()*data.mask).sum(1)/lengths).sum()
        advantage_square = (data.weights*((error-mean_error[:, None]).square()*data.mask).sum(1)/lengths).sum()
        chosen = prediction.masked_fill(data.invalid_actions, -torch.inf).argmax(1)
        best_value, best_action = data.best_value, data.best_action
        chosen_value = data.y.gather(1, chosen[:, None]).squeeze(1)
        regret = (data.weights*(best_value-chosen_value)).sum()*TARGET_SCALE
        agreement = (data.weights*(chosen == best_action).float()).sum()
        hard_regret = (data.hard_weights*(best_value-chosen_value)).sum()*TARGET_SCALE
        hard_agreement = (data.hard_weights*(chosen == best_action).float()).sum()
        targeted_regret = (data.targeted_weights*(best_value-chosen_value)).sum()*TARGET_SCALE
        metrics = torch.stack((loss*TARGET_SCALE**2, q_square.sqrt(), advantage_square.sqrt(),
                               regret, agreement, hard_regret, hard_agreement, targeted_regret, anchor*TARGET_SCALE**2, advantage*TARGET_SCALE**2, rank*TARGET_SCALE**2)).cpu().numpy()
    if not np.isfinite(metrics).all():
        raise FloatingPointError("Nonfinite torch training/development metrics")
    return {"objective_version": settings["objective_version"],
            "q_anchor_loss": float(metrics[8]), "advantage_loss": float(metrics[9]), "ranking_loss": float(metrics[10]),
            **(data.transfer_diagnostics.evaluate(chosen.cpu().numpy()) if settings["transfer_diagnostics"] else {}),
            **data.confidence_diagnostics, "selection_loss": float(metrics[0]), "q_rmse": float(metrics[1]),
            "advantage_rmse": float(metrics[2]), "label_regret": float(metrics[3]),
            "label_action_agreement": float(metrics[4]),
            "roots": data.root_count, "families": data.family_count,
            "hard_roots": data.hard_count,
            "hard_label_regret": float(metrics[5]) if data.hard_count else None,
            "hard_label_action_agreement": float(metrics[6]) if data.hard_count else None,
            "targeted_roots": data.targeted_count,
            "targeted_label_regret": float(metrics[7]) if data.targeted_count else None}


@torch.inference_mode()
def torch_teacher_predictions(model, roots, device):
    """One static FP32 pass before any optimizer update, with TF32 disabled."""
    with training_precision(device, tf32=False):
        for start in range(0, len(roots), 64):
            batch = roots[start:start+64]
            x = torch.from_numpy(np.concatenate([root["x"] for root in batch])).to(device)
            prediction = torch.cat([model(x[i:i+4096], amp=False)
                                    for i in range(0, len(x), 4096)]).cpu().numpy()
            cursor = 0
            for root in batch:
                count = len(root["q"])
                yield prediction[cursor:cursor+count]
                cursor += count


def fit_candidate_torch(teacher_model: QNetwork, roots, config=None, callback=None,
                        stop_event=None, *, device="cuda", runtime=None) -> FitResult:
    """Fit on CUDA (or CPU for numerical tests), export the same safe NPZ model.

    CUDA uses larger complete-root batches sampled with replacement according
    to the existing equal-stratum/family/root weights. The number of optimizer
    steps per epoch is at least the NumPy budget; increasing the GPU batch does
    not silently reduce the learning-step count. Consequently an epoch may
    make several balanced passes over a small dataset. This is reported.
    """
    started = time.monotonic()
    settings = _settings(config)
    options = {**TORCH_DEFAULTS, **(config or {})}
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but PyTorch cannot use CUDA")
    if device.type == "cuda" and device.index is None:
        # Tensor.device is cuda:N, never the unindexed spelling cuda. Resolve
        # once so resident data cannot be mistaken for host-resident data.
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Only CUDA and CPU torch devices are supported")
    for key in ("gpu_batch_roots", "gpu_min_steps_per_epoch", "gpu_memory_limit_mb"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    interval = float(options["gpu_callback_interval_seconds"])
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("GPU callback interval must be nonnegative")
    if type(options["gpu_amp"]) is not bool or type(options["gpu_tf32"]) is not bool:
        raise ValueError("gpu_amp and gpu_tf32 must be booleans")
    runtime = fit_runtime_options(options, runtime)
    before = teacher_model.checksum()
    train, val, teacher_id = prepare_roots(roots, teacher_model.input_dim, settings)
    capture_identity = settings["objective_version"] != "q_advantage_ranking_v1" or settings["refresh_student_actions"] or settings["transfer_diagnostics"]
    all_roots = train+val
    max_actions = max(len(root["q"]) for root in all_roots)
    model = TorchQ(teacher_model, device)
    predict = lambda selected: torch_teacher_predictions(model, selected, device)
    if settings["refresh_student_actions"] or settings["transfer_diagnostics"]:
        initialize_student_diagnostics(train+val, predict, teacher_model.checksum(), settings)
    probabilities = mistake_sampling_weights(train, settings, predict)
    mistake_sampling_weights(val, settings, predict)
    # Build on CPU once to measure every tensor (including pair indices and
    # confidence) before committing GPU memory. Host fallback keeps these same
    # arrays and transfers only selected complete-root batches.
    train_data = ResidentRoots(train, "cpu", max_actions=max_actions, settings=settings)
    val_data = ResidentRoots(val, "cpu", max_actions=max_actions, settings=settings)
    data_placement, data_budget, placement_reason = place_root_data(train_data, val_data, device, runtime)
    sampling_weights = torch.tensor(probabilities, dtype=torch.float32, device=device)
    amp = bool(device.type == "cuda" and options["gpu_amp"] and torch.cuda.is_bf16_supported())
    tf32 = bool(device.type == "cuda" and options["gpu_tf32"])
    original_batches = max(1, math.ceil(len(train)/settings["batch_roots"]))
    if device.type == "cuda":
        batch_roots = int(options["gpu_batch_roots"])
        batches = max(original_batches, math.ceil(len(train)/batch_roots), int(options["gpu_min_steps_per_epoch"]))
    else:
        # CPU mode exists for portable backend verification, not a forced
        # replacement for the already efficient NumPy execution path.
        batch_roots, batches = settings["batch_roots"], original_batches
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU (torch numerical verification)"
    information = {"backend": "torch_cuda" if device.type == "cuda" else "torch_cpu",
                   "device_name": device_name, "device": str(device),
                   "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                   "precision": "BF16 hidden matmul / FP32 head, loss and parameters" if amp else "FP32",
                   "tf32_training": tf32, "development_precision": "FP32, TF32 disabled",
                   "game_evaluation_backend": "NumPy CPU FP32",
                   "resident_data_mib": (train_data.bytes+val_data.bytes)/1024**2,
                   "data_placement": data_placement, "data_placement_reason": placement_reason,
                   "data_budget_mib": data_budget/1024**2, "runtime_placement": runtime,
                   "effective_batch_roots": batch_roots, "steps_per_epoch": batches,
                   "roots_sampled_per_epoch": batches*batch_roots,
                   "sampling": ("equal-stratum, family/root base with bounded teacher-mistake weights, "
                                "with replacement, complete action sets" if settings["mistake_weight_strength"] > 0 else
                                "equal-stratum/family/root, with replacement, complete action sets"),
                   "mistake_weight_strength": settings["mistake_weight_strength"],
                   "mistake_weight_scale": settings["mistake_weight_scale"],
                   "mistake_weight_max": settings["mistake_weight_max"],
                   "hard_train_roots": train_data.hard_count,
                   "hard_val_roots": val_data.hard_count}
    groups = [
        {"params": [p for key, p in model.weights.items() if not key.startswith("b") and not key.endswith("_b")],
         "weight_decay": settings["l2"]},
        {"params": [p for key, p in model.weights.items() if key.startswith("b") or key.endswith("_b")],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=settings["learning_rate"], betas=(.9, .999), eps=1e-8,
                                  fused=(device.type == "cuda"))
    generator = torch.Generator(device=device).manual_seed(settings["seed"])
    history = [{"kind": "epoch", "epoch": 0, "eligible_candidate": False,
                "train": evaluate_torch(model, train_data, settings),
                "val": evaluate_torch(model, val_data, settings),
                "selection_metric": settings["checkpoint_metric"],
                "elapsed_seconds": time.monotonic()-started, **information}]
    history[0]["selection_score"] = history[0]["val"][settings["checkpoint_metric"]]
    interrupted = _stopped(stop_event)
    if callback is not None and callback(dict(history[0])) is False:
        interrupted = True
    best, best_epoch, best_score, stale, steps = None, 0, math.inf, 0, 0
    pool = CandidatePool(settings["checkpoint_candidates"], before, settings["preserve_early_checkpoints"],
                         settings["prefer_first_four_checkpoints"])
    total_steps = batches*settings["epochs"]
    last_report = time.monotonic()
    for epoch in range(1, settings["epochs"]+1):
        if interrupted or _stopped(stop_event):
            interrupted = True
            break
        model.train()
        # Keep the same device RNG and with-replacement sampling under either
        # placement. Resident data gathers stay on device; host data gathers
        # use one copied epoch index table and bounded pinned batch buffers.
        order = torch.multinomial(sampling_weights, batches*batch_roots,
                                  replacement=True, generator=generator).reshape(batches, batch_roots)
        if train_data.x.device != device:
            # Preserve CUDA multinomial and generator sequence. One epoch copy
            # avoids a synchronizing GPU->CPU index transfer for every batch.
            order = order.cpu()
        epoch_loss = torch.zeros((), device=device, dtype=torch.float32)
        for batch_index in range(batches):
            if _stopped(stop_event):
                interrupted = True
                break
            x, target, mask, pairs, pair_indices, lengths, confidence = train_data.training_batch(
                order[batch_index], device)
            optimizer.zero_grad(set_to_none=True)
            progress = steps/max(1, total_steps-1)
            learning_rate = settings["learning_rate"]*(.25+.75*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            with training_precision(device, tf32=tf32):
                prediction = model(x.flatten(0, 1), amp=amp).reshape_as(target)
                loss = root_loss(prediction, target, mask,
                                 advantage_weight=settings["advantage_weight"],
                                 huber_delta=settings["huber_delta"]/TARGET_SCALE,
                                 q_anchor_weight=settings["q_anchor_weight"],
                                 confidence_weights=confidence, lengths=lengths)
                loss = loss + ranking_root_loss(prediction,
                    pairs, settings,
                    pair_indices=(pair_indices[:, :, 0], pair_indices[:, :, 1]))
                loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"],
                                                          error_if_nonfinite=False, foreach=True)
            optimizer.step()
            steps += 1
            epoch_loss += loss.detach()
            # Avoid a CUDA synchronization on every tiny optimizer step.
            should_report = (time.monotonic()-last_report >= float(options["gpu_callback_interval_seconds"])
                             or batch_index+1 == batches)
            if should_report:
                scalars = torch.stack((loss.detach()*TARGET_SCALE**2, gradient_norm.detach())).cpu().numpy()
                if not np.isfinite(scalars).all():
                    raise FloatingPointError("Nonfinite CUDA training loss/gradient")
                if callback is not None:
                    message = {"kind": "train_step", "epoch": epoch, "epochs": settings["epochs"],
                               "batch": batch_index+1, "batches": batches, "step": steps,
                               "steps_total": total_steps, "loss": float(scalars[0]),
                               "learning_rate": learning_rate, "gradient_norm": float(scalars[1]),
                               "elapsed_seconds": time.monotonic()-started, **information}
                    if callback(message) is False:
                        interrupted = True
                        break
                last_report = time.monotonic()
        if interrupted:
            break
        metrics = {"kind": "epoch", "epoch": epoch, "epochs": settings["epochs"],
                   "train": evaluate_torch(model, train_data, settings),
                   "val": evaluate_torch(model, val_data, settings), "eligible_candidate": True,
                   "elapsed_seconds": time.monotonic()-started,
                   "mean_batch_loss": float((epoch_loss*TARGET_SCALE**2/batches).cpu()),
                   "optimizer_steps": steps, "selection_metric": settings["checkpoint_metric"],
                   **information}
        metrics["selection_score"] = metrics["val"][settings["checkpoint_metric"]]
        history.append(metrics)
        pool.add(model.export(teacher_model), metrics)
        if metrics["selection_score"] < best_score:
            best = {key: value.detach().clone() for key, value in model.weights.items()}
            best_epoch, best_score, stale = epoch, metrics["selection_score"], 0
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
        raise AssertionError("Torch training mutated the frozen teacher")
    exported = model.export(teacher_model, best)
    exported.metadata.update({"trained_against_teacher": teacher_id, "best_epoch": best_epoch,
                              "objective_version": settings["objective_version"],
                              "q_anchor_weight": settings["q_anchor_weight"],
                              "optimizer_steps": steps, "trained_candidate": best_epoch > 0,
                              "training_interrupted": interrupted,
                              "checkpoint_metric": settings["checkpoint_metric"],
                              "training_backend": information})
    checkpoints = pool.selected()
    for point in checkpoints:
        point["model"].metadata.update({"trained_against_teacher": teacher_id,
            "best_epoch": point["epoch"], "optimizer_steps": point["optimizer_steps"],
            "objective_version": settings["objective_version"], "q_anchor_weight": settings["q_anchor_weight"],
            "trained_candidate": True, "checkpoint_metric": settings["checkpoint_metric"]})
    return FitResult(exported, best_epoch, history, train_data.family_count, val_data.family_count,
                     interrupted, len(train), len(val), teacher_id, time.monotonic()-started,
                     steps, settings["checkpoint_metric"], checkpoints, initial_parameter_sha256=before if capture_identity else None)


# Compatible spelling for callers that explicitly import this backend.
fit_candidate = fit_candidate_torch
