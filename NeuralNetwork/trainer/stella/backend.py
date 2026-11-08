"""Optional CUDA dispatcher; importing it never imports torch eagerly."""
from __future__ import annotations

import importlib.util
import os

from .fit import fit_candidate as fit_numpy


def backend_info(requested="auto", *, probe=True) -> dict:
    """Choose a backend from actual availability, not an nvidia-smi label.

    Auto mode remains fully usable offline when PyTorch or CUDA is absent.
    A tiny deterministic CUDA matmul checks the installed wheel/driver/kernel
    combination without consuming any training or evaluation random stream.
    """
    if requested not in ("auto", "numpy", "cuda"):
        raise ValueError("training.backend must be auto, numpy or cuda")
    forced = os.environ.get("STELLA_FORCE_CPU", "").strip().lower() in ("1", "true", "yes")
    if requested == "numpy" or forced:
        return {"backend": "numpy", "device": "cpu", "device_name": "CPU / NumPy FP32",
                "reason": "CPU override" if forced else "Configured NumPy backend"}
    if importlib.util.find_spec("torch") is None:
        if requested == "cuda":
            raise RuntimeError("CUDA backend requested, but PyTorch is not installed")
        return {"backend": "numpy", "device": "cpu", "device_name": "CPU / NumPy FP32",
                "reason": "PyTorch unavailable; portable CPU backend remains active"}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access a CUDA device")
        device = torch.device("cuda")
        if probe:
            with torch.inference_mode():
                matrix = torch.ones((16, 16), device=device, dtype=torch.float32)
                result = matrix @ matrix
                torch.cuda.synchronize(device)
                if float(result[0, 0].cpu()) != 16.0:
                    raise RuntimeError("CUDA numerical probe failed")
        return {"backend": "torch_cuda", "device": "cuda",
                "device_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                "capability": list(torch.cuda.get_device_capability(device)),
                "reason": "Actual CUDA tensor execution verified" if probe else "CUDA device available"}
    except (ImportError, OSError, RuntimeError) as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA backend unavailable: {exc}") from exc
        return {"backend": "numpy", "device": "cpu", "device_name": "CPU / NumPy FP32",
                "reason": f"CUDA unavailable; CPU fallback: {type(exc).__name__}: {exc}"}


def fit_candidate(teacher_model, roots, config=None, callback=None, stop_event=None):
    settings = dict(config or {})
    # A CUDA memory fallback must be able to iterate the same immutable labels
    # again even when an external caller supplied a one-shot generator.
    roots = roots if isinstance(roots, (list, tuple)) else list(roots)
    requested = settings.get("backend", "auto")
    information = backend_info(requested)
    if callback is not None:
        callback({"kind": "backend", **information})
    if settings.get("objective_version") == "action_gap_calibration_v3":
        # The shared action-calibration objective uses the same portable network
        # and bounded residency helpers on both hardware profiles.
        from .fit_calibration import fit_candidate_calibration
        target = "cuda" if information["backend"] == "torch_cuda" else "cpu"
        result = fit_candidate_calibration(teacher_model, roots, settings, callback, stop_event, device=target)
        if target == "cuda":
            import torch
            torch.cuda.empty_cache()
        return result
    if information["backend"] == "torch_cuda":
        from .fit_torch import fit_candidate_torch
        try:
            result = fit_candidate_torch(teacher_model, roots, settings, callback, stop_event, device="cuda")
        except Exception as exc:
            # Only memory exhaustion gets an automatic retry. Invalid labels,
            # nonfinite gradients and programming errors must remain visible.
            import torch
            if requested != "auto" or not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            torch.cuda.empty_cache()
            information = {"backend": "numpy", "device": "cpu", "device_name": "CPU / NumPy FP32",
                           "reason": f"CUDA memory budget exceeded; fresh CPU fit: {exc}"}
            if callback is not None:
                callback({"kind": "backend", **information})
        else:
            # Fitting returns CPU NumPy weights and scalar history. Its local
            # CUDA tensors have now left scope. The next rollout/evaluation
            # runs in a separate process, which cannot reuse this allocator's
            # cached blocks. Release only unused blocks once at this boundary.
            # Keep this outside the fit exception handler: a CUDA error during
            # handoff must remain visible rather than trigger a CPU retraining.
            import torch
            handoff = {
                "scope": "post_fit_parent_process_unused_allocator_cache",
                "reserved_before_bytes": int(torch.cuda.memory_reserved()),
                "allocated_before_bytes": int(torch.cuda.memory_allocated()),
            }
            torch.cuda.empty_cache()
            handoff.update(reserved_after_bytes=int(torch.cuda.memory_reserved()),
                           allocated_after_bytes=int(torch.cuda.memory_allocated()))
            result.model.metadata["cuda_allocator_handoff"] = handoff
            if result.history:
                result.history[-1]["cuda_allocator_handoff"] = dict(handoff)
            return result
    result = fit_numpy(teacher_model, roots, settings, callback, stop_event)
    result.model.metadata["training_backend"] = information
    for row in result.history:
        row.update({"backend": "numpy", "device_name": information["device_name"]})
    return result
