"""Verify the installed CUDA runtime without downloading or installing.

The clean source packages use the active Python environment on both Colab and
the desktop. The probe runs in the launcher's separate process, so it cannot
change training RNG state or precision settings. No bundled interpreter is
required. Environment variables used by the probe are restored before return.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from .io_utils import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def probe_current_runtime():
    report = {"python_executable": sys.executable, "ok": False,
              "packages_installed": False}
    try:
        if sys.version_info < (3, 10):
            raise RuntimeError("The trainer requires Python 3.10 or newer")
        import numpy as np
        import torch
        report.update(torch_version=torch.__version__, numpy_version=np.__version__,
                      cuda_runtime=torch.version.cuda)
        version = tuple(int(x) for x in re.findall(r"\d+", torch.__version__)[:2])
        if version < (2, 7):
            raise RuntimeError("The CUDA trainer requires PyTorch 2.7 or newer")
        if not torch.cuda.is_available():
            raise RuntimeError("The active Python cannot use CUDA")
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        properties = torch.cuda.get_device_properties(device)
        report.update(device_name=properties.name, capability=list(capability),
                      total_memory_bytes=properties.total_memory)
        from .hardware_profiles import profile_name, validate_gpu_profile
        selected_profile = profile_name()
        if selected_profile:
            report.update(validate_gpu_profile(selected_profile, properties.total_memory, properties.name))
        runtime_version = tuple(int(x) for x in re.findall(r"\d+", str(torch.version.cuda))[:2])
        if capability[0] >= 12 and runtime_version < (12, 8):
            raise RuntimeError("This Blackwell GPU requires a CUDA 12.8+ PyTorch build")
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            x = torch.ones((32, 32), device="cuda", dtype=torch.float32, requires_grad=True)
            y = (x @ x.T).sum()
            y.backward()
            torch.cuda.synchronize()
            if not torch.equal(x.grad, torch.full_like(x, 64)):
                raise RuntimeError("CUDA FP32 forward/backward verification failed")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        from .cuda_runtime import CudaModule
        source = ('extern "C" __global__ void stella_probe(long long* x, int n) {'
                  'int i=blockIdx.x*blockDim.x+threadIdx.x;'
                  'if(i<n) x[i]=(long long)2147483647+i+1;}')
        previous_cache = os.environ.get("STELLA_GPU_CACHE_DIR")
        module = None
        with tempfile.TemporaryDirectory(prefix="stella-nvrtc-probe-") as cache:
            try:
                os.environ["STELLA_GPU_CACHE_DIR"] = cache
                module = CudaModule(source)
                output = torch.zeros(32, dtype=torch.int64, device="cuda")
                module.launch("stella_probe", 32, [output, 32])
                torch.cuda.synchronize()
                if not torch.equal(output, torch.arange(32, dtype=torch.int64, device="cuda")+2**31):
                    raise RuntimeError("NVRTC int64 kernel verification failed")
                report.update(nvrtc_library=module.nvrtc_library,
                              nvrtc_version=list(module.nvrtc_version),
                              nvrtc_execution_verified=True, fp32_backward_verified=True)
            finally:
                try:
                    if module is not None:
                        module.close()
                finally:
                    if previous_cache is None:
                        os.environ.pop("STELLA_GPU_CACHE_DIR", None)
                    else:
                        os.environ["STELLA_GPU_CACHE_DIR"] = previous_cache
        report.update(ok=True, backend="torch_cuda", precision="FP32")
    except Exception as error:
        report["reason"] = type(error).__name__ + ": " + str(error)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(ROOT / "runs/current"))
    parser.add_argument("--retry", action="store_true", help="Compatibility option; every call probes the current runtime")
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU execution; skip the CUDA probe")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if (run_dir / "stop.flag").exists():
        result = {"state": "cancelled", "message": "Runtime verification cancelled", "backend": "pending"}
    elif args.cpu:
        result = {"state": "cpu", "ok": True, "message": "CPU execution explicitly requested", "backend": "numpy",
                  "python_executable": sys.executable, "packages_installed": False}
    else:
        result = probe_current_runtime()
        result.update(state="ready" if result["ok"] else "unavailable",
                      message="Existing CUDA runtime verified" if result["ok"] else "Existing CUDA runtime is unavailable")
        if (run_dir / "stop.flag").exists():
            result.update(state="cancelled", message="Runtime verification cancelled")
    result.update(elapsed_seconds=time.monotonic()-started, updated_at=time.time())
    atomic_json(run_dir / "gpu_setup.json", result)
    print(result["message"] + (": " + result["reason"] if result.get("reason") else ""), flush=True)
    return 0 if result["state"] in {"ready", "cpu"} else (10 if result["state"] == "cancelled" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
