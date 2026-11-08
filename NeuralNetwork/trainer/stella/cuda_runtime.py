"""Compile device-only kernels with the NVRTC library shipped with Torch."""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import tempfile
import sys
from pathlib import Path

import torch

from .indexing import require_integer

INT32_MAX = 2**31 - 1


def nvrtc_candidates(torch_library, cuda_version, *, windows=None, search_paths=None,
                     environment=None):
    """Find wheel-owned NVRTC without assuming pip libraries are in ldconfig.

    Prefer the CUDA major used by this Torch build. The bare loader names remain
    a final option for system/Colab installations with preconfigured loader paths.
    """
    windows = os.name == "nt" if windows is None else windows
    environment = os.environ if environment is None else environment
    search_paths = sys.path if search_paths is None else search_paths
    major = str(cuda_version or "12").split(".")[0]
    folders = [Path(torch_library)]
    for item in search_paths:
        if not item:
            continue
        root = Path(item)
        folders.extend((root / "nvidia/cuda_nvrtc/lib", root / "nvidia/cuda_nvrtc/bin",
                        root / "nvidia/cu13/lib", root / "nvidia/cu12/lib"))
    for item in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if item:
            folders.append(Path(item))
    cuda_root = environment.get("CUDA_PATH") or environment.get("CUDA_HOME")
    if cuda_root:
        folders.extend((Path(cuda_root) / "bin", Path(cuda_root) / "lib64"))
    pattern = "nvrtc64_*.dll" if windows else "libnvrtc.so*"
    found = []
    for folder in folders:
        if folder.is_dir():
            found.extend(str(p) for p in sorted(folder.glob(pattern)) if p.is_file())
    def matches(path):
        name = Path(path).name
        return (name.startswith("nvrtc64_" + major) if windows
                else name == "libnvrtc.so." + major or name.startswith("libnvrtc.so." + major + "."))
    found.sort(key=lambda p: (not matches(p), windows and p.lower().endswith(".alt.dll")))
    names = ([f"nvrtc64_{major}0_0.dll", "nvrtc64_120_0.dll", "nvrtc64_130_0.dll"] if windows
             else [f"libnvrtc.so.{major}", "libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"])
    override = environment.get("STELLA_NVRTC_LIBRARY")
    return list(dict.fromkeys(([override] if override else [])
        + [p for p in found if matches(p)] + names[:1]
        + [p for p in found if not matches(p)] + names[1:]))


class CudaModule:
    def __init__(self, source):
        torch.cuda.init()
        self.device = torch.device("cuda", torch.cuda.current_device())
        library = Path(torch.__file__).parent / "lib"
        self._dll_directories = []
        loader = C.WinDLL if os.name == "nt" else C.CDLL
        failures = []
        for candidate in nvrtc_candidates(library, torch.version.cuda):
            try:
                if os.name == "nt" and Path(candidate).is_absolute() and hasattr(os, "add_dll_directory"):
                    self._dll_directories.append(os.add_dll_directory(str(Path(candidate).parent)))
                self.nvrtc = loader(candidate)
                self.nvrtc_library = candidate
                break
            except OSError as error:
                failures.append(f"{candidate}: {error}")
        else:
            raise RuntimeError("Cannot load NVRTC for Torch CUDA " + str(torch.version.cuda)
                               + "; tried " + " | ".join(failures))
        self.driver = loader("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        major, minor = C.c_int(), C.c_int()
        self.nvrtc.nvrtcVersion.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int)]
        self._nv(self.nvrtc.nvrtcVersion(C.byref(major), C.byref(minor)))
        self.nvrtc_version = (major.value, minor.value)
        self.functions = {}
        capability = "".join(map(str, torch.cuda.get_device_capability()))
        options = [f"--gpu-architecture=sm_{capability}", "--std=c++17", "--fmad=false"]
        identity = hashlib.sha256((source + str(options) + torch.__version__ + str(self.nvrtc_version)).encode()).hexdigest()
        cache = Path(os.environ.get("STELLA_GPU_CACHE_DIR") or
                     Path(__file__).resolve().parents[1] / "runtime/cuda-cache")
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"games-{identity}.cubin"
        if path.exists():
            binary = path.read_bytes()
        else:
            program = C.c_void_p()
            self._nv(self.nvrtc.nvrtcCreateProgram(C.byref(program), source.encode(), b"games.cu", 0, None, None))
            args = (C.c_char_p * len(options))(*(x.encode() for x in options))
            code = self.nvrtc.nvrtcCompileProgram(program, len(options), args)
            size = C.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size))
            log = C.create_string_buffer(size.value)
            self.nvrtc.nvrtcGetProgramLog(program, log)
            if code:
                self.nvrtc.nvrtcDestroyProgram(C.byref(program))
                raise RuntimeError(log.value.decode(errors="replace"))
            self._nv(self.nvrtc.nvrtcGetCUBINSize(program, C.byref(size)))
            result = C.create_string_buffer(size.value)
            self._nv(self.nvrtc.nvrtcGetCUBIN(program, result))
            binary = result.raw
            self.nvrtc.nvrtcDestroyProgram(C.byref(program))
            # Concurrent evaluator workers can compile the same identity. Publish
            # only complete binaries, so another worker never reads a partial file.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache, prefix="games-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(binary)
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.module = C.c_void_p()
        self._cu(self.driver.cuModuleLoadData(C.byref(self.module), binary))
        self.identity = identity

    def _nv(self, code):
        if code:
            self.nvrtc.nvrtcGetErrorString.restype = C.c_char_p
            raise RuntimeError(self.nvrtc.nvrtcGetErrorString(code).decode())

    def _cu(self, code):
        if code:
            error = C.c_char_p()
            self.driver.cuGetErrorString(code, C.byref(error))
            raise RuntimeError(f"CUDA {code}: {error.value.decode()}")

    def launch(self, name, count, args, block=128, *, blocks=None):
        self.bind(name, count, args, block, blocks=blocks)()

    def bind(self, name, count, args, block=128, *, blocks=None):
        """Prepare stable arguments once for repeated launches in one rollout.

        The closure owns tensor references and ctypes cells until its last call.
        Stream selection stays dynamic, matching launch(). Never mutate/resize
        a bound tensor; a new rollout binds its own exact-size storage.
        """
        if not self.module:
            raise RuntimeError("Cannot bind a kernel from a closed CUDA module")
        count = require_integer(count, "kernel count", maximum=INT32_MAX)
        block = require_integer(block, "block size", minimum=1, maximum=1024)
        grid = ((count + block - 1)//block if blocks is None else
                require_integer(blocks, "grid blocks", maximum=INT32_MAX))
        args = tuple(args)  # Hold the exact pointees even if a caller mutates its list.
        values, tensors = [], []
        for position, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                if arg.device != self.device or not arg.is_contiguous():
                    raise ValueError(
                        f"Kernel argument {position} must be contiguous on {self.device}, "
                        f"got {arg.device} with shape {tuple(arg.shape)}")
                values.append(C.c_void_p(arg.data_ptr()))
                tensors.append(arg)
            else:
                # Kernel scalar ABI is signed int32; ctypes otherwise silently
                # wraps e.g. 2**32 into zero and launches with an invalid n.
                value = require_integer(arg, f"kernel scalar {position}",
                                        minimum=-(2**31), maximum=INT32_MAX)
                values.append(C.c_int(value))
        function = self.functions.get(name)
        if function is None:
            function = C.c_void_p()
            with torch.cuda.device(self.device):
                self._cu(self.driver.cuModuleGetFunction(C.byref(function), self.module, name.encode()))
            self.functions[name] = function
        pointers = (C.c_void_p * len(values))(*(C.addressof(v) for v in values))
        recorded_streams = set()
        def invoke():
            # Retain pointees and argument cells, not just integer addresses.
            _keepalive = (args, values)
            if not self.module:
                raise RuntimeError("Cannot launch a kernel from a closed CUDA module")
            if grid:
                with torch.cuda.device(self.device):
                    stream = torch.cuda.current_stream(self.device)
                    if stream.cuda_stream not in recorded_streams:
                        # Raw driver launches are invisible to the Torch allocator.
                        # Record the first launch too: an input tensor may have
                        # been allocated on another stream before binding.
                        # The caller still orders cross-stream dependencies.
                        for tensor in tensors:
                            tensor.record_stream(stream)
                        recorded_streams.add(stream.cuda_stream)
                    self._cu(self.driver.cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0,
                              C.c_void_p(stream.cuda_stream), pointers, None))
        return invoke

    def close(self):
        if self.module:
            with torch.cuda.device(self.device):
                self._cu(self.driver.cuModuleUnload(self.module))
            self.module = None
