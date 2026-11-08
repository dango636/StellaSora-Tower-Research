"""One spawn-isolated FP32 inference service for CPU rollout workers.

Only features and predicted values cross this boundary. Each worker owns one
bounded shared-memory slot and at most one outstanding request. Torch/CUDA is
never imported in simulation workers, and the service exits before training.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time

import numpy as np

from .features import TARGET_SCALE
from .network import QNetwork

TIE_ATOL = 1e-4
TIE_RTOL = 2e-6


class InferenceUnavailable(RuntimeError):
    pass


def requested_backend(compute, variable="STELLA_LABEL_BACKEND"):
    requested = os.environ.get(variable, compute.get("backend", "auto")).lower()
    if requested not in ("auto", "numpy", "cuda"):
        raise ValueError(f"{variable} must be auto, numpy or cuda")
    if os.environ.get("STELLA_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return "numpy"
    return requested


def _serve(model_paths, requested, spec, connections, ready, stopping, counters, batch_rows):
    """Top-level target so Windows spawn never inherits a CUDA context."""
    try:
        sources = {name: QNetwork.load(path, frozen=True) for name, path in model_paths.items()}
        source = next(iter(sources.values()))
        if requested == "numpy":  # Used to exercise the transport without CUDA.
            information = {"backend": "numpy", "device_name": "CPU transport test"}
            def predict(x, name):
                return sources[name].predict(x)
        else:
            from .backend import backend_info
            information = backend_info(requested)
            if information["backend"] != "torch_cuda":
                ready.send({"error": information["reason"]})
                return
            import torch
            from .fit_torch import TorchQ
            torch.set_num_threads(1)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            models = {name: TorchQ(model, "cuda").eval().requires_grad_(False)
                      for name, model in sources.items()}
            host = torch.empty((batch_rows, source.input_dim), dtype=torch.float32, pin_memory=True)
            device = torch.empty_like(host, device="cuda")
            staging = host.numpy()

            def predict(x, name):
                n = len(x)
                with torch.inference_mode():
                    device[:n].copy_(host[:n], non_blocking=True)
                    return (models[name](device[:n], amp=False) * TARGET_SCALE).cpu().numpy()

            warm_rows = min(32, batch_rows)
            staging[:warm_rows] = 0
            for name in models:
                predict(staging[:warm_rows], name)
        inputs = [np.frombuffer(b, np.float32).reshape(spec["rows"], source.input_dim)
                  for b in spec["inputs"]]
        outputs = [np.frombuffer(b, np.float32) for b in spec["outputs"]]
        if requested == "numpy":
            staging = np.empty((batch_rows, source.input_dim), dtype=np.float32)
        information.update(precision="FP32", amp=False, tf32=False,
                           transport="shared_memory", service_pid=os.getpid(),
                           model_parameter_sha256=source.checksum(),
                           models={name: model.checksum() for name, model in sources.items()},
                           tie_atol=TIE_ATOL, tie_rtol=TIE_RTOL, version=2)
        ready.send({"information": information})
        indexes = {connection: i for i, connection in enumerate(connections)}

        def execute(group, name):
            offset = 0
            for i, n in group:
                staging[offset:offset+n] = inputs[i][:n]
                offset += n
            values = predict(staging[:offset], name)
            if values.shape != (offset,) or not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite or incomplete GPU prediction")
            counters[0] += len(group)
            counters[1] += 1
            counters[2] += offset
            counters[3] = max(counters[3], offset)
            offset = 0
            for i, n in group:
                outputs[i][:n] = values[offset:offset+n]
                offset += n
                connections[i].send(None)

        while not stopping.is_set():
            available = wait(connections, timeout=.05)
            if not available:
                continue
            # Drain ready requests without adding a timer wait to every batch.
            others = [c for c in connections if c not in available]
            if others:
                available += wait(others, timeout=0)
            by_model = {}
            for connection in available:
                request = connection.recv()
                name, n = request if isinstance(request, tuple) else ("teacher", request)
                if type(n) is not int or not 0 < n <= spec["rows"]:
                    raise ValueError("Invalid shared inference request")
                by_model.setdefault(name, []).append((indexes[connection], n))
            for name, requests in by_model.items():
                group, rows = [], 0
                for item in requests:
                    if group and rows+item[1] > batch_rows:
                        execute(group, name)
                        group, rows = [], 0
                    group.append(item)
                    rows += item[1]
                execute(group, name)
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        try:
            ready.send({"error": message})
        except (EOFError, OSError):
            pass
        for connection in connections:
            try:
                connection.send(message)
            except (EOFError, OSError):
                pass
    finally:
        spec["alive"].clear()
        ready.close()
        for connection in connections:
            connection.close()


class InferenceService:
    def __init__(self, model_path, workers, requested="auto", *, rows=4096, batch_rows=8192):
        if workers < 1 or rows < 1 or batch_rows < rows:
            raise ValueError("Invalid inference capacity")
        model_paths = model_path if isinstance(model_path, dict) else {"teacher": str(model_path)}
        sources = {name: QNetwork.load(path, frozen=True) for name, path in model_paths.items()}
        source = next(iter(sources.values()))
        context = mp.get_context("spawn")
        pairs = [context.Pipe(duplex=True) for _ in range(workers)]
        self.connections = [pair[1] for pair in pairs]
        self.spec = {"rows": rows, "input_dim": source.input_dim,
                     "inputs": [context.RawArray("f", rows*source.input_dim) for _ in pairs],
                     "outputs": [context.RawArray("f", rows) for _ in pairs],
                     "connections": [pair[0] for pair in pairs],
                     "next_slot": context.Value("i", 0), "alive": context.Event(),
                     "refined": context.RawArray("q", workers),
                     "refined_batches": context.RawArray("q", workers),
                     "model_paths": model_paths, "model_checksum": source.checksum(),
                     "model_checksums": {name: model.checksum() for name, model in sources.items()}}
        self.stopping = context.Event()
        self.counters = context.RawArray("q", 4)
        self.ready, child_ready = context.Pipe(duplex=False)
        self.child_ready = child_ready
        self.process = context.Process(target=_serve, name="stella-label-cuda",
            args=(model_paths, requested, self.spec, self.connections, child_ready,
                  self.stopping, self.counters, batch_rows), daemon=True)
        self.information = None

    def start(self, control=None, timeout=60):
        self.spec["alive"].set()
        self.process.start()
        deadline = time.monotonic()+timeout
        try:
            while not self.ready.poll(.1):
                if control:
                    control()
                if self.ready.poll():  # A reply can arrive during a long pause.
                    break
                if not self.process.is_alive() or time.monotonic() >= deadline:
                    raise InferenceUnavailable("GPU inference startup failed or timed out")
            result = self.ready.recv()
            if "error" in result:
                raise InferenceUnavailable(result["error"])
            self.information = result["information"]
            self.spec["information"] = self.information
            return self
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        return {**(self.information or {}), "requests": self.counters[0],
                "batches": self.counters[1], "action_rows": self.counters[2],
                "max_batch_rows": self.counters[3],
                "cpu_refined_decisions": sum(self.spec["refined"]),
                "cpu_refined_batches": sum(self.spec["refined_batches"])}

    def close(self):
        self.stopping.set()
        if self.process.pid is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
        self.spec["alive"].clear()
        for connection in self.connections + self.spec["connections"] + [self.ready, self.child_ready]:
            connection.close()


class InferenceClient:
    def __init__(self, spec, control=None):
        with spec["next_slot"].get_lock():
            self.slot = spec["next_slot"].value
            spec["next_slot"].value += 1
        if self.slot >= len(spec["inputs"]):
            raise RuntimeError("No free inference worker slot")
        self.spec, self.control = spec, control
        self.connection = spec["connections"][self.slot]
        self.x = np.frombuffer(spec["inputs"][self.slot], np.float32).reshape(spec["rows"], spec["input_dim"])
        self.y = np.frombuffer(spec["outputs"][self.slot], np.float32)

    def predict(self, x, name="teacher"):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.spec["input_dim"] or not np.isfinite(x).all():
            raise ValueError("Invalid public feature matrix")
        result = np.empty(len(x), dtype=np.float32)
        for offset in range(0, len(x), self.spec["rows"]):
            block = x[offset:offset+self.spec["rows"]]
            if not self.spec["alive"].is_set():
                raise RuntimeError("GPU inference service stopped; completed labels are preserved")
            self.x[:len(block)] = block
            self.connection.send((name, len(block)))
            deadline = time.monotonic()+30
            while not self.connection.poll(.05):
                if self.control:
                    self.control()
                if self.connection.poll():  # Do not time out a reply received while paused.
                    break
                if not self.spec["alive"].is_set() or time.monotonic() >= deadline:
                    raise RuntimeError("GPU inference service stopped or request timed out")
            error = self.connection.recv()
            if error is not None:
                raise RuntimeError(f"GPU inference failed: {error}")
            result[offset:offset+len(block)] = self.y[:len(block)]
        return result


class RemoteQModel:
    """Resolve close action rankings with the original NumPy frozen teacher."""
    def __init__(self, client, source, name="teacher"):
        checksum = client.spec.get("model_checksums", {"teacher": client.spec["model_checksum"]})[name]
        if source.checksum() != checksum:
            raise ValueError("GPU service does not match the frozen CPU teacher")
        self.client, self.source, self.name = client, source, name
        self.information = client.spec["information"]

    def predict(self, x):
        return self.client.predict(x) if self.name == "teacher" else self.client.predict(x, self.name)

    def predict_actions(self, x, sizes):
        values = self.predict(x)
        sizes = np.asarray(sizes, dtype=np.int64)
        if sizes.ndim != 1 or not len(sizes) or np.any(sizes < 1) or sizes.sum() != len(values):
            raise ValueError("Invalid action group sizes")
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        maxima = np.maximum.reduceat(values, offsets[:-1])
        tolerance = TIE_ATOL + TIE_RTOL*np.abs(maxima)
        close = values >= np.repeat(maxima-tolerance, sizes)
        ambiguous = np.flatnonzero(np.add.reduceat(close.astype(np.int32), offsets[:-1]) > 1)
        if len(ambiguous):
            # Preserve the original CPU batch shape. BLAS can round a tiny
            # subset differently, even turning an exact CPU tie into a winner.
            if self.name == "teacher":
                values = self.source.predict(x)
            else:
                # Evaluation previously inferred one decision at a time.
                for i in ambiguous:
                    start, end = offsets[i:i+2]
                    values[start:end] = self.source.predict(x[start:end])
            self.client.spec["refined"][self.client.slot] += len(ambiguous)
            self.client.spec["refined_batches"][self.client.slot] += 1
        return values
