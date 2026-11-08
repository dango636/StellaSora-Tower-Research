"""Spawn-isolated CUDA game generator used by the resumable training engine."""
from __future__ import annotations

import multiprocessing as mp
import json
import os
from pathlib import Path
import traceback
import queue
import threading
import time
import sys

from .io_utils import Control, StopRequested
from .label_inference import InferenceUnavailable


def _serve(model_paths, run_dir, connection, operation, pipeline):
    generator = None
    try:
        from .backend import backend_info
        information = backend_info("auto")
        if information["backend"] != "torch_cuda":
            connection.send({"unavailable": information["reason"]})
            return
        control = Control(run_dir)
        if operation == "evaluation":
            from .gpu_games import GpuPairedGames
            generator = GpuPairedGames(model_paths)
            generate = generator.evaluate_pairs
        elif operation == "sampling":
            from .gpu_sampling import GpuFamilySampler
            generator = GpuFamilySampler(model_paths)
            generate = generator.sample_families
        elif operation == "mining":
            from .hard_mining import GpuHardMiner
            generator = GpuHardMiner(model_paths)
            generate = generator.mine
        else:
            from .gpu_games import GpuGames
            generator = GpuGames(model_paths)
            generate = generator.label_roots
        import torch
        config_path = Path(run_dir) / "resolved_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        cache_limit_mb = int(os.environ.get("STELLA_GPU_CACHE_MB",
            config.get("compute", {}).get("gpu_memory_limit_mb", 3072)))
        if cache_limit_mb < 1:
            raise ValueError("CUDA allocator cache limit must be positive")
        cache_trims = 0
        generator.information["cache_limit_mb"] = cache_limit_mb
        generator.information["label_pipeline"] = pipeline
        connection.send({"information": generator.information})
        prepared_queue = queue.Queue(maxsize=1)
        from .network import QNetwork
        model_cache = {}
        active_paths = model_paths
        model_switches = 0
        model_switch_seconds = 0.0

        def unpack_request(request):
            if isinstance(request, list):
                return request, None, None
            tasks, paths = request["tasks"], request["model_paths"]
            named_paths = paths if isinstance(paths, dict) else {"teacher": paths}
            sources = {}
            for name, path in named_paths.items():
                if path not in model_cache:
                    model_cache[path] = QNetwork.load(path, frozen=True)
                sources[name] = model_cache[path]
            return tasks, paths, sources

        def switch_models(paths, sources):
            nonlocal active_paths, model_switches, model_switch_seconds
            if paths is None or paths == active_paths:
                return
            started = time.perf_counter()
            if operation == "label":
                generator.set_model(sources["teacher"])
            elif operation == "sampling":
                generator.set_models(paths, sources)
            else:
                raise ValueError("Resident model switching supports label and sampling operations")
            active_paths = paths
            model_switches += 1
            model_switch_seconds += time.perf_counter()-started
        if pipeline:
            # RNG packing is Python-heavy; give CUDA's launch thread frequent
            # access to the GIL instead of blocking it for the default 5 ms.
            sys.setswitchinterval(.0005)
            def prefetch():
                try:
                    while True:
                        request = connection.recv()
                        if request is None:
                            prepared_queue.put(None)
                            return
                        control.check()
                        started = time.perf_counter()
                        tasks, paths, sources = unpack_request(request)
                        prepared = generator.prepare_label_roots(tasks)
                        prepared_queue.put((tasks, prepared, time.perf_counter()-started, paths, sources))
                except BaseException as exc:
                    prepared_queue.put(exc)
            threading.Thread(target=prefetch, name="label-input-prefetch", daemon=True).start()
        prepare_seconds = wait_seconds = 0.0
        while True:
            if pipeline:
                started = time.perf_counter()
                item = prepared_queue.get()
                wait_seconds += time.perf_counter()-started
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                tasks, prepared, elapsed, paths, sources = item
                prepare_seconds += elapsed
                switch_models(paths, sources)
                rows = generator.label_roots(tasks, control.check, prepared=prepared)
                del prepared, item
            else:
                request = connection.recv()
                if request is None:
                    break
                tasks, paths, sources = unpack_request(request)
                switch_models(paths, sources)
                rows = generate(tasks, control.check)
            # Dynamic action counts leave differently sized temporary blocks in
            # the allocator. Release those free blocks between complete batches
            # before WDDM starts paging an otherwise small live working set.
            reserved_mb = torch.cuda.memory_reserved() / 2**20
            peak_mb = torch.cuda.max_memory_allocated() / 2**20
            if reserved_mb > cache_limit_mb:
                torch.cuda.empty_cache()
                cache_trims += 1
            from .hardware_profiles import memory_limit_summary
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            generator.information["memory_limits"] = memory_limit_summary(
                allocator_cache_mb=cache_limit_mb, device_free_bytes=free_bytes,
                device_total_bytes=total_bytes)
            statistics = dict(generator.snapshot(), cache_trims=cache_trims,
                              resident_model_switches=model_switches,
                              model_switch_seconds=model_switch_seconds,
                              cached_cpu_models=len(model_cache),
                              input_prepare_seconds=prepare_seconds,
                              input_wait_seconds=wait_seconds,
                              reserved_before_trim_mb=reserved_mb,
                              reserved_after_trim_mb=torch.cuda.memory_reserved() / 2**20,
                              allocated_mb=torch.cuda.memory_allocated() / 2**20,
                              peak_allocated_mb=peak_mb)
            torch.cuda.reset_peak_memory_stats()
            connection.send({"rows": rows, "stats": statistics})
    except StopRequested:
        connection.send({"stopped": True})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        if generator:
            generator.close()
        connection.close()


class CudaGameService:
    def __init__(self, model_paths, run_dir, operation="label", pipeline=None):
        context = mp.get_context("spawn")
        self.connection, self.child = context.Pipe()
        paths = ({key: str(path) for key, path in model_paths.items()}
                 if isinstance(model_paths, dict) else str(model_paths))
        self.pipeline = operation == "label" and (True if pipeline is None else bool(pipeline))
        self.process = context.Process(target=_serve, name=f"stella-cuda-{operation}",
            args=(paths, str(run_dir), self.child, operation, self.pipeline))
        self.information, self.stats = {}, {}

    def receive(self, control):
        while not self.connection.poll(.2):
            control()
            if not self.process.is_alive():
                raise RuntimeError(f"CUDA game process exited: {self.process.exitcode}")
        result = self.connection.recv()
        if result.get("stopped"):
            raise StopRequested()
        if "unavailable" in result:
            raise InferenceUnavailable(result["unavailable"])
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def start(self, control):
        self.process.start()
        self.child.close()
        try:
            self.information = self.receive(control)["information"]
            return self
        except BaseException:
            self.close()
            raise

    def generate(self, tasks, control, model_paths=None):
        self.submit(tasks, model_paths=model_paths)
        return self.finish(control)

    def submit(self, tasks, model_paths=None):
        self.connection.send(tasks if model_paths is None else {"tasks": tasks, "model_paths": model_paths})

    def finish(self, control):
        result = self.receive(control)
        self.stats = result["stats"]
        return result["rows"]

    def snapshot(self):
        return dict(self.information, **self.stats)

    def close(self):
        try:
            self.connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        if self.process.pid is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
        self.connection.close()
