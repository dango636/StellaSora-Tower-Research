"""One bounded evaluation worker shared by the six official checkpoints.

CPU input preparation is pipelined with GPU games. Model boundaries are FIFO
commands on the same pipe: all outstanding results are drained before switching
weights, and cached inputs contain no model-dependent values.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

from .catalogue_inputs import CatalogueInputs
from .io_utils import Control, StopRequested
from .presets import Preset


def _enabled(name, default=True):
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in ("0", "false", "no", "off")


def _worker(presets, run_dir, connection, seed, games, cache_bytes, pipeline,
            generator_factory=None):
    generator = None
    try:
        if generator_factory is None:
            from .gpu_games import GpuGames
            generator_factory = GpuGames
        disk_cache = None
        if _enabled("STELLA_EVAL_DISK_CACHE", default=False):
            configured = os.environ.get("STELLA_EVAL_DISK_CACHE_DIR")
            disk_cache = Path(configured) if configured else (Path(run_dir)/"cache" if run_dir is not None else None)
        inputs = CatalogueInputs(presets, seed, games, cache_bytes,
            disk_cache_dir=disk_cache,
            disk_cache_bytes=int(os.environ.get("STELLA_EVAL_DISK_CACHE_MB", "16384"))*2**20,
            disk_reserve_bytes=int(os.environ.get("STELLA_EVAL_DISK_RESERVE_MB", "16384"))*2**20)
        check = Control(run_dir).check if run_dir is not None else (lambda: None)
        prepared = queue.Queue(maxsize=1)
        stop = threading.Event()

        def put(item):
            while not stop.is_set():
                try:
                    prepared.put(item, timeout=.2)
                    return
                except queue.Full:
                    pass

        def prepare(request):
            if request is not None and request["kind"] == "batch":
                check()
                start, end = request["start"], request["stop"]
                return request, inputs.prepare(start, end)
            return request, None

        def producer():
            try:
                while not stop.is_set():
                    request = connection.recv()
                    put(prepare(request))
                    if request is None:
                        return
            except BaseException as error:
                put(error)

        if pipeline:
            # Only CPU-local RNGs are touched by the producer. Frequent GIL
            # handoff keeps Python CUDA launches from waiting for Random packing.
            sys.setswitchinterval(.0005)
            threading.Thread(target=producer, name="catalogue-input-prefetch", daemon=True).start()
        connection.send({"ready": True})
        input_wait = 0.0
        baseline_cache = inputs.snapshot()
        while True:
            began_wait = time.perf_counter()
            item = prepared.get() if pipeline else prepare(connection.recv())
            input_wait += time.perf_counter()-began_wait
            if isinstance(item, BaseException):
                raise item
            request, arrays = item
            if request is None:
                break
            check()
            if request["kind"] == "presets":
                inputs = CatalogueInputs(request["presets"], seed, games, cache_bytes,
                    disk_cache_dir=disk_cache,
                    disk_cache_bytes=int(os.environ.get("STELLA_EVAL_DISK_CACHE_MB", "16384"))*2**20,
                    disk_reserve_bytes=int(os.environ.get("STELLA_EVAL_DISK_RESERVE_MB", "16384"))*2**20)
                connection.send({"rebound": True})
            elif request["kind"] == "model":
                model_path = request["path"]
                if generator is None:
                    generator = generator_factory(model_path)
                else:
                    from .network import QNetwork
                    generator.set_model(QNetwork.load(model_path, frozen=True))
                # All per-model statistics start at zero. Session/cache counters
                # are explicitly labelled cumulative in the result.
                generator.stats = {key: type(value)(0) for key, value in generator.stats.items()}
                input_wait = 0.0
                baseline_cache = inputs.snapshot()
                information = dict(generator.information, resident_evaluation=True,
                                   input_pipeline=pipeline, input_cache_limit_bytes=cache_bytes,
                                   input_cache_mode=inputs.cache_mode,
                                   persistent_input_cache=inputs.disk_store.snapshot()
                                       if inputs.disk_store is not None else {"enabled": False})
                connection.send({"information": information})
            elif request["kind"] == "batch":
                if generator is None:
                    raise RuntimeError("Evaluation batch submitted before model selection")
                started = time.perf_counter()
                scores, steps = generator._run(*arrays, None, control=check)
                cache = inputs.snapshot()
                statistics = dict(generator.snapshot(), input_wait_seconds=input_wait,
                                  input_cache=cache,
                                  input_prepare_seconds=cache["prepare_seconds"]-baseline_cache["prepare_seconds"],
                                  input_cache_hits=cache["hits"]-baseline_cache["hits"],
                                  input_cache_misses=cache["misses"]-baseline_cache["misses"])
                connection.send({"start": request["start"], "stop": request["stop"],
                                 "scores": scores.astype("int16"), "steps": steps.astype("int16"),
                                 "statistics": statistics,
                                 "batch_seconds": time.perf_counter()-started})
                # Do not retain an uncached previous batch while waiting for the
                # next one. Cache-owned arrays remain valid for all six models.
                del arrays, item
            else:
                raise ValueError(f"Unknown catalogue worker command: {request['kind']}")
        stop.set()
    except StopRequested:
        try:
            connection.send({"stopped": True})
        except (BrokenPipeError, EOFError, OSError):
            pass
    except BaseException:
        try:
            connection.send({"error": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if "stop" in locals():
            stop.set()
        if generator is not None:
            generator.close()
        connection.close()


class CatalogueEvaluationSession:
    """Lazy, spawn-isolated CUDA lifetime; safe to create for cached evaluations.

    Reuse this object only for identical ordered presets, seed and game count.
    It does not modify run configuration or model files. At most two batches are
    in flight, so pipe, CPU temporary and interruption work stay bounded.
    """
    def __init__(self, presets, *, seed, games_per_preset=256, run_dir=None,
                 cache_mb=None, pipeline=None, _worker_target=None):
        # Canonicalize and copy nested pools. A caller mutating its original
        # dictionary after spawn must never change the parent's cache identity.
        self.presets = tuple(Preset.from_dict(p.to_dict() if isinstance(p, Preset) else p).to_dict()
                             for p in presets)
        self.seed, self.games = int(seed), int(games_per_preset)
        self.run_dir = str(Path(run_dir).resolve()) if run_dir is not None else None
        if cache_mb is None:
            cache_mb = int(os.environ.get("STELLA_EVAL_INPUT_CACHE_MB", "8192"))
        if int(cache_mb) < 0:
            raise ValueError("STELLA_EVAL_INPUT_CACHE_MB must be nonnegative")
        self.cache_bytes = int(cache_mb) * 2**20
        self.pipeline = _enabled("STELLA_EVAL_PREFETCH") if pipeline is None else bool(pipeline)
        self.depth = 2 if self.pipeline else 1
        self.worker_target = _worker_target or _worker
        self.process = self.connection = None
        self.pending = 0
        self.information = {}
        self.closed = False

    def matches(self, presets, seed, games_per_preset, run_dir):
        return (self.seed == int(seed) and self.games == int(games_per_preset)
                and self.presets == tuple(Preset.from_dict(
                    p.to_dict() if isinstance(p, Preset) else p).to_dict() for p in presets)
                and self.run_dir == (str(Path(run_dir).resolve()) if run_dir is not None else None))

    def rebind_presets(self, presets, control=None):
        """Change only prepared inputs between drained model evaluations."""
        if self.closed or self.pending:
            raise RuntimeError("Cannot rebind a closed or busy evaluation session")
        presets = tuple(Preset.from_dict(p.to_dict() if isinstance(p, Preset) else p).to_dict()
                        for p in presets)
        if self.process is not None:
            self.connection.send({"kind": "presets", "presets": presets})
            self._receive(control)
        self.presets = presets
        self.information = {}

    def _receive(self, control=None):
        while not self.connection.poll(.2):
            if control:
                control()
            if not self.process.is_alive():
                # A final error message may arrive between poll() and exit.
                if self.connection.poll():
                    break
                raise RuntimeError(f"Catalogue CUDA process exited: {self.process.exitcode}")
        try:
            result = self.connection.recv()
        except EOFError as error:
            raise RuntimeError("Catalogue CUDA process closed without a result") from error
        if result.get("stopped"):
            raise StopRequested()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def begin_model(self, path, control=None):
        if self.closed:
            raise RuntimeError("Evaluation session is already closed")
        if self.pending:
            raise RuntimeError("Cannot switch model while evaluation batches are outstanding")
        if self.process is None:
            context = mp.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(target=self.worker_target, name="stella-cuda-catalogue",
                args=(self.presets, self.run_dir, child, self.seed, self.games,
                      self.cache_bytes, self.pipeline))
            try:
                self.process.start()
            finally:
                child.close()
            self._receive(control)
        self.connection.send({"kind": "model", "path": str(Path(path).resolve())})
        self.information = self._receive(control)["information"]
        return self.information

    def submit(self, start, stop):
        if self.closed or self.process is None or not self.information:
            raise RuntimeError("Evaluation model is not ready")
        if self.pending >= self.depth:
            raise RuntimeError("Catalogue pipeline capacity exceeded")
        self.connection.send({"kind": "batch", "start": int(start), "stop": int(stop)})
        self.pending += 1

    def receive(self, control=None):
        if not self.pending:
            raise RuntimeError("No outstanding evaluation batch")
        result = self._receive(control)
        self.pending -= 1
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
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
        self.pending = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
