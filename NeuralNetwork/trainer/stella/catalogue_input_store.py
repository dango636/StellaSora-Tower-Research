"""Verified local-disk cache for reproducible, model-independent RNG inputs.

This cache never stores predictions, labels or game scores. A damaged block is
removed only inside its owned cache directory and rebuilt from the same public
preset and seed by CatalogueInputs. Fixed inputs can be reused in later rounds.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

from .io_utils import atomic_json, digest, dumps, file_hash


SCHEMA = "catalogue_rng_latent_disk_v1"
OWNED_DIRECTORY = "catalogue-inputs-v1"
FILES = {"randoms": "randoms.npy", "latent": "latent.npy"}


def source_identity():
    """Code that controls seed derivation, latent draws and packed-state ABI."""
    directory = Path(__file__).resolve().parent
    return {name: file_hash(directory/name) for name in
            ("presets.py", "random_streams.py", "gpu_game_tables.py", "catalogue_inputs.py", "vendor/stella_sim.py")}


class CatalogueInputStore:
    def __init__(self, parent, identity, *, budget_bytes=16*2**30, reserve_bytes=16*2**30):
        self.identity = json.loads(dumps(dict(identity, cache_schema=SCHEMA,
                             python_random_runtime=list(sys.version_info[:2]), sources=source_identity())))
        self.identity_hash = digest(self.identity)
        self.parent = Path(parent).expanduser().resolve()
        self.root = self.parent/OWNED_DIRECTORY
        self.directory = self.root/self.identity_hash
        self.budget_bytes, self.reserve_bytes = int(budget_bytes), int(reserve_bytes)
        if self.budget_bytes < 0 or self.reserve_bytes < 0:
            raise ValueError("Input disk cache budget and disk reserve must be nonnegative")
        self.enabled = self.budget_bytes > 0 and os.name == "posix"
        self.disabled_reason = None
        if os.name != "posix":
            self.disabled_reason = "local POSIX cache locking unavailable"
        if self.parent == Path("/content/drive") or Path("/content/drive") in self.parent.parents:
            self.enabled = False
            self.disabled_reason = "mounted Google Drive is excluded from the training input cache"
        marker_bytes = len(dumps(self.identity).encode())+1
        if 0 < self.budget_bytes < marker_bytes:
            self.enabled = False
            self.disabled_reason = "input cache budget is smaller than its identity metadata"
        self.lock = threading.Lock()
        self.stats = dict(hits=0, misses=0, rebuilt_blocks=0, writes=0, capacity_bypasses=0,
                          free_space_bypasses=0, io_bypasses=0, read_bytes=0, written_bytes=0,
                          read_seconds=0., write_seconds=0.)
        if self.enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                if self.root.is_symlink():
                    raise ValueError("Input cache directory must not be a symlink")
                with self._disk_lock():
                    self.directory.mkdir(exist_ok=True)
                    if self.directory.is_symlink():
                        raise ValueError("Input cache identity directory must not be a symlink")
                    marker = self.directory/"identity.json"
                    if marker.exists():
                        if marker.is_symlink() or json.loads(marker.read_text()) != self.identity:
                            raise ValueError("Input cache identity marker does not match its namespace")
                    elif any(self.directory.iterdir()):
                        raise ValueError("Refusing an unmarked nonempty input cache namespace")
                    else:
                        if self._used_bytes()+marker_bytes > self.budget_bytes:
                            self.enabled = False
                            self.disabled_reason = "input cache capacity already exhausted"
                        elif shutil.disk_usage(self.root).free-marker_bytes < self.reserve_bytes:
                            self.enabled = False
                            self.disabled_reason = "input cache disk reserve would be crossed"
                        else:
                            atomic_json(marker, self.identity)
            except OSError as error:
                self.enabled = False
                self.disabled_reason = f"local cache unavailable: {type(error).__name__}"

    @contextmanager
    def _disk_lock(self):
        import fcntl
        lock_path = self.root/"cache.lock"
        if lock_path.is_symlink():
            raise ValueError("Input cache lock must not be a symlink")
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _update(self, **values):
        with self.lock:
            for key, value in values.items():
                self.stats[key] += value

    def _block(self, start, stop):
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
            raise ValueError("Cache intervals must be increasing nonnegative integers")
        block = self.directory/f"block_{start:08d}_{stop:08d}"
        if block.is_symlink():
            raise ValueError("Input cache block must not be a symlink")
        return block

    @staticmethod
    def _remove_owned_files(block):
        removed = 0
        if not block.exists():
            return removed
        for path in block.iterdir():
            owned = (path.name in (*FILES.values(), "manifest.json") or
                     (path.name.endswith(".tmp") and
                      any(path.name.startswith(name+".") for name in (*FILES.values(), "manifest.json"))))
            if not owned:
                continue
            if path.is_symlink():
                raise ValueError("Refusing to remove a symlink from the input cache")
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    def load(self, start, stop):
        if not self.enabled:
            return None
        began = time.perf_counter()
        try:
            return self._load(start, stop, began)
        except OSError:
            # A transient local-disk problem must not abort a valid training
            # run. Keep the cache optional; unsafe symlinks still raise ValueError.
            self._update(misses=1, io_bypasses=1, read_seconds=time.perf_counter()-began)
            return None

    def _load(self, start, stop, began):
        block = self._block(start, stop)
        marker = block/"manifest.json"
        if not marker.exists():
            with self._disk_lock():
                # A writer may have completed while this reader waited. Only
                # uncommitted files are recoverable as a partial write here.
                if not marker.exists():
                    removed = self._remove_owned_files(block)
                    self._update(misses=1, rebuilt_blocks=int(bool(removed)))
                    return None
        try:
            with self._disk_lock():
                if marker.is_symlink():
                    raise ValueError("Input cache manifest must not be a symlink")
                metadata = json.loads(marker.read_text())
                if (metadata.get("schema") != SCHEMA or metadata.get("identity") != self.identity_hash
                        or metadata.get("start") != start or metadata.get("stop") != stop
                        or set(metadata.get("arrays", {})) != set(FILES)):
                    raise ValueError("Input cache block metadata does not match the requested interval")
                arrays, read_bytes = [], 0
                for name, filename in FILES.items():
                    path = block/filename
                    shape = (stop-start, 625 if name == "randoms" else 5)
                    dtype = np.dtype(np.uint32 if name == "randoms" else np.int32)
                    entry = metadata["arrays"][name]
                    expected_bytes = int(np.prod(shape))*dtype.itemsize
                    size = path.stat().st_size
                    if (path.is_symlink() or entry.get("shape") != list(shape)
                            or entry.get("dtype") != dtype.str or entry.get("size_bytes") != size
                            or not expected_bytes <= size <= expected_bytes+4096):
                        raise ValueError("Input cache shape/dtype/file length mismatch")
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                        raise ValueError("Input cache byte checksum mismatch")
                    # A normal ndarray keeps future external disk edits from
                    # mutating already-verified resident inputs through mmap.
                    array = np.load(io.BytesIO(payload), allow_pickle=False, max_header_size=4096)
                    if array.shape != shape or array.dtype != dtype:
                        raise ValueError("Input cache NPY header mismatch")
                    if name == "randoms" and not array.flags.f_contiguous:
                        raise ValueError("Cached MT state must retain Fortran storage")
                    owner = array
                    while isinstance(owner.base, np.ndarray):
                        owner = owner.base
                    owner.setflags(write=False)
                    array.setflags(write=False)
                    arrays.append(array)
                    read_bytes += len(payload)
                self._update(hits=1, read_bytes=read_bytes, read_seconds=time.perf_counter()-began)
                return tuple(arrays)
        except (ValueError, KeyError, TypeError, AttributeError, EOFError):
            # These are deterministic inputs, so rebuilding is safe. Remove
            # only our three known block files; never an external target or
            # another namespace, teacher, checkpoint, score or run artifact.
            with self._disk_lock():
                self._remove_owned_files(block)
            self._update(misses=1, rebuilt_blocks=1, read_seconds=time.perf_counter()-began)
            return None

    def _used_bytes(self):
        total = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Input cache inventory contains a symlink")
            if path.is_file():
                total += path.stat().st_size
        return total

    @staticmethod
    def _write_array(path, array):
        fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return dict(shape=list(array.shape), dtype=array.dtype.str, size_bytes=path.stat().st_size,
                    sha256=file_hash(path))

    def save(self, start, stop, randoms, latent):
        if not self.enabled:
            return False
        began = time.perf_counter()
        block = self._block(start, stop)
        randoms, latent = np.asarray(randoms), np.asarray(latent)
        if (randoms.shape != (stop-start, 625) or randoms.dtype != np.uint32
                or latent.shape != (stop-start, 5) or latent.dtype != np.int32
                or not randoms.flags.f_contiguous):
            raise ValueError("Only canonical RNG/latent inputs may enter the disk cache")
        # Includes both NPY headers, JSON metadata and transient atomic files.
        required = randoms.nbytes+latent.nbytes+16384
        try:
            with self._disk_lock():
                if self._used_bytes()+required > self.budget_bytes:
                    self._update(capacity_bypasses=1)
                    return False
                if shutil.disk_usage(self.root).free-required < self.reserve_bytes:
                    self._update(free_space_bypasses=1)
                    return False
                block.mkdir(exist_ok=True)
                metadata = dict(schema=SCHEMA, identity=self.identity_hash, start=start, stop=stop,
                                arrays={})
                for name, array in (("randoms", randoms), ("latent", latent)):
                    path = block/FILES[name]
                    if path.is_symlink():
                        raise ValueError("Input cache array path must not be a symlink")
                    metadata["arrays"][name] = self._write_array(path, array)
                atomic_json(block/"manifest.json", metadata)
                self._update(writes=1, written_bytes=sum(v["size_bytes"] for v in metadata["arrays"].values()),
                             write_seconds=time.perf_counter()-began)
                return True
        except OSError:
            self._update(io_bypasses=1, write_seconds=time.perf_counter()-began)
            return False

    def snapshot(self):
        with self.lock:
            return dict(self.stats, enabled=self.enabled, disabled_reason=self.disabled_reason,
                        identity=self.identity_hash, budget_bytes=self.budget_bytes,
                        disk_reserve_bytes=self.reserve_bytes,
                        contents="RNG state and latent draws only; no models, predictions or scores")
