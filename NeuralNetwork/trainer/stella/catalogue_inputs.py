"""Bounded immutable fixed-seed inputs for repeated catalogue evaluations.

The compact mode caches RNG/latent arrays; a high-memory profile may also keep
public states and beliefs. Optional verified local storage retains RNG/latent
inputs across rounds. One producer owns prepare(); snapshot is safe from the
CUDA consumer thread. No cached value depends on model weights.
"""
from __future__ import annotations

from collections import OrderedDict
import os
import threading
import time

import numpy as np

from .gpu_game_tables import LAYOUT, pack_state
from .presets import Preset, canonical_seed
from .random_streams import pack_streams
from .vendor import stella_sim as sim

SEED_DOMAIN = "fixed_catalogue_environment_v1"


class CatalogueInputs:
    def __init__(self, presets, seed, games_per_preset, cache_bytes=0, cache_mode=None,
                 *, disk_cache_dir=None, disk_cache_bytes=16*2**30, disk_reserve_bytes=16*2**30):
        self.presets = tuple(Preset.from_dict(p.to_dict() if isinstance(p, Preset) else p)
                             for p in presets)
        self.seed, self.games_per_preset = int(seed), int(games_per_preset)
        if not self.presets or self.games_per_preset < 1 or int(cache_bytes) < 0:
            raise ValueError("Nonempty presets, positive games and nonnegative cache required")
        self.roots = tuple(sim.initial_from_pools(p.pools) for p in self.presets)
        self.packed = tuple(pack_state(root, i) for i, root in enumerate(self.roots))
        self.fingerprints = tuple(p.fingerprint for p in self.presets)
        self.total = len(self.presets) * self.games_per_preset
        self.cache_bytes = int(cache_bytes)
        self.cache_mode = (os.environ.get("STELLA_EVAL_INPUT_CACHE_MODE", "compact")
                           if cache_mode is None else str(cache_mode))
        if self.cache_mode not in ("compact", "full"):
            raise ValueError("Evaluation input cache mode must be compact or full")
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.prototype_bytes = sum(a.nbytes for pair in self.packed for a in pair)
        self.disk_store = None
        if disk_cache_dir is not None:
            from .catalogue_input_store import CatalogueInputStore
            identity = dict(seed_domain=SEED_DOMAIN, seed=self.seed, games_per_preset=self.games_per_preset,
                            preset_fingerprints=list(self.fingerprints), rules_version=sim.MODEL_VERSION,
                            state_size=len(self.packed[0][0]), belief_size=len(self.packed[0][1]),
                            mt_state_words=625, latent_fields=["gE", "fE", "gC", "fC", "G1"])
            self.disk_store = CatalogueInputStore(disk_cache_dir, identity,
                budget_bytes=disk_cache_bytes, reserve_bytes=disk_reserve_bytes)
        self.stats = dict(hits=0, misses=0, evictions=0, bytes=0, peak_bytes=0,
                          prepared_games=0, reused_games=0, prepare_seconds=0.0,
                          rng_prepare_seconds=0.0, reconstruct_seconds=0.0)

    def _segments(self, start, stop):
        cursor = start
        while cursor < stop:
            index, game = divmod(cursor, self.games_per_preset)
            n = min(stop-cursor, self.games_per_preset-game)
            yield index, game, cursor-start, cursor-start+n
            cursor += n

    def _reconstruct(self, start, stop, randoms, latent):
        state0, belief0 = self.packed[0]
        count = stop-start
        states = np.empty((count, len(state0)), dtype=state0.dtype)
        beliefs = np.empty((count, len(belief0)), dtype=belief0.dtype)
        for index, _, begin, end in self._segments(start, stop):
            state, belief = self.packed[index]
            states[begin:end] = state
            beliefs[begin:end] = belief
        states[:, LAYOUT["gE"]:LAYOUT["gE"]+5] = latent
        states.setflags(write=False)
        beliefs.setflags(write=False)
        # Return a view over a read-only owner. Normal callers cannot turn this
        # view writable and accidentally corrupt later candidates' RNG streams.
        random_view = randoms.view()
        random_view.setflags(write=False)
        # Views over read-only owners also prevent ordinary callers from
        # re-enabling writeability on the tensors retained for later models.
        return states.view(), beliefs.view(), random_view

    def prepare(self, start, stop):
        began = time.perf_counter()
        start, stop = int(start), int(stop)
        if not 0 <= start < stop <= self.total:
            raise ValueError(f"Invalid catalogue interval [{start}, {stop})/{self.total}")
        key = (start, stop)
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
        hit = cached is not None
        rng_seconds = 0.0
        if not hit:
            stored = self.disk_store.load(start, stop) if self.disk_store is not None else None
            if stored is not None:
                randoms, latent = stored
            else:
                rng_started = time.perf_counter()
                count = stop-start
                # CUDA consumes randoms.T. Fortran storage makes that transpose
                # contiguous, avoiding consumer-thread repacking.
                randoms = np.empty((count, 625), dtype=np.uint32, order="F")
                latent = np.empty((count, 5), dtype=np.int32)
                for index, game, begin, end in self._segments(start, stop):
                    rng, draws = pack_streams(self.roots[index], [
                        canonical_seed(SEED_DOMAIN, self.seed, self.fingerprints[index], k)
                        for k in range(game, game+end-begin)])
                    randoms[begin:end] = rng
                    latent[begin:end] = draws
                randoms.setflags(write=False)
                latent.setflags(write=False)
                rng_seconds = time.perf_counter()-rng_started
                if self.disk_store is not None:
                    self.disk_store.save(start, stop, randoms, latent)
        reconstruction_started = time.perf_counter()
        if hit and self.cache_mode == "full":
            arrays = cached
        else:
            if hit:
                randoms, latent = cached
            arrays = self._reconstruct(start, stop, randoms, latent)
            cached = arrays if self.cache_mode == "full" else (randoms, latent)
        reconstruction_seconds = time.perf_counter()-reconstruction_started
        size = sum(array.nbytes for array in cached)
        with self.lock:
            if not hit:
                self.stats["misses"] += 1
                self.stats["prepared_games"] += stop-start
                if size <= self.cache_bytes:
                    while self.cache and self.stats["bytes"]+size > self.cache_bytes:
                        _, evicted = self.cache.popitem(last=False)
                        self.stats["bytes"] -= sum(array.nbytes for array in evicted)
                        self.stats["evictions"] += 1
                    self.cache[key] = cached
                    self.stats["bytes"] += size
                    self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.stats["bytes"])
            else:
                self.stats["hits"] += 1
                self.stats["reused_games"] += stop-start
            # Full preparation time includes reconstruction on both cache hits
            # and misses. RNG-only and reconstruction-only times are separate.
            self.stats["prepare_seconds"] += time.perf_counter()-began
            self.stats["rng_prepare_seconds"] += rng_seconds
            self.stats["reconstruct_seconds"] += reconstruction_seconds
        return arrays

    def snapshot(self):
        with self.lock:
            return dict(self.stats, limit_bytes=self.cache_bytes,
                        cached_batches=len(self.cache), total_games=self.total,
                        prototype_bytes=self.prototype_bytes,
                        cache_bytes_per_game=(sum(a.nbytes for a in self.packed[0])+2500
                                              if self.cache_mode == "full" else 2520),
                        cache_mode=self.cache_mode,
                        disk_cache=self.disk_store.snapshot() if self.disk_store is not None else {"enabled": False},
                        prepare_timing_scope="all_prepare_calls_including_hit_reconstruction",
                        schema="catalogue_immutable_inputs_v3")
