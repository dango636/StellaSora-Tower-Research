"""Persistent seeds are reusable only with their complete byte identity."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.catalogue_input_store import CatalogueInputStore, OWNED_DIRECTORY
from stella.catalogue_inputs import CatalogueInputs
from stella.preset_catalogue import catalogue_presets


def inputs(path, *, seed=123, games=4, mode="full", budget=2**20, reserve=0):
    return CatalogueInputs(catalogue_presets()[:2], seed, games, cache_bytes=0, cache_mode=mode,
                           disk_cache_dir=path, disk_cache_bytes=budget, disk_reserve_bytes=reserve)


def equal_arrays(a, b):
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_new_session_reuses_exact_inputs_without_rng_generation(tmp_path, monkeypatch):
    first = inputs(tmp_path)
    expected = first.prepare(0, 8)
    assert first.snapshot()["disk_cache"]["writes"] == 1
    second = inputs(tmp_path, mode="compact")
    import stella.catalogue_inputs as module
    monkeypatch.setattr(module, "pack_streams", lambda *_: (_ for _ in ()).throw(AssertionError("RNG must be reused")))
    actual = second.prepare(0, 8)
    equal_arrays(expected, actual)
    stats = second.snapshot()
    assert stats["disk_cache"]["hits"] == 1
    assert stats["rng_prepare_seconds"] == 0
    assert actual[2].T.flags.c_contiguous
    assert not any(word in path.name.lower() for path in tmp_path.rglob("*")
                   for word in ("teacher", "score", "prediction", "model"))
    for array in actual:
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_byte_corruption_rebuilds_only_reproducible_owned_block(tmp_path):
    first = inputs(tmp_path)
    expected = first.prepare(0, 8)
    store = first.disk_store
    block = store._block(0, 8)
    unrelated = tmp_path/"teacher.npz"
    unrelated.write_bytes(b"private weights that must survive")
    extra = block/"unrelated.txt"
    extra.write_text("do not delete")
    path = block/"randoms.npy"
    damaged = bytearray(path.read_bytes())
    damaged[-7] ^= 0x55
    path.write_bytes(damaged)
    second = inputs(tmp_path)
    actual = second.prepare(0, 8)
    equal_arrays(expected, actual)
    stats = second.snapshot()["disk_cache"]
    assert stats["rebuilt_blocks"] == 1 and stats["writes"] == 1
    assert unrelated.read_bytes() == b"private weights that must survive"
    assert extra.read_text() == "do not delete"
    third = inputs(tmp_path)
    equal_arrays(expected, third.prepare(0, 8))
    assert third.snapshot()["disk_cache"]["hits"] == 1


@pytest.mark.parametrize("kind", ["interval", "shape", "dtype", "size", "json"])
def test_corrupt_metadata_is_rebuilt_with_identical_seeds(tmp_path, kind):
    first = inputs(tmp_path)
    expected = first.prepare(0, 8)
    path = first.disk_store._block(0, 8)/"manifest.json"
    data = json.loads(path.read_text())
    if kind == "interval":
        data["start"] = 1
    elif kind == "shape":
        data["arrays"]["randoms"]["shape"] = [8, 624]
    elif kind == "dtype":
        data["arrays"]["latent"]["dtype"] = "<f4"
    elif kind == "size":
        data["arrays"]["latent"]["size_bytes"] = 1
    else:
        path.write_text("{broken")
    if kind != "json":
        path.write_text(json.dumps(data))
    second = inputs(tmp_path)
    equal_arrays(expected, second.prepare(0, 8))
    assert second.snapshot()["disk_cache"]["rebuilt_blocks"] == 1


def test_seed_and_preset_order_use_distinct_namespaces(tmp_path):
    original = inputs(tmp_path)
    first = original.prepare(0, 8)
    changed = inputs(tmp_path, seed=124)
    second = changed.prepare(0, 8)
    assert changed.disk_store.identity_hash != original.disk_store.identity_hash
    assert not np.array_equal(first[2], second[2])
    reordered = CatalogueInputs(tuple(reversed(catalogue_presets()[:2])), 123, 4,
                               disk_cache_dir=tmp_path, disk_cache_bytes=2**20, disk_reserve_bytes=0)
    assert reordered.disk_store.identity_hash not in (original.disk_store.identity_hash, changed.disk_store.identity_hash)


def test_source_hash_changes_invalidate_input_namespace(tmp_path, monkeypatch):
    original = inputs(tmp_path)
    original.prepare(0, 8)
    import stella.catalogue_input_store as module
    monkeypatch.setattr(module, "source_identity", lambda: {"rules": "new-code"})
    second = inputs(tmp_path)
    assert second.disk_store.identity_hash != original.disk_store.identity_hash
    second.prepare(0, 8)
    assert second.snapshot()["disk_cache"]["misses"] == 1


def test_capacity_and_free_space_protect_disk_but_keep_training_inputs(tmp_path, monkeypatch):
    # Identity metadata fits, while the 20 KiB input payload does not.
    constrained = inputs(tmp_path/"limited", budget=5000)
    expected = constrained.prepare(0, 8)
    assert constrained.snapshot()["disk_cache"]["capacity_bypasses"] == 1
    assert not list((tmp_path/"limited").rglob("*.npy"))
    import stella.catalogue_input_store as module
    # Permit namespace creation, then run out of safe headroom before a block.
    low_disk = inputs(tmp_path/"low_disk", reserve=5000)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=6000))
    equal_arrays(expected, low_disk.prepare(0, 8))
    assert low_disk.snapshot()["disk_cache"]["free_space_bypasses"] == 1
    assert not list((tmp_path/"low_disk").rglob("*.npy"))


def test_budget_counts_other_identities_without_deleting_them(tmp_path):
    first = inputs(tmp_path)
    first.prepare(0, 8)
    existing = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    used = sum(map(len, existing.values()))
    second = inputs(tmp_path, seed=124, budget=used+5000)
    second.prepare(0, 8)
    assert second.snapshot()["disk_cache"]["capacity_bypasses"] == 1
    for path, payload in existing.items():
        assert path.read_bytes() == payload


def test_loaded_inputs_are_detached_from_later_file_edits(tmp_path):
    first = inputs(tmp_path)
    first.prepare(0, 8)
    second = inputs(tmp_path)
    arrays = second.prepare(0, 8)
    frozen = arrays[2].copy()
    path = second.disk_store._block(0, 8)/"randoms.npy"
    payload = bytearray(path.read_bytes())
    payload[-4:] = b"\x00\x00\x00\x00"
    path.write_bytes(payload)
    np.testing.assert_array_equal(arrays[2], frozen)


def test_partial_atomic_write_is_never_accepted_and_can_be_rebuilt(tmp_path, monkeypatch):
    first = inputs(tmp_path)
    original = first.disk_store._write_array
    calls = []
    def interrupted(path, array):
        calls.append(path.name)
        if len(calls) == 2:
            raise OSError("simulated interruption after the first durable input array")
        return original(path, array)
    monkeypatch.setattr(first.disk_store, "_write_array", interrupted)
    expected = first.prepare(0, 8)
    block = first.disk_store._block(0, 8)
    assert (block/"randoms.npy").exists()
    assert not (block/"manifest.json").exists()
    second = inputs(tmp_path)
    equal_arrays(expected, second.prepare(0, 8))
    assert second.snapshot()["disk_cache"]["rebuilt_blocks"] == 1
    assert (block/"manifest.json").exists()


def test_refuses_symlink_array_without_deleting_external_file(tmp_path):
    first = inputs(tmp_path)
    first.prepare(0, 8)
    path = first.disk_store._block(0, 8)/"randoms.npy"
    outside = tmp_path/"outside.npy"
    outside.write_bytes(path.read_bytes())
    original = outside.read_bytes()
    path.unlink()
    path.symlink_to(outside)
    second = inputs(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        second.prepare(0, 8)
    assert outside.read_bytes() == original


def test_drive_cache_disabled_and_tiny_budget_writes_nothing(tmp_path):
    store = CatalogueInputStore("/content/drive/MyDrive/cache", {"seed": 1}, reserve_bytes=0)
    assert not store.enabled
    assert "Drive" in store.disabled_reason
    tiny = CatalogueInputStore(tmp_path, {"seed": 1}, budget_bytes=1, reserve_bytes=0)
    assert not tiny.enabled
    assert not (tmp_path/OWNED_DIRECTORY).exists()


def test_cache_lock_io_failure_bypasses_without_stopping_valid_inputs(tmp_path, monkeypatch):
    producer = inputs(tmp_path)
    expected = producer.prepare(0, 8)
    reader = inputs(tmp_path)
    @contextmanager
    def broken_lock():
        raise OSError("transient local filesystem failure")
        yield
    monkeypatch.setattr(reader.disk_store, "_disk_lock", broken_lock)
    equal_arrays(expected, reader.prepare(0, 8))
    assert reader.snapshot()["disk_cache"]["io_bypasses"] == 2  # read and optional write


def test_initial_manifest_probe_io_failure_is_recoverable(tmp_path, monkeypatch):
    producer = inputs(tmp_path)
    producer.prepare(0, 8)
    original = Path.exists
    def broken_probe(path):
        if path.name == "manifest.json":
            raise OSError("transient metadata failure")
        return original(path)
    monkeypatch.setattr(Path, "exists", broken_probe)
    assert producer.disk_store.load(0, 8) is None
    assert producer.disk_store.snapshot()["io_bypasses"] == 1


def test_corrupt_cache_cleanup_io_failure_is_recoverable(tmp_path, monkeypatch):
    producer = inputs(tmp_path)
    producer.prepare(0, 8)
    path = producer.disk_store._block(0, 8)/"manifest.json"
    path.write_text("{broken")
    def failed_cleanup(_):
        raise OSError("temporary inability to remove a recomputable cache block")
    monkeypatch.setattr(producer.disk_store, "_remove_owned_files", failed_cleanup)
    assert producer.disk_store.load(0, 8) is None
    assert producer.disk_store.snapshot()["io_bypasses"] == 1


def test_actual_prefetch_worker_reuses_verified_disk_inputs_across_lifetimes(tmp_path, monkeypatch):
    """Exercise the real FIFO/prefetch/cache wiring with a CPU stand-in for games."""
    from stella.catalogue_service import _worker
    monkeypatch.setenv("STELLA_EVAL_DISK_CACHE", "1")
    monkeypatch.setenv("STELLA_EVAL_DISK_CACHE_MB", "2")
    monkeypatch.setenv("STELLA_EVAL_DISK_RESERVE_MB", "0")
    monkeypatch.setenv("STELLA_EVAL_INPUT_CACHE_MODE", "full")
    monkeypatch.delenv("STELLA_EVAL_DISK_CACHE_DIR", raising=False)
    monkeypatch.setattr(sys, "setswitchinterval", lambda _: None)
    class FakeGames:
        def __init__(self, _):
            self.information = {"backend": "test_cpu_game_standin"}
            # Nonzero stand-in counters verify that model selection resets all
            # per-model fields without discarding newer host timing counters.
            self.stats = dict(roots=0, games=0, decisions=0, action_rows=0, seconds=0.,
                              host_input_validation_calls=7, host_input_validation_seconds=1.5)
        def _run(self, states, beliefs, randoms, boundaries, control=None):
            assert boundaries is None
            assert all(not a.flags.writeable for a in (states, beliefs, randoms))
            assert randoms.T.flags.c_contiguous
            self.stats["games"] += len(states)
            self.stats["host_input_validation_calls"] += 1
            self.stats["host_input_validation_seconds"] += 0.25
            return (randoms[:, 0] % 133).astype(np.int32), np.ones(len(states), dtype=np.int32)
        def snapshot(self):
            return dict(self.information, **self.stats)
        def close(self):
            pass
    def session():
        parent, child = multiprocessing.Pipe()
        thread = threading.Thread(target=_worker,
            args=([p.to_dict() for p in catalogue_presets()[:2]], str(tmp_path), child,
                  123, 4, 2**20, True, FakeGames), daemon=True)
        thread.start()
        def receive():
            assert parent.poll(10), "prefetch worker failed to return a result"
            row = parent.recv()
            assert "error" not in row, row.get("error")
            return row
        try:
            assert receive()["ready"]
            parent.send({"kind": "model", "path": "fake-model"})
            assert receive()["information"]["persistent_input_cache"]["enabled"]
            parent.send({"kind": "batch", "start": 0, "stop": 8})
            result = receive()
            parent.send(None)
            return result
        finally:
            thread.join(10)
            parent.close()
            assert not thread.is_alive()
    first = session()
    second = session()
    np.testing.assert_array_equal(first["scores"], second["scores"])
    for result in (first, second):
        assert result["statistics"]["host_input_validation_calls"] == 1
        assert result["statistics"]["host_input_validation_seconds"] == 0.25
    assert first["statistics"]["input_cache"]["disk_cache"]["writes"] == 1
    assert second["statistics"]["input_cache"]["disk_cache"]["hits"] == 1
