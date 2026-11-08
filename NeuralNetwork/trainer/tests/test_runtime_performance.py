"""Exact-data and active-row contracts behind the hardware optimizations."""
from __future__ import annotations

from pathlib import Path
from random import Random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.catalogue_inputs import CatalogueInputs
from stella.environment import Episode, legal_actions, validate_state_preset
from stella.features import FEATURE_DIM, KINDS, PHASES
from stella.gpu_game_tables import LAYOUT, MAX_ACTIONS, STATE_SIZE, action_array, pack_state
from stella.gpu_games import GpuGames, rollout_memory_budget
from stella.hard_examples import classify_decision
from stella.hardware_profiles import PROFILES, apply_profile, memory_limit_summary
from stella.preset_catalogue import catalogue_presets
from stella.presets import Preset, canonical_seed
from stella.random_streams import pack_streams
from stella.sampling_descriptors import HARD_KIND_NAMES, hard_category_codes
from stella.vendor import stella_sim as sim


def _tasks():
    rows = []
    for i, preset_id in enumerate((0, 167, 2100, 4959)):
        preset = catalogue_presets()[preset_id]
        episode = Episode(preset, 20261106+i)
        chooser = Random(70+i)
        for _ in range(10+i*7):
            if episode.done:
                break
            episode.step(chooser.choice(episode.actions))
        assert not episode.done
        rows.append(dict(preset=preset.to_dict(), state=sim.to_jsonable(episode.state),
                         action_keys=[sim.action_key(a) for a in episode.actions],
                         rollouts_per_action=(1, 3, 5, 7)[i], rollout_seed_offset=i*3,
                         label_seed=20261106+i))
    return rows


def _original_prepare(tasks):
    """The old implementation as an independent branch-order/RNG oracle."""
    states, beliefs, randoms, first = [], [], [], []
    boundaries, roots, root_actions = [0], [], []
    for group, task in enumerate(tasks):
        preset = Preset.from_dict(task["preset"])
        root = sim.state_from_dict(task["state"])
        validate_state_preset(root, preset)
        actions = legal_actions(root)
        n, offset = task["rollouts_per_action"], task.get("rollout_seed_offset", 0)
        seeds = [canonical_seed("complete_teacher_branch", task["label_seed"], k)
                 for k in range(offset, offset+n)]
        rng, latent = pack_streams(root, seeds)
        state, belief = pack_state(root, group)
        count = len(actions)*n
        batch = np.repeat(state[None], count, axis=0)
        batch[:, LAYOUT["gE"]:LAYOUT["gE"]+5] = np.tile(latent, (len(actions), 1))
        states.append(batch)
        beliefs.append(np.repeat(belief[None], count, axis=0))
        randoms.append(np.tile(rng, (len(actions), 1)))
        first.append(np.repeat(np.asarray([action_array(a) for a in actions], dtype=np.int32), n, axis=0))
        boundaries.append(boundaries[-1]+count)
        roots.append(root); root_actions.append(actions)
    return (np.concatenate(states), np.concatenate(beliefs), np.concatenate(randoms), boundaries,
            np.concatenate(first), roots, root_actions)


def test_preallocated_labels_keep_every_branch_and_paired_rng_byte():
    tasks = _tasks()
    original = _original_prepare(tasks)
    optimized = GpuGames.prepare_label_roots(None, tasks)
    for index in (0, 1, 2, 4):
        np.testing.assert_array_equal(original[index], optimized[index])
        assert original[index].dtype == optimized[index].dtype
    assert original[3] == optimized[3]
    assert original[5:] == optimized[5:]
    assert optimized[2].flags.f_contiguous
    assert optimized[2].T.flags.c_contiguous
    assert np.shares_memory(optimized[2], np.ascontiguousarray(optimized[2].T))
    # Action branches share complete MT state and latent draws by rollout ID.
    for i, task in enumerate(tasks):
        start, end = optimized[3][i:i+2]
        n = task["rollouts_per_action"]
        stream = optimized[2][start:end].reshape(-1, n, 625)
        for action in stream[1:]:
            np.testing.assert_array_equal(action, stream[0])


@pytest.mark.parametrize("key,value", [("rollouts_per_action", 1.5),
    ("rollouts_per_action", True), ("rollouts_per_action", 0),
    ("rollout_seed_offset", -1), ("rollout_seed_offset", 1.5)])
def test_label_preparation_rejects_truncated_integer_fields(key, value):
    task = dict(_tasks()[0], **{key: value})
    with pytest.raises((ValueError, TypeError)):
        GpuGames.prepare_label_roots(None, [task])


def test_empty_label_inputs_and_rollout_shape_mismatch():
    arrays = GpuGames.prepare_label_roots(None, [])
    assert arrays[0].shape == (0, STATE_SIZE)
    assert arrays[2].shape == (0, 625)
    assert arrays[3] == [0]
    assert GpuGames.rollout(None, [], []) == []
    with pytest.raises(ValueError, match="equal length"):
        GpuGames.rollout(None, [], [1])
    root = sim.state_from_dict(_tasks()[0]["state"])
    with pytest.raises(ValueError, match="equal length"):
        GpuGames.rollout(None, [root], [1], [])
    with pytest.raises(ValueError, match="not legal"):
        GpuGames.rollout(None, [root], [1], [sim.Action("SELECT", index=-1)])


def _buffers(capacity):
    return dict(counts=torch.zeros(capacity, dtype=torch.int32),
                sizes=torch.zeros(capacity, dtype=torch.int32),
                offsets=torch.zeros(capacity, dtype=torch.int32),
                actions=torch.zeros((capacity, MAX_ACTIONS, 4), dtype=torch.int32),
                choices=torch.zeros(capacity, dtype=torch.int32),
                flags=torch.zeros(3, dtype=torch.int32),
                features=torch.zeros((capacity*MAX_ACTIONS, FEATURE_DIM)),
                results=torch.zeros((2, capacity), dtype=torch.int32))


def test_smaller_workspace_reuses_storage_but_excludes_stale_tail():
    capacity, active = 11, 3
    full = _buffers(capacity)
    for value in full.values():
        value.fill_(123)
    views = GpuGames._active_buffers(full, active)
    assert views["features"].shape == (active*MAX_ACTIONS, FEATURE_DIM)
    assert views["actions"].shape == (active, MAX_ACTIONS, 4)
    assert views["results"].shape == (2, active)
    assert views["results"][0].is_contiguous() and views["results"][1].is_contiguous()
    assert views["features"].is_contiguous() and views["actions"].is_contiguous()
    for name, view in views.items():
        assert view.untyped_storage().data_ptr() == full[name].untyped_storage().data_ptr()
    views["features"].fill_(7)
    views["results"][1].fill_(9)
    assert torch.all(full["features"][active*MAX_ACTIONS:] == 123)
    assert torch.all(full["results"][1, active:] == 123)
    assert torch.all(full["results"][0] == 123)


def test_memory_budget_credits_active_views_not_entire_capacity():
    full = _buffers(11)
    active = GpuGames._active_buffers(full, 3)
    reusable = sum(t.numel()*t.element_size() for t in active.values())
    budget = rollout_memory_budget(3, 0, 0, 0, reusable_bytes=reusable)
    assert budget["required_bytes"] > 0  # new states, posterior and MT inputs still required
    assert not budget["fits"]
    assert reusable < sum(t.numel()*t.element_size() for t in full.values())


def test_capacity_reuse_and_cross_stream_abort_contract(monkeypatch):
    game = GpuGames.__new__(GpuGames)
    game.workspace_reuse = True
    game._workspace_complete = True
    game._workspace_reuses = game._workspace_allocations = 0
    game._workspace = (11, _buffers(11), 7)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=7))
    views = game._buffers(3)
    assert len(views["counts"]) == 3 and game._workspace[0] == 11
    assert game._workspace_reuses == 1 and game._workspace_allocations == 0
    game._workspace_complete = False
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=9))
    with pytest.raises(RuntimeError, match="different stream"):
        game._release_incompatible_workspace(3)
    assert game._workspace is not None


def test_host_validation_timing_counts_guards_separately_from_completed_rollouts(monkeypatch):
    from stella import gpu_games as module
    game = GpuGames.__new__(GpuGames)
    game.stats = dict(roots=0, games=0, decisions=0, action_rows=0, seconds=0.0,
                      host_input_validation_calls=0, host_input_validation_seconds=0.0,
                      host_input_validation_failed_calls=0, host_input_validation_failed_seconds=0.0)
    game.information = {}
    game.graph_forward = game.profile = game._workspace = None
    game._workspace_allocations = game._workspace_reuses = 0
    arrays = CatalogueInputs(catalogue_presets()[:1], 57, 2).prepare(0, 2)
    reached_after_guard = []

    class StopBeforeCuda(RuntimeError):
        pass

    def stop_before_cuda(n):
        reached_after_guard.append(n)
        raise StopBeforeCuda

    monkeypatch.setattr(game, "_release_incompatible_workspace", stop_before_cuda)
    # Each nonempty run reads the clock at entry, before and after its guard.
    ticks = iter((10.0, 10.2, 10.5, 20.0, 20.3, 21.0,
                  30.0, 30.1, 30.8, 40.0))
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    for _ in range(2):
        with pytest.raises(StopBeforeCuda):
            game._run(*arrays, None)
    invalid = arrays[0][:, :-1]
    with pytest.raises(ValueError, match="shape"):
        game._run(invalid, arrays[1], arrays[2], None)
    game._run(*(array[:0] for array in arrays), None)

    snapshot = game.snapshot()
    assert reached_after_guard == [2, 2]  # rejected/empty inputs never reach CUDA
    assert snapshot["host_input_validation_calls"] == 2
    assert snapshot["host_input_validation_seconds"] == pytest.approx(1.0)
    assert snapshot["host_input_validation_failed_calls"] == 1
    assert snapshot["host_input_validation_failed_seconds"] == pytest.approx(0.7)
    assert snapshot["games"] == snapshot["decisions"] == snapshot["seconds"] == 0
    assert snapshot["gpu_profile"] == {"enabled": False}


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_eval_cache_is_immutable_model_independent_and_chunk_independent(mode):
    presets = [catalogue_presets()[i] for i in (0, 167, 4959)]
    inputs = CatalogueInputs(presets, 65537, 4, cache_bytes=2**20, cache_mode=mode)
    first = inputs.prepare(0, 8)
    repeated = inputs.prepare(0, 8)
    assert inputs.snapshot()["hits"] == 1
    assert inputs.snapshot()["misses"] == 1
    assert inputs.snapshot()["cache_mode"] == mode
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a, b)
        with pytest.raises(ValueError):
            b.setflags(write=True)
    alternate = CatalogueInputs(presets, 65537, 4, cache_bytes=0, cache_mode="compact")
    other = [alternate.prepare(0, 3), alternate.prepare(3, 8)]
    for index, array in enumerate(first):
        np.testing.assert_array_equal(array, np.concatenate([part[index] for part in other]))
    assert repeated[2].T.flags.c_contiguous
    if mode == "full":
        assert all(np.shares_memory(a, b) for a, b in zip(first, repeated))


def test_eval_full_cache_eviction_budget_includes_public_states():
    presets = catalogue_presets()[:2]
    a, b = pack_state(sim.initial_from_pools(presets[0].pools))
    bytes_per_game = a.nbytes+b.nbytes+625*4
    inputs = CatalogueInputs(presets, 3, 4, cache_bytes=bytes_per_game*4, cache_mode="full")
    inputs.prepare(0, 4)
    inputs.prepare(4, 8)
    snapshot = inputs.snapshot()
    assert snapshot["bytes"] == bytes_per_game*4
    assert snapshot["evictions"] == 1
    assert snapshot["cache_bytes_per_game"] == bytes_per_game
    inputs.prepare(0, 4)
    assert inputs.snapshot()["misses"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_readonly_cached_host_inputs_are_not_mutated_by_cuda_working_storage():
    from stella.gpu_games import device
    cache = CatalogueInputs(catalogue_presets()[:2], 3, 4, cache_bytes=2**20, cache_mode="full")
    arrays = cache.prepare(0, 8)
    snapshots = [array.copy() for array in arrays]
    for array in arrays:
        tensor = device(array.T if array.dtype == np.uint32 else array)
        tensor.zero_()
    torch.cuda.synchronize()
    for original, saved in zip(arrays, snapshots):
        np.testing.assert_array_equal(original, saved)
    for repeated, saved in zip(cache.prepare(0, 8), snapshots):
        np.testing.assert_array_equal(repeated, saved)


@pytest.mark.parametrize("target", ["cpu", "cuda"])
def test_device_hard_categories_equal_original_for_padding_prices_and_offers(target):
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(20261106)
    size = 3000
    phases = rng.integers(0, len(PHASES), size=size, dtype=np.int32)
    phases[:1500] = PHASES.index("SHOP")
    phases[1500:2500] = PHASES.index("PAGE")
    counts = rng.integers(0, MAX_ACTIONS+1, size=size, dtype=np.int32)
    choices = np.asarray([rng.integers(max(1, n)) for n in counts], dtype=np.int32)
    offer_counts = rng.integers(0, 4, size=size, dtype=np.int32)
    offers = rng.integers(0, 4, size=(size, 3, 7), dtype=np.int32)
    offers[:2000, :, 0] = 1
    offers[:2000, :, 3] = offers[:2000, :, 5] = 3
    actions = np.zeros((size, MAX_ACTIONS, 4), dtype=np.int32)
    action_kinds = [KINDS.index(k) for k in ("SELECT", "REROLL", "BUY", "ENHANCE", "RESTOCK", "LEAVE")]
    actions[:, :, 0] = rng.choice(action_kinds, size=(size, MAX_ACTIONS))
    actions[:, :, 1] = rng.integers(1, 4, size=(size, MAX_ACTIONS))
    actions[:, :, 2] = rng.choice([0, 160, 180], size=(size, MAX_ACTIONS))
    # Deterministic precedence checks, including both purchase types and padding.
    for i, chosen in enumerate(range(4)):
        phases[i], counts[i], choices[i] = PHASES.index("SHOP"), 4, chosen
        actions[i, :4, :3] = [(KINDS.index("BUY"), 0, 160), (KINDS.index("ENHANCE"), 0, 180),
                              (KINDS.index("RESTOCK"), 0, 0), (KINDS.index("LEAVE"), 0, 0)]
    expected = []
    for p, off, on, acts, n, c in zip(phases, offers, offer_counts, actions, counts, choices):
        name = (classify_decision(PHASES[p], off[:on],
                [(KINDS[a[0]], int(a[1]), int(a[2])) for a in acts[:n]], int(c)) if n > 1 else None)
        expected.append(HARD_KIND_NAMES.index(name))
    tensors = [torch.from_numpy(x).to(target) for x in (phases, offers, offer_counts, actions, counts, choices)]
    actual = hard_category_codes(*tensors).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual[:4].tolist() == [3, 5, 4, 4]


def test_hardware_profiles_keep_training_semantics_and_honor_user_overrides(monkeypatch):
    for key in set().union(*(set(value) for value in PROFILES.values())):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("STELLA_CPU_WORKERS", "2")
    monkeypatch.setenv("STELLA_HARDWARE_PROFILE", "local_5070")
    monkeypatch.setenv("STELLA_GPU_ROOT_BATCH", "17")
    result = apply_profile("colab_blackwell96")
    assert result["environment"]["STELLA_GPU_ROOT_BATCH"] == "17"
    assert result["environment"]["STELLA_GPU_CACHE_MB"] == "73728"
    assert result["environment"]["STELLA_EVAL_INPUT_CACHE_MODE"] == "full"
    assert not result["learning_config_changed"]
    assert not any(k.endswith(("_AMP", "_TF32", "_BATCH_ROOTS")) for k in result["environment"])
    summary = memory_limit_summary(device_free_bytes=73*2**30, device_total_bytes=80*2**30)
    assert summary["device_total_bytes"] == 80*2**30
    assert summary["device_free_bytes_at_snapshot"] == 73*2**30
    assert summary["rollout_allocator_cache_ceiling_bytes"] == 72*2**30
    assert summary["fit_resident_data_ceiling_bytes"] == 72*2**30
    assert summary["fit_free_memory_reserve_bytes"] == 8*2**30
    assert "separate lifetimes" in summary["budget_semantics"]
