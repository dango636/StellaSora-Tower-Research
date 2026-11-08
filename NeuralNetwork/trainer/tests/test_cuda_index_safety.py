"""Reproduce negative gather indices and reject invalid host ABI inputs."""
from contextlib import nullcontext
import ctypes as C
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stella.cuda_runtime import CudaModule
from stella.fit import _settings, prepare_roots, ranking_pairs
from stella.fit_diagnostics import initialize_student_diagnostics
from stella.fit_torch import ResidentRoots, ranking_root_loss
from stella.features import KINDS, PHASES
from stella.gpu_game_tables import LAYOUT, pack_state
from stella.gpu_validation import (MAX_BATCH_GAMES, validate_batch_size,
                                   validate_rollout_inputs)
from stella.indexing import action_index, require_integer
from stella.preset_catalogue import catalogue_presets
from stella.random_streams import pack_streams
from stella.vendor import stella_sim as sim


def root(**extra):
    return dict(x=np.zeros((2, 3), np.float32), q=np.array([10., 0.], np.float32),
                action_returns=np.array([[10., 10., 10.], [0., 0., 0.]], np.float32),
                family_id="train", teacher_id="teacher", preset_group="all", split="train", **extra)


def test_negative_student_sentinel_is_not_a_real_last_action():
    settings = _settings({"ranking_weight": .25})
    unknown = root(student_action_index=-1)
    missing = root()
    pairs = ranking_pairs(unknown, settings)
    np.testing.assert_array_equal(pairs, ranking_pairs(missing, settings))
    assert np.all(pairs[:, :2] >= 0)
    prediction = torch.zeros((1, 2), requires_grad=True)
    loss = ranking_root_loss(prediction, torch.from_numpy(pairs[None]), settings)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
    # The old path generated [winner=0, rival=-1] and crashed torch.gather.
    with pytest.raises(RuntimeError, match="out of bounds"):
        prediction.gather(1, torch.tensor([[-1]]))


@pytest.mark.parametrize("bad", [-2, 2, 1.0, 1.5, True, "1", 2**40, np.bool_(False)])
@pytest.mark.parametrize("field", ["student_action_index", "teacher_action_index"])
def test_malformed_archived_actions_fail_before_training(bad, field):
    a = root(**{field: bad})
    b = dict(root(), family_id="validation", split="val")
    with pytest.raises(ValueError, match=field):
        prepare_roots([a, b], 3, _settings({}))


@pytest.mark.parametrize("unknown", [None, -1, np.int64(-1)])
def test_unknown_teacher_never_claims_last_action_evidence(unknown):
    settings = _settings({"refresh_student_actions": True, "transfer_diagnostics": True})
    item = root(teacher_action_index=unknown)
    predictions = lambda roots: (np.array([0., 1.], np.float32) for _ in roots)
    initialize_student_diagnostics([item], predictions, "frozen", settings)
    assert not item["_student_baseline"]["disagreement"]
    assert not item["_student_baseline"]["teacher_supported"]
    assert "teacher_student_disagreement" not in item


def test_resident_index_layout_is_checked_without_device_value_reads():
    data = ResidentRoots([root(student_action_index=-1)], "cpu", settings={"ranking_weight": .25})
    assert data.pair_indices.dtype == torch.int64
    assert torch.all(data.pair_indices >= 0)
    for bad in (torch.zeros(1), torch.zeros((1, 1), dtype=torch.int64),
                torch.empty(1, dtype=torch.int64, device="meta")):
        with pytest.raises(ValueError, match="Root indices"):
            data.training_batch(bad, torch.device("cpu"))
    result = data.training_batch(torch.tensor([0], dtype=torch.int64), torch.device("cpu"))
    assert result[0].shape == (1, 2, 3)
    with pytest.raises(ValueError, match="max_actions"):
        ResidentRoots([root()], "cpu", max_actions=1)


def packed_batch(n=2):
    state = sim.initial_from_pools(catalogue_presets()[0].pools)
    packed, belief = pack_state(state)
    randoms, latent = pack_streams(state, list(range(n)))
    states = np.repeat(packed[None], n, 0)
    states[:, LAYOUT["gE"]:LAYOUT["gE"]+5] = latent
    return states, np.repeat(belief[None], n, 0), np.asfortranarray(randoms), [0, n], None


def test_valid_packed_inputs_preserve_fortran_rng_and_immutable_owners():
    inputs = packed_batch()
    for arr in inputs[:3]:
        arr.setflags(write=False)
    result = validate_rollout_inputs(*inputs)
    assert result[2].flags.f_contiguous
    assert np.shares_memory(result[2], inputs[2])
    assert not result[2].flags.writeable
    for a, b in zip(result[:3], inputs[:3]):
        np.testing.assert_array_equal(a, b)


def test_catalogue_evaluation_may_omit_branch_grouping():
    states, beliefs, randoms, _, _ = packed_batch()
    normalized = validate_rollout_inputs(states, beliefs, randoms, None)
    assert normalized[3] == (0, len(states))


def test_int32_count_bound_supports_wide_addresses_without_wrapping():
    assert validate_batch_size(np.int64(MAX_BATCH_GAMES)) == MAX_BATCH_GAMES
    assert validate_batch_size(500_000) == 500_000  # > 2^31 feature elements are valid.
    for bad in (MAX_BATCH_GAMES+1, -1, True, 1.5, "128"):
        with pytest.raises(ValueError):
            validate_batch_size(bad)


@pytest.mark.parametrize("case", ["state_float", "rng_overflow", "rng_position", "posterior_nan", "posterior_mass",
                                   "pool_length", "negative_inventory", "event_id", "bound_role", "boundary", "first_select"])
def test_bad_packed_inputs_fail_on_host(case):
    states, beliefs, randoms, boundaries, first = packed_batch()
    if case == "state_float": states = states.astype(float)
    elif case == "rng_overflow":
        randoms = randoms.astype(np.uint64); randoms[0, 0] = 2**32
    elif case == "rng_position": randoms[0, 624] = 625
    elif case == "posterior_nan": beliefs[0, 0] = np.nan
    elif case == "posterior_mass": beliefs[0, :24] *= 2
    elif case == "pool_length": states[0, LAYOUT["pool_n"]] = 10
    elif case == "negative_inventory": states[0, LAYOUT["H"]] = -1
    elif case == "event_id": states[0, LAYOUT["event"]] = 999
    elif case == "bound_role":
        states[0, LAYOUT["source"]] = 1; states[0, LAYOUT["mode"]] = 2; states[0, LAYOUT["bound"]] = 0
    elif case == "boundary": boundaries = [0, 3]
    elif case == "first_select": first = np.zeros((2, 4), np.int32)  # SELECT index 0.
    with pytest.raises(ValueError):
        validate_rollout_inputs(states, beliefs, randoms, boundaries, first)


def test_packed_event_first_action_cannot_index_an_unmodeled_table_slot():
    states, beliefs, randoms, boundaries, _ = packed_batch(1)
    eid = next(e for e, data in sim.EVENT_DATA.items() if any(data[2]))
    valid = next(oid for oid in sim.EVENT_DATA[eid][2] if oid)
    states[:, LAYOUT["phase"]] = PHASES.index("EVENT")
    states[:, LAYOUT["event"]] = eid
    first = np.array([[KINDS.index("EVENT_OPTION"), valid, 0, eid]], np.int32)
    validate_rollout_inputs(states, beliefs, randoms, boundaries, first)
    first[0, 1] = eid * 100 + 99
    with pytest.raises(ValueError, match="event-table"):
        validate_rollout_inputs(states, beliefs, randoms, boundaries, first)
    first[0] = KINDS.index("GENERATE_PAGE"), 0, 0, 0
    with pytest.raises(ValueError, match="incompatible"):
        validate_rollout_inputs(states, beliefs, randoms, boundaries, first)


def test_potential_phase_requires_source_before_raw_kernel():
    states, beliefs, randoms, boundaries, _ = packed_batch(1)
    states[:, LAYOUT["phase"]] = PHASES.index("DRAW")
    states[:, LAYOUT["source"]] = 0
    with pytest.raises(ValueError, match="retain its source"):
        validate_rollout_inputs(states, beliefs, randoms, boundaries)


def module_stub(monkeypatch):
    module = CudaModule.__new__(CudaModule)
    module.device = torch.device("cuda:0")
    module.module = C.c_void_p(1)
    module.functions = {"test": C.c_void_p(2)}
    calls = []
    module.driver = SimpleNamespace(cuLaunchKernel=lambda *args: calls.append(args) or 0)
    module._cu = lambda code: None
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=13))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return module, calls


def test_cuda_bind_rejects_int32_truncation_before_driver(monkeypatch):
    module, calls = module_stub(monkeypatch)
    for scalar in (2**31, -(2**31)-1, 1.5, True):
        with pytest.raises(ValueError, match="kernel scalar"):
            module.bind("test", 1, (scalar,))
    for count in (-1, 2**31, 1.0):
        with pytest.raises(ValueError, match="kernel count"):
            module.bind("test", count, (1,))
    assert not calls
    launch = module.bind("test", 513, (np.int64(513),))
    launch()
    assert calls[-1][1] == 5  # Complete tail, no dropped game.


def test_cuda_bind_rejects_host_pointer_and_bad_grid(monkeypatch):
    module, calls = module_stub(monkeypatch)
    with pytest.raises(ValueError, match="contiguous on cuda:0"):
        module.bind("test", 1, (torch.zeros(1), 1))
    for block in (0, 1025, 1.5):
        with pytest.raises(ValueError, match="block size"):
            module.bind("test", 1, (1,), block=block)
    with pytest.raises(ValueError, match="grid blocks"):
        module.bind("test", 1, (1,), blocks=2**31)
    assert not calls


def test_raw_kernel_pointees_record_first_and_changed_stream_once(monkeypatch):
    module, calls = module_stub(monkeypatch)
    # A host tensor stands in only for pointer metadata; no driver/GPU executes.
    module.device = torch.device("cpu")
    tensor = torch.zeros(1)
    stream = SimpleNamespace(cuda_stream=13)
    recorded = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(torch.Tensor, "record_stream",
                        lambda self, used: recorded.append((id(self), used.cuda_stream)))
    invoke = module.bind("test", 1, (tensor, 1))
    assert not recorded
    invoke()
    invoke()
    stream = SimpleNamespace(cuda_stream=14)
    invoke()
    assert recorded == [(id(tensor), 13), (id(tensor), 14)]
    assert len(calls) == 3


def test_closed_module_rejects_new_and_already_bound_kernels(monkeypatch):
    module, calls = module_stub(monkeypatch)
    invoke = module.bind("test", 1, (1,))
    module.module = None
    with pytest.raises(RuntimeError, match="closed CUDA module"):
        module.bind("test", 1, (1,))
    with pytest.raises(RuntimeError, match="closed CUDA module"):
        invoke()
    assert not calls


def sampler_without_cuda():
    from stella.gpu_sampling import GpuFamilySampler
    sampler = GpuFamilySampler.__new__(GpuFamilySampler)
    sampler._action_columns = torch.arange(5)
    return sampler


def test_sampling_compact_padding_preserves_argmax_and_runner_gap():
    sampler = sampler_without_cuda()
    counts = torch.tensor([0, 1, 3, 2, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 0, 3, 5, 5], dtype=torch.int32)
    teacher = torch.tensor([1.1, 3.3, 2.2, 5., 5.])
    student = torch.tensor([5., 1., 4., 0., 2.])
    gaps = sampler._gaps(teacher, counts, offsets)
    torch.testing.assert_close(gaps, torch.tensor([0., 0., 1.1, 0., 0.]))
    choices, margin = sampler._policy_comparison(teacher, student, counts, offsets)
    torch.testing.assert_close(choices, torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1], [0, 0]]))
    torch.testing.assert_close(margin, torch.tensor([0., 0., 2.2, 0., 0.]))
    empty, gap = sampler._policy_comparison(torch.empty(0), torch.empty(0),
                                           torch.tensor([0, 1]), torch.tensor([0, 0]))
    assert empty.device.type == "cpu" and gap.device.type == "cpu"


@pytest.mark.parametrize("offset", [2, 100])
def test_sampling_never_clamps_or_wraps_invalid_live_prefix(offset):
    sampler = sampler_without_cuda()
    counts, offsets = torch.tensor([3], dtype=torch.int32), torch.tensor([offset], dtype=torch.int32)
    values = torch.tensor([1., 2., 3.])
    with pytest.raises(RuntimeError, match="out of bounds"):
        sampler._gaps(values, counts, offsets)
    with pytest.raises(RuntimeError, match="out of bounds"):
        sampler._policy_comparison(values, values, counts, offsets)


def test_sampling_rejects_different_policy_value_counts():
    sampler = sampler_without_cuda()
    with pytest.raises(ValueError):
        sampler._policy_comparison(torch.ones(3), torch.ones(2), torch.tensor([3]), torch.tensor([3]))
