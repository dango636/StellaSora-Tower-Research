"""GPU-resident complete games and counterfactual frozen-teacher rollouts.

Initial public states and Python MT19937 streams are packed once. Device
kernels own state transitions, action enumeration and feature encoding.
All continuation decisions use the resident FP32 GPU teacher.
"""
from __future__ import annotations

from pathlib import Path
import os
import time

import numpy as np
import torch

from .cuda_runtime import CudaModule
from .environment import legal_actions, score, validate_state_preset
from .features import FEATURE_DIM, TARGET_SCALE, encode_actions
from .fit_torch import TorchQ
from .gpu_forward import FixedGraphForward
from .gpu_profile import RolloutProfile
from .gpu_validation import require_integer, validate_batch_size, validate_rollout_inputs
from .gpu_game_tables import LAYOUT, MAX_ACTIONS, STATE_SIZE, action_array, coin_tables, header, pack_state
from .network import QNetwork
from .random_streams import pack_streams
from .presets import PRESET_VERSION, Preset, canonical_seed
from .vendor import stella_sim as sim


def device(array):
    host = np.ascontiguousarray(array)
    if not host.flags.writeable:
        # Cached host inputs are intentionally immutable. torch.tensor creates
        # independent CUDA storage without exposing a writable NumPy alias.
        return torch.tensor(host, device="cuda")
    return torch.from_numpy(host).to("cuda")


def rollout_memory_budget(n, free_bytes, reserved_bytes, allocated_bytes,
                          reusable_bytes=0, reserve_bytes=0):
    """Conservative per-batch budget; allocator slack can fund new storage.

    Budget the current input layout, all workspace rows, optional first actions
    and two compact Q vectors (old/new forwards may briefly overlap). The three
    status ints are added separately. Model/context allocations already reduce
    free_bytes; bounded neural-network activations use reserve_bytes.
    """
    available = int(free_bytes) + max(0, int(reserved_bytes)-int(allocated_bytes))
    input_per_branch = STATE_SIZE*4 + 72*8 + 625*4 + 4*4
    workspace_per_branch = (4 + MAX_ACTIONS*4 + 2)*4 + MAX_ACTIONS*FEATURE_DIM*4
    value_per_branch = 2*MAX_ACTIONS*4
    fixed = int(n)*(input_per_branch + workspace_per_branch + value_per_branch) + 12
    required = max(0, fixed-int(reusable_bytes)) + int(reserve_bytes)
    return dict(branches=int(n), fixed_bytes=fixed, reusable_bytes=int(reusable_bytes),
                bytes_per_branch=input_per_branch + workspace_per_branch + value_per_branch,
                reserve_bytes=int(reserve_bytes), available_bytes=available,
                required_bytes=required, fits=required <= available)


class GpuGames:
    def __init__(self, model_path):
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from .hardware_profiles import profile_name, validate_gpu_profile
        self.hardware_profile = profile_name()
        self.hardware_device = None
        if self.hardware_profile is not None:
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.hardware_device = validate_gpu_profile(self.hardware_profile,
                properties.total_memory, properties.name)
        self.source = model_path if isinstance(model_path, QNetwork) else QNetwork.load(model_path, frozen=True)
        self.model = TorchQ(self.source, "cuda").eval().requires_grad_(False)
        self.module = CudaModule(header() + Path(__file__).with_suffix(".cu").read_text(encoding="utf-8"))
        self.tables = tuple(device(a) for a in coin_tables())
        # Hidden activations dominate the rollout working set. Bounded FP32
        # forwards avoid allocator fragmentation and WDDM paging as root batches grow.
        self.inference_rows = int(os.environ.get("STELLA_GPU_INFERENCE_ROWS", "16384"))
        if self.inference_rows < 1:
            raise ValueError("STELLA_GPU_INFERENCE_ROWS must be positive")
        self.optimizations = os.environ.get("STELLA_GPU_OPTIMIZATIONS", "1") != "0"
        self.workspace_reuse = os.environ.get("STELLA_GPU_WORKSPACE_REUSE", "1") != "0"
        self.bound_launch = os.environ.get("STELLA_GPU_BOUND_LAUNCH", "1") != "0"
        self.single_sync = os.environ.get("STELLA_GPU_SINGLE_SYNC", "1") != "0"
        self._workspace = None
        self._workspace_complete = False
        self._workspace_allocations = 0
        self._workspace_reuses = 0
        self.profile = RolloutProfile() if os.environ.get("STELLA_GPU_PROFILE", "0") == "1" else None
        self.graph_forward = (FixedGraphForward(self.inference_rows)
                              if os.environ.get("STELLA_GPU_CUDA_GRAPH", "0") == "1" else None)
        self.information = {"backend": "torch_cuda", "simulation_backend": "cuda_games",
                            "rules_version": sim.MODEL_VERSION,
                            "device_name": torch.cuda.get_device_name(), "precision": "FP32",
                            "posterior_precision": "FP64", "amp": False, "tf32": False,
                            "rng": "CPython MT19937 integer draws", "kernel_sha256": self.module.identity,
                            "nvrtc_version": list(self.module.nvrtc_version),
                            "nvrtc_library": self.module.nvrtc_library,
                            "model_parameter_sha256": self.source.checksum(),
                            "decision_mode": "gpu", "version": 1}
        self.information.update(hardware_profile=self.hardware_profile,
                                hardware_device=self.hardware_device,
                                inference_rows=self.inference_rows,
                                rollout_optimizations=self.optimizations,
                                workspace_reuse=self.workspace_reuse,
                                single_status_sync=self.single_sync and self.optimizations,
                                bound_launch=self.bound_launch and self.optimizations,
                                cuda_graph_requested=self.graph_forward is not None,
                                gpu_profile_enabled=self.profile is not None,
                                host_input_validation_timing_scope=(
                                    "host perf_counter before device upload; successful and failed guards "
                                    "separate; contained in completed-rollout wall time, not GPU event time; "
                                    "successful guard calls are not completed-rollout counts"))
        self.stats = dict(roots=0, games=0, decisions=0, action_rows=0, seconds=0.0,
                          host_input_validation_calls=0, host_input_validation_seconds=0.0,
                          host_input_validation_failed_calls=0, host_input_validation_failed_seconds=0.0)

    def snapshot(self):
        result = dict(self.information, **self.stats)
        result["cuda_graph"] = (self.graph_forward.snapshot() if self.graph_forward is not None
                                else {"requested": False})
        result["gpu_profile"] = self.profile.snapshot() if self.profile is not None else {"enabled": False}
        result["rollout_workspace"] = dict(
            capacity_branches=self._workspace[0] if self._workspace is not None else 0,
            allocations=self._workspace_allocations, reuses=self._workspace_reuses,
            strategy="one high-water allocation; active rows only")
        return result

    @torch.inference_mode()
    def set_model(self, source):
        """Switch a frozen teacher without rebuilding kernels, tables or CUDA context."""
        compatible = (self.optimizations and self.model is not None
                      and self.model.input_dim == source.input_dim
                      and tuple(self.model.hidden) == tuple(source.hidden)
                      and set(self.model.weights) == set(source.params)
                      and all(tuple(self.model.weights[k].shape) == tuple(v.shape)
                              for k, v in source.params.items()))
        if compatible:
            # Same stream as subsequent inference; copy every FP32 parameter,
            # preserving the allocation and the original matrix shapes.
            for key, value in source.params.items():
                host = np.ascontiguousarray(value)
                if not host.flags.writeable:
                    # Frozen CPU teachers are intentionally read-only. Copy to
                    # writable staging rather than exposing a writable tensor
                    # alias or emitting torch's read-only NumPy warning.
                    host = host.copy()
                self.model.weights[key].copy_(torch.from_numpy(host))
        else:
            self.model = TorchQ(source, "cuda").eval().requires_grad_(False)
        self.source = source
        if self.graph_forward is not None:
            self.graph_forward.set_model(self.model)
        self.information = dict(self.information, model_parameter_sha256=source.checksum())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        torch.cuda.synchronize()
        if self.graph_forward is not None:
            self.graph_forward.close()
        self.model = None
        self._workspace = None
        self.tables = ()
        self.module.close()
        torch.cuda.empty_cache()

    def _forward(self, features):
        if self.graph_forward is not None:
            return self.graph_forward(self.model, features, TARGET_SCALE)
        return self.model(features, amp=False)*TARGET_SCALE

    @torch.inference_mode()
    def _values(self, features):
        rows = len(features)
        limit = self.inference_rows
        if not limit or rows <= limit:
            return self._forward(features)
        values = torch.empty(rows, device=features.device, dtype=torch.float32)
        for start in range(0, rows, limit):
            end = min(start+limit, rows)
            values[start:end] = self._forward(features[start:end])
        return values

    def _release_incompatible_workspace(self, n):
        # Drop stale feature storage before uploading the next batch's inputs,
        # avoiding an unnecessary old-feature + new-input allocation peak.
        # Slots are only ever used on their allocation stream. Even an aborted
        # call can release its slot without a host sync: allocator reuse on that
        # stream remains ordered after its queued kernels. Cross-stream reuse is
        # prohibited by _buffers; callers must still be sequential/non-reentrant.
        cached = self._workspace
        stream = torch.cuda.current_stream().cuda_stream
        if cached is not None and not self._workspace_complete and cached[2] != stream:
            raise RuntimeError("An interrupted CUDA rollout cannot resume on a different stream")
        if cached is not None and (not self.workspace_reuse or not self._workspace_complete
                or cached[0] < n or cached[2] != stream):
            self._workspace = None

    @staticmethod
    def _active_buffers(buffers, n):
        """Views expose exactly n branches; no kernel sees retained tail rows.

        Feature/action views remain C contiguous. Result rows are separate
        contiguous vectors passed independently to game_finish; the parent
        2-by-n view may have its original high-water row stride.
        """
        return {key: (value if key == "flags" else value[:n*MAX_ACTIONS]
                      if key == "features" else value[:, :n] if key == "results"
                      else value[:n]) for key, value in buffers.items()}

    def _buffers(self, n):
        # A single high-water slot avoids several large feature slabs. Smaller
        # batches reuse leading views without padding decisions or changing n.
        # Restrict reuse to the allocation stream. Custom CUDA launches are not
        # visible to the PyTorch allocator's stream tracking; an interrupted run
        # on another stream must not release storage while kernels still use it.
        # As before, a GpuGames object is for sequential, non-reentrant calls.
        n = validate_batch_size(n)
        stream = torch.cuda.current_stream().cuda_stream
        cached = self._workspace
        if (self.workspace_reuse and self._workspace_complete
                and cached is not None and cached[0] >= n and cached[2] == stream):
            self._workspace_reuses += 1
            return self._active_buffers(cached[1], n)
        self._workspace = None
        del cached
        buffers = dict(
            counts=torch.empty(n, dtype=torch.int32, device="cuda"),
            sizes=torch.empty(n, dtype=torch.int32, device="cuda"),
            offsets=torch.empty(n, dtype=torch.int32, device="cuda"),
            actions=torch.empty((n, MAX_ACTIONS, 4), dtype=torch.int32, device="cuda"),
            choices=torch.empty(n, dtype=torch.int32, device="cuda"),
            flags=torch.empty(3, dtype=torch.int32, device="cuda"),
            features=torch.empty((n*MAX_ACTIONS, FEATURE_DIM), dtype=torch.float32, device="cuda"),
            results=torch.empty((2, n), dtype=torch.int32, device="cuda"))
        if self.workspace_reuse:
            self._workspace = (n, buffers, stream)
        self._workspace_allocations += 1
        return buffers

    def _check_memory_budget(self, n):
        # Legacy callers without an explicit hardware profile retain their
        # original allocation behavior. Check once per run, never per decision.
        if self.hardware_profile is None:
            return
        reusable = 0
        cached = self._workspace
        if (self.workspace_reuse and self._workspace_complete and cached is not None
                and cached[0] >= n and cached[2] == torch.cuda.current_stream().cuda_stream):
            # Only active views replace this run's allocations. Counting the
            # entire retained capacity would under-budget newly uploaded inputs.
            reusable = sum(t.numel()*t.element_size()
                           for t in self._active_buffers(cached[1], n).values())
        free, total = torch.cuda.mem_get_info()
        from .hardware_profiles import memory_limit_summary
        self.information["memory_limits"] = memory_limit_summary(
            device_free_bytes=free, device_total_bytes=total)
        reserve = (2048 if self.hardware_profile == "colab_blackwell96" else 768)*2**20
        budget = rollout_memory_budget(n, free, torch.cuda.memory_reserved(),
                                       torch.cuda.memory_allocated(), reusable, reserve)
        self.information["last_rollout_memory_budget"] = budget
        if not budget["fits"]:
            raise RuntimeError(
                f"GPU memory budget exceeded for {self.hardware_profile}: {n} branches "
                f"need {budget['required_bytes']/2**30:.2f} GiB additional budget "
                f"including network workspace, available {budget['available_bytes']/2**30:.2f} GiB. "
                "Free other GPU workloads or reduce STELLA_GPU_ROOT_BATCH / "
                "STELLA_GPU_EVAL_BATCH / STELLA_GPU_SAMPLING_BATCH; keep the matching hardware profile. "
                "No rollout branches were dropped.")

    @torch.inference_mode()
    def _run(self, states, beliefs, randoms, boundaries, initial_actions=None, control=None, observe=None):
        started = time.perf_counter()
        n = len(states)
        if not n:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        validation_started = time.perf_counter()
        try:
            states, beliefs, randoms, boundaries, initial_actions = validate_rollout_inputs(
                states, beliefs, randoms, boundaries, initial_actions)
        except Exception:
            validation_elapsed = time.perf_counter()-validation_started
            self.stats["host_input_validation_failed_calls"] += 1
            self.stats["host_input_validation_failed_seconds"] += validation_elapsed
            raise
        else:
            validation_elapsed = time.perf_counter()-validation_started
            self.stats["host_input_validation_calls"] += 1
            self.stats["host_input_validation_seconds"] += validation_elapsed
        self._release_incompatible_workspace(n)
        self._check_memory_budget(n)
        states, beliefs, randoms = device(states), device(beliefs), device(randoms.T)
        buffers = self._buffers(n)
        self._workspace_complete = False
        counts, actions, choices, flags, features = (buffers[k] for k in
            ("counts", "actions", "choices", "flags", "features"))
        sizes, offsets = buffers["sizes"], buffers["offsets"]
        choices.zero_()
        launch = self.module.launch
        profile = self.profile
        fused_status = self.single_sync and self.optimizations
        # These pointers/shapes stay fixed for the complete run. Binding once
        # removes per-decision ctypes cells and tensor.data_ptr dispatches.
        if self.optimizations:
            def bind(name, count, args, **kwargs):
                if self.bound_launch:
                    return self.module.bind(name, count, args, **kwargs)
                return lambda: launch(name, count, args, **kwargs)
            enumerate_actions = bind("game_actions_compact", n,
                (states, actions, counts, sizes, flags, n))
            feature_name = "game_features_status" if fused_status else "game_features"
            feature_args = (states, beliefs, actions, counts, offsets, features)
            encode_features = bind(feature_name, n,
                (*feature_args, flags, n) if fused_status else (*feature_args, n), blocks=n)
            step_game = bind("game_step", n,
                (states, beliefs, randoms, actions, choices, *self.tables, n, 0))
        if profile is not None:
            profile.discard_failed_run()
            timing = profile.start("initial_state")
        if initial_actions is not None:
            first = device(np.asarray(initial_actions, dtype=np.int32))
            launch("game_step", n, (states, beliefs, randoms, first, choices, *self.tables, n, 1))
        else:
            launch("game_start", n, (states, beliefs, randoms, *self.tables, n))
        if profile is not None:
            profile.stop(timing)
        while True:
            if control:
                control()
            if profile is not None:
                timing = profile.start("actions_prefix")
            if self.optimizations:
                enumerate_actions()
                torch.cumsum(sizes, 0, dtype=torch.int32, out=offsets)
            else:
                launch("game_actions", n, (states, actions, counts, n))
                sizes = torch.where(counts > 1, counts, 0)
                offsets = torch.cumsum(sizes, 0, dtype=torch.int32)
            if profile is not None:
                profile.stop(timing)
                wait_started = time.perf_counter()
            if fused_status:
                # Write the same compact feature rows before the sole status
                # readback. No inference shape/padding or FP32 arithmetic changes.
                if profile is not None:
                    timing = profile.start("features_status")
                encode_features()
                if profile is not None:
                    profile.stop(timing)
                status = flags.cpu().numpy()
                rows = int(status[2])
            else:
                rows = int(offsets[-1].item())
            if profile is not None:
                profile.cpu_wait("features_status_copy" if fused_status else "prefix_item",
                                 time.perf_counter()-wait_started)
                timing = profile.start("features")
            if fused_status:
                if status[0]:
                    raise RuntimeError(f"CUDA game rule error {status[0]}")
                if not status[1]:
                    break
            else:
                if self.optimizations:
                    encode_features()
                else:
                    launch("game_features", n, (states, beliefs, actions, counts, offsets, features, n), blocks=n)
            x = features[:rows]
            if profile is not None:
                profile.stop(timing)
                timing = profile.start("forward")
            values = self._values(x) if rows else torch.empty(0, device="cuda")
            if profile is not None:
                profile.stop(timing)
                timing = profile.start("choose")
            if fused_status:
                launch("game_choose_only", n, (counts, offsets, values, choices, n))
            else:
                if not self.optimizations:
                    flags.zero_()
                launch("game_choose", n, (states, counts, offsets, values, choices, flags, n))
            if profile is not None:
                profile.stop(timing)
                wait_started = time.perf_counter()
            if not fused_status:
                status = flags.cpu().numpy()
                if profile is not None:
                    profile.cpu_wait("flags_copy", time.perf_counter()-wait_started)
                if status[0]:
                    raise RuntimeError(f"CUDA game rule error {status[0]}")
                if not status[1]:
                    break
            if observe:
                observe(states, beliefs, randoms, actions, counts, offsets, features, choices, values)
            self.stats["action_rows"] += rows
            if profile is not None:
                timing = profile.start("step")
            if self.optimizations:
                step_game()
            else:
                launch("game_step", n, (states, beliefs, randoms, actions, choices, *self.tables, n, 0))
            if profile is not None:
                profile.stop(timing)
        if profile is not None:
            timing = profile.start("finish")
        # Adjacent result rows permit one readback and reuse across batches.
        results = buffers["results"]
        scores, steps = results[0], results[1]
        launch("game_finish", n, (states, scores, steps, n))
        if profile is not None:
            profile.stop(timing)
            wait_started = time.perf_counter()
        scores, steps = results.cpu().numpy()
        if profile is not None:
            profile.cpu_wait("finish_copies", time.perf_counter()-wait_started)
            profile.collect_completed_run()
        self._workspace_complete = True
        self.stats["games"] += n
        self.stats["decisions"] += int(steps.sum())
        self.stats["seconds"] += time.perf_counter()-started
        return scores, steps

    def label_roots(self, tasks, control=None, observe=None, prepared=None):
        """Label every action, then spend explicit refinement quotas on close shop decisions."""
        rows = self._label_roots_once(tasks, control, observe, prepared=prepared)
        pending = []
        for i, (task, row) in enumerate(zip(tasks, rows)):
            n = row["returns"].shape[1]
            limit = int(task.get("refine_rollouts_per_action", n))
            if limit <= n:
                continue
            root = sim.state_from_dict(task["state"])
            actions = legal_actions(root)
            predictions = self.source.predict(row["x"])
            chosen = int(predictions.argmax())
            buys = [j for j, a in enumerate(actions)
                    if (a.kind == "BUY" and a.price == 160)
                    or (a.kind == "ENHANCE" and a.price == 180)]
            adaptive = task.get("refinement_mode") == "critical_conflicts_v2"
            if adaptive:
                buys = [j for j in range(len(actions)) if j != chosen]
                limit = min(limit, n + int(task["refine_branch_budget"])//len(actions))
            elif root.phase != "SHOP" or actions[chosen].kind not in ("RESTOCK", "LEAVE") or not buys:
                continue
            row["metadata"]["refinement"] = dict(
                requested_rollouts=limit, base_rollouts=n, actual_rollouts=n, teacher_action_index=chosen,
                purchase_action_indices=buys, seed_start=0, seed_stop=n, rounds=[])
            if adaptive:
                row["metadata"]["refinement"].update(
                    mode="critical_conflicts_v2", alternative_action_indices=buys,
                    branch_budget=int(task["refine_branch_budget"]), extra_branch_rollouts=0)
            pending.append((i, chosen, buys, limit))
        while pending:
            extra_tasks, selected = [], []
            for i, chosen, buys, limit in pending:
                row = rows[i]
                returns = row["returns"].astype(np.float64)
                n = returns.shape[1]
                differences = returns[buys] - returns[chosen]
                means = differences.mean(axis=1)
                errors = differences.std(axis=1, ddof=1) / np.sqrt(n)
                gap = float(tasks[i].get("refine_competition_gap", .3))
                sigma = float(tasks[i].get("refine_competition_sigma", 2.0))
                competitive = np.abs(means) <= np.maximum(gap, sigma * errors)
                if tasks[i].get("refinement_mode") == "critical_conflicts_v2":
                    # Identical paired returns and deterministic differences contain
                    # no unresolved simulation noise; more identical seeds add no value.
                    competitive &= errors > 0
                if n >= limit or not competitive.any():
                    continue
                amount = min(limit-n, int(tasks[i].get("refine_batch_rollouts", 512)))
                extra_tasks.append(dict(tasks[i], rollouts_per_action=amount,
                                        rollout_seed_offset=n))
                selected.append((i, chosen, buys, limit))
                row["metadata"]["refinement"]["rounds"].append(dict(
                    seed_start=n, seed_stop=n+amount,
                    purchase_minus_teacher=means.tolist(), paired_standard_error=errors.tolist()))
            if not extra_tasks:
                break
            extras = self._label_roots_once(extra_tasks, control, observe)
            for (i, chosen, buys, limit), extra in zip(selected, extras):
                row = rows[i]
                row["returns"] = np.concatenate((row["returns"], extra["returns"]), axis=1)
                returns = row["returns"]
                n = returns.shape[1]
                row["q"] = returns.mean(axis=1, dtype=np.float64).astype(np.float32)
                row["standard_error"] = returns.std(axis=1, ddof=1)/np.sqrt(n)
                row["metadata"].update(rollouts_per_action=n, complete_branches=int(returns.size),
                    branch_decisions=row["metadata"]["branch_decisions"]+extra["metadata"]["branch_decisions"])
                refinement = row["metadata"]["refinement"]
                difference = returns[buys].astype(np.float64)-returns[chosen].astype(np.float64)
                if tasks[i].get("refinement_mode") == "critical_conflicts_v2":
                    refinement["extra_branch_rollouts"] = (n-refinement["base_rollouts"])*len(returns)
                refinement.update(seed_stop=n, actual_rollouts=n,
                    purchase_minus_teacher=difference.mean(axis=1).tolist(),
                    paired_standard_error=(difference.std(axis=1, ddof=1)/np.sqrt(n)).tolist())
            pending = selected
        return rows

    def prepare_label_roots(self, tasks):
        """Prepare one contiguous allocation per input, in unchanged branch order.

        Two passes avoid retaining a complete list of per-root branch arrays
        while concatenating another full copy. MT storage is Fortran ordered:
        the CUDA kernel consumes ``randoms.T`` and therefore receives a C
        contiguous view without a second multi-gigabyte transpose copy.
        All RNG calls, action order and paired rollout columns are unchanged.
        This function is CPU-only and remains safe in the prefetch thread.
        """
        tasks = list(tasks)
        boundaries, roots, root_actions = [0], [], []
        for group, task in enumerate(tasks):
            preset = Preset.from_dict(task["preset"])
            root = sim.state_from_dict(task["state"])
            validate_state_preset(root, preset)
            actions = legal_actions(root)
            if [sim.action_key(a) for a in actions] != task["action_keys"]:
                raise ValueError("Root legal actions changed")
            n = require_integer(task["rollouts_per_action"], "rollouts_per_action", minimum=1)
            offset = require_integer(task.get("rollout_seed_offset", 0), "rollout_seed_offset", minimum=0)
            if not actions:
                raise ValueError("Cannot label a root with no legal actions")
            boundaries.append(boundaries[-1]+len(actions)*n)
            roots.append(root); root_actions.append(actions)
        total = boundaries[-1]
        if total:
            validate_batch_size(total)
        states = np.empty((total, STATE_SIZE), dtype=np.int32)
        beliefs = np.empty((total, 72), dtype=np.float64)
        randoms = np.empty((total, 625), dtype=np.uint32, order="F")
        first = np.empty((total, 4), dtype=np.int32)
        for group, (task, root, actions) in enumerate(zip(tasks, roots, root_actions)):
            n = int(task["rollouts_per_action"])
            offset = int(task.get("rollout_seed_offset", 0))
            seeds = [canonical_seed("complete_teacher_branch", task["label_seed"], k) for k in range(offset, offset+n)]
            rng, latent = pack_streams(root, seeds)
            state, belief = pack_state(root, group)
            start, end = boundaries[group:group+2]
            states[start:end] = state
            beliefs[start:end] = belief
            for j, action in enumerate(actions):
                begin, stop = start+j*n, start+(j+1)*n
                states[begin:stop, LAYOUT["gE"]:LAYOUT["gE"]+5] = latent
                randoms[begin:stop] = rng
                first[begin:stop] = action_array(action)
        return states, beliefs, randoms, boundaries, first, roots, root_actions

    def _label_roots_once(self, tasks, control=None, observe=None, prepared=None):
        if prepared is None:
            prepared = self.prepare_label_roots(tasks)
        states, beliefs, randoms, boundaries, first, roots, root_actions = prepared
        scores, steps = self._run(states, beliefs, randoms, boundaries, first, control, observe)
        results = []
        for i, (task, root, actions) in enumerate(zip(tasks, roots, root_actions)):
            start, end = boundaries[i:i+2]
            returns = (scores[start:end]-score(root)).astype(np.float32).reshape(len(actions), -1)
            metadata = dict(task, label_semantics="E[terminal ordinary score - root ordinary score | root action, frozen teacher]",
                            rules_version=sim.MODEL_VERSION,
                            preset_version=PRESET_VERSION, complete_branches=int(returns.size),
                            branch_decisions=int(steps[start:end].sum()), root_score=score(root),
                            schema="complete_root_q_v1", label_inference=self.information)
            results.append({"x": encode_actions(root, actions), "q": returns.mean(axis=1, dtype=np.float64).astype(np.float32),
                            "returns": returns, "standard_error": returns.std(axis=1, ddof=1)/np.sqrt(returns.shape[1]),
                            "metadata": metadata})
        self.stats["roots"] += sum(not task.get("rollout_seed_offset", 0) for task in tasks)
        return results

    def rollout(self, public_states, seeds, first_actions=None, control=None):
        """Finish independent public states, optionally applying a supplied first action."""
        public_states, seeds = list(public_states), list(seeds)
        if len(public_states) != len(seeds):
            raise ValueError("Public states and environment seeds must have equal length")
        if first_actions is not None:
            first_actions = list(first_actions)
            if len(first_actions) != len(public_states):
                raise ValueError("Initial actions and public states must have equal length")
            for root, action in zip(public_states, first_actions):
                if action not in legal_actions(root):
                    raise ValueError("Initial rollout action is not legal in its public state")
        if not public_states:
            return []
        states, beliefs, randoms = [], [], []
        for group, (root, seed) in enumerate(zip(public_states, seeds)):
            state, belief = pack_state(root, group)
            rng, latent = pack_streams(root, [seed])
            state[LAYOUT["gE"]:LAYOUT["gE"]+5] = latent[0]
            states.append(state); beliefs.append(belief); randoms.append(rng[0])
        scores, steps = self._run(np.asarray(states), np.asarray(beliefs), np.asarray(randoms),
                                 list(range(len(states)+1)),
                                 [action_array(a) for a in first_actions] if first_actions is not None else None,
                                 control=control)
        return [{"score": int(score), "steps": int(step)} for score, step in zip(scores, steps)]

    def play_games(self, presets, seeds, control=None):
        """Generate complete games from full 36-slot presets and independent seeds."""
        return self.rollout([sim.initial_from_pools(p.pools) for p in presets], seeds, control=control)


class GpuPairedGames:
    """Complete paired evaluations with both frozen policies resident on CUDA."""

    def __init__(self, model_paths: dict[str, str]):
        self.games = {name: GpuGames(model_paths[name]) for name in ("candidate", "reference")}
        self.information = {
            key: value for key, value in self.games["candidate"].information.items()
            if key != "model_parameter_sha256"
        }
        self.information.update(
            models={name: game.information["model_parameter_sha256"]
                    for name, game in self.games.items()},
            pairing="same preset and environment_seed",
        )
        self.stats = dict(pairs=0, batches=0)

    def evaluate_pairs(self, tasks, control=None):
        tasks = list(tasks)
        if not tasks:
            return []
        presets = [Preset.from_dict(task["preset"]) for task in tasks]
        seeds = [int(task["environment_seed"]) for task in tasks]
        candidate = self.games["candidate"].play_games(presets, seeds, control=control)
        reference = self.games["reference"].play_games(presets, seeds, control=control)
        rows = [
            {"pair_id": task["pair_id"], "preset": preset.to_dict(), "seed": seed,
             "rules_version": sim.MODEL_VERSION,
             "candidate_score": c["score"], "teacher_score": r["score"],
             "candidate_steps": c["steps"], "teacher_steps": r["steps"],
             "delta": c["score"]-r["score"], "complete": True,
             "evaluation_inference": self.information}
            for task, preset, seed, c, r in zip(tasks, presets, seeds, candidate, reference)
        ]
        self.stats["pairs"] += len(rows)
        self.stats["batches"] += 1
        return rows

    def snapshot(self):
        totals = {key: sum(game.stats[key] for game in self.games.values())
                  for key in ("games", "decisions", "action_rows", "seconds",
                              "host_input_validation_calls", "host_input_validation_seconds",
                              "host_input_validation_failed_calls", "host_input_validation_failed_seconds")}
        return dict(self.information, **self.stats, **totals)

    def close(self):
        for game in self.games.values():
            game.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
