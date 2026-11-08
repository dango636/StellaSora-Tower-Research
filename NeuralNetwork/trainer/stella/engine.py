"""Autonomous, resumable frozen-teacher policy iteration.

Run with ``python -m stella.engine --run-dir runs/current --config config.json``.
Zero (default) --rounds means keep iterating until the user stops the process.
"""
from __future__ import annotations

from .run_paths import run_path, portable_basename

# These MUST precede NumPy import, including in spawned worker processes.
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import math
import multiprocessing
from pathlib import Path
from random import Random
import shutil
import signal
import time
import traceback

from .config import VERSION, resolve, round_budget
from .hardware_profiles import apply_profile, runtime_workers
from .evaluation import evaluate_pair, evaluate_pairs, summarize_pairs, summarize_audit
from .backend import fit_candidate
from .io_utils import (atomic_json, read_json, append_jsonl, read_jsonl, upsert_history,
                       write_gzip_json, read_gzip_json, digest, file_hash,
                       StopRequested)
from .network import QNetwork
from .environment import sim
from .presets import (PRESET_VERSION, ROLE_TARGET_RANGE,
                      evaluation_presets, sample_preset, canonical_seed, Preset)
from .sampling import (FAMILY_SPLIT_VERSION, assign_family_splits,
                       sample_family, label_root, save_label, load_label)
from . import workers

ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cuda_task_batches(tasks, operation, batch_size):
    """Keep high-precision root batches within the previous game-buffer size."""
    batch, games = [], 0
    game_limit = batch_size*5*256
    for task in tasks:
        count = len(task["action_keys"])*task["rollouts_per_action"] if operation == "label" else 0
        if batch and (len(batch) >= batch_size or (operation == "label" and games+count > game_limit)):
            yield batch
            batch, games = [], 0
        batch.append(task)
        games += count
    if batch:
        yield batch


class Engine:
    def __init__(self, run_dir, config, max_rounds=0, max_hours=0):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.max_rounds = max_rounds
        if not math.isfinite(max_hours) or max_hours < 0:
            raise ValueError("max_hours must be finite and nonnegative")
        self.max_seconds = max_hours * 3600
        self.status_path = self.run_dir / "status.json"
        self.state_path = self.run_dir / "run_state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else None
        self.started = time.monotonic()
        self.base_elapsed = self.state.get("elapsed_seconds", 0.0) if self.state else 0.0
        self.last_publish = 0.0
        self.last_state_save = 0.0
        self.phase_started = self.started
        self.phase_initial_completed = 0
        self.round_started = self.started
        self.pool = None
        self._pool_inference = None
        self._serial_initialized = False
        self.execution_profile = apply_profile()
        self.worker_count = runtime_workers(config["workers"])
        if self.execution_profile["profile"]:
            atomic_json(self.run_dir / "execution_profile.json", self.execution_profile)
        self.status = {"status": "bootstrap", "phase": "bootstrap", "round": 0,
                       "workers": config["workers"], "version": VERSION,
                       "scope": "完整36普通槽位：每名角色18–54级，三人合计72–144级，不含核心",
                       "preset_version": PRESET_VERSION,
                       "rules_version": sim.MODEL_VERSION,
                       "legacy_comparison_available": False, "model_parameters": None,
                       "message": "正在加载模型和运行断点"}

    def elapsed(self):
        return self.base_elapsed + time.monotonic()-self.started

    def log(self, message, level="info"):
        append_jsonl(self.run_dir / "events.jsonl", {"time": utc_now(), "level": level, "message": str(message)},
                     self.config["retention"]["event_rows"])
        print(f"[{utc_now()}] {message}", flush=True)

    def publish(self, force=False, **updates):
        self.status.update(updates)
        now = time.monotonic()
        if not force and now-self.last_publish < .5:
            return
        self.last_publish = now
        self.status.update(elapsed_seconds=self.elapsed(),
                           session_elapsed_seconds=now-self.started,
                           session_limit_seconds=self.max_seconds or None,
                           session_remaining_seconds=max(0, self.max_seconds-(now-self.started)) if self.max_seconds else None,
                           round_elapsed_seconds=now-self.round_started,
                           phase_elapsed_seconds=now-self.phase_started, updated_at=utc_now())
        if self.state:
            self.status.update(teacher_id=self.state["teacher_id"], best_model_id=self.state["teacher_id"])
            self.status.update(rules_version=self.state.get("rules_version"),
                               rules_effective_round=self.state.get("rules_effective_round"))
            self.state["elapsed_seconds"] = self.elapsed()
            if force or now-self.last_state_save >= 5:
                atomic_json(self.state_path, self.state)
                self.last_state_save = now
        atomic_json(self.status_path, self.status)

    def stage(self, phase, total=0, completed=0, message=""):
        self.phase_started = time.monotonic()
        self.phase_initial_completed = completed
        self.publish(True, status=phase, phase=phase, phase_total=total, phase_completed=completed,
                     phase_progress=completed/total if total else 0.0,
                     eta_seconds=None, message=message)

    def progress(self, completed, total, **extras):
        new = completed-self.phase_initial_completed
        elapsed = time.monotonic()-self.phase_started
        eta = (total-completed)*elapsed/new if new > 0 else None
        self.publish(phase_total=total, phase_completed=completed,
                     phase_progress=completed/total if total else 1.0,
                     eta_seconds=eta, phase_units_per_second=new/max(elapsed, 1e-9) if new > 0 else None, **extras)

    def control(self):
        if (self.run_dir / "stop.flag").exists():
            self.stop_reason = "requested"
            raise StopRequested("已收到安全停止请求。")
        self.check_time_limit()
        was_paused = False
        while (self.run_dir / "pause.flag").exists():
            self.check_time_limit()
            was_paused = True
            self.publish(status="paused", message="已暂停；点击继续可恢复当前阶段。")
            if (self.run_dir / "stop.flag").exists():
                raise StopRequested("在暂停期间安全停止。")
            time.sleep(.2)
        if was_paused:
            self.publish(True, status=self.status["phase"], message="继续当前阶段。")

    def check_time_limit(self):
        if self.max_seconds and time.monotonic()-self.started >= self.max_seconds:
            self.stop_reason = "time_limit"
            raise StopRequested("本次运行时限已到，已完成的数据将保存；未完成评测不晋升。")

    def parallel(self, tasks, function, paths, on_result, inference=None):
        """Bound in-flight work; save each complete work unit immediately."""
        if not tasks:
            return
        if self._pool_inference is not inference:
            self.close_pool()
            self._pool_inference = inference
        count = self.worker_count
        if count == 1:
            if not self._serial_initialized:
                workers.initialize(str(self.run_dir), {}, inference)
                self._serial_initialized = True
            for task in tasks:
                self.control()
                on_result(task, workers.invoke(function, task, paths))
            return
        if self.pool is None:
            from .cpu_affinity import create_plan
            affinity = create_plan(count, self.run_dir)
            if affinity:
                self.log("CPU 工作进程按物理核心分配到逻辑 CPU：" +
                         ", ".join(str(slot["logical_cpu"]) for slot in affinity["slots"]))
            else:
                self.log("本机未应用 CPU 核心绑定；原因见 worker_affinity.json。")
            self.pool = ProcessPoolExecutor(max_workers=count,
                                   mp_context=multiprocessing.get_context("spawn"),
                                   initializer=workers.initialize,
                                   initargs=(str(self.run_dir), {}, inference, affinity))
        pool = self.pool
        iterator, pending = iter(tasks), {}
        try:
            for _ in range(count * 2):
                task = next(iterator, None)
                if task is None:
                    break
                pending[pool.submit(workers.invoke, function, task, paths)] = task
            while pending:
                self.control()
                ready, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                if not ready:
                    self.publish()
                    continue
                for future in ready:
                    task = pending.pop(future)
                    on_result(task, future.result())
                    nxt = next(iterator, None)
                    if nxt is not None:
                        pending[pool.submit(workers.invoke, function, nxt, paths)] = nxt
        except BaseException:
            # Workers observe the same flag at bounded decision intervals.
            (self.run_dir / "stop.flag").touch()
            (self.run_dir / "pause.flag").unlink(missing_ok=True)
            for future in pending:
                future.cancel()
            raise

    def close_pool(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self._serial_initialized = False
        self._pool_inference = None

    def cuda_batches(self, tasks, paths, on_result, directory, operation):
        """Run complete sampling families, root labels or paired games on CUDA."""
        from .gpu_label_service import CudaGameService
        from .label_inference import InferenceUnavailable, requested_backend
        options = {
            "sampling": ("STELLA_SAMPLING_BACKEND", "STELLA_GPU_SAMPLING_BATCH", 256,
                         "sampling_inference.json", "可达轨迹采样", "条轨迹"),
            "label": ("STELLA_LABEL_BACKEND", "STELLA_GPU_ROOT_BATCH", 32,
                      "label_inference.json", "完整根标签", "个根"),
            "evaluation": ("STELLA_EVAL_BACKEND", "STELLA_GPU_EVAL_BATCH", 512,
                           "inference.json", "独立配对整局", "组配对"),
        }
        backend_env, batch_env, default_batch, filename, title, unit = options[operation]
        requested = requested_backend(self.config["compute"], backend_env)
        simulation = os.environ.get("STELLA_SIM_BACKEND", "auto").lower()
        if simulation not in ("auto", "cuda", "python"):
            raise ValueError("STELLA_SIM_BACKEND must be auto, cuda or python")
        if requested == "numpy" or simulation == "python":
            return False
        self.close_pool()
        self.publish(True, message=f"正在加载 GPU {title}生成器")
        try:
            generator = CudaGameService(paths, self.run_dir, operation).start(self.control)
        except InferenceUnavailable as exc:
            if requested == "cuda" or simulation == "cuda":
                raise
            self.log(f"{title} GPU 不可用，使用 CPU：{exc}")
            return False
        started = utc_now()
        batch_size = max(1, int(os.environ.get(batch_env, str(default_batch))))
        simulation_field = {"label": "simulation_backend_actual",
                            "sampling": "sampling_simulation_backend",
                            "evaluation": "evaluation_simulation_backend"}[operation]
        batch_field = {"label": "label_batch_roots", "sampling": "sampling_batch_families",
                       "evaluation": "evaluation_batch_pairs"}[operation]
        self.publish(True, **{f"{operation}_backend_actual": "torch_cuda",
                              f"{operation}_device": generator.information["device_name"],
                              f"{operation}_decision_mode": "gpu", simulation_field: "cuda_games",
                              batch_field: batch_size, "workers": 1,
                              "message": f"GPU {title}；每批 {batch_size} {unit}"})
        self.log(f"{title}：GPU 完整对局生成；每批 {batch_size} {unit}；GPU 原生 FP32 决策")
        if generator.pipeline:
            self.log("标注流水线：CPU预备下一批，GPU计算与上一批标签保存重叠")
        atomic_json(run_path(directory, filename), {"started_at": started, **generator.information})
        try:
            batches = iter(cuda_task_batches(tasks, operation, batch_size))
            if generator.pipeline:
                # Two outstanding batches: one on CUDA, one CPU-prepared ahead.
                # Submit the next batch before persisting completed results.
                from collections import deque
                pending = deque()
                for _ in range(2):
                    batch = next(batches, None)
                    if batch is not None:
                        generator.submit(batch)
                        pending.append(batch)
                while pending:
                    self.control()
                    batch = pending.popleft()
                    results = generator.finish(self.control)
                    following = next(batches, None)
                    if following is not None:
                        generator.submit(following)
                        pending.append(following)
                    for task, row in zip(batch, results):
                        on_result(task, row)
            else:
                for batch in batches:
                    self.control()
                    results = generator.generate(batch, self.control)
                    for task, row in zip(batch, results):
                        on_result(task, row)
        finally:
            generator.close()
            record = {"started_at": started, "ended_at": utc_now(),
                      "batch_size": batch_size, **generator.snapshot()}
            atomic_json(run_path(directory, filename), record)
            append_jsonl(directory / (Path(filename).stem + "_sessions.jsonl"), record)
            self.publish(True, workers=self.worker_count)
        return True

    def label_parallel(self, tasks, paths, on_result, directory):
        """Change only the execution backend; frozen quotas/seeds stay intact."""
        if not tasks:
            return
        if self.cuda_batches(tasks, paths["teacher"], on_result, directory, "label"):
            return
        from .label_inference import InferenceService, InferenceUnavailable, requested_backend
        requested = requested_backend(self.config["compute"])
        service = None
        information = {"backend": "numpy", "device_name": "CPU / NumPy FP32",
                       "reason": "CPU label inference selected"}
        if requested != "numpy":
            self.close_pool()
            self.publish(True, message="正在启动共享 GPU 标签推理服务，已有完整标签保留")
            try:
                service = InferenceService(paths["teacher"], self.worker_count, requested).start(self.control)
                information = service.information
            except InferenceUnavailable as exc:
                if requested == "cuda":
                    raise
                information["reason"] = f"GPU label inference unavailable; CPU fallback: {exc}"
        started = utc_now()
        self.publish(True, label_backend_actual=information["backend"],
                     simulation_backend_actual="python", label_decision_mode="cpu_compatible",
                     label_device=information["device_name"], label_backend_reason=information.get("reason", ""),
                     message="完整根标签继续生成；" + ("CPU 模拟 + GPU 批量推理" if service else "CPU 推理"))
        self.log(f"标签推理：{information['backend']} / {information['device_name']}；{information.get('reason', '')}")
        atomic_json(directory / "label_inference.json", {"started_at": started, **information})
        try:
            self.parallel(tasks, label_root, paths, on_result, service.spec if service else None)
        finally:
            if service:
                try:
                    self.close_pool()
                finally:
                    service.close()
                information = service.snapshot()
            record = {"started_at": started, "ended_at": utc_now(), **information}
            atomic_json(directory / "label_inference.json", record)
            append_jsonl(directory / "label_inference_sessions.jsonl", record)

    def bootstrap(self):
        self.control()
        if self.state and self.state.get("rules_version") != sim.MODEL_VERSION:
            raise ValueError("运行断点属于旧状态空间；请先记录规则迁移并隔离旧规则标签。")
        if self.state and self.state.get("preset_version") != PRESET_VERSION:
            raise ValueError("已有任务的预设范围与当前版本不同。请在新任务目录从已入选检查点继续，保留旧任务数据。")
        config_file = self.run_dir / "resolved_config.json"
        if config_file.exists():
            stored = read_json(config_file)
            if stored != self.config:
                raise ValueError("已有任务的配置已冻结。请使用原配置恢复，或为修改后的配置指定新的 --run-dir。")
        else:
            atomic_json(config_file, self.config)
        if self.state:
            if self.state.get("version") != VERSION or self.state["config_hash"] != digest(self.config):
                raise ValueError("运行断点版本/配置不匹配")
            model = QNetwork.load(run_path(self.run_dir, self.state["teacher_checkpoint"]))
            self.log(f"恢复任务：下一轮 {self.state['next_round']}，当前教师 {self.state['teacher_id']}")
        else:
            model_path = self.run_dir / "checkpoints" / "M000000.npz"
            supplied = ROOT / "models" / "bootstrap.npz"
            model_config = self.config.get("model", {})
            if model_config.get("initialization") == "checkpoint":
                supplied = Path(model_config["initial_checkpoint"]).expanduser()
                if not supplied.is_absolute():
                    supplied = ROOT / supplied
                model = QNetwork.load(supplied)
                if tuple(model_config["hidden"]) != model.hidden:
                    raise ValueError("初始检查点结构与配置不匹配")
                model.metadata["warm_start_source"] = {
                    "path": str(supplied.resolve()), "sha256": file_hash(supplied),
                    "parameter_sha256": model.checksum(),
                    "model_id": model.metadata.get("model_id"),
                    "preset_version": model.metadata.get("preset_version", "full36_72_144_v1")}
                initialization = "warm start from accepted checkpoint: " + str(supplied)
            elif model_config.get("initialization", "packaged") == "packaged" and supplied.exists():
                model = QNetwork.load(supplied)
                if "hidden" in model_config and tuple(model_config["hidden"]) != model.hidden:
                    raise ValueError("随包模型结构不匹配；新结构请使用 model.initialization=fresh")
                initialization = model.metadata.get("initialization", "packaged independently evaluated bootstrap")
            else:
                model = QNetwork(hidden=tuple(model_config.get("hidden", (128, 96))), seed=self.config["seed"])
                initialization = "learnable public heuristic prior; no historical model weights"
            model.save(model_path, {"model_id": "M000000", "initialization": initialization,
                                   "created_at": utc_now(), "scope": [72, 144],
                                   "rules_version": sim.MODEL_VERSION,
                                   "role_scope": list(ROLE_TARGET_RANGE), "preset_version": PRESET_VERSION})
            self.state = {"version": VERSION, "config_hash": digest(self.config),
                          "preset_version": PRESET_VERSION,
                          "rules_version": sim.MODEL_VERSION,
                          "teacher_id": "M000000", "teacher_checkpoint": "checkpoints/M000000.npz",
                          "next_round": 1, "rejected_rounds": 0, "elapsed_seconds": 0.0,
                          "created_at": utc_now(), "initialization": initialization,
                          "audit_presets": [], "audit_model_id": None}
            atomic_json(self.state_path, self.state)
            self.log(f"初始化 {model.parameter_count:,} 参数神经网络。起点：{initialization}")
        self.publish(True, model_parameters=model.parameter_count,
                     model_architecture=[model.input_dim, *model.hidden, 1])
        if self.config["compute"]["auto_tune_workers"]:
            from .performance import calibrate_workers
            self.stage("tuning", 3, 0, "按本机完整标签吞吐选择CPU进程数（首次运行一次）")
            calibration = calibrate_workers(
                self.run_dir, run_path(self.run_dir, self.state["teacher_checkpoint"]), self.config["workers"],
                callback=lambda info: self.publish(True, **info), control=self.control)
            self.worker_count = calibration["selected_workers"]
            self.log(f"CPU实际标签吞吐校准：选用 {self.worker_count} 个常驻工作进程。")
        self.publish(True, workers=self.worker_count)
        panel_file = self.run_dir / "audit_panel.json"
        if not panel_file.exists():
            panel = evaluation_presets(canonical_seed("fixed_audit", self.config["seed"]),
                                       self.config["evaluation"]["audit_presets_per_bucket"], True)
            atomic_json(panel_file, [p.to_dict() for p in panel])
        if self.state["audit_model_id"] is None:
            audit = self.audit(run_path(self.run_dir, self.state["teacher_checkpoint"]), self.run_dir / "initial_audit")
            self.state.update(audit_presets=audit, audit_model_id=self.state["teacher_id"])
            initial = {"round": 0, "initial": True, "teacher_id": "M000000",
                       "rules_version": sim.MODEL_VERSION,
                       "candidate_id": "M000000", "best_model_id": "M000000", "promoted": False,
                       "audit_presets": audit, "audit_reused": False, "elapsed_seconds": self.elapsed(),
                       "initialization": self.state["initialization"]}
            upsert_history(self.run_dir / "history.jsonl", initial, self.config["retention"]["history_rows"])
            self.publish(True, audit_presets=audit, message="初始独立审计完成，即将冻结教师并开始迭代。")
        if self.state.get("rules_audit_pending"):
            directory = self.run_dir / "rules_revisions" / sim.MODEL_VERSION
            audit = self.audit(run_path(self.run_dir, self.state["teacher_checkpoint"]), directory / "audit_pairs")
            revision = {"round": self.state["rules_effective_round"]-1,
                        "rules_revision": True, "rules_version": sim.MODEL_VERSION,
                        "previous_rules_version": self.state["previous_rules_version"],
                        "teacher_id": self.state["teacher_id"], "candidate_id": None,
                        "best_model_id": self.state["teacher_id"], "promoted": False,
                        "primary_delta": None, "audit_presets": audit, "audit_reused": False,
                        "audit_model_id": self.state["teacher_id"],
                        "elapsed_seconds": self.elapsed(), "completed_at": utc_now()}
            atomic_json(directory / "baseline.json", revision)
            upsert_history(self.run_dir / "history.jsonl", revision, self.config["retention"]["history_rows"])
            self.state.update(audit_presets=audit, audit_model_id=self.state["teacher_id"],
                              rules_audit_pending=False)
            self.publish(True, audit_presets=audit, message="新规则下的教师基线完成，继续训练。")
            self.log(f"规则 {sim.MODEL_VERSION} 的 {self.state['teacher_id']} 基线完成；权重沿用。")

    def evaluation_tasks(self, presets, n, domain, round_number=0):
        tasks = []
        for p in presets:
            for i in range(n):
                tasks.append({"pair_id": f"p{p.fingerprint[:16]}_{i:05d}", "preset": p.to_dict(),
                              "rules_version": sim.MODEL_VERSION,
                              "environment_seed": canonical_seed(domain, self.config["seed"], round_number,
                                                                 p.fingerprint, i)})
        return tasks

    def evaluate(self, tasks, candidate_path, reference_path, directory, description):
        directory.mkdir(parents=True, exist_ok=True)
        pending, rows = [], []
        for task in tasks:
            path = directory / f"{task['pair_id']}.json"
            if path.exists():
                rows.append(read_json(path))
            else:
                pending.append(task)
        self.stage("evaluating", len(tasks), len(rows), description)
        if not pending:
            information_path = directory / "inference.json"
            if information_path.exists():
                information = read_json(information_path)
                self.publish(True, evaluation_backend_actual=information["backend"],
                             evaluation_device=information["device_name"],
                             evaluation_simulation_backend=information.get("simulation_backend", "python"),
                             evaluation_decision_mode=information.get("decision_mode", "cpu_compatible"))
            return rows
        def complete(task, row):
            atomic_json(directory / f"{task['pair_id']}.json", row)
            rows.append(row)
            self.progress(len(rows), len(tasks), evaluated_pairs=len(rows),
                          message=f"{description}：{len(rows)}/{len(tasks)} 配对整局")
        paths = {"candidate": str(candidate_path), "reference": str(reference_path)}
        if self.cuda_batches(pending, paths, complete, directory, "evaluation"):
            self.control()
            return rows
        service = None
        if pending:
            from .label_inference import InferenceService, InferenceUnavailable, requested_backend
            requested = requested_backend(self.config["compute"], "STELLA_EVAL_BACKEND")
            if requested != "numpy":
                self.close_pool()
                try:
                    service = InferenceService(paths, self.worker_count, requested).start(self.control)
                except InferenceUnavailable as exc:
                    if requested == "cuda":
                        raise
                    self.log(f"GPU 评测推理不可用，使用 CPU：{exc}")
        self.publish(True, evaluation_backend_actual="torch_cuda" if service else "numpy",
                     evaluation_simulation_backend="python", evaluation_decision_mode="cpu_compatible",
                     evaluation_device=service.information["device_name"] if service else "CPU / NumPy FP32")
        if service:
            self.log("独立评测：CPU 并行模拟 + 双模型 GPU 批量推理，每批 32 组配对。")
            def completed_batch(batch, results):
                for task, row in zip(batch, results):
                    complete(task, row)
            try:
                batches = [pending[i:i+32] for i in range(0, len(pending), 32)]
                self.parallel(batches, evaluate_pairs, paths, completed_batch, service.spec)
            finally:
                self.close_pool()
                service.close()
                atomic_json(directory / "inference.json", service.snapshot())
        else:
            self.parallel(pending, evaluate_pair, paths, complete)
        self.control()
        return rows

    def audit(self, model_path, directory):
        presets = [Preset.from_dict(p) for p in read_json(self.run_dir / "audit_panel.json")]
        tasks = self.evaluation_tasks(presets, self.config["evaluation"]["audit_games_per_preset"],
                                     "fixed_audit_games")
        rows = self.evaluate(tasks, model_path, self.run_dir / "checkpoints/M000000.npz",
                             directory, "固定面板审计（仅趋势展示，不参与晋升）")
        # Assert the quota, full vectors and exact paired seeds before charting.
        summarize_pairs(rows, tasks, bootstrap_replicates=20, seed=self.config["seed"])
        result = summarize_audit(rows)
        atomic_json(directory / "summary.json", result)
        return result

    def apply_next_round_config(self, number, directory):
        path = self.run_dir / "pending_config.json"
        if not path.exists() or (directory / "manifest.json").exists():
            return
        change = read_json(path)
        if number < change["effective_round"]:
            return
        if change["previous_config_hash"] != digest(self.config):
            raise ValueError("Pending training revision does not match the active config")
        updated = resolve(change["config"])
        record = dict(change, applied_round=number, applied_at=utc_now(),
                      previous_config=self.config, resolved_config=updated, config_hash=digest(updated))
        atomic_json(self.run_dir / "config_revisions" / (change["revision"]+".json"), record)
        self.close_pool()
        atomic_json(self.run_dir / "resolved_config.json", updated)
        for target in change["config_paths"]:
            atomic_json(Path(target), change["config"])
        self.config = updated
        self.worker_count = runtime_workers(updated["workers"])
        self.state.update(config_hash=digest(updated), config_revision=change["revision"],
                          training_revision_effective_round=number)
        atomic_json(self.state_path, self.state)
        path.unlink()
        self.log(f"第 {number} 轮起启用 {change['revision']}：" +
                 change.get("description", "按本轮冻结配置继续训练"))

    def mine_hard_profile(self, number, directory):
        profile_path = directory / "hard_mining.json"
        if profile_path.exists():
            return read_json(profile_path)["category_weights"]
        previous = self.run_dir / "rounds" / f"round_{number-1:06d}"
        if not (previous / "summary.json").exists():
            atomic_json(profile_path, dict(rules_version=sim.MODEL_VERSION, roots=0,
                        category_weights={}, reason="no completed previous round under the current rules"))
            return {}
        previous_manifest = read_json(previous / "manifest.json")
        if previous_manifest.get("rules_version") != sim.MODEL_VERSION:
            return {}
        tasks = []
        for task in previous_manifest["sampling_tasks"]:
            if task["split"] != "train":
                continue
            family = read_gzip_json(previous / "sampling" / (task["family_id"]+".json.gz"))
            for root in family["roots"]:
                if root["phase"] in ("PAGE", "SHOP"):
                    tasks.append({"path": str(previous / "labels" / (root["root_id"]+".npz"))})
        Random(canonical_seed("hard_mining", self.config["seed"], number)).shuffle(tasks)
        tasks = tasks[:self.config["sampling"]["hard_mining_roots"]]
        if not tasks:
            return {}
        from .gpu_label_service import CudaGameService
        from .hard_examples import SPECIFIC_KINDS
        self.stage("mining", len(tasks), 0, "从上一轮训练标签挖掘误选；开发轨迹不参与")
        miner = CudaGameService(run_path(self.run_dir, previous_manifest["teacher_checkpoint"]),
                               self.run_dir, "mining").start(self.control)
        rows = []
        try:
            for batch in cuda_task_batches(tasks, "mining", 256):
                rows.extend(miner.generate(batch, self.control))
                self.progress(len(rows), len(tasks))
        finally:
            miner.close()
        categories, weights = {}, {}
        for category in SPECIFIC_KINDS:
            local = [row for row in rows if row["category"] == category]
            signal = sum(row["priority_signal"] for row in local)/max(1, len(local))
            weights[category] = min(4.0, 1.0+signal/.1)
            categories[category] = dict(roots=len(local), mistakes=sum(r["priority_signal"] > 0 for r in local),
                                        mean_effective_regret=signal)
        mistakes = sorted((r for r in rows if r["priority_signal"] > 0),
                          key=lambda row: row["priority_signal"], reverse=True)[:256]
        # Retained public examples guide category sampling; old teacher labels
        # never become a new teacher's numerical training targets.
        write_gzip_json(directory / "hard_example_bank.json.gz", mistakes)
        atomic_json(profile_path, dict(source_round=number-1, source_split="train",
                    rules_version=sim.MODEL_VERSION,
                    source_teacher_id=previous_manifest["teacher_id"], roots=len(rows), bank_roots=len(mistakes),
                    categories=categories, category_weights=weights, inference=miner.snapshot(),
                    use="training-category priorities; fresh roots relabeled under the new frozen teacher",
                    development_sampling="equal public categories; no mined weights"))
        self.log(f"难例挖掘：{len(rows)} 个训练状态，保留 {len(mistakes)} 个误选案例供类型采样")
        return weights

    def prepare_manifest(self, number, directory):
        path = directory / "manifest.json"
        if path.exists():
            manifest = read_json(path)
            if manifest.get("preset_version") != PRESET_VERSION:
                raise ValueError("已有轮次使用不同的预设范围，不能混入当前任务")
            if manifest.get("rules_version") != sim.MODEL_VERSION:
                raise ValueError("已有轮次属于不同的状态空间规则")
            if (manifest["teacher_id"] != self.state["teacher_id"] or
                    manifest["teacher_sha256"] != file_hash(run_path(self.run_dir, self.state["teacher_checkpoint"]))):
                raise ValueError("A partially completed round no longer matches its frozen teacher")
            return manifest
        budget = round_budget(self.config, number, self.state["rejected_rounds"])
        hard_sampling = self.config["sampling"].get("hard_sampling", False)
        hard_weights = self.mine_hard_profile(number, directory) if hard_sampling else {}
        # The promotion panel and all quotas are frozen BEFORE any labels/train.
        promotion = []
        occupied = {p["fingerprint"] for p in read_json(self.run_dir / "audit_panel.json")}
        # Development games must also be independent of the replay being fitted.
        for prior in range(max(1, number-self.config["sampling"]["replay_rounds"]+1), number):
            prior_path = self.run_dir / "rounds" / f"round_{prior:06d}" / "manifest.json"
            if prior_path.exists():
                occupied.update(t["preset"]["fingerprint"]
                                for t in read_json(prior_path)["sampling_tasks"])
        for group in range(3):
            rng = Random(canonical_seed("fresh_promotion_presets", self.config["seed"], number, group))
            for j in range(budget["presets_per_bucket"]):
                while True:
                    p = sample_preset(rng, group, name=f"R{number}-P{group+1}-{j+1}")
                    if p.fingerprint not in occupied:
                        break
                occupied.add(p.fingerprint)
                promotion.append(p)
        development = []
        evaluation_config = self.config["evaluation"]
        if self.config["training"].get("checkpoint_candidates", 1) > 1:
            for group in range(3):
                rng = Random(canonical_seed("epoch_development_presets", self.config["seed"], number, group))
                for j in range(evaluation_config["development_presets_per_bucket"]):
                    while True:
                        p = sample_preset(rng, group, name=f"R{number}-D{group+1}-{j+1}")
                        if p.fingerprint not in occupied:
                            break
                    occupied.add(p.fingerprint)
                    development.append(p)
        tasks = []
        for i in range(budget["families"]):
            group = i % 3
            rng = Random(canonical_seed("training_presets", self.config["seed"], number, i))
            while True:
                p = sample_preset(rng, group, name=f"R{number}-F{i:05d}")
                if p.fingerprint not in occupied:
                    break
            occupied.add(p.fingerprint)
            mode = (i // 3) % 10
            have_previous = bool(self.state.get("last_candidate_checkpoint"))
            use_previous = mode == 9 and have_previous
            tasks.append({"family_id": f"r{number:06d}_f{i:05d}", "ordinal": i,
                          "rules_version": sim.MODEL_VERSION,
                          "preset": p.to_dict(),
                          "balanced_phase_sampling": self.config["sampling"].get("balanced_phase_sampling", False),
                          "epsilon": 0.0 if mode < 6 or use_previous else (.15 if mode < 8 else .5),
                          "behavior_source": "last_candidate" if use_previous else "teacher",
                          "roots_per_family": budget["roots_per_family"],
                          "teacher_id": self.state["teacher_id"],
                          "environment_seed": canonical_seed("source_environment", self.config["seed"], number, i),
                          "exploration_seed": canonical_seed("source_exploration", self.config["seed"], number, i),
                          "selection_seed": canonical_seed("public_root_selection", self.config["seed"], number, i),
                          "label_seed": canonical_seed("training_full_returns", self.config["seed"], number, i)})
        assign_family_splits(tasks, self.config["seed"], number)
        for task in tasks:
            task["rollouts_per_action"] = (
                budget.get("validation_rollouts_per_action", budget["rollouts_per_action"])
                if task["split"] == "val" else budget["rollouts_per_action"])
            if hard_sampling:
                task.update(hard_sampling=True,
                            hard_rollouts_per_action=budget["hard_validation_rollouts_per_action"]
                                if task["split"] == "val" else budget["hard_rollouts_per_action"])
                if task["split"] == "train":
                    task["hard_category_weights"] = hard_weights
        manifest = {"version": VERSION, "round": number, "budget": budget,
                    "rules_version": sim.MODEL_VERSION,
                    "preset_version": PRESET_VERSION,
                    "teacher_id": self.state["teacher_id"],
                    "teacher_checkpoint": self.state["teacher_checkpoint"],
                    "teacher_sha256": file_hash(run_path(self.run_dir, self.state["teacher_checkpoint"])),
                    "config_hash": digest(self.config), "created_at": utc_now(),
                    "behavior_checkpoint": self.state.get("last_candidate_checkpoint"),
                    "behavior_sha256": file_hash(run_path(self.run_dir, self.state["last_candidate_checkpoint"]))
                        if self.state.get("last_candidate_checkpoint") else None,
                    "sampling_tasks": tasks, "family_split_version": FAMILY_SPLIT_VERSION,
                    "promotion_tasks": self.evaluation_tasks(promotion, budget["games_per_preset"],
                                                              "independent_promotion_games", number),
                    "selection_condition": "estimated three-stratum mean delta > 0 after full quota",
                    "ci_is_gate": False, "action_gain_lock": False}
        if development:
            manifest["development_tasks"] = self.evaluation_tasks(
                development, evaluation_config["development_games_per_preset"],
                "epoch_development_games", number)
            manifest["epoch_selection_rule"] = (
                "highest three-stratum mean on the complete independent development quota; "
                "ties retain the label-metric winner; promotion uses separate untouched games")
        if hard_sampling:
            manifest.update(training_config=dict(self.config["training"]),
                            compute_config=dict(self.config["compute"]), hard_sampling=True,
                            hard_mining="hard_mining.json", training_revision=self.state.get("config_revision"))
        atomic_json(path, manifest)
        return manifest

    def refinement_tasks(self, roots, manifest, directory):
        """Freeze a bounded, stratified set before observing any rollout returns."""
        budget = manifest["budget"]
        limit = budget.get("refinement_roots", 0)
        if not limit:
            return
        if budget.get("refinement_mode") == "critical_conflicts_v2":
            return self.critical_refinement_tasks(roots, manifest, directory)
        path = directory / "refinement_plan.json"
        if path.exists():
            selected = set(read_json(path)["root_ids"])
        else:
            eligible = [r for r in roots if r.get("hard_kind") in
                        ("shop_buy160_skip", "shop_enhance180_skip")]
            eligible.sort(key=lambda r: canonical_seed("refinement_quota", manifest.get("workflow", {}).get("base_seed", self.config["seed"]),
                                                       manifest["round"], r["root_id"]))
            chosen = []
            for split, fraction in (("train", .75), ("val", .25)):
                quota = int(limit*fraction)//3
                for group in range(3):
                    chosen.extend(r["root_id"] for r in
                                  [r for r in eligible if r["split"] == split and r["preset_group"] == group][:quota])
            selected = set(chosen)
            selected.update(r["root_id"] for r in
                            [r for r in eligible if r["root_id"] not in selected][:limit-len(selected)])
            atomic_json(path, {"root_ids": sorted(selected), "eligible_roots": len(eligible),
                              "maximum_roots": limit, "selected_before_labels": True,
                              "train_limit": budget["refinement_rollouts_per_action"],
                              "validation_limit": budget["refinement_validation_rollouts_per_action"]})
        for root in roots:
            if root["root_id"] in selected:
                root["refine_rollouts_per_action"] = budget[
                    "refinement_validation_rollouts_per_action" if root["split"] == "val"
                    else "refinement_rollouts_per_action"]
        self.publish(True, refinement_candidate_roots=len(selected))

    def critical_refinement_tasks(self, roots, manifest, directory):
        """Freeze broader candidates under the legacy plan's exact branch ceiling."""
        budget = manifest["budget"]
        limit = budget["refinement_roots"]
        path = directory / "refinement_plan.json"
        if path.exists():
            plan = read_json(path)
        else:
            def seed(root):
                return canonical_seed("refinement_quota", manifest.get("workflow", {}).get("base_seed", self.config["seed"]),
                                      manifest["round"], root["root_id"])
            def stratified(items, diversified=False):
                items = sorted(items, key=seed)
                if diversified:
                    # Interleave public categories before applying the original quotas.
                    groups = {}
                    for root in items:
                        category = root.get("hard_kind") or root["phase"].lower() + "_general"
                        groups.setdefault(category, []).append(root)
                    items = []
                    categories = sorted(groups)
                    while any(groups.values()):
                        for category in categories:
                            if groups[category]:
                                items.append(groups[category].pop(0))
                chosen = []
                for split, fraction in (("train", .75), ("val", .25)):
                    quota = int(limit*fraction)//3
                    for group in range(3):
                        chosen.extend([r for r in items if r["split"] == split
                                       and r["preset_group"] == group][:quota])
                ids = {r["root_id"] for r in chosen}
                chosen.extend([r for r in items if r["root_id"] not in ids][:limit-len(chosen)])
                return chosen
            def maximum(root):
                return budget["refinement_validation_rollouts_per_action"
                              if root["split"] == "val" else "refinement_rollouts_per_action"]
            legacy = stratified([r for r in roots if r.get("hard_kind") in
                                 ("shop_buy160_skip", "shop_enhance180_skip")])
            ceilings = {split: sum(max(0, maximum(r)-r["rollouts_per_action"])*len(r["action_keys"])
                                   for r in legacy if r["split"] == split)
                        for split in ("train", "val")}
            eligible = [r for r in roots if len(r["action_keys"]) > 1 and
                        (r.get("hard_kind") or r["phase"] in ("SHOP", "PAGE", "ENHANCE_PAGE", "EVENT"))]
            chosen = stratified(eligible, diversified=True)
            allocations = {r["root_id"]: 0 for r in chosen}
            # One common-random-number seed costs one branch per legal action.
            # Separate split pools prevent development labels consuming train quota.
            for split in ("train", "val"):
                remaining = ceilings[split]
                active = [r for r in chosen if r["split"] == split]
                while active:
                    following = []
                    for root in active:
                        key, count = root["root_id"], len(root["action_keys"])
                        capacity = max(0, maximum(root)-root["rollouts_per_action"])*count
                        if allocations[key] + count <= capacity and count <= remaining:
                            allocations[key] += count
                            remaining -= count
                            following.append(root)
                    active = following
            plan = dict(mode="critical_conflicts_v2", root_ids=sorted(allocations),
                        eligible_roots=len(eligible), maximum_roots=limit,
                        selected_before_labels=True, branch_budget_by_split=ceilings,
                        branch_budget=sum(ceilings.values()), branch_allocations=allocations,
                        allocated_branches=sum(allocations.values()),
                        budget_reference="legacy stratified two-skip-kind maximum extra branches")
            atomic_json(path, plan)
        for root in roots:
            allocation = plan["branch_allocations"].get(root["root_id"], 0)
            if allocation:
                root.update(refinement_mode="critical_conflicts_v2",
                            refine_branch_budget=allocation,
                            refine_rollouts_per_action=root["rollouts_per_action"] + allocation//len(root["action_keys"]))
        self.publish(True, refinement_mode=plan["mode"],
                     refinement_candidate_roots=len(plan["root_ids"]),
                     refinement_branch_budget=plan["branch_budget"],
                     refinement_branch_budget_by_split=plan["branch_budget_by_split"],
                     refinement_allocated_branches=plan["allocated_branches"])

    def collect(self, manifest, directory):
        sample_dir, label_dir = directory / "sampling", directory / "labels"
        sample_dir.mkdir(exist_ok=True); label_dir.mkdir(exist_ok=True)
        pending, families = [], []
        for task in manifest["sampling_tasks"]:
            path = sample_dir / f"{task['family_id']}.json.gz"
            if path.exists():
                families.append(read_gzip_json(path))
            else:
                pending.append(task)
        self.stage("sampling", len(manifest["sampling_tasks"]), len(families), "采样可达状态，按完整预设和轨迹隔离训练/开发")
        def sampled(task, family):
            write_gzip_json(sample_dir / f"{task['family_id']}.json.gz", family)
            families.append(family)
            self.progress(len(families), len(manifest["sampling_tasks"]))
        paths = {"teacher": str(run_path(self.run_dir, manifest["teacher_checkpoint"]))}
        if manifest.get("behavior_checkpoint"):
            previous = run_path(self.run_dir, manifest["behavior_checkpoint"])
            if file_hash(previous) != manifest["behavior_sha256"]:
                raise ValueError("Frozen source-trajectory behavior model changed")
            paths["behavior"] = str(previous)
        if pending and not self.cuda_batches(pending, paths, sampled, directory, "sampling"):
            self.publish(True, sampling_backend_actual="numpy", sampling_simulation_backend="python",
                         sampling_device="CPU / NumPy FP32", workers=self.worker_count)
            self.parallel(pending, sample_family, paths, sampled)
        roots = sorted([r for f in families for r in f["roots"]], key=lambda r: r["root_id"])
        self.refinement_tasks(roots, manifest, directory)
        if manifest.get("hard_sampling"):
            self.publish(True, targeted_roots=sum(bool(r.get("hard_sample_slot")) for r in roots),
                         hard_sampling_enabled=True)
        pending = [r for r in roots if not (label_dir / f"{r['root_id']}.npz").exists()]
        complete_count = len(roots)-len(pending)
        refinement_used = {"train": 0, "val": 0}
        adaptive_refinement = manifest["budget"].get("refinement_mode") == "critical_conflicts_v2"
        if adaptive_refinement:
            import numpy as np
            # Only candidate metadata is read, never the large return arrays.
            for root in roots:
                saved = label_dir / f"{root['root_id']}.npz"
                if root.get("refine_branch_budget") and saved.exists():
                    with np.load(saved, allow_pickle=False) as archive:
                        metadata = json.loads(str(archive["metadata"].item()))
                    refinement_used[root["split"]] += metadata.get("refinement", {}).get("extra_branch_rollouts", 0)
            self.publish(True, refinement_used_branches=sum(refinement_used.values()),
                         refinement_used_branches_by_split=dict(refinement_used))
        self.stage("labeling", len(roots), complete_count, "冻结教师生成全部合法动作的完整终局收益标签")
        def labeled(task, row):
            nonlocal complete_count
            save_label(label_dir / f"{task['root_id']}.npz", row)
            complete_count += 1
            if adaptive_refinement:
                refinement_used[task["split"]] += row["metadata"].get("refinement", {}).get("extra_branch_rollouts", 0)
                self.status.update(refinement_used_branches=sum(refinement_used.values()),
                                   refinement_used_branches_by_split=dict(refinement_used))
            self.progress(complete_count, len(roots), samples=complete_count,
                          label_rollouts_per_action=row["metadata"]["rollouts_per_action"],
                          message=f"完整根标签 {complete_count}/{len(roots)}；本根每动作 {row['metadata']['rollouts_per_action']} 次终局续局")
        self.label_parallel(pending, paths, labeled, directory)
        if adaptive_refinement:
            atomic_json(directory / "refinement_usage.json", dict(
                mode="critical_conflicts_v2", extra_branch_rollouts=sum(refinement_used.values()),
                extra_branch_rollouts_by_split=refinement_used,
                branch_budget=self.status["refinement_branch_budget"],
                allocated_branches=self.status["refinement_allocated_branches"]))
        self.control()
        return len(roots)

    def replay(self, number, teacher_id):
        config = self.config["sampling"]
        available = []
        for r in range(number, max(0, number-config["replay_rounds"]), -1):
            directory = self.run_dir / "rounds" / f"round_{r:06d}"
            manifest_path = directory / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = read_json(manifest_path)
            if (manifest.get("preset_version") != PRESET_VERSION or
                    manifest.get("rules_version") != sim.MODEL_VERSION or
                    manifest["teacher_id"] != teacher_id):
                continue
            for path in sorted((directory / "labels").glob("*.npz")):
                available.append(path)
        # Keep whole families when limiting replay; never split train/dev or mix teachers.
        roots, families = [], {}
        for path in available:
            family_id = path.stem.rsplit("_root", 1)[0]
            families.setdefault(family_id, []).append(path)
        for family_id, paths in families.items():
            if roots and len(roots)+len(paths) > config["replay_max_roots"]:
                break
            self.control()
            family = [load_label(path, teacher_id) for path in paths]
            if any(row["family_id"] != family_id for row in family):
                raise ValueError("Replay filename and family identity disagree")
            roots.extend(family)
        return roots

    def fit_checkpoints(self, manifest, directory):
        from .fit_cache import identity, prepare, validate_pool
        from .learner_initialization import fit_initial_model
        initial = fit_initial_model(manifest, self.run_dir)
        cache_identity = prepare(directory, identity(manifest, self.config["seed"]))
        pool_path = directory / "fit_candidates.json"
        if pool_path.exists():
            cached = read_json(pool_path)
            validate_pool(directory, cached, cache_identity)
            return cached
        initial_student = QNetwork.load(run_path(self.run_dir, initial["checkpoint"])).clone(frozen=True)
        checksum = initial_student.checksum()
        roots = self.replay(manifest["round"], manifest["teacher_id"])
        input_digest = None
        if manifest.get("workflow", {}).get("protocol") == "paired_anchor_v2" or cache_identity is not None:
            from .workflow import label_input_digest
            input_digest = label_input_digest(roots)
        settings = dict(manifest.get("training_config", self.config["training"]),
                        **manifest.get("compute_config", self.config["compute"]),
                        seed=(manifest["workflow"]["optimizer_seed"] if manifest.get("workflow", {}).get("protocol") == "paired_anchor_v2"
                              else canonical_seed("optimizer", self.config["seed"], manifest["round"])))
        if input_digest is not None or manifest.get("catalogue_coverage"):
            settings.update(initial_student_model_id=initial["model_id"], training_round=manifest["round"])
        if "learner_initialization" in manifest:
            self.log(f"学生初始化：{initial['model_id']}（{initial['source']}）；正式最佳与逐预设教师路由保持冻结")
        branch_name = {"control": "对照分支", "action": "动作优势分支"}.get(manifest.get("workflow_branch"), "")
        self.stage("training", settings["epochs"], 0,
                   f"{branch_name}训练候选：{len(roots)} 个{'逐预设冻结教师' if 'teacher_bank' in manifest else '同教师'}根样本；整轨迹开发集独立")
        loss_history = []
        def callback(metrics):
            self.control()
            if metrics.get("kind") == "backend":
                self.log(f"训练后端：{metrics.get('backend')} / {metrics.get('device_name')}；{metrics.get('reason', '')}")
                self.publish(True, training_backend_actual=metrics.get("backend"),
                             training_device=metrics.get("device_name", metrics.get("device")),
                             backend_reason=metrics.get("reason"))
            elif metrics.get("kind") == "epoch":
                train_metrics = metrics.get("train", {})
                val_metrics = metrics.get("val", metrics.get("validation", {}))
                item = {"step": metrics.get("epoch", 0),
                        "training_loss": train_metrics.get("selection_loss"),
                        "validation_loss": val_metrics.get("selection_loss")}
                loss_history.append(item)
                self.progress(metrics.get("epoch", 0), settings["epochs"],
                              training_loss=item["training_loss"], validation_loss=item["validation_loss"],
                              loss_history=loss_history, validation_metrics=val_metrics,
                              **({"message": f"{branch_name}训练：epoch {metrics.get('epoch', 0)}/{settings['epochs']}"} if branch_name else {}),
                              checkpoint_metric=settings["checkpoint_metric"],
                              checkpoint_score=val_metrics.get(settings["checkpoint_metric"]),
                              training_precision=metrics.get("precision"),
                              training_tf32=metrics.get("tf32_training"),
                              training_backend_actual=metrics.get("backend", self.status.get("training_backend_actual", "numpy")),
                              training_device=metrics.get("device_name", self.status.get("training_device", "CPU")))
            else:
                self.publish()
        result = fit_candidate(initial_student, roots, settings, callback=callback,
                               stop_event=lambda: (self.run_dir / "stop.flag").exists())
        if result.interrupted:
            raise StopRequested("训练已中断；标签已保存，恢复后重启本轮优化阶段。")
        if initial_student.checksum() != checksum:
            raise RuntimeError("Frozen initial student changed during candidate optimization")
        points = result.candidate_checkpoints or [dict(epoch=result.best_epoch,
                  model=result.model, optimizer_steps=result.optimizer_steps)]
        candidates = []
        for point in points:
            checkpoint = directory / "fit_checkpoints" / f"epoch_{point['epoch']:03d}.npz"
            point["model"].save(checkpoint, {"model_id": f"M{manifest['round']:06d}",
                                           "rules_version": sim.MODEL_VERSION,
                                           "teacher_id": manifest["teacher_id"],
                                           "teacher_sha256": manifest["teacher_sha256"],
                                           "scope": [84, 132] if "teacher_bank" in manifest else [72, 144], "round": manifest["round"],
                                           "role_scope": list(ROLE_TARGET_RANGE), "preset_version": PRESET_VERSION,
                                           "best_epoch": point["epoch"],
                                           "teacher_bank_digest": digest(manifest["teacher_bank"]) if "teacher_bank" in manifest else None,
                                           "label_semantics": "complete terminal action returns under each preset's frozen teacher" if "teacher_bank" in manifest else "complete terminal action returns under frozen teacher",
                                           **({"learner_initialization": dict(initial), "initial_student_model_id": initial["model_id"],
                                               "initial_student_checkpoint": initial["checkpoint"], "initial_student_sha256": initial["sha256"]}
                                              if "learner_initialization" in manifest else {})})
            candidates.append(dict(epoch=point["epoch"], checkpoint=str(checkpoint.relative_to(directory)),
                                   sha256=file_hash(checkpoint), optimizer_steps=point["optimizer_steps"]))
        summary = result.summary()
        if input_digest is not None:
            summary["frozen_label_input_sha256"] = input_digest
        if "learner_initialization" in manifest:
            summary.update(learner_initialization=dict(initial), initial_student_model_id=initial["model_id"],
                           initial_student_checkpoint=initial["checkpoint"], initial_student_sha256=initial["sha256"])
        pool = {"fit": summary, "candidates": candidates,
                "teacher_id": manifest["teacher_id"], "rules_version": sim.MODEL_VERSION}
        if cache_identity is not None:
            pool["fit_cache_identity"] = cache_identity
            summary["fit_cache_identity"] = cache_identity
        atomic_json(pool_path, pool)
        return pool

    def train(self, manifest, directory, candidate_path):
        metrics_path = directory / "fit.json"
        if candidate_path.exists() and metrics_path.exists():
            QNetwork.load(candidate_path)
            return read_json(metrics_path)
        pool = self.fit_checkpoints(manifest, directory)
        summary = pool["fit"]
        label_best_epoch = summary["best_epoch"]
        candidates = sorted(pool["candidates"], key=lambda p: (p["epoch"] != label_best_epoch, p["epoch"]))
        selected = candidates[0]
        tasks = manifest.get("development_tasks", [])
        scores = []
        if tasks and len(candidates) > 1:
            best_mean = -math.inf
            for index, point in enumerate(candidates, 1):
                self.control()
                checkpoint = run_path(directory, point["checkpoint"])
                if file_hash(checkpoint) != point["sha256"]:
                    raise ValueError("Development checkpoint changed after fitting")
                evaluation_dir = directory / "development" / f"epoch_{point['epoch']:03d}"
                score_path = evaluation_dir / "summary.json"
                if score_path.exists():
                    score = read_json(score_path)
                else:
                    self.publish(True, checkpoint_candidate_index=index,
                                 checkpoint_candidates=len(candidates), checkpoint_epoch=point["epoch"])
                    rows = self.evaluate(tasks, checkpoint, run_path(self.run_dir, manifest["teacher_checkpoint"]),
                                         evaluation_dir,
                                         f"开发整局选训练时点 {index}/{len(candidates)}（epoch {point['epoch']}）")
                    result = summarize_pairs(rows, tasks, bootstrap_replicates=20,
                                             seed=canonical_seed("epoch_development_ci", self.config["seed"], manifest["round"]))
                    score = {"epoch": point["epoch"], "checkpoint_sha256": point["sha256"],
                             "candidate_mean": result["primary_candidate_mean"],
                             "teacher_mean": result["primary_teacher_mean"],
                             "delta": result["primary_delta"], "pairs": result["pairs"],
                             "presets": result["presets"], "purpose": "epoch selection only"}
                    atomic_json(score_path, score)
                if score["checkpoint_sha256"] != point["sha256"]:
                    raise ValueError("Development results belong to a different checkpoint")
                scores.append(score)
                if score["candidate_mean"] > best_mean:
                    selected, best_mean = point, score["candidate_mean"]
            self.log(f"开发整局选中 epoch {selected['epoch']}，均分 {best_mean:.4f}；随后执行独立正式晋升评测。")
        QNetwork.load(run_path(directory, selected["checkpoint"])).save(candidate_path, {
            "best_epoch": selected["epoch"], "label_best_epoch": label_best_epoch,
            "checkpoint_selection": "independent_development_games" if scores else "label_metric"})
        summary.update(best_epoch=selected["epoch"], label_best_epoch=label_best_epoch,
                       checkpoint_selection={"method": "independent_development_games" if scores else "label_metric",
                                             "selected_epoch": selected["epoch"], "candidates": scores,
                                             "development_pairs_per_candidate": len(tasks),
                                             "promotion_games_used": False})
        atomic_json(metrics_path, summary)
        return summary

    def run_round(self, number):
        self.round_started = time.monotonic()
        directory = self.run_dir / "rounds" / f"round_{number:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.apply_next_round_config(number, directory)
        summary_path = directory / "summary.json"
        self.publish(True, round=number, candidate_id=f"M{number:06d}", loss_history=[],
                     training_loss=None, validation_loss=None, validation_metrics={},
                     checkpoint_score=None, targeted_roots=None, samples=0,
                     checkpoint_epoch=None, checkpoint_selection_epoch=None,
                     refinement_candidate_roots=None, refinement_mode=None,
                     refinement_branch_budget=None, refinement_allocated_branches=None,
                     refinement_used_branches=None, refinement_branch_budget_by_split=None,
                     refinement_used_branches_by_split=None)
        if summary_path.exists():
            summary = read_json(summary_path)
            self.commit(summary)
            return
        manifest = self.prepare_manifest(number, directory)
        training = manifest.get("training_config", self.config["training"])
        compute = manifest.get("compute_config", self.config["compute"])
        self.publish(True, round_budget=manifest["budget"],
                     config_revision=manifest.get("training_revision"),
                     development_pairs_per_candidate=len(manifest.get("development_tasks", [])),
                     checkpoint_candidates=training.get("checkpoint_candidates", 1),
                     preserve_early_checkpoints=training.get("preserve_early_checkpoints", False),
                     advantage_confidence_weighting=training.get("advantage_confidence_weighting", False),
                     checkpoint_metric=training["checkpoint_metric"],
                     training_precision="BF16 hidden / FP32 head" if compute["gpu_amp"] else "FP32",
                     training_tf32=compute["gpu_tf32"],
                     hard_sampling_enabled=bool(manifest.get("hard_sampling")))
        self.log(f"第 {number} 轮冻结 {manifest['teacher_id']}；预算 {manifest['budget']}")
        candidate_path = self.run_dir / "checkpoints" / f"M{number:06d}.npz"
        metrics_path = directory / "fit.json"
        if not (candidate_path.exists() and metrics_path.exists()):
            self.collect(manifest, directory)
        metrics = self.train(manifest, directory, candidate_path)
        self.publish(True, checkpoint_epoch=None, checkpoint_selection_epoch=metrics["best_epoch"])
        self.control()
        if file_hash(run_path(self.run_dir, manifest["teacher_checkpoint"])) != manifest["teacher_sha256"]:
            raise RuntimeError("Teacher checkpoint bytes changed during the round")
        evaluation_path = directory / "evaluation.json"
        if evaluation_path.exists():
            result = read_json(evaluation_path)
        else:
            rows = self.evaluate(manifest["promotion_tasks"], candidate_path,
                                 run_path(self.run_dir, manifest["teacher_checkpoint"]), directory / "evaluation_pairs",
                                 "独立新预设晋升评测（全部预定配额完成后选模）")
            result = summarize_pairs(rows, manifest["promotion_tasks"],
                                     self.config["evaluation"]["bootstrap_replicates"],
                                     canonical_seed("round_ci", self.config["seed"], number))
            result.update(candidate_sha256=file_hash(candidate_path),
                          teacher_sha256=manifest["teacher_sha256"], manifest_hash=digest(manifest))
            atomic_json(evaluation_path, result)
        if (result["candidate_sha256"] != file_hash(candidate_path) or
                result["teacher_sha256"] != manifest["teacher_sha256"] or
                result["manifest_hash"] != digest(manifest)):
            raise ValueError("Completed evaluation does not match candidate/teacher/protocol")
        self.publish(True, primary_delta=result["primary_delta"], preset_results=result["presets"],
                     ci_low=result["ci_low"], ci_high=result["ci_high"])
        promoted = bool(result["eligible"] and result["primary_delta"] > 0.0)
        if promoted:
            self.log(f"第 {number} 轮候选平均变化 {result['primary_delta']:+.4f}，达到均值晋升条件；完成趋势审计后提交。")
            audit = self.audit(candidate_path, directory / "audit_pairs")
        else:
            self.log(f"第 {number} 轮候选平均变化 {result['primary_delta']:+.4f}，保留 {manifest['teacher_id']}。")
            audit = self.state["audit_presets"]
        best_epoch = metrics["best_epoch"]
        best_metrics = next((m for m in metrics["history"] if m.get("epoch") == best_epoch), {})
        summary = dict(result, round=number, teacher_id=manifest["teacher_id"],
                       rules_version=sim.MODEL_VERSION,
                       candidate_id=f"M{number:06d}", best_model_id=f"M{number:06d}" if promoted else manifest["teacher_id"],
                       promoted=promoted, audit_presets=audit, audit_reused=not promoted,
                       audit_model_id=f"M{number:06d}" if promoted else self.state["audit_model_id"],
                       elapsed_seconds=self.elapsed(), round_seconds=time.monotonic()-self.round_started,
                       best_epoch=best_epoch, fit=metrics, training_loss=best_metrics.get("train", {}).get("selection_loss"),
                       validation_loss=best_metrics.get("val", best_metrics.get("validation", {})).get("selection_loss"),
                       completed_at=utc_now(), candidate_checkpoint=f"checkpoints/M{number:06d}.npz")
        atomic_json(summary_path, summary)
        self.commit(summary)

    def commit(self, summary):
        number = summary["round"]
        # Summary -> idempotent history -> state is a recoverable commit order.
        history = {k: v for k, v in summary.items() if k != "fit"}
        upsert_history(self.run_dir / "history.jsonl", history, self.config["retention"]["history_rows"])
        if summary["promoted"]:
            self.state.update(teacher_id=summary["candidate_id"], teacher_checkpoint=summary["candidate_checkpoint"],
                              rejected_rounds=0, audit_model_id=summary["candidate_id"])
        else:
            self.state["rejected_rounds"] += 1
        self.state.update(next_round=number+1, elapsed_seconds=self.elapsed(), audit_presets=summary["audit_presets"],
                          last_candidate_checkpoint=summary["candidate_checkpoint"])
        atomic_json(self.state_path, self.state)
        # Stable external pointer; only fully accepted models can change it.
        atomic_json(self.run_dir / "best_model.json", {"model_id": self.state["teacher_id"],
                    "evaluation_rules_version": sim.MODEL_VERSION,
                    "checkpoint": self.state["teacher_checkpoint"], "accepted_round": int(self.state["teacher_id"][1:]),
                    "last_assessed_round": number,
                    "sha256": file_hash(run_path(self.run_dir, self.state["teacher_checkpoint"])),
                    "meaning": "current champion selected by complete independent estimated mean; not a proof of true optimality"})
        self.publish(True, status="promoting", phase="promoting", phase_progress=1.0,
                     audit_presets=summary["audit_presets"], audit_reused=summary["audit_reused"],
                     primary_delta=summary["primary_delta"], preset_results=summary["presets"],
                     message=f"本轮完成，当前最佳已入选模型 {self.state['teacher_id']}；继续下一轮。")
        self.prune()

    def prune(self):
        from .retention import retention_plan, apply_retention
        plan = retention_plan(self.run_dir, self.state,
            keep_rounds=max(self.config["retention"]["rounds"],
                            self.config["sampling"]["replay_rounds"] + 1),
            keep_checkpoints=self.config["retention"]["checkpoints"])
        atomic_json(self.run_dir / "retention_last.json", apply_retention(plan))

    def run(self):
        try:
            self.bootstrap()
            completed = 0
            while not self.max_rounds or completed < self.max_rounds:
                self.control()
                self.run_round(self.state["next_round"])
                completed += 1
            self.publish(True, status="stopped", message=f"已完成命令行指定的 {completed} 轮；双击可继续。")
            self.log("已完成本次指定轮数。")
            return 0
        except (StopRequested, KeyboardInterrupt) as exc:
            (self.run_dir / "stop.flag").touch()
            (self.run_dir / "pause.flag").unlink(missing_ok=True)
            self.log(str(exc) or "安全停止。")
            self.publish(True, status="stopped", stop_reason=getattr(self, "stop_reason", "requested"),
                         message=(str(exc) or "已安全停止。") + " 再次启动将恢复已完成的工作。")
            return 0
        except Exception as exc:
            details = traceback.format_exc()
            self.log(f"任务暂停于错误：{exc}", "error")
            (self.run_dir / "error.log").write_text(details, encoding="utf-8")
            self.publish(True, status="error", error=str(exc), message="未晋升不完整候选；错误详情见 error.log。修复后可从断点恢复。")
            print(details, flush=True)
            return 1
        finally:
            self.close_pool()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continuous frozen-teacher neural policy iteration")
    parser.add_argument("--run-dir", default=str(ROOT / "runs/role18_54"))
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--rounds", type=int, default=0, help="0: until stopped; positive: additional complete rounds")
    parser.add_argument("--hours", type=float, default=0, help="Wall-clock hours per worker session, including pauses; 0: unlimited")
    parser.add_argument("--smoke", action="store_true", help="Small REAL complete run; not a model quality claim")
    parser.add_argument("--profile", choices=["local_5070", "colab_blackwell96"])
    args = parser.parse_args(argv)
    apply_profile(args.profile)
    if args.rounds < 0:
        parser.error("--rounds must be nonnegative")
    if not math.isfinite(args.hours) or args.hours < 0:
        parser.error("--hours must be finite and nonnegative")
    raw = read_json(args.config) if Path(args.config).exists() else {}
    config = resolve(raw, smoke=args.smoke)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # Lock the engine independently of the GUI, including headless launches.
    from launcher import InstanceLock
    lock = InstanceLock(run_dir / "engine.lock")
    if not lock.acquire():
        print("同一任务已有训练进程，拒绝重复写入。", flush=True)
        lock.close()
        return 2
    def stop_handler(_sig, _frame):
        (run_dir / "stop.flag").touch()
        (run_dir / "pause.flag").unlink(missing_ok=True)
    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)
    try:
        if config.get("catalogue_mode"):
            from .catalogue_engine import CatalogueEngine
            engine_class = CatalogueEngine
        else:
            engine_class = Engine
        return engine_class(run_dir, config, args.rounds, args.hours).run()
    finally:
        lock.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
