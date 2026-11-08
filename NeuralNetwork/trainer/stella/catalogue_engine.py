"""Continuous training with one frozen champion per catalogue structure.

The fixed catalogue evaluation deliberately overlaps training structures. Its
environment seeds never enter source trajectories, labels or epoch selection.
Only completed summaries change the bank; a summary can be committed repeatedly
after interruption without applying its updates twice.
"""
from __future__ import annotations

from .run_paths import run_path, portable_basename

from collections import defaultdict, deque
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
from random import Random
import time

import numpy as np

from .engine import Engine, ROOT, utc_now, cuda_task_batches
from .environment import sim
from .io_utils import (atomic_json, read_json, write_gzip_json, read_gzip_json,
                       digest, file_hash, upsert_history, append_jsonl)
from .network import QNetwork
from .presets import PRESET_VERSION, canonical_seed
from .preset_catalogue import catalogue_presets, catalogue_id
from .sampling import sample_family, save_label, load_label
from .label_writer import LabelWriter
from .catalogue_evaluation import evaluate_catalogue, select_champions, _atomic_npz
from .catalogue_service import CatalogueEvaluationSession, _enabled
from .workflow import (PAIRED_ANCHOR_V2, frozen_workflow, frozen_seed,
                       build_workflow, verify_frozen_models, compare_paired_official)
from .learner_initialization import freeze_initialization, fit_initial_model


PROTOCOL = "catalogue4960_per_preset_champions_v1"
GAMES = 256
FAMILIES = 4960
JOINT_EVALUATION = "six_checkpoint_official_v1"


def teacher_age_sample(bank):
    indices = np.linspace(0, len(bank)-1, 512, dtype=int).tolist()
    return {"preset_ids": indices, "teacher_rounds": [int(bank[i].split("_")[0][1:]) for i in indices]}



class CatalogueEngine(Engine):
    def bootstrap(self):
        self.control()
        self.presets = catalogue_presets()
        self.preset_indices = {p.fingerprint: i for i, p in enumerate(self.presets)}
        config_path = self.run_dir / "resolved_config.json"
        if config_path.exists() and read_json(config_path) != self.config:
            raise ValueError("已有目录配置已冻结，请使用原配置恢复")
        atomic_json(config_path, self.config)
        if self.state:
            if self.state.get("protocol") != PROTOCOL:
                raise ValueError("目录不是4,960逐预设教师运行")
            self.log(f"恢复4,960逐预设教师：下一轮 {self.state['next_round']}")
        else:
            supplied = Path(self.config["model"]["initial_checkpoint"]).expanduser()
            if not supplied.is_absolute():
                supplied = ROOT / supplied
            checkpoint = self.run_dir / "checkpoints/M000000.npz"
            model = QNetwork.load(supplied)
            model.save(checkpoint, {"model_id": "M000000", "protocol": PROTOCOL,
                "rules_version": sim.MODEL_VERSION, "preset_version": PRESET_VERSION,
                "warm_start_source": str(supplied), "warm_start_sha256": file_hash(supplied)})
            self.state = {"version": self.config["version"], "protocol": PROTOCOL,
                "config_hash": digest(self.config), "preset_version": PRESET_VERSION,
                "rules_version": sim.MODEL_VERSION, "next_round": 1,
                "teacher_id": "M000000", "teacher_checkpoint": "checkpoints/M000000.npz",
                "best_candidate_mean": None, "teacher_bank": ["M000000"] * len(self.presets),
                "models": {"M000000": "checkpoints/M000000.npz"},
                "teacher_scores": None, "rejected_rounds": 0, "elapsed_seconds": 0,
                "initialization": str(supplied), "audit_presets": []}
            atomic_json(self.run_dir / "catalogue.json", {"protocol": PROTOCOL,
                "presets": [p.to_dict() for p in self.presets], "equal_weight": True,
                "games_per_preset": GAMES, "minimum_gain": .05,
                "required_integer_sum_gain": 13,
                "evaluation_structure_overlap_with_training": "deliberately authorized",
                "evaluation_seed_domain": "fixed_catalogue_environment_v1",
                "source_checkpoint": str(supplied), "created_at": utc_now()})
            atomic_json(self.state_path, self.state)
        if (self.state.get("rules_version") != sim.MODEL_VERSION or
                self.state.get("migration_baseline_pending")):
            self.migrate_rules()
        self.publish(True, catalogue_mode=True, catalogue_size=len(self.presets),
                     scope="4,960种固定完整原始结构等权；团队84–132普通目标级数，不含核心",
                     model_parameters=QNetwork.load(run_path(self.run_dir, self.state["teacher_checkpoint"])).parameter_count,
                     best_candidate_mean=self.state.get("best_candidate_mean"),
                     composite_teacher_mean=self.state.get("composite_teacher_mean"),
                     distinct_teacher_count=len(set(self.state["teacher_bank"])))
        if not self.state.get("teacher_scores"):
            scores = self.evaluate_matrix(run_path(self.run_dir, self.state["teacher_checkpoint"]),
                self.presets, self.run_dir / "initial_evaluation", "初始化4,960预设教师基线")
            path = self.run_dir / "teacher_scores/round_000000.npz"
            _atomic_npz_mkdir(path, scores=scores)
            mean = float(scores.mean())
            self.state.update(teacher_scores=str(path.relative_to(self.run_dir)),
                              teacher_scores_rules_version=sim.MODEL_VERSION,
                              best_candidate_mean=mean, composite_teacher_mean=mean)
            atomic_json(self.state_path, self.state)
            upsert_history(self.run_dir / "history.jsonl", {"round": 0, "initial": True,
                "protocol": PROTOCOL, "candidate_mean": mean, "best_candidate_mean": mean,
                "composite_teacher_mean": mean, "champion_updated_count": 0,
                "distinct_teacher_count": 1, "candidate_id": "M000000", "teacher_id": "M000000",
                "teacher_age_sample": teacher_age_sample(self.state["teacher_bank"]),
                "elapsed_seconds": self.elapsed()}, self.config["retention"]["history_rows"])


    def migrate_rules(self):
        """Reprice frozen weights under current rules; evaluation chunks resume normally."""
        if not self.state.get("migration_baseline_pending"):
            self.state.update(previous_rules_version=self.state.get("rules_version"),
                rules_effective_round=self.state["next_round"],
                migration_baseline_pending=True)
        self.state.update(rules_version=sim.MODEL_VERSION,
                          migration_target_rules_version=sim.MODEL_VERSION)
        atomic_json(self.state_path, self.state)
        directory = self.run_dir / "rules_revisions" / sim.MODEL_VERSION
        groups = defaultdict(list)
        for index, teacher_id in enumerate(self.state["teacher_bank"]):
            groups[teacher_id].append(index)
        scores = np.empty((len(self.presets), GAMES), dtype=np.int16)
        with CatalogueEvaluationSession(self.presets,
                seed=canonical_seed("catalogue_final", self.config["seed"]),
                games_per_preset=GAMES, run_dir=self.run_dir) as session:
            for teacher_id, indices in sorted(groups.items()):
                presets = [self.presets[i] for i in indices]
                session.rebind_presets(presets, self.control)
                scores[indices] = self.evaluate_matrix(
                    run_path(self.run_dir, self.state["models"][teacher_id]),
                    presets, directory / "teachers" / teacher_id,
                    f"新规则教师基线：{teacher_id}，{len(indices)}预设",
                    count_teacher_cache=False, session=session)
        best_scores = self.evaluate_matrix(
            run_path(self.run_dir, self.state["teacher_checkpoint"]), self.presets,
            directory / "best_single", "新规则最佳单模型基线：4,960预设×256局",
            count_teacher_cache=False)
        path = directory / "teacher_scores.npz"
        _atomic_npz_mkdir(path, scores=scores)
        baseline = {"rules_version": sim.MODEL_VERSION,
            "previous_rules_version": self.state.get("previous_rules_version"),
            "rules_effective_round": self.state["rules_effective_round"],
            "teacher_bank_digest": digest(self.state["teacher_bank"]),
            "teacher_id": self.state["teacher_id"],
            "best_candidate_mean": float(best_scores.mean()),
            "composite_teacher_mean": float(scores.mean()),
            "teacher_scores": str(path.relative_to(self.run_dir)),
            "teacher_scores_sha256": file_hash(path), "completed_at": utc_now()}
        atomic_json(directory / "baseline.json", baseline)
        atomic_json(self.run_dir / "best_model.json", {
            "model_id": self.state["teacher_id"], "checkpoint": self.state["teacher_checkpoint"],
            "candidate_mean": baseline["best_candidate_mean"], "rules_version": sim.MODEL_VERSION,
            "last_assessed_round": self.state["next_round"] - 1, "protocol": PROTOCOL})
        atomic_json(self.run_dir / "champion_bank.json", {
            "round": self.state["next_round"] - 1, "protocol": PROTOCOL,
            "rules_version": sim.MODEL_VERSION, "teacher_bank": self.state["teacher_bank"],
            "models": self.state["models"], "teacher_scores": baseline["teacher_scores"],
            "composite_teacher_mean": baseline["composite_teacher_mean"]})
        atomic_json(self.run_dir / "preset_results.json", {
            "round": self.state["next_round"] - 1, "rules_version": sim.MODEL_VERSION,
            "rules_revision": True,
            "presets": [{"catalogue_id": i, "name": p.name, "fingerprint": p.fingerprint,
                "cap": p.total, "candidate_mean": float(best_scores[i].mean()),
                "single_candidate_mean": float(best_scores[i].mean()),
                "candidate_id": self.state["teacher_id"], "teacher_mean": float(scores[i].mean()),
                "delta": float(best_scores[i].mean() - scores[i].mean()), "n": GAMES,
                "updated": False, "teacher_id": self.state["teacher_bank"][i],
                "previous_teacher_id": self.state["teacher_bank"][i]}
                for i, p in enumerate(self.presets)]})
        self.state.update(teacher_scores=baseline["teacher_scores"],
            teacher_scores_rules_version=sim.MODEL_VERSION,
            best_candidate_mean=baseline["best_candidate_mean"],
            composite_teacher_mean=baseline["composite_teacher_mean"],
            migration_baseline_pending=False)
        atomic_json(self.state_path, self.state)
        self.publish(True, preset_results_path="preset_results.json",
                     best_candidate_mean=baseline["best_candidate_mean"],
                     composite_teacher_mean=baseline["composite_teacher_mean"])
        self.log(f"新规则 {sim.MODEL_VERSION} 基线完成，保留权重并重新开始第{self.state['next_round']}轮。")

    def evaluate_matrix(self, checkpoint, presets, directory, message, games=GAMES, development=False, session=None, manifest=None, count_teacher_cache=True):
        self.close_pool()
        total = len(presets)*games
        self.stage("evaluating", total, 0, message)
        self.publish(True, evaluation_total_presets=len(presets), evaluation_completed_presets=0,
                     evaluation_games_total=total, evaluation_games_completed=0,
                     evaluation_backend_actual="torch_cuda", evaluation_simulation_backend="cuda_games")
        def progress(row):
            statistics = row.get("statistics", {})
            cached_teacher = len(self.presets)*GAMES if self.state.get("teacher_scores") and not development and count_teacher_cache else 0
            hits = statistics.get("cache_hits", 0)+row.get("cached_games", 0)+cached_teacher
            misses = statistics.get("cache_misses", 0)+row["completed_games"]-row.get("cached_games", 0)
            self.progress(row["completed_games"], total,
                evaluation_completed_presets=row["completed_games"]//games,
                evaluation_games_completed=row["completed_games"], cache_hits=hits,
                cache_misses=misses, cache_hit_rate=hits/max(1, hits+misses),
                evaluation_cached_games=row.get("cached_games", 0),
                evaluation_device=row.get("information", {}).get("device_name", "CUDA"),
                evaluation_batch_games_per_second=row.get("batch_games_per_second"))
        result = evaluate_catalogue(checkpoint, presets, directory, run_dir=self.run_dir,
            games_per_preset=games, batch_size=max(1, int(os.environ.get("STELLA_GPU_EVAL_BATCH", "8192"))),
            seed=(frozen_seed(manifest, self.config, "development" if development else "evaluation")
                  if manifest else canonical_seed("catalogue_development" if development else "catalogue_final", self.config["seed"])), control=self.control, progress=progress, session=session)
        self.control()
        return result["scores"]

    def prepare_manifest(self, number, directory):
        path = directory / "manifest.json"
        if path.exists():
            manifest = read_json(path)
            if manifest.get("rules_version") != sim.MODEL_VERSION:
                raise ValueError("未完成轮次属于旧规则，请移除该轮目录后重新采样")
            if manifest["teacher_bank"] != self.state["teacher_bank"]:
                raise ValueError("未完成轮次的逐预设教师已改变")
            verify_frozen_models(manifest, self.run_dir)
            fit_initial_model(manifest, self.run_dir)
            return manifest
        sampling = self.config["sampling"]
        budget = {"families": FAMILIES, "roots_per_family": sampling["roots_per_family"],
            "rollouts_per_action": sampling["rollouts_per_action"], "games_per_preset": GAMES,
            "validation_rollouts_per_action": sampling["rollouts_per_action"] * sampling.get("validation_rollouts_multiplier", 1)}
        if sampling.get("hard_sampling"):
            budget.update(hard_roots_per_family=sampling["roots_per_family"]//4,
                          hard_rollouts_per_action=sampling["hard_rollouts_per_action"],
                          hard_validation_rollouts_per_action=sampling["hard_validation_rollouts_per_action"])
        if sampling.get("refinement_roots"):
            for key in ("refinement_roots", "refinement_rollouts_per_action",
                        "refinement_validation_rollouts_per_action", "refinement_mode"):
                if key in sampling:
                    budget[key] = sampling[key]
        tasks = []
        # Randomized cycles give every structure exactly the same sampling
        # probability and avoid accidental repeated structures within a round.
        indices = list(range(len(self.presets)))
        Random(canonical_seed("catalogue_uniform_families", self.config["seed"], number)).shuffle(indices)
        behavior_order = list(range(len(self.presets)))
        Random(canonical_seed("catalogue_behavior_partition", self.config["seed"])).shuffle(behavior_order)
        half = len(behavior_order)//2
        student_presets = set(behavior_order[:half] if number % 2 else behavior_order[half:])
        for i in range(FAMILIES):
            index = indices[i % len(indices)]
            p = self.presets[index]
            split = "val" if canonical_seed("catalogue_preset_split", self.config["seed"], p.fingerprint) % 4 == 0 else "train"
            mode = i % 10
            tasks.append({"family_id": f"r{number:06d}_f{i:05d}", "ordinal": i,
                "rules_version": sim.MODEL_VERSION, "preset": p.to_dict(), "catalogue_id": index,
                "teacher_id": self.state["teacher_bank"][index], "split": split,
                "balanced_phase_sampling": sampling.get("balanced_phase_sampling", False),
                "phase_sampling_index": i, "epsilon": 0.0 if mode < 6 else (.15 if mode < 8 else .5),
                "behavior_source": "best_candidate" if index in student_presets else "teacher",
                "behavior_model_id": self.state["teacher_id"] if index in student_presets else self.state["teacher_bank"][index],
                "student_model_id": self.state["teacher_id"], "disagreement_sampling": True,
                "roots_per_family": budget["roots_per_family"],
                "rollouts_per_action": budget["validation_rollouts_per_action"] if split == "val" else budget["rollouts_per_action"],
                "environment_seed": canonical_seed("catalogue_source_environment", self.config["seed"], number, i),
                "exploration_seed": canonical_seed("catalogue_source_exploration", self.config["seed"], number, i),
                "selection_seed": canonical_seed("catalogue_root_selection", self.config["seed"], number, i),
                "label_seed": canonical_seed("catalogue_full_labels", self.config["seed"], number, i)})
            if sampling.get("hard_sampling"):
                tasks[-1].update(hard_sampling=True,
                    hard_rollouts_per_action=budget["hard_validation_rollouts_per_action"] if split == "val"
                                            else budget["hard_rollouts_per_action"])
        coverage = None
        if "catalogue_coverage" in self.config:
            from .catalogue_coverage import transform_sampling_tasks
            tasks, coverage = transform_sampling_tasks(
                tasks, self.config["seed"], number, self.config["catalogue_coverage"], self.state["teacher_bank"])
            budget.update(families=len(tasks), unique_presets=FAMILIES,
                          train_roots_per_preset=coverage["train_roots_per_preset"],
                          val_roots_per_preset=coverage["val_roots_per_preset"],
                          total_new_roots=coverage["roots"])
        models = {key: {"checkpoint": self.state["models"][key],
                       "sha256": file_hash(run_path(self.run_dir, self.state["models"][key]))}
                  for key in set(self.state["teacher_bank"])}
        manifest = {"version": self.config["version"], "protocol": PROTOCOL, "round": number,
            "rules_version": sim.MODEL_VERSION, "preset_version": PRESET_VERSION,
            "teacher_bank": list(self.state["teacher_bank"]), "teacher_models": models,
            "teacher_id": self.state["teacher_id"], "teacher_checkpoint": self.state["teacher_checkpoint"],
            "teacher_sha256": file_hash(run_path(self.run_dir, self.state["teacher_checkpoint"])),
            "teacher_scores": self.state["teacher_scores"], "budget": budget,
            "teacher_scores_rules_version": sim.MODEL_VERSION,
            "teacher_scores_sha256": file_hash(run_path(self.run_dir, self.state["teacher_scores"])),
            "sampling_tasks": tasks, "evaluation_mode": JOINT_EVALUATION,
            "evaluation_candidates": 6, "evaluation_games_per_preset": GAMES,
            "evaluation_seed_domain": "catalogue_final",
            "selection_uses_formal_evaluation": True,
            "behavior_checkpoint": self.state["teacher_checkpoint"],
            "behavior_sha256": file_hash(run_path(self.run_dir, self.state["teacher_checkpoint"])),
            "sampling_policy": "half_teacher_half_best_alternating_disagreement_v1",
            "hard_sampling": sampling.get("hard_sampling", False),
            "training_config": dict(self.config["training"]), "compute_config": dict(self.config["compute"]),
            "family_split_version": "stable_full_catalogue_preset_v1", "created_at": utc_now(),
            "selection_condition": "per-preset paired integer sum gain >= 13 of 256 games",
            "evaluation_structure_overlap_with_training": True, "ci_is_gate": False}
        manifest["training_config"]["checkpoint_candidates"] = 6
        manifest["training_config"]["per_preset_teacher_bank"] = True
        if coverage is not None:
            manifest["catalogue_coverage"] = coverage
            manifest["family_split_version"] = coverage["protocol"]
            manifest["training_config"]["validation_unit"] = "trajectory"
            manifest["training_config"]["initial_student_model_id"] = manifest["teacher_id"]
            manifest["training_config"]["training_round"] = number
            manifest["sampling_policy"] = "all_presets_independent_train_val_v3"
        if self.config.get("workflow", {}).get("protocol") == PAIRED_ANCHOR_V2:
            manifest["workflow"] = build_workflow(self.config, number)
        initialization = freeze_initialization(manifest["training_config"], self.state, self.run_dir, number)
        if initialization is not None:
            manifest["learner_initialization"] = initialization
            manifest["training_config"]["initial_student_model_id"] = initialization["model_id"]
        atomic_json(path, manifest)
        # A small durable provenance record survives bulky root-data retention.
        atomic_json(self.run_dir / "teacher_banks" / f"round_{number:06d}.json",
                    {key: manifest[key] for key in ("round", "protocol", "teacher_bank", "teacher_models", "teacher_id", "teacher_checkpoint", "learner_initialization")
                     if key in manifest})
        return manifest

    def cuda_teacher_batches(self, groups, manifest, on_result, directory, operation):
        """Keep one CUDA worker alive across the complete frozen teacher bank."""
        if not groups:
            return
        from .gpu_label_service import CudaGameService
        self.close_pool()
        sampling = operation == "sampling"
        batch_size = max(1, int(os.environ.get(
            "STELLA_GPU_SAMPLING_BATCH" if sampling else "STELLA_GPU_ROOT_BATCH",
            "1024" if sampling else "128")))

        def batches():
            for teacher_id, tasks in groups.items():
                paths = {"teacher": str(run_path(self.run_dir, manifest["teacher_models"][teacher_id]["checkpoint"]))}
                if sampling and manifest.get("behavior_checkpoint"):
                    paths["behavior"] = str(run_path(self.run_dir, manifest["behavior_checkpoint"]))
                model_paths = paths if sampling else paths["teacher"]
                for batch in cuda_task_batches(tasks, operation, batch_size):
                    yield teacher_id, batch, model_paths

        stream = iter(batches())
        first = next(stream)
        started = utc_now()
        generator = CudaGameService(first[2], self.run_dir, operation).start(self.control)
        self.publish(True, **{
            f"{operation}_backend_actual": "torch_cuda",
            f"{operation}_device": generator.information["device_name"],
            f"{operation}_decision_mode": "gpu",
            "sampling_simulation_backend" if sampling else "simulation_backend_actual": "cuda_games",
            "sampling_batch_families" if sampling else "label_batch_roots": batch_size,
            "resident_teacher_service": True, "resident_teacher_groups": len(groups), "workers": 1,
            "message": f"GPU常驻{'采样' if sampling else '标注'}：{len(groups)}个教师共用进程与CUDA缓存"})
        self.log(f"GPU常驻{'采样' if sampling else '标注'}：跨{len(groups)}个教师复用进程，批次提前提交")
        pending = deque()
        group_stats = {}
        previous = {}

        def submit(item):
            teacher_id, batch, model_paths = item
            generator.submit(batch, model_paths=model_paths)
            pending.append((teacher_id, batch))

        try:
            submit(first)
            if generator.pipeline:
                following = next(stream, None)
                if following is not None:
                    submit(following)
            while pending:
                self.control()
                teacher_id, batch = pending.popleft()
                rows = generator.finish(self.control)
                following = next(stream, None)
                if following is not None:
                    submit(following)
                stats = generator.snapshot()
                group = group_stats.setdefault(teacher_id, {"batches": 0, "tasks": 0})
                group["batches"] += 1
                group["tasks"] += len(batch)
                for key in ("games", "decisions", "action_rows", "seconds", "input_wait_seconds"):
                    group[key] = group.get(key, 0)+stats.get(key, 0)-previous.get(key, 0)
                previous = stats
                self.publish(resident_teacher_id=teacher_id,
                             resident_model_switches=stats.get("resident_model_switches", 0),
                             resident_model_switch_seconds=stats.get("model_switch_seconds", 0),
                             resident_input_wait_seconds=stats.get("input_wait_seconds", 0))
                for task, row in zip(batch, rows):
                    on_result(task, row)
        finally:
            generator.close()
            record = {"started_at": started, "ended_at": utc_now(), "resident_teacher_service": True,
                      "batch_size": batch_size, "teacher_groups": group_stats, **generator.snapshot()}
            atomic_json(directory / f"resident_{operation}_inference.json", record)
            append_jsonl(directory / f"resident_{operation}_sessions.jsonl", record)
            self.publish(True, workers=self.worker_count)

    def collect(self, manifest, directory):
        verify_frozen_models(manifest, self.run_dir)
        sample_dir, label_dir = directory / "sampling", directory / "labels"
        sample_dir.mkdir(exist_ok=True)
        label_dir.mkdir(exist_ok=True)
        pending = defaultdict(list)
        families = []
        for task in manifest["sampling_tasks"]:
            path = sample_dir / (task["family_id"]+".json.gz")
            if path.exists():
                families.append(read_gzip_json(path))
            else:
                pending[task["teacher_id"]].append(task)
        family_count = len(manifest["sampling_tasks"])
        self.stage("sampling", family_count, len(families), "按逐预设冻结教师分组采样完整轨迹")
        def sampled(task, family):
            family["teacher_id"] = task["teacher_id"]
            family["catalogue_id"] = task["catalogue_id"]
            write_gzip_json(sample_dir / (task["family_id"]+".json.gz"), family)
            families.append(family)
            self.progress(len(families), family_count)
        self.cuda_teacher_batches(pending, manifest, sampled, directory, "sampling")
        roots = sorted([root for family in families for root in family["roots"]], key=lambda r: r["root_id"])
        if manifest.get("sampling_policy"):
            sampling_summary = {
                "policy": manifest["sampling_policy"], "families": len(families), "roots": len(roots),
                "student_families": sum(f.get("behavior_source") == "best_candidate" for f in families),
                "teacher_families": sum(f.get("behavior_source") == "teacher" for f in families),
                "disagreement_roots": sum(bool(r.get("teacher_student_disagreement")) for r in roots),
                "disagreement_hard_roots": sum(bool(r.get("teacher_student_disagreement")) and
                                               bool(r.get("hard_sample_slot")) for r in roots)}
            atomic_json(directory / "sampling_summary.json", sampling_summary)
            self.publish(True, sampling_summary=sampling_summary)
        # The new deployment-proxy validation uses a fixed label quota. Keep
        # adaptive precision selection entirely within training roots.
        refinement_roots = ([root for root in roots if root["split"] == "train"]
                            if manifest.get("catalogue_coverage") else roots)
        self.refinement_tasks(refinement_roots, manifest, directory)
        pending = defaultdict(list)
        for root in roots:
            # Persist catalogue identity and teacher checksum with every label.
            index = catalogue_id(root["preset"]["pools"])
            root["catalogue_id"] = index
            root["teacher_sha256"] = manifest["teacher_models"][root["teacher_id"]]["sha256"]
            if frozen_workflow(manifest):
                root["student_sha256"] = manifest["teacher_sha256"]
                root["behavior_sha256"] = (manifest["teacher_sha256"] if root.get("behavior_source") == "best_candidate"
                                              else root["teacher_sha256"])
            if not (label_dir / (root["root_id"]+".npz")).exists():
                pending[root["teacher_id"]].append(root)
        completed = len(roots)-sum(map(len, pending.values()))
        self.stage("labeling", len(roots), completed, "逐预设冻结教师生成完整终局动作标签")
        already_saved = completed
        writer = LabelWriter(max_pending=2048)
        def labeled(task, row):
            nonlocal completed
            # Both CPU and CUDA must preserve this teacher routing provenance.
            row["metadata"].update(teacher_id=task["teacher_id"], catalogue_id=task["catalogue_id"],
                                   teacher_sha256=task["teacher_sha256"])
            for key in ("student_sha256", "behavior_sha256"):
                if key in task:
                    row["metadata"][key] = task[key]
            writer.submit(label_dir / (task["root_id"]+".npz"), row)
            completed += 1
            statistics = writer.snapshot()
            self.progress(already_saved+writer.saved, len(roots), samples=already_saved+writer.saved,
                          generated_labels=completed, **statistics)
        self.publish(True, label_background_writer=True)
        try:
            self.cuda_teacher_batches(pending, manifest, labeled, directory, "label")
        finally:
            writer.close()
            statistics = writer.snapshot()
            atomic_json(directory / "label_write_statistics.json", statistics)
            self.progress(already_saved+writer.saved, len(roots), samples=already_saved+writer.saved,
                          generated_labels=completed, **statistics)
        self.control()
        return len(roots)

    def replay(self, number, teacher_id):
        """Whole families, stable preset split, and only matching current teachers."""
        current = read_json(self.run_dir / "rounds" / f"round_{number:06d}" / "manifest.json")
        roots = []
        policy = frozen_workflow(current).get("replay", {})
        limit = policy.get("max_roots", self.config["sampling"]["replay_max_roots"])
        rounds = policy.get("rounds", self.config["sampling"]["replay_rounds"])
        for previous in range(number, max(0, number-rounds), -1):
            directory = self.run_dir / "rounds" / f"round_{previous:06d}"
            path = directory / "manifest.json"
            if not path.exists():
                continue
            manifest = read_json(path)
            if (manifest.get("protocol") != PROTOCOL or
                    manifest.get("rules_version") != sim.MODEL_VERSION):
                continue
            family_paths = defaultdict(list)
            for label_path in sorted((directory / "labels").glob("*.npz")):
                family_paths[label_path.stem.rsplit("_root", 1)[0]].append(label_path)
            for task in manifest["sampling_tasks"]:
                index = task["catalogue_id"]
                if (current.get("catalogue_coverage") and previous != number and
                        task["split"] == "val"):
                    # Do not dilute the new current-student/natural proxy with
                    # historical exploratory/hard validation trajectories.
                    continue
                if task["teacher_id"] != current["teacher_bank"][index]:
                    continue
                paths = family_paths.get(task["family_id"], [])
                if roots and len(roots)+len(paths) > limit:
                    return roots
                self.control()
                for path in paths:
                    row = load_label(path, task["teacher_id"])
                    if (row.get("family_id") != task["family_id"] or row.get("split") != task["split"]
                            or row.get("root_id") != path.stem):
                        raise ValueError("Replay label identity/split differs from its frozen source task")
                    for key, task_key in (("source_environment_seed", "environment_seed"),
                                          ("source_exploration_seed", "exploration_seed")):
                        if key in row and row[key] != task[task_key]:
                            raise ValueError("Replay trajectory provenance differs from frozen manifest")
                    if frozen_workflow(current) or current.get("catalogue_coverage"):
                        expected = current["teacher_models"][task["teacher_id"]]["sha256"]
                        if row.get("teacher_sha256") != expected:
                            raise ValueError("Replay label teacher checksum differs from frozen bank")
                    if catalogue_id(row["preset"]["pools"]) != index:
                        raise ValueError("Replay full original preset disagrees with its catalogue identity")
                    row["catalogue_id"] = index
                    row["preset_group"] = f"catalogue_{index:04d}"
                    # Old labels can remain in their original split. Recover
                    # provenance from their frozen source manifest, never infer
                    # a seed from a file name or reassign old validation data.
                    row["source_environment_seed"] = task["environment_seed"]
                    row["source_exploration_seed"] = task["exploration_seed"]
                    row["source_round"] = previous
                    roots.append(row)
        return roots

    def train(self, manifest, directory, candidate_path):
        verify_frozen_models(manifest, self.run_dir)
        fit_initial_model(manifest, self.run_dir)
        from .fit_cache import identity, prepare, validate_winner
        cache_identity = prepare(directory, identity(manifest, self.config["seed"]))
        path = directory / "fit.json"
        if candidate_path.exists() and path.exists():
            cached = read_json(path)
            validate_winner(candidate_path, cached, cache_identity)
            if frozen_workflow(manifest):
                if cached.get("workflow_digest") != digest(manifest["workflow"]) or cached.get("candidate_sha256") != file_hash(candidate_path):
                    raise ValueError("Cached winner differs from frozen paired workflow")
            return cached
        if manifest.get("evaluation_mode") == JOINT_EVALUATION:
            return self.train_official_candidates(manifest, directory, candidate_path)
        pool = self.fit_checkpoints(manifest, directory)
        candidates = pool["candidates"][:manifest["training_config"].get("checkpoint_candidates", 3)]
        scores = []
        best = None
        best_mean = -float("inf")
        presets = [self.presets[i] for i in manifest["development_indices"]]
        games = manifest.get("development_games_per_preset", 128)
        method = manifest.get("development_selection", "catalogue36_independent_development")
        for i, point in enumerate(candidates, 1):
            self.publish(True, checkpoint_candidate_index=i, checkpoint_candidates=len(candidates),
                         checkpoint_epoch=point["epoch"])
            result = self.evaluate_matrix(run_path(directory, point["checkpoint"]), presets,
                directory / "development" / f"epoch_{point['epoch']:03d}",
                f"独立开发{len(presets)}预设×{games}局选择训练时点 {i}/{len(candidates)}", games=games, development=True)
            mean = float(result.mean())
            scores.append({"epoch": point["epoch"], "candidate_mean": mean, "games": int(result.size)})
            if mean > best_mean:
                best, best_mean = point, mean
        QNetwork.load(run_path(directory, best["checkpoint"])).save(candidate_path,
            {"best_epoch": best["epoch"], "checkpoint_selection": method,
             "teacher_bank_digest": digest(manifest["teacher_bank"]), "protocol": PROTOCOL})
        summary = dict(pool["fit"], best_epoch=best["epoch"],
            checkpoint_selection={"method": method, "candidates": scores,
                                  "presets": len(presets), "games_per_preset": games,
                                  "promotion_games_used": False, "selected_epoch": best["epoch"]})
        if cache_identity is not None:
            summary.update(fit_cache_identity=cache_identity, candidate_sha256=file_hash(candidate_path))
        atomic_json(path, summary)
        return summary

    def fit_checkpoints(self, manifest, directory):
        if not frozen_workflow(manifest):
            return super().fit_checkpoints(manifest, directory)
        verify_frozen_models(manifest, self.run_dir)
        from .fit_cache import identity as calibration_identity, prepare
        cache_identity = prepare(directory, calibration_identity(manifest, self.config["seed"]))
        candidates, branch_fits = [], {}
        for branch in manifest["workflow"]["branches"]:
            tag = branch["tag"]
            branch_dir = directory / "branches" / tag
            branch_dir.mkdir(parents=True, exist_ok=True)
            branch_manifest = copy.deepcopy(manifest)
            branch_manifest["training_config"] = dict(branch["training_config"])
            branch_manifest["workflow_branch"] = tag
            identity = {"workflow": manifest["workflow"], "branch": tag,
                        "initial_sha256": manifest["teacher_sha256"],
                        "teacher_models": manifest["teacher_models"],
                        "teacher_bank": manifest["teacher_bank"],
                        "training_config": branch["training_config"],
                        "compute_config": manifest["compute_config"]}
            identity_path = branch_dir / "workflow_identity.json"
            if identity_path.exists():
                if read_json(identity_path) != identity:
                    raise ValueError("Cached branch fit differs from frozen workflow")
            else:
                if (branch_dir / "fit_candidates.json").exists():
                    raise ValueError("Unidentified cached branch fit")
                atomic_json(identity_path, identity)
            pool = super().fit_checkpoints(branch_manifest, branch_dir)
            branch_fits[tag] = pool["fit"]
            if len(pool["candidates"]) != 3:
                raise ValueError("Paired workflow requires three distinct actual checkpoints per branch")
            for point in pool["candidates"]:
                path = run_path(branch_dir, point["checkpoint"])
                if file_hash(path) != point["sha256"]:
                    raise ValueError("Cached branch checkpoint checksum changed")
                candidates.append(dict(point, branch=tag, checkpoint=str(path.relative_to(directory))))
        digests = {fit.get("frozen_label_input_sha256") for fit in branch_fits.values()}
        if len(digests) != 1 or None in digests:
            raise ValueError("Paired branches did not train on identical ordered frozen label inputs")
        initial_parameters = {fit.get("initial_parameter_sha256") for fit in branch_fits.values()}
        if len(initial_parameters) != 1 or None in initial_parameters:
            raise ValueError("Paired branches did not use identical initial model parameters")
        pool = {"fit": branch_fits["control"], "branch_fits": branch_fits,
                "candidates": candidates, "workflow_protocol": PAIRED_ANCHOR_V2}
        if cache_identity is not None:
            pool["fit_cache_identity"] = cache_identity
        atomic_json(directory / "fit_candidates.json", pool)
        return pool

    def train_official_candidates(self, manifest, directory, candidate_path):
        """One shared-seed assessment serves both single-model and preset selection."""
        pool = self.fit_checkpoints(manifest, directory)
        candidates = pool["candidates"][:manifest["evaluation_candidates"]]
        results, matrices = [], []
        # The six models use the exact same environment inputs. One session
        # retains their ~3 GiB compact CPU input cache and CUDA context, never their
        # predictions; all candidate games still execute independently.
        evaluation_session = (CatalogueEvaluationSession(self.presets,
            seed=frozen_seed(manifest, self.config, "evaluation"),
            games_per_preset=GAMES, run_dir=self.run_dir)
            if _enabled("STELLA_EVAL_RESIDENT") else nullcontext(None))
        with evaluation_session as session:
            for i, point in enumerate(candidates, 1):
                self.control()
                branch = point.get("branch")
                suffix = f"_B{branch}" if branch else ""
                model_id = f"M{manifest['round']:06d}{suffix}_E{point['epoch']:03d}"
                checkpoint = self.run_dir / "checkpoints" / (model_id+".npz")
                if checkpoint.exists() and (frozen_workflow(manifest) or manifest.get("catalogue_coverage")):
                    if QNetwork.load(checkpoint).checksum() != QNetwork.load(run_path(directory, point["checkpoint"])).checksum():
                        raise ValueError("Official candidate cache belongs to different branch weights")
                if not checkpoint.exists():
                    QNetwork.load(run_path(directory, point["checkpoint"])).save(checkpoint, {
                        "model_id": model_id, "best_epoch": point["epoch"],
                        "checkpoint_selection": JOINT_EVALUATION,
                        "teacher_bank_digest": digest(manifest["teacher_bank"]), "protocol": PROTOCOL,
                        **({"workflow_protocol": PAIRED_ANCHOR_V2, "branch": branch} if branch else {})})
                self.publish(True, checkpoint_candidate_index=i, checkpoint_candidates=len(candidates),
                             checkpoint_epoch=point["epoch"], evaluation_mode=JOINT_EVALUATION)
                branch_name = {"control": "对照分支", "action": "动作优势分支"}.get(branch, "")
                scores = self.evaluate_matrix(checkpoint, self.presets,
                    directory / "official_candidates" / (f"branch_{branch}" if branch else "") / f"epoch_{point['epoch']:03d}",
                    f"{branch_name}正式候选 {i}/{len(candidates)}：4,960预设×256局（epoch {point['epoch']}）", session=session, manifest=manifest)
                matrices.append(scores)
                results.append({"epoch": point["epoch"], "model_id": model_id,
                                "checkpoint": str(checkpoint.relative_to(self.run_dir)),
                                "candidate_mean": float(scores.mean()), "games": int(scores.size),
                                **({"branch": branch, "source_model_id": model_id,
                                    "source_checkpoint": str(checkpoint.relative_to(self.run_dir)), "scores_identity": {
                                    "model_sha256": file_hash(checkpoint),
                                    "seed": frozen_seed(manifest, self.config, "evaluation"),
                                    "seed_domain": "fixed_catalogue_environment_v1", "rules_version": sim.MODEL_VERSION,
                                    "preset_fingerprints_sha256": digest([getattr(p, "fingerprint", None) for p in self.presets])}} if branch else {})})
        workflow_comparison = (compare_paired_official(results, np.stack(matrices))
                               if frozen_workflow(manifest) else None)
        best_index = int(np.argmax([point["candidate_mean"] for point in results]))
        best = results[best_index]
        # Keep the usual round ID for the global winner; other checkpoints
        # have epoch suffixes and can independently remain in the teacher bank.
        QNetwork.load(run_path(self.run_dir, best["checkpoint"])).save(candidate_path, {
            "model_id": f"M{manifest['round']:06d}", "source_checkpoint_id": best["model_id"]})
        best.update(model_id=f"M{manifest['round']:06d}",
                    checkpoint=str(candidate_path.relative_to(self.run_dir)))
        _atomic_npz_mkdir(directory / "official_candidate_scores.npz", scores=np.stack(matrices))
        selected_fit = pool.get("branch_fits", {}).get(best.get("branch"), pool["fit"])
        summary = dict(selected_fit, best_epoch=best["epoch"], checkpoint_selection={
            "method": JOINT_EVALUATION, "candidates": results, "selected_index": best_index,
            "selected_epoch": best["epoch"], "presets": len(self.presets), "games_per_preset": GAMES,
            "promotion_games_used": True, "independent_post_selection_evaluation": False})
        if frozen_workflow(manifest):
            summary.update(workflow_comparison=workflow_comparison, workflow_protocol=PAIRED_ANCHOR_V2,
                           workflow_checkpoint_policy=manifest["workflow"]["checkpoint_policy"], selected_branch=best["branch"],
                           branch_fits=pool["branch_fits"], workflow_digest=digest(manifest["workflow"]),
                           candidate_sha256=file_hash(candidate_path))
        if pool.get("fit_cache_identity") is not None:
            summary.update(fit_cache_identity=pool["fit_cache_identity"],
                           candidate_sha256=file_hash(candidate_path))
        atomic_json(directory / "fit.json", summary)
        return summary

    def run_round(self, number):
        self.round_started = time.monotonic()
        directory = self.run_dir / "rounds" / f"round_{number:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.apply_next_round_config(number, directory)
        path = directory / "summary.json"
        self.publish(True, round=number, candidate_id=f"M{number:06d}", loss_history=[],
                     training_loss=None, validation_loss=None, samples=0,
                     champion_updated_count=0, distinct_teacher_count=len(set(self.state["teacher_bank"])))
        if path.exists():
            completed = read_json(path)
            if completed.get("rules_version") != sim.MODEL_VERSION:
                raise ValueError("已完成轮次属于旧规则，不能提交到当前规则")
            if "learner_initialization" in completed:
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    raise ValueError("Completed continuation summary lost its frozen manifest")
                frozen = read_json(manifest_path)
                if completed["learner_initialization"] != frozen.get("learner_initialization"):
                    raise ValueError("Completed summary has a different frozen learner initialization")
                fit_initial_model(frozen, self.run_dir)
            self.commit(completed)
            return
        manifest = self.prepare_manifest(number, directory)
        self.publish(True, round_budget=manifest["budget"],
                     evaluation_mode=manifest.get("evaluation_mode", "independent_development_then_formal"),
                     checkpoint_candidates=manifest["training_config"].get("checkpoint_candidates", 3),
                     preserve_early_checkpoints=manifest["training_config"].get("preserve_early_checkpoints", False),
                     advantage_confidence_weighting=manifest["training_config"].get("advantage_confidence_weighting", False))
        candidate = self.run_dir / "checkpoints" / f"M{number:06d}.npz"
        if not (candidate.exists() and (directory / "fit.json").exists()):
            self.collect(manifest, directory)
        metrics = self.train(manifest, directory, candidate)
        joint = manifest.get("evaluation_mode") == JOINT_EVALUATION
        candidate_models = {}
        candidate_results = []
        if joint:
            with np.load(directory / "official_candidate_scores.npz", allow_pickle=False) as data:
                all_scores = data["scores"]
            candidates = metrics["checkpoint_selection"]["candidates"]
            scores = all_scores[metrics["checkpoint_selection"]["selected_index"]]
            winners = all_scores.sum(axis=2, dtype=np.int64).argmax(axis=0)
            champion_scores = all_scores[winners, np.arange(len(self.presets))]
            champion_ids = [candidates[int(index)]["model_id"] for index in winners]
            candidate_models = {point["model_id"]: point["checkpoint"] for point in candidates}
            evaluated_games = int(all_scores.size)
        else:
            scores = self.evaluate_matrix(candidate, self.presets, directory / "catalogue_evaluation",
                                          "正式评测全部4,960预设×256局")
            champion_scores = scores
            champion_ids = [f"M{number:06d}"] * len(self.presets)
            evaluated_games = int(scores.size)
        score_path = run_path(self.run_dir, manifest["teacher_scores"])
        if (manifest.get("teacher_scores_rules_version", manifest["rules_version"]) != sim.MODEL_VERSION or
                (manifest.get("teacher_scores_sha256") and file_hash(score_path) != manifest["teacher_scores_sha256"])):
            raise ValueError("教师评测矩阵与本轮冻结规则或校验值不同")
        with np.load(score_path, allow_pickle=False) as data:
            teacher = data["scores"].copy()
        selection = select_champions(teacher, scores)
        champion_selection = select_champions(teacher, champion_scores) if joint else selection
        previous_teacher_means = teacher.mean(axis=1)
        updated = champion_selection["promote"]
        bank = list(manifest["teacher_bank"])
        for index in np.flatnonzero(updated):
            bank[int(index)] = champion_ids[int(index)]
        teacher[updated] = champion_scores[updated]
        if joint:
            candidate_results = [dict(point, champion_updated_count=int(np.sum(updated & (winners == i))))
                                 for i, point in enumerate(candidates)]
        score_path = self.run_dir / "teacher_scores" / f"round_{number:06d}.npz"
        _atomic_npz_mkdir(score_path, scores=teacher)
        selection_arrays = {key: value for key, value in champion_selection.items() if isinstance(value, np.ndarray)}
        if joint:
            selection_arrays["winning_checkpoint_index"] = winners
        _atomic_npz_mkdir(directory / "selection.npz", **selection_arrays)
        mean = selection["candidate_mean"]
        best_mean = self.state["best_candidate_mean"]
        best = mean > best_mean
        best_id = f"M{number:06d}" if best else self.state["teacher_id"]
        best_metrics = next((item for item in metrics.get("history", []) if item.get("epoch") == metrics["best_epoch"]), {})
        preset_results = [{"catalogue_id": i, "name": preset.name, "fingerprint": preset.fingerprint,
            "cap": preset.total, "candidate_mean": float(champion_scores[i].mean()),
            "candidate_id": champion_ids[i], "single_candidate_mean": float(scores[i].mean()),
            "teacher_mean": float(previous_teacher_means[i]), "delta": float(champion_selection["mean_difference"][i]),
            "ci_low": float(champion_selection["ci95_low"][i]), "ci_high": float(champion_selection["ci95_high"][i]),
            "n": GAMES, "updated": bool(updated[i]), "previous_teacher_id": manifest["teacher_bank"][i],
            "teacher_id": bank[i]} for i, preset in enumerate(self.presets)]
        atomic_json(directory / "preset_results.json", {"round": number,
                    "rules_version": sim.MODEL_VERSION, "presets": preset_results})
        summary = {"protocol": PROTOCOL, "round": number, "teacher_id": manifest["teacher_id"],
            "candidate_id": f"M{number:06d}", "candidate_checkpoint": str(candidate.relative_to(self.run_dir)),
            "candidate_mean": mean, "best_candidate_mean": mean if best else best_mean,
            "evaluation_mode": manifest.get("evaluation_mode", "independent_development_then_formal"),
            "candidate_models": candidate_models, "checkpoint_results": candidate_results,
            "assessed_checkpoints": len(candidate_results) if joint else 1,
            "best_model_id": best_id, "promoted": best, "best_single_updated": best,
            "composite_teacher_mean": float(teacher.mean()), "previous_composite_teacher_mean": selection["teacher_mean"],
            "primary_delta": selection["equal_preset_mean_difference"],
            "champion_updated_count": int(updated.sum()), "distinct_teacher_count": len(set(bank)),
            "teacher_age_sample": teacher_age_sample(bank),
            "teacher_bank": bank, "teacher_scores": str(score_path.relative_to(self.run_dir)),
            "best_epoch": metrics["best_epoch"], "fit": metrics, "rules_version": sim.MODEL_VERSION,
            "training_loss": best_metrics.get("train", {}).get("selection_loss"),
            "validation_loss": best_metrics.get("val", best_metrics.get("validation", {})).get("selection_loss"),
            "evaluation_completed_presets": len(self.presets), "evaluation_total_presets": len(self.presets),
            "evaluation_games_completed": evaluated_games, "evaluation_games_total": evaluated_games,
            "elapsed_seconds": self.elapsed(), "round_seconds": time.monotonic()-self.round_started,
            "completed_at": utc_now(), "selection_required_sum_gain": 13,
            "ci_is_gate": False, "audit_presets": []}
        if "learner_initialization" in manifest:
            summary["learner_initialization"] = dict(manifest["learner_initialization"])
        atomic_json(path, summary)
        self.commit(summary)

    def commit(self, summary):
        number = summary["round"]
        history = {key: value for key, value in summary.items() if key not in ("fit", "teacher_bank")}
        upsert_history(self.run_dir / "history.jsonl", history, self.config["retention"]["history_rows"])
        models = dict(self.state["models"])
        models.update(summary.get("candidate_models", {}))
        models[summary["candidate_id"]] = summary["candidate_checkpoint"]
        best_id = summary["best_model_id"]
        protected = set(summary["teacher_bank"]) | {best_id, summary["candidate_id"], "M000000"}
        self.state.update(next_round=number+1, teacher_bank=summary["teacher_bank"],
            teacher_scores=summary["teacher_scores"], teacher_scores_rules_version=sim.MODEL_VERSION,
            teacher_id=best_id, teacher_checkpoint=models[best_id],
            best_candidate_mean=summary["best_candidate_mean"],
            composite_teacher_mean=summary["composite_teacher_mean"],
            models={key: value for key, value in models.items() if key in protected},
            last_candidate_checkpoint=summary["candidate_checkpoint"], elapsed_seconds=self.elapsed())
        atomic_json(self.state_path, self.state)
        atomic_json(self.run_dir / "best_model.json", {"model_id": best_id,
            "checkpoint": self.state["teacher_checkpoint"], "candidate_mean": summary["best_candidate_mean"],
            "last_assessed_round": number, "protocol": PROTOCOL})
        atomic_json(self.run_dir / "champion_bank.json", {"round": number, "protocol": PROTOCOL,
            "teacher_bank": summary["teacher_bank"], "models": self.state["models"],
            "teacher_scores": summary["teacher_scores"], "composite_teacher_mean": summary["composite_teacher_mean"]})
        result_path = self.run_dir / "rounds" / f"round_{number:06d}" / "preset_results.json"
        if result_path.exists():
            atomic_json(self.run_dir / "preset_results.json", read_json(result_path))
        self.publish(True, **{key: summary[key] for key in ("candidate_mean", "best_candidate_mean",
            "composite_teacher_mean", "champion_updated_count", "distinct_teacher_count", "primary_delta")},
            status="promoting", phase="promoting", phase_progress=1.0,
            preset_results_path="preset_results.json",
            message=f"本轮完成：{summary['champion_updated_count']}个预设更新教师；继续下一轮")
        self.log(f"第{number}轮：候选{summary['candidate_mean']:.5f}，最佳单模型{summary['best_candidate_mean']:.5f}，"
                 f"组合教师{summary['composite_teacher_mean']:.5f}，更新{summary['champion_updated_count']}个预设")
        self.prune()

    def prune(self):
        from .retention import retention_plan, apply_retention
        # Completed rounds can be removed on Linux without a desktop trash
        # service. All live bank and retained manifest references are protected.
        plan = retention_plan(self.run_dir, self.state,
            keep_rounds=max(self.config["retention"]["rounds"],
                            self.config["sampling"]["replay_rounds"] + 1),
            keep_checkpoints=self.config["retention"]["checkpoints"])
        report = apply_retention(plan)
        atomic_json(self.run_dir / "retention_last.json", report)


def _atomic_npz_mkdir(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(path, **arrays)
