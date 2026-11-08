"""Frozen, bounded budgets; three equal target-total strata."""
from __future__ import annotations

import copy
import os

VERSION = "stella_continuous_qpi_v1"
DEFAULT = {
    "version": VERSION,
    "seed": 202610267701,
    "workers": 0,
    "compute": {"backend": "auto", "auto_tune_workers": True,
                "gpu_batch_roots": 128, "gpu_min_steps_per_epoch": 8,
                "gpu_amp": False, "gpu_tf32": False, "gpu_memory_limit_mb": 1024,
                "gpu_callback_interval_seconds": 0.25},
    "sampling": {"families": 96, "roots_per_family": 4,
                 "max_families": 240, "rollouts_per_action": 4,
                 "max_rollouts_per_action": 16, "replay_rounds": 3,
                 "replay_max_roots": 2400},
    "training": {"epochs": 16, "batch_roots": 24, "learning_rate": 0.0008,
                 "advantage_weight": 0.75, "l2": 0.00001,
                 "huber_delta": 2.0, "gradient_clip": 5.0,
                 "min_epochs": 5, "patience": 5, "checkpoint_metric": "advantage_rmse"},
    "evaluation": {"presets_per_bucket": 4, "games_per_preset": 32,
                   "max_games_per_preset": 128, "audit_presets_per_bucket": 2,
                   "audit_games_per_preset": 32, "bootstrap_replicates": 1000},
    "retention": {"rounds": 8, "checkpoints": 12, "history_rows": 2000,
                  "event_rows": 5000},
}


def _merge(target, source):
    for key, value in source.items():
        if key not in target:
            raise ValueError(f"未知配置项: {key}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"配置 {key} 必须是对象")
            _merge(target[key], value)
        else:
            target[key] = value


def resolve(raw=None, smoke=False):
    raw = raw or {}
    value = copy.deepcopy(DEFAULT)
    if "catalogue_mode" in raw:
        value["catalogue_mode"] = False
    # Optional additions keep legacy resolved configs/hashes unchanged.
    if "workflow" in raw:
        value["workflow"] = {"protocol": "paired_anchor_v2"}
    if "catalogue_coverage" in raw:
        value["catalogue_coverage"] = {"train_roots":3,"val_roots":1,
            "train_rollouts_per_action":256,"val_rollouts_per_action":512,
            "train_hard_rollouts_per_action":512,"val_hard_rollouts_per_action":1024,
            "validation_on_policy":True,"validation_root_mode":"natural"}
    if "model" in raw:
        value["model"] = {"hidden": [128, 96], "initialization": "packaged"}
        if "initial_checkpoint" in raw["model"]:
            value["model"]["initial_checkpoint"] = ""
    for name, default in (("validation_rollouts_multiplier", 1), ("balanced_phase_sampling", False),
                          ("hard_sampling", False), ("hard_rollouts_per_action", 512),
                          ("hard_validation_rollouts_per_action", 1024), ("hard_mining_roots", 2048),
                          ("refinement_roots", 384), ("refinement_rollouts_per_action", 1024),
                          ("refinement_validation_rollouts_per_action", 2048),
                          ("refinement_mode", "legacy")):
        if name in raw.get("sampling", {}):
            value["sampling"][name] = default
    for name, default in (("mistake_weight_strength", 0.0), ("mistake_weight_scale", .25),
                          ("mistake_weight_max", 3.0), ("ranking_weight", 0.0),
                          ("ranking_min_gap", .03), ("ranking_confidence_z", 1.5),
                          ("ranking_top_k", 3), ("ranking_temperature", .1),
                          ("ranking_critical_boost", 2.0), ("checkpoint_candidates", 1),
                          ("preserve_early_checkpoints", False),
                          ("prefer_first_four_checkpoints", False),
                          ("advantage_confidence_weighting", False),
                          ("confidence_floor", .25), ("confidence_z", 1.5),
                          ("q_anchor_weight", None), ("objective_version", "q_advantage_ranking_v1"),
                          ("refresh_student_actions",False), ("transfer_diagnostics",False),
                          ("validation_unit","preset"), ("calibration_require_all_presets",True),
                          ("calibration_visits_per_preset",0),
                          ("calibration_sampling","replacement"),
                          ("calibration_coverage_priority_correction",False),
                          ("calibration_stopping_metric","checkpoint_metric"),
                          ("learner_initialization","best"),
                          ("calibration_current_fraction",0.5)):
        if name in raw.get("training", {}):
            value["training"][name] = default
    if "calibration" in raw.get("training",{}):
        from dataclasses import asdict
        from .action_calibration import CalibrationConfig
        value["training"]["calibration"] = asdict(CalibrationConfig())
        # A new opt-in loss setting must not alter old resolved manifests or
        # their hashes merely because the dataclass acquired a default field.
        if (not isinstance(raw["training"]["calibration"], dict)
                or "reliability_floor" not in raw["training"]["calibration"]):
            del value["training"]["calibration"]["reliability_floor"]
    for name, default in (("development_presets_per_bucket", 4),
                          ("development_games_per_preset", 128)):
        if name in raw.get("evaluation", {}):
            value["evaluation"][name] = default
    _merge(value, raw)
    if "catalogue_coverage" in value:
        coverage = value["catalogue_coverage"]
        if not value.get("catalogue_mode"):
            raise ValueError("catalogue_coverage requires the fixed catalogue mode")
        if "workflow" in value:
            raise ValueError("Enable catalogue_coverage separately from the legacy paired_anchor_v2 workflow")
        for key in ("train_roots","val_roots","train_rollouts_per_action","val_rollouts_per_action",
                    "train_hard_rollouts_per_action","val_hard_rollouts_per_action"):
            if type(coverage[key]) is not int or coverage[key] < (1 if key.endswith("roots") else 2):
                raise ValueError(f"Invalid catalogue_coverage.{key}")
        if type(coverage["validation_on_policy"]) is not bool or coverage["validation_root_mode"] not in ("natural","rotating"):
            raise ValueError("Invalid independent validation behavior or root mode")
        needed = 4960*(coverage["train_roots"]+coverage["val_roots"])
        if value["sampling"]["replay_max_roots"] < needed:
            raise ValueError("Replay capacity must fit the entire new all-preset round")
    if "calibration_sampling" in value["training"] and value["training"]["calibration_sampling"] not in (
            "coverage_without_replacement", "stratified_resampling", "replacement"):
        raise ValueError("Invalid training.calibration_sampling")
    if ("calibration_coverage_priority_correction" in value["training"] and
            type(value["training"]["calibration_coverage_priority_correction"]) is not bool):
        raise ValueError("training.calibration_coverage_priority_correction must be boolean")
    if value["training"].get("calibration_stopping_metric", "checkpoint_metric") not in (
            "checkpoint_metric", "advantage_rmse", "checkpoint_or_advantage_rmse"):
        raise ValueError("training.calibration_stopping_metric must be checkpoint_metric, advantage_rmse or checkpoint_or_advantage_rmse")
    if "calibration_current_fraction" in value["training"]:
        import math
        fraction = value["training"]["calibration_current_fraction"]
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError("training.calibration_current_fraction must be finite in [0,1]")
    if "calibration" in value["training"]:
        from .action_calibration import CalibrationConfig
        loss_config = CalibrationConfig(**value["training"]["calibration"])
        if loss_config.target_scale != 20.0:
            raise ValueError("The portable checkpoint scale is fixed at 20")
    if "workflow" in value and value["workflow"]["protocol"] != "paired_anchor_v2":
        raise ValueError("workflow.protocol must be paired_anchor_v2")
    if "workflow" in value and not value.get("catalogue_mode"):
        raise ValueError("paired_anchor_v2 requires catalogue_mode")
    initialization = value["training"].get("learner_initialization", "best")
    if initialization not in ("best", "last_candidate"):
        raise ValueError("training.learner_initialization must be best or last_candidate")
    if initialization == "last_candidate" and (not value.get("catalogue_mode") or "workflow" in value):
        raise ValueError("last_candidate learner initialization requires catalogue_mode without legacy workflow")
    if value["version"] != VERSION:
        raise ValueError("配置版本与程序不符")
    if type(value["workers"]) is not int or value["workers"] < 0:
        raise ValueError("workers 须为非负整数；0 自动选择")
    if value["workers"] == 0:
        value["workers"] = min(8, max(1, (os.cpu_count() or 2) - 1))
    else:
        value["compute"]["auto_tune_workers"] = False
    if value["compute"]["backend"] not in ("auto", "numpy", "cuda"):
        raise ValueError("compute.backend 必须为 auto、numpy 或 cuda")
    if "model" in value:
        model = value["model"]
        if (not isinstance(model["hidden"], list) or not 1 <= len(model["hidden"]) <= 6 or
                any(type(w) is not int or not 1 <= w <= 4096 for w in model["hidden"])):
            raise ValueError("model.hidden 须为1–6层、每层1–4096个单元")
        if model["initialization"] not in ("fresh", "packaged", "checkpoint"):
            raise ValueError("model.initialization 必须为 fresh、packaged 或 checkpoint")
        if model["initialization"] == "checkpoint":
            if not isinstance(model.get("initial_checkpoint"), str) or not model["initial_checkpoint"].strip():
                raise ValueError("checkpoint 初始化需要 model.initial_checkpoint 路径")
        elif "initial_checkpoint" in model:
            raise ValueError("initial_checkpoint 仅用于 checkpoint 初始化")
    if smoke:
        value["workers"] = min(2, value["workers"])
        value["compute"]["auto_tune_workers"] = False
        value["sampling"].update(families=12, roots_per_family=2, max_families=12,
                                  rollouts_per_action=2, max_rollouts_per_action=2,
                                  replay_max_roots=100)
        value["training"].update(epochs=3, min_epochs=2, patience=3, batch_roots=6)
        value["evaluation"].update(presets_per_bucket=1, games_per_preset=3,
                                   max_games_per_preset=3, audit_presets_per_bucket=1,
                                   audit_games_per_preset=3, bootstrap_replicates=200)
    for section in ("sampling", "evaluation", "retention"):
        for key, val in value[section].items():
            if section == "sampling" and key == "refinement_mode":
                if val not in ("legacy", "critical_conflicts_v2"):
                    raise ValueError("sampling.refinement_mode 必须为 legacy 或 critical_conflicts_v2")
                continue
            if section == "sampling" and key in ("balanced_phase_sampling", "hard_sampling"):
                if type(val) is not bool:
                    raise ValueError(f"sampling.{key} 须为布尔值")
                continue
            if type(val) is not int or val < 1:
                raise ValueError(f"{section}.{key} 须为正整数")
    if value["sampling"]["families"] < 12:
        raise ValueError("families 至少12，保证训练/开发均覆盖三个区间")
    if value["sampling"]["max_families"] < value["sampling"]["families"]:
        raise ValueError("max_families 不能小于 families")
    if value["sampling"]["rollouts_per_action"] < 2:
        raise ValueError("每动作至少2次独立完整续局")
    if value["sampling"]["max_rollouts_per_action"] < value["sampling"]["rollouts_per_action"]:
        raise ValueError("max_rollouts_per_action 太小")
    if value["evaluation"]["games_per_preset"] < 2:
        raise ValueError("每预设至少2局配对评测")
    if value["evaluation"]["max_games_per_preset"] < value["evaluation"]["games_per_preset"]:
        raise ValueError("max_games_per_preset 不能小于 games_per_preset")
    if value["sampling"]["replay_max_roots"] < value["sampling"]["max_families"] * value["sampling"]["roots_per_family"]:
        raise ValueError("replay_max_roots 必须容纳完整的新一轮根样本")
    value["smoke"] = bool(smoke)
    return value


def round_budget(config, round_number, rejected):
    """More evidence after stalls, with explicit memory/compute ceilings."""
    s, e = config["sampling"], config["evaluation"]
    growth = min(2.5, 1.0 + (round_number - 1) // 8 * .25 + rejected // 3 * .5)
    families = min(s["max_families"], int(s["families"] * growth))
    families = max(12, families // 3 * 3)
    precision = 2 ** min(2, (round_number - 1) // 12 + rejected // 4)
    budget = {"families": families, "roots_per_family": s["roots_per_family"],
            "rollouts_per_action": min(s["max_rollouts_per_action"], s["rollouts_per_action"] * precision),
            "games_per_preset": min(e["max_games_per_preset"],
                                     e["games_per_preset"] * (2 ** min(2, (round_number-1)//12))),
            "presets_per_bucket": e["presets_per_bucket"]}
    if "validation_rollouts_multiplier" in s:
        budget["validation_rollouts_per_action"] = budget["rollouts_per_action"] * s["validation_rollouts_multiplier"]
    if s.get("hard_sampling"):
        budget.update(hard_roots_per_family=s["roots_per_family"]//4,
                      hard_rollouts_per_action=s["hard_rollouts_per_action"],
                      hard_validation_rollouts_per_action=s["hard_validation_rollouts_per_action"])
    if s.get("refinement_roots"):
        budget.update(refinement_roots=s["refinement_roots"],
                      refinement_rollouts_per_action=s["refinement_rollouts_per_action"],
                      refinement_validation_rollouts_per_action=s["refinement_validation_rollouts_per_action"])
        if "refinement_mode" in s:
            budget["refinement_mode"] = s["refinement_mode"]
    return budget
