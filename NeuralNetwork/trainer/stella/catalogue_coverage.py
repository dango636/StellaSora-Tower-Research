"""Independent train/development trajectories for every catalogue preset.

This module only expands a *complete frozen* per-preset sampling manifest. It
does not choose teachers, modify environment rules, or promote checkpoints.
Training preserves the existing behavior mixture; development defaults to the
current student's unperturbed trajectories. The caller must use ``root_selection_schedule`` when
selecting/serializing roots; the legacy ``ordinal % 4`` sampler is insufficient
for the default three-training/one-development-root allocation.

The same preset in both splits is intentional: the evaluation target is the
fixed 4,960-preset population. Families, environment draws, exploration draws,
root selection and rollout-label seeds remain disjoint. The fitter must still
normalize each split by preset, not by its number of action rows.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from typing import Mapping, Sequence

from .preset_catalogue import CATALOGUE_SIZE, catalogue_id
from .presets import Preset, canonical_seed


COVERAGE_PROTOCOL = "catalogue4960_independent_trajectories_v1"
ROOT_ROLES = ("natural", "phase", "hard")
SEED_FIELDS = ("environment_seed", "exploration_seed", "selection_seed", "label_seed")
_ALLOWED_CONFIG = {
    "train_roots", "val_roots", "train_rollouts_per_action",
    "val_rollouts_per_action", "train_hard_rollouts_per_action",
    "val_hard_rollouts_per_action", "validation_on_policy", "validation_root_mode",
}


def _integer(value, name, *, minimum=0):
    # bool is an int subclass, but accepting it as a sampling quota is a typo.
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _quota(tasks, split, field, config, setting):
    if setting in config:
        return _integer(config[setting], setting, minimum=2)
    values = {task[field] for task in tasks if task["split"] == split and field in task}
    if len(values) != 1:
        raise ValueError(f"Cannot infer one {setting}; pass it explicitly")
    return _integer(values.pop(), setting, minimum=2)


def _schedule(count, preset_id, round_number, split, validation_root_mode):
    # Every training family with >=3 roots has one of each kind. A short
    # Natural development samples estimate action calibration on the student's
    # visited states. Rotating development is an explicit diagnostic alternative,
    # never an accidental consequence of a short root quota.
    if split == "val" and validation_root_mode == "natural":
        return ["natural"] * count
    start = 0 if split == "train" and count >= 3 else (preset_id + round_number - 1) % 3
    return [ROOT_ROLES[(start + i) % len(ROOT_ROLES)] for i in range(count)]


def transform_sampling_tasks(tasks, seed, round_number, config, bank):
    """Return independent train/val tasks and an auditable coverage summary.

    ``tasks`` must contain exactly one original family for each catalogue ID
    0..4959 and the teacher selected in ``bank`` for that ID. ``config`` accepts
    ``train_roots``/``val_roots`` (default 3/1), plus optional split-specific
    ``*_rollouts_per_action`` and ``*_hard_rollouts_per_action``. Validation uses
    ``validation_on_policy=True`` and ``validation_root_mode='natural'`` by
    default. Set the former False to retain the source behavior mixture, or use
    ``validation_root_mode='rotating'`` for phase/hard diagnostic coverage.
    These change validation state sampling, never the frozen label teacher.
    Omitted rollout
    quotas are inferred from the original manifest's corresponding split, so
    expanding an old validation preset does not give its training copy a larger
    rollout budget than other training presets.

    Input objects are never mutated. Output order follows input scheduling
    order, train then val, but family IDs and all statistical seeds depend only
    on base seed, round, original preset fingerprint, split, and purpose. Seeds
    never contain teacher/model IDs or the input's previous split assignment.
    """
    _integer(seed, "seed")
    _integer(round_number, "round_number", minimum=1)
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("coverage config must be a mapping")
    unknown = set(config) - _ALLOWED_CONFIG
    if unknown:
        raise ValueError(f"Unknown coverage settings: {sorted(unknown)}")
    counts = {
        "train": _integer(config.get("train_roots", 3), "train_roots", minimum=1),
        "val": _integer(config.get("val_roots", 1), "val_roots", minimum=1),
    }
    validation_on_policy = config.get("validation_on_policy", True)
    if type(validation_on_policy) is not bool:
        raise ValueError("validation_on_policy must be boolean")
    validation_root_mode = config.get("validation_root_mode", "natural")
    if validation_root_mode not in ("natural", "rotating"):
        raise ValueError("validation_root_mode must be 'natural' or 'rotating'")
    if not isinstance(bank, Sequence) or isinstance(bank, (str, bytes)) or len(bank) != CATALOGUE_SIZE:
        raise ValueError(f"A complete {CATALOGUE_SIZE}-preset teacher bank is required")
    if any(not isinstance(teacher, str) or not teacher for teacher in bank):
        raise ValueError("Teacher bank entries must be nonempty model IDs")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or len(tasks) != CATALOGUE_SIZE:
        raise ValueError(f"Expected one source family for each of {CATALOGUE_SIZE} presets")

    source_ids, source_families, source_ordinals = set(), set(), set()
    original_seeds = set()
    validated = []
    rules = set()
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("Every source task must be a mapping")
        required = {"catalogue_id", "family_id", "ordinal", "teacher_id", "preset", "split",
                    "roots_per_family", "rollouts_per_action", "rules_version", "behavior_source", "behavior_model_id",
                    "student_model_id", "epsilon", *SEED_FIELDS}
        missing = required - set(task)
        if missing:
            raise ValueError(f"Incomplete source task: {sorted(missing)}")
        index = _integer(task["catalogue_id"], "catalogue_id")
        if index >= CATALOGUE_SIZE or index in source_ids:
            raise ValueError("Catalogue IDs must cover 0..4959 exactly once")
        source_ids.add(index)
        if task["teacher_id"] != bank[index]:
            raise ValueError(f"Teacher routing differs from frozen bank at preset {index}")
        family = task["family_id"]
        if not isinstance(family, str) or not family or "_root" in family or family in source_families:
            raise ValueError("Source family IDs must be unique and cannot contain '_root'")
        source_families.add(family)
        ordinal = _integer(task["ordinal"], "ordinal")
        if ordinal in source_ordinals:
            raise ValueError("Source task ordinals must be unique")
        source_ordinals.add(ordinal)
        _integer(task["roots_per_family"], "roots_per_family", minimum=1)
        _integer(task["rollouts_per_action"], "rollouts_per_action", minimum=2)
        if task.get("hard_sampling"):
            if "hard_rollouts_per_action" not in task:
                raise ValueError("Hard-sampling source task lacks hard_rollouts_per_action")
            _integer(task["hard_rollouts_per_action"], "hard_rollouts_per_action", minimum=2)
        if task["split"] not in ("train", "val"):
            raise ValueError("Source splits must be explicit train/val")
        if task["behavior_source"] not in ("teacher", "best_candidate", "last_candidate"):
            raise ValueError("Unknown source behavior policy")
        for field in ("behavior_model_id", "student_model_id", "rules_version"):
            if not isinstance(task[field], str) or not task[field]:
                raise ValueError(f"{field} must be a nonempty string")
        if task["behavior_source"] == "teacher" and task["behavior_model_id"] != task["teacher_id"]:
            raise ValueError("Teacher behavior task has a different behavior_model_id")
        epsilon = task["epsilon"]
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be a finite number in [0,1]")
        rules.add(task["rules_version"])
        try:
            preset = Preset.from_dict(task["preset"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Source task contains an invalid full preset") from exc
        if catalogue_id(preset.pools) != index:
            raise ValueError(f"Source preset does not match catalogue_id {index}")
        for field in SEED_FIELDS:
            original_seeds.add(_integer(task[field], field))
        if "root_id" in task or "state" in task:
            raise ValueError("Coverage expansion takes family tasks, not already sampled roots")
        validated.append((task, preset))
    if source_ids != set(range(CATALOGUE_SIZE)) or len(rules) != 1:
        raise ValueError("A complete single-rules-version catalogue is required")

    quotas = {}
    if any(type(task.get("hard_sampling", False)) is not bool for task in tasks):
        raise ValueError("hard_sampling must be boolean")
    hard_enabled_values = {task.get("hard_sampling", False) for task in tasks}
    if len(hard_enabled_values) != 1:
        raise ValueError("Source tasks must agree on hard_sampling")
    hard_enabled = next(iter(hard_enabled_values))
    for split in ("train", "val"):
        base = _quota(tasks, split, "rollouts_per_action", config, f"{split}_rollouts_per_action")
        hard = (_quota(tasks, split, "hard_rollouts_per_action", config,
                       f"{split}_hard_rollouts_per_action") if hard_enabled else base)
        if hard < base:
            raise ValueError("Hard-root rollout quota cannot be below the base quota")
        quotas[split] = {"rollouts_per_action": base, "hard_rollouts_per_action": hard}

    output, generated_seeds, family_ids = [], set(), set()
    role_counts = {split: Counter() for split in counts}
    source_counts = {split: Counter() for split in counts}
    for source, preset in validated:
        index = source["catalogue_id"]
        for split in ("train", "val"):
            family_id = f"r{round_number:06d}_f{index:05d}_{split}"
            if family_id in family_ids:
                raise ValueError("Generated family ID collision")
            family_ids.add(family_id)
            task = deepcopy(dict(source))
            schedule = _schedule(counts[split], index, round_number, split, validation_root_mode)
            task.update(
                family_id=family_id, ordinal=len(output), split=split,
                roots_per_family=counts[split],
                root_selection_schedule=schedule,
                # Every preset rotates through all four phases and three floor
                # bands over twelve rounds; val is offset from its train copy.
                phase_sampling_index=(index + round_number - 1 + (6 if split == "val" else 0)) % 12,
                balanced_phase_sampling=True,
                coverage_protocol=COVERAGE_PROTOCOL,
                coverage_source_family_id=source["family_id"],
                coverage_source_ordinal=source["ordinal"],
                **quotas[split],
            )
            if split == "val" and validation_on_policy:
                task.update(behavior_source="best_candidate",
                            behavior_model_id=source["student_model_id"], epsilon=0.0)
            for field in SEED_FIELDS:
                value = canonical_seed(COVERAGE_PROTOCOL, seed, round_number,
                                       preset.fingerprint, split, field)
                if value in generated_seeds or value in original_seeds:
                    raise ValueError("Sampling seed collision; refusing ambiguous statistical streams")
                generated_seeds.add(value)
                task[field] = value
            if task["teacher_id"] != bank[index]:
                raise AssertionError("Coverage expansion changed a frozen teacher")
            role_counts[split].update(schedule)
            source_counts[split][task["behavior_source"]] += 1
            output.append(task)

    original_roots = sum(task["roots_per_family"] for task in tasks)
    new_roots = CATALOGUE_SIZE * sum(counts.values())
    bank_digest = hashlib.sha256(json.dumps(list(bank), separators=(",", ":")).encode()).hexdigest()
    summary = {
        "protocol": COVERAGE_PROTOCOL,
        "source_families": len(tasks), "families": len(output),
        "presets": CATALOGUE_SIZE, "train_presets": CATALOGUE_SIZE, "val_presets": CATALOGUE_SIZE,
        "train_families": CATALOGUE_SIZE, "val_families": CATALOGUE_SIZE,
        "train_roots_per_preset": counts["train"], "val_roots_per_preset": counts["val"],
        "train_roots": CATALOGUE_SIZE * counts["train"],
        "val_roots": CATALOGUE_SIZE * counts["val"], "roots": new_roots,
        "original_roots": original_roots, "root_count_ratio": new_roots / original_roots,
        "split_unit": "independent_full_source_trajectory",
        "same_presets_in_both_splits": True, "shared_families": 0,
        "seed_namespace": COVERAGE_PROTOCOL, "distinct_sampling_seeds": len(generated_seeds),
        "sampling_seed_collisions": 0,
        "teacher_routes_unchanged": True, "teacher_bank_sha256": bank_digest,
        "label_teacher_routes_unchanged": True,
        "training_behavior_preserved": True,
        "validation_on_policy": validation_on_policy,
        "validation_root_mode": validation_root_mode,
        "validation_behavior_changed_families": sum(
            task["behavior_source"] != source["behavior_source"]
            or task["behavior_model_id"] != source["behavior_model_id"]
            or task["epsilon"] != source["epsilon"]
            for task, (source, _) in zip(output[1::2], validated)),
        "root_role_counts": {key: dict(value) for key, value in role_counts.items()},
        "behavior_source_counts": {key: dict(value) for key, value in source_counts.items()},
        "rollout_quotas": quotas,
        "required_fitter_weighting": "equal preset mass within each split",
        "required_sampler_support": "root_selection_schedule",
    }
    return output, summary
