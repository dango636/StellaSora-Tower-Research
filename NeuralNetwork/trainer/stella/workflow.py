"""Explicit, frozen paired training workflow; absent on historical manifests."""
from __future__ import annotations

from .run_paths import run_path, portable_basename

from .presets import canonical_seed
from .io_utils import file_hash

PAIRED_ANCHOR_V2 = "paired_anchor_v2"


def frozen_workflow(manifest):
    value = manifest.get("workflow", {})
    return value if value.get("protocol") == PAIRED_ANCHOR_V2 else {}


def frozen_seed(manifest, config, domain):
    workflow = frozen_workflow(manifest)
    if workflow:
        return workflow[domain + "_seed"]
    return canonical_seed({"evaluation": "catalogue_final", "development": "catalogue_development",
                           "optimizer": "optimizer"}[domain], config["seed"],
                          *([manifest["round"]] if domain == "optimizer" else []))


def build_workflow(config, number):
    base = config["seed"]
    common = dict(config["training"], advantage_weight=.75, ranking_weight=.25,
                  checkpoint_candidates=3, preserve_early_checkpoints=True,
                  prefer_first_four_checkpoints=False, per_preset_teacher_bank=True, objective_version="q_anchor_transfer_v2",
                  refresh_student_actions=True, transfer_diagnostics=True)
    return {"protocol": PAIRED_ANCHOR_V2, "base_seed": base,
            "optimizer_seed": canonical_seed("optimizer", base, number),
            "evaluation_seed": canonical_seed("catalogue_final", base),
            "development_seed": canonical_seed("catalogue_development", base),
            "replay": {"rounds": config["sampling"]["replay_rounds"],
                       "max_roots": config["sampling"]["replay_max_roots"]},
            "checkpoint_policy": "existing_three_point_early_1_2_4_and_label_best",
            "branches": [{"tag": tag, "training_config": dict(common, q_anchor_weight=anchor)}
                         for tag, anchor in (("control", .25), ("action", .05))]}


def verify_frozen_models(manifest, run_dir):
    if not frozen_workflow(manifest) and not manifest.get("catalogue_coverage"):
        return
    for path, checksum in [(manifest["teacher_checkpoint"], manifest["teacher_sha256"])] + [
            (model["checkpoint"], model["sha256"]) for model in manifest["teacher_models"].values()]:
        if file_hash(run_path(run_dir, path)) != checksum:
            raise ValueError("Frozen workflow checkpoint checksum changed: " + path)


def label_input_digest(roots):
    """Hash ordered frozen inputs once per fit, before normalization/priority mutation."""
    import hashlib
    import json
    import numpy as np
    digest = hashlib.sha256()
    for root in roots:
        identity = {key: root.get(key) for key in ("root_id", "family_id", "split", "teacher_id",
                    "teacher_sha256", "catalogue_id", "preset_group", "action_keys", "preset",
                    "student_action_index", "teacher_action_index", "decision_regret", "decision_regret_se",
                    "hard_sample_slot", "hard_kind", "training_priority", "behavior_source", "behavior_model_id",
                    "student_model_id", "behavior_sha256", "student_sha256", "teacher_student_disagreement",
                    "source_round", "source_environment_seed", "source_exploration_seed", "label_seed",
                    "coverage_protocol", "root_selection_role")}
        digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        for key in ("x", "q", "action_returns", "pair_se", "standard_error"):
            value = np.ascontiguousarray(root.get(key, []))
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def compare_paired_official(results, matrices):
    """Describe same-epoch action minus control scores on the fixed full catalogue.

    Inputs are aligned candidate records and complete [candidate,preset,game]
    integer matrices. This does not select models or change promotion decisions.
    """
    import hashlib
    import math
    import numpy as np
    scores = np.asarray(matrices)
    if (scores.ndim != 3 or scores.shape[0] != len(results) or scores.shape[1] < 1
            or scores.shape[2] < 2 or not np.issubdtype(scores.dtype, np.integer)):
        raise ValueError("Paired official comparison requires aligned complete integer score matrices")
    indexed = {}
    def identity(index):
        point = results[index]
        matrix = np.ascontiguousarray(scores[index])
        return {"model_id": point["model_id"], "branch": point["branch"], "epoch": int(point["epoch"]),
                "checkpoint": point.get("checkpoint"), "scores_identity": {
                    **point.get("scores_identity", {}), "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                    "shape": list(matrix.shape), "dtype": str(matrix.dtype)}}
    for index, point in enumerate(results):
        tag, epoch = point.get("branch"), int(point.get("epoch", 0))
        if tag not in ("control", "action") or epoch < 1:
            continue
        key = (tag, epoch)
        if key in indexed:
            raise ValueError("Duplicate actual checkpoint for paired branch/epoch")
        indexed[key] = index
    pairs = []
    common = sorted({e for b, e in indexed if b == "control"} & {e for b, e in indexed if b == "action"})
    p, g = scores.shape[1:]
    for epoch in common:
        c, a = indexed[("control", epoch)], indexed[("action", epoch)]
        # Cast each operand first: unsigned score arrays must support negative gains.
        difference = scores[a].astype(np.int64) - scores[c].astype(np.int64)
        preset_sums = difference.sum(axis=1, dtype=np.int64)
        mean = float(preset_sums.sum(dtype=np.int64)/(p*g))
        se = math.sqrt(float(difference.var(axis=1, ddof=1).sum())/g)/p
        pairs.append({"epoch": epoch, "control": identity(c), "action": identity(a),
            "mean_difference_action_minus_control": mean, "standard_error": se,
            "ci95_low": mean-1.96*se, "ci95_high": mean+1.96*se,
            "preset_wins": int((preset_sums > 0).sum()), "preset_ties": int((preset_sums == 0).sum()),
            "preset_losses": int((preset_sums < 0).sum())})
    best = {}
    for tag in ("control", "action"):
        indices = [i for (branch, _), i in indexed.items() if branch == tag]
        if indices:
            index = max(indices, key=lambda i: float(scores[i].mean()))
            best[tag] = dict(identity(index), official_mean=float(scores[index].mean()),
                             post_selection_descriptive=True)
    return {"protocol": PAIRED_ANCHOR_V2, "direction": "action_minus_control", "presets": p,
            "games_per_preset": g, "same_epoch_pairs": pairs, "branch_best_official": best,
            "same_fixed_seeds": True, "exploratory": True, "independent_evaluation": False,
            "promotion_gate": False, "ci_is_descriptive": True,
            "standard_error_formula": "sqrt(sum_p(var_games(action-control, ddof=1))/G)/P",
            "interpretation": "Exploratory same-fixed-seed comparison; not independent evaluation or a promotion gate. Branch-best means are post-selection descriptive values."}
