"""Reachable public roots and full-terminal frozen-teacher action labels.

The label for every legal action is E[terminal ordinary score - root score].
All root actions use the same independent seed list. No teacher continuation
is replaced with a one-transition DeltaV or a truncated bootstrap target.
"""
from __future__ import annotations

from random import Random
import json
from pathlib import Path
import numpy as np

from .hard_examples import classify_hard_state, choose_hard_root
from . import workers
from .environment import Episode, branch, legal_actions, score, sim, validate_state_preset
from .features import encode_actions
from .io_utils import atomic_bytes, dumps
from .presets import PRESET_VERSION, Preset, canonical_seed


FAMILY_SPLIT_VERSION = "bucket_behavior_family_75_25_v2"


def assign_family_splits(tasks, seed, round_number):
    """Split whole trajectories within target bucket and actual behavior source.

    Largest-remainder quotas preserve the 25% development budget per bucket;
    each source's quota differs from its ideal by less than one family.
    Independent seeded shuffles avoid coupling the split to exploration mode.
    """
    strata = {}
    for task in tasks:
        bucket = Preset.from_dict(task["preset"]).bucket
        key = (bucket, task["behavior_source"], task["epsilon"])
        strata.setdefault(key, []).append(task)
    for bucket in range(3):
        keys = sorted(key for key in strata if key[0] == bucket)
        Random(canonical_seed("family_split_quota", seed, round_number, bucket)).shuffle(keys)
        quotas = {key: len(strata[key]) // 4 for key in keys}
        remaining = sum(len(strata[key]) for key in keys) // 4 - sum(quotas.values())
        for key in sorted(keys, key=lambda k: len(strata[k]) % 4, reverse=True)[:remaining]:
            quotas[key] += 1
        for key in keys:
            families = sorted(strata[key], key=lambda t: t["family_id"])
            Random(canonical_seed("family_split_shuffle", seed, round_number, *key)).shuffle(families)
            for index, task in enumerate(families):
                task["split"] = "val" if index < quotas[key] else "train"
    # Schedule phase/floor coverage separately in each bucket and split. A
    # shuffled validation subset must not lose the twelve requested phase/bands.
    counters = {}
    for task in sorted(tasks, key=lambda t: t["ordinal"]):
        if task.get("balanced_phase_sampling"):
            key = (Preset.from_dict(task["preset"]).bucket, task["split"])
            task["phase_sampling_index"] = counters.get(key, 0)
            counters[key] = task["phase_sampling_index"] + 1


def root_selection_role(task, ordinal):
    """Explicit roles keep 3+1 trajectory coverage from losing its hard slot."""
    schedule = task.get("root_selection_schedule")
    if schedule is None:
        return ("natural", "natural", "phase", "hard")[ordinal % 4]
    if (len(schedule) != task["roots_per_family"] or
            any(role not in ("natural", "phase", "hard") for role in schedule)):
        raise ValueError("Invalid explicit root-selection schedule")
    return schedule[ordinal]


def select_family_roots(task, eligible, gap, classify=None, disagreement=None):
    """Select public roots; state-like rows need only phase and floor fields."""
    selection = Random(task["selection_seed"])
    if not eligible:
        raise RuntimeError("No nontrivial reachable root in a complete episode")
    eligible = list(eligible)
    selection.shuffle(eligible)
    selected = []
    target = task["roots_per_family"]
    # Half natural visitation; one quarter phase/floor stratified; the rest
    # disagreement / hard decisions. Only public state and model predictions,
    # never final returns, rank roots.
    for ordinal in range(target):
        remaining = [(step, state) for step, state in eligible if step not in {s[0] for s in selected}]
        if not remaining:
            break
        role = root_selection_role(task, ordinal)
        if role == "natural":
            chosen = remaining[0]
        elif role == "phase":
            if task.get("balanced_phase_sampling"):
                # Old frozen manifests retain their original root selection.
                schedule_index = task.get("phase_sampling_index", task["ordinal"] // 12)
                phase_index = schedule_index % 4
                band = (schedule_index // 4) % 3
            else:
                phase_index = task["ordinal"] % 4
                band = task["ordinal"] % 3
            phase = ("SHOP", "EVENT", "PAGE", "ENHANCE_PAGE")[phase_index]
            def rank(item):
                _, state = item
                actual_band = 0 if state.floor <= 7 else (1 if state.floor <= 14 else 2)
                return state.phase != phase, actual_band != band
            chosen = min(remaining, key=rank)
        else:
            disputed = ([item for item in remaining if disagreement(item[1]) is not None]
                        if task.get("disagreement_sampling") and disagreement is not None else [])
            if disputed:
                band = task.get("phase_sampling_index", task["ordinal"]) % 3
                def disagreement_rank(item):
                    state = item[1]
                    actual_band = 0 if state.floor <= 7 else (1 if state.floor <= 14 else 2)
                    return actual_band != band, -disagreement(state)
                chosen = min(disputed, key=disagreement_rank)
            elif task.get("hard_sampling"):
                chosen = choose_hard_root(task, remaining, gap, classify)
            else:
                candidates = remaining[:6]
                chosen = min(candidates, key=lambda item: gap(item[1]))
        selected.append(chosen)
    return selected


def build_sampled_family(task, preset, selected, trajectory_steps, trajectory_score,
                         sampling_inference=None):
    """Serialize selected public roots with their original family assignment."""
    roots = []
    for j, (step, state) in enumerate(selected):
        validate_state_preset(state, preset)
        actions = legal_actions(state)
        roots.append({"root_id": f"{task['family_id']}_root{j:02d}",
                      "rules_version": sim.MODEL_VERSION,
                      "family_id": task["family_id"], "split": task["split"],
                      "preset": preset.to_dict(), "preset_group": preset.bucket,
                      "cap": preset.total, "state": sim.to_jsonable(state),
                      "step": step, "phase": state.phase, "floor": state.floor,
                      "action_keys": [sim.action_key(a) for a in actions],
                      "teacher_id": task["teacher_id"],
                      "label_seed": canonical_seed(task["label_seed"], step),
                      "rollouts_per_action": task["rollouts_per_action"]})
        if task.get("disagreement_sampling"):
            roots[-1].update(task["_policy_decisions"][step],
                             behavior_source=task["behavior_source"],
                             behavior_model_id=task["behavior_model_id"],
                             student_model_id=task["student_model_id"])
        if task.get("coverage_protocol"):
            roots[-1].update(coverage_protocol=task["coverage_protocol"],
                             catalogue_id=task["catalogue_id"],
                             source_environment_seed=task["environment_seed"],
                             source_exploration_seed=task["exploration_seed"],
                             root_selection_role=root_selection_role(task, j))
        if task.get("hard_sampling") and root_selection_role(task, j) == "hard":
            roots[-1]["hard_sample_slot"] = True
            roots[-1]["hard_example"] = True
            roots[-1]["hard_kind"] = task.get("_hard_kinds", {}).get(step) or "small_gap_fallback"
            roots[-1]["hard_selection"] = ("student_teacher_disagreement"
                if roots[-1].get("teacher_student_disagreement") else "hard_category_fallback")
            roots[-1]["rollouts_per_action"] = task.get("hard_rollouts_per_action", task["rollouts_per_action"])
        if sampling_inference is not None:
            roots[-1]["sampling_inference"] = sampling_inference
    family = {"family_id": task["family_id"], "preset": preset.to_dict(),
            "rules_version": sim.MODEL_VERSION,
            "split": task["split"], "roots": roots, "trajectory_steps": trajectory_steps,
            "trajectory_score": trajectory_score, "epsilon": task["epsilon"],
            "behavior_source": task.get("behavior_source", "teacher")}
    if sampling_inference is not None:
        family["sampling_inference"] = sampling_inference
    return family


def sample_family(task):
    workers.check()
    actor = workers.POLICIES["teacher"]
    behavior = workers.POLICIES.get("behavior", actor) if task.get("behavior_source") in ("last_candidate", "best_candidate") else actor
    preset = Preset.from_dict(task["preset"])
    episode = Episode(preset, task["environment_seed"])
    explore = Random(task["exploration_seed"])
    eligible = []
    policy_decisions = {}
    disagreement_gaps = {}
    student = workers.POLICIES.get("behavior", actor)
    while not episode.done:
        workers.check()
        state, actions = episode.state, episode.actions
        if len(actions) > 1 and score(state) < preset.total:
            eligible.append((episode.steps, state))
            if task.get("disagreement_sampling"):
                teacher_q = actor.values(state, actions)
                student_q = teacher_q if student is actor else student.values(state, actions)
                teacher_choice, student_choice = int(np.argmax(teacher_q)), int(np.argmax(student_q))
                policy_decisions[episode.steps] = {"teacher_action_index": teacher_choice,
                    "student_action_index": student_choice,
                    "teacher_student_disagreement": teacher_choice != student_choice,
                    "teacher_student_predicted_gap": float(teacher_q[teacher_choice]-teacher_q[student_choice])}
                disagreement_gaps[id(state)] = (float(teacher_q[teacher_choice]-teacher_q[student_choice])
                                                  if teacher_choice != student_choice else None)
        if explore.random() < task["epsilon"]:
            action = explore.choice(actions)
        else:
            action = behavior.choose(state, actions)
        episode.step(action)
    def gap(state):
        q = np.sort(actor.values(state))
        return float(q[-1] - q[-2])
    def classify(state):
        actions = legal_actions(state)
        return classify_hard_state(state, actions, int(np.argmax(actor.values(state))))
    selected = select_family_roots(task, eligible, gap, classify,
        disagreement=(lambda state: disagreement_gaps.get(id(state))) if task.get("disagreement_sampling") else None)
    if task.get("disagreement_sampling"):
        task = dict(task, _policy_decisions=policy_decisions)
    if task.get("hard_sampling"):
        task = dict(task, _hard_kinds={step: classify(state) for j, (step, state) in enumerate(selected)
                                     if root_selection_role(task, j) == "hard"})
    return build_sampled_family(task, preset, selected, episode.steps, episode.score)


def label_root(task, actor=None):
    workers.check()
    actor = workers.POLICIES["teacher"] if actor is None else actor
    preset = Preset.from_dict(task["preset"])
    root = sim.state_from_dict(task["state"])
    validate_state_preset(root, preset)
    actions = legal_actions(root)
    if [sim.action_key(a) for a in actions] != task["action_keys"]:
        raise ValueError("Root legal-action set changed; refusing partial labels")
    n = int(task["rollouts_per_action"])
    if n < 2:
        raise ValueError("Full action-value labels need at least two rollout seeds")
    seeds = [canonical_seed("complete_teacher_branch", task["label_seed"], k) for k in range(n)]
    episodes, positions = [], []
    returns = np.empty((len(actions), n), dtype=np.float32)
    branch_steps = 0
    for ai, action in enumerate(actions):
        for k, seed in enumerate(seeds):
            # branch() sees the public posterior, never the original latent.
            ep = branch(root, action, seed, preset)
            episodes.append(ep)
            positions.append((ai, k))
    while episodes:
        workers.check()
        active, active_positions = [], []
        for ep, (ai, k) in zip(episodes, positions):
            if ep.done:
                returns[ai, k] = ep.score - score(root)
                branch_steps += ep.steps
            else:
                active.append(ep)
                active_positions.append((ai, k))
        episodes, positions = active, active_positions
        if not episodes:
            break
        # All continuation decisions use precisely the frozen round teacher.
        chosen = actor.choose_batch([ep.state for ep in episodes], [ep.actions for ep in episodes])
        if len(chosen) != len(episodes):
            raise ValueError("Incomplete teacher batch")
        for ep, action in zip(episodes, chosen):
            ep.step(action)
    if not np.isfinite(returns).all() or np.any(returns < 0) or np.any(returns > preset.total-score(root)):
        raise ValueError("Invalid complete-return labels")
    q = returns.mean(axis=1, dtype=np.float64).astype(np.float32)
    metadata = dict(task, label_semantics="E[terminal ordinary score - root ordinary score | root action, frozen teacher]",
                    rules_version=sim.MODEL_VERSION,
                    preset_version=PRESET_VERSION,
                    complete_branches=int(returns.size), branch_decisions=branch_steps,
                    root_score=score(root), schema="complete_root_q_v1")
    inference = getattr(getattr(actor, "model", None), "information", None)
    if inference is not None:
        metadata["label_inference"] = dict(inference)
    return {"x": encode_actions(root, actions), "q": q, "returns": returns,
            "standard_error": returns.std(axis=1, ddof=1)/np.sqrt(n), "metadata": metadata}


def save_label(path, row):
    import io
    import zipfile
    buffer = io.BytesIO()
    arrays = dict(x=row["x"], q=row["q"], returns=row["returns"],
                  standard_error=row["standard_error"],
                  metadata=np.asarray(dumps(row["metadata"])))
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


def load_label(path, teacher_id=None):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("rules_version") != sim.MODEL_VERSION:
            raise ValueError("Replay label belongs to a different state-space rule version")
        if metadata.get("preset_version") != PRESET_VERSION:
            raise ValueError("Replay label belongs to a different preset scope")
        preset = Preset.from_dict(metadata["preset"])
        if metadata["schema"] != "complete_root_q_v1":
            raise ValueError("Unsupported replay schema")
        if teacher_id is not None and metadata["teacher_id"] != teacher_id:
            raise ValueError("A replay label belongs to another teacher")
        state = sim.state_from_dict(metadata["state"])
        validate_state_preset(state, preset)
        x, q, returns = archive["x"].copy(), archive["q"].copy(), archive["returns"].copy()
        actions = legal_actions(state)
        if metadata["action_keys"] != [sim.action_key(a) for a in actions]:
            raise ValueError("Replay action identities/order changed")
        if returns.shape != (len(actions), metadata["rollouts_per_action"]):
            raise ValueError("Replay has an incomplete branch quota")
        if metadata["complete_branches"] != returns.size:
            raise ValueError("Replay complete-branch count disagrees")
        if (not np.isfinite(returns).all() or np.any(returns < 0) or
                np.any(returns > preset.total-score(state))):
            raise ValueError("Replay contains invalid terminal returns")
        expected_x = encode_actions(state, actions)
        if x.shape != expected_x.shape or not np.array_equal(x, expected_x):
            raise ValueError("Replay features do not match the original public state/actions")
        if returns.shape[0] != len(q) or not np.allclose(q, returns.mean(axis=1), atol=1e-5):
            raise ValueError("Replay Q disagrees with complete branch returns")
        if len(q) != len(legal_actions(state)) or x.shape[0] != len(q):
            raise ValueError("Replay lacks legal action labels")
        return dict(metadata, x=x, q=q, action_returns=returns,
                    preset_group=preset.bucket, cap=preset.total)
