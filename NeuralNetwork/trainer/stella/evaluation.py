"""Independent complete paired games and diagnostic hierarchical intervals."""
from __future__ import annotations

import math
from fractions import Fraction
import numpy as np
from . import workers
from .environment import Episode, sim
from .presets import Preset, canonical_seed


def play_game(actor, preset, seed):
    episode = Episode(preset, seed)
    while not episode.done:
        workers.check()
        action = actor.choose(episode.state, episode.actions)
        episode.step(action)
    if not 0 <= episode.score <= preset.total:
        raise ValueError("Invalid terminal ordinary score")
    return episode.score, episode.steps


def evaluate_pair(task):
    workers.check()
    preset = Preset.from_dict(task["preset"])
    # Seed depends on domain / round / preset / episode only, never model ID.
    seed = int(task["environment_seed"])
    candidate = workers.POLICIES["candidate"]
    reference = workers.POLICIES["reference"]
    c, cs = play_game(candidate, preset, seed)
    r, rs = play_game(reference, preset, seed)
    return {"pair_id": task["pair_id"], "preset": preset.to_dict(), "seed": seed,
            "rules_version": sim.MODEL_VERSION,
            "candidate_score": c, "teacher_score": r, "candidate_steps": cs,
            "teacher_steps": rs, "delta": c-r, "complete": True}


def evaluate_pairs(tasks):
    """Interleave independent games so each policy predicts a useful GPU batch."""
    presets = [Preset.from_dict(task["preset"]) for task in tasks]
    outcomes = []
    for name in ("candidate", "reference"):
        actor = workers.POLICIES[name]
        episodes = [Episode(preset, int(task["environment_seed"])) for preset, task in zip(presets, tasks)]
        active = episodes
        while active:
            workers.check()
            actions = actor.choose_batch([ep.state for ep in active], [ep.actions for ep in active])
            for episode, action in zip(active, actions):
                episode.step(action)
            active = [episode for episode in active if not episode.done]
        outcomes.append([(ep.score, ep.steps) for ep in episodes])
    return [{"pair_id": task["pair_id"], "preset": preset.to_dict(), "seed": int(task["environment_seed"]),
             "rules_version": sim.MODEL_VERSION,
             "candidate_score": c, "teacher_score": r, "candidate_steps": cs,
             "teacher_steps": rs, "delta": c-r, "complete": True}
            for task, preset, (c, cs), (r, rs) in zip(tasks, presets, outcomes[0], outcomes[1])]


def _normal_interval(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean - 1.96 * se, mean + 1.96 * se


def summarize_pairs(rows, expected_tasks, bootstrap_replicates=1000, seed=0):
    expected = {t["pair_id"]: t for t in expected_tasks}
    if len(expected) != len(expected_tasks):
        raise ValueError("Duplicate pair IDs in the predeclared evaluation quota")
    actual = {r["pair_id"]: r for r in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("Only the entire predeclared evaluation quota can select a teacher")
    by_preset = {}
    for key, task in expected.items():
        row = actual[key]
        if row.get("rules_version") != task.get("rules_version", sim.MODEL_VERSION):
            raise ValueError("Evaluation rule version differs from its frozen task")
        p = Preset.from_dict(task["preset"])
        if (not row.get("complete") or row["seed"] != task["environment_seed"] or
                Preset.from_dict(row["preset"]).fingerprint != p.fingerprint):
            raise ValueError("Evaluation provenance mismatch or incomplete game")
        for name in ("candidate_score", "teacher_score"):
            if (not math.isfinite(row[name]) or not 0 <= row[name] <= p.total
                    or int(row[name]) != row[name]):
                raise ValueError("Evaluation has a nonterminal/invalid score")
        by_preset.setdefault(p.fingerprint, (p, []))[1].append(row)
    presets, groups = [], {0: [], 1: [], 2: []}
    for p, values in by_preset.values():
        c = np.asarray([r["candidate_score"] for r in values], dtype=float)
        t = np.asarray([r["teacher_score"] for r in values], dtype=float)
        d = c-t
        lo, hi = _normal_interval(d)
        entry = {"name": p.name, "fingerprint": p.fingerprint, "group": p.bucket,
                 "cap": p.total, "candidate_mean": float(c.mean()),
                 "teacher_mean": float(t.mean()), "delta": float(d.mean()),
                 "ci_low": lo, "ci_high": hi, "n": len(d)}
        presets.append(entry)
        groups[p.bucket].append((c, t, d))
    if any(not g for g in groups.values()):
        raise ValueError("All three supported target-total strata must be evaluated")
    # The terminal scores are integers. Exact rational aggregation distinguishes
    # a real improvement from a floating-point +7e-15 on a true tie, without
    # introducing any minimum-gain threshold or statistical significance lock.
    def exact_mean(column):
        strata = []
        for g in groups.values():
            means = [Fraction(sum(int(x) for x in triple[column]), len(triple[column])) for triple in g]
            strata.append(sum(means, Fraction()) / len(means))
        return sum(strata, Fraction()) / 3
    exact_c, exact_t = exact_mean(0), exact_mean(1)
    exact_delta = exact_c-exact_t
    primary_c, primary_t = float(exact_c), float(exact_t)
    # Resample preset clusters and paired episodes within each fixed stratum.
    # This interval is descriptive and is NEVER a promotion veto.
    rng = np.random.default_rng(canonical_seed("diagnostic_interval", seed))
    replicas = []
    for _ in range(bootstrap_replicates):
        means = []
        for g in groups.values():
            indices = rng.integers(0, len(g), len(g))
            means.append(np.mean([rng.choice(g[i][2], len(g[i][2]), replace=True).mean() for i in indices]))
        replicas.append(float(np.mean(means)))
    lo, hi = np.quantile(replicas, [.025, .975])
    delta = float(exact_delta)
    return {"presets": presets, "primary_candidate_mean": primary_c,
            "rules_version": sim.MODEL_VERSION,
            "primary_teacher_mean": primary_t, "primary_delta": delta,
            "ci_low": float(lo), "ci_high": float(hi), "pairs": len(rows),
            "games": len(rows)*2, "eligible": True, "promoted": bool(exact_delta > 0),
            "exact_delta": {"numerator": exact_delta.numerator, "denominator": exact_delta.denominator},
            "selection_rule": "complete quota AND estimated equal-stratum mean delta strictly > 0; ties retain teacher",
            "ci_role": "diagnostic only; hierarchical paired bootstrap",
            "strata": [{"group": i,
                        "candidate_mean": float(np.mean([c.mean() for c, _, _ in g])),
                        "teacher_mean": float(np.mean([t.mean() for _, t, _ in g])),
                        "delta": float(np.mean([d.mean() for _, _, d in g]))} for i, g in groups.items()]}


def summarize_audit(rows):
    groups = {}
    for row in rows:
        if not row.get("complete"):
            raise ValueError("Incomplete fixed audit game")
        p = Preset.from_dict(row["preset"])
        groups.setdefault(p.fingerprint, (p, []))[1].append(row)
    result = []
    for p, values in groups.values():
        current = np.asarray([r["candidate_score"] for r in values], dtype=float)
        initial = np.asarray([r["teacher_score"] for r in values], dtype=float)
        lo, hi = _normal_interval(current-initial)
        result.append({"name": f"{p.name} · {p.total}", "group": p.group,
                       "bucket": p.bucket, "cap": p.total, "mean": float(current.mean()),
                       "baseline_mean": float(initial.mean()),
                       "delta_vs_initial": float((current-initial).mean()),
                       "ci_low": lo, "ci_high": hi, "n": len(values),
                       "fingerprint": p.fingerprint})
    return result
