"""Candidate-only CUDA catalogue evaluation with compact resumable score matrices.

Rows are presets in supplied order, columns are fixed environment seeds. Cached
champion rows are retained by the caller; evaluating a challenger never runs them.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .presets import Preset, canonical_seed
from .catalogue_inputs import SEED_DOMAIN



def environment_seed(seed, preset, game_index):
    return canonical_seed(SEED_DOMAIN, int(seed), preset.fingerprint, int(game_index))


def _atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _validate_chunk(scores, steps, expected):
    scores, steps = np.asarray(scores), np.asarray(steps)
    if (scores.shape != (expected,) or steps.shape != (expected,)
            or not np.issubdtype(scores.dtype, np.integer)
            or not np.issubdtype(steps.dtype, np.integer)):
        raise ValueError("Invalid cached catalogue chunk shape/dtype")
    if (scores < 0).any() or (scores > 132).any() or (steps < 0).any():
        raise ValueError("Invalid cached catalogue scores or step counts")
    return scores, steps


def evaluate_catalogue(model_path, presets, output_dir, *, run_dir=None,
                       games_per_preset=256, batch_size=8192, seed=20261103,
                       control=None, progress=None, session=None):
    """Return resumable scores/steps matrices with an optional shared CUDA session.

    The original disk chunk boundaries, seed domain and manifest identity are
    retained. A different device may execute a missing disk chunk in smaller
    sub-batches without invalidating completed work. A cached load never creates a GPU process. Supplying a
    CatalogueEvaluationSession allows the six candidates to reuse CPU inputs and
    a CUDA context, while each candidate still plays every uncached game.
    """
    from collections import deque
    from .catalogue_service import CatalogueEvaluationSession
    from .vendor import stella_sim as sim
    started = time.perf_counter()
    presets = [Preset.from_dict(p.to_dict() if isinstance(p, Preset) else p) for p in presets]
    games_per_preset, batch_size, seed = int(games_per_preset), int(batch_size), int(seed)
    if not presets or games_per_preset < 1 or batch_size < 1:
        raise ValueError("Catalogue, games_per_preset and batch_size must be nonempty/positive")
    if session is not None and not session.matches(presets, seed, games_per_preset, run_dir):
        raise ValueError("Evaluation session presets/seed/games/run differ from request")
    model_path = Path(model_path).resolve()
    identity = {"schema": "cuda_catalogue_matrix_v1", "model_sha256": hashlib.sha256(
        model_path.read_bytes()).hexdigest(), "preset_fingerprints": [p.fingerprint for p in presets],
        "games_per_preset": games_per_preset, "seed": seed, "seed_domain": SEED_DOMAIN,
        "batch_size": batch_size, "rules_version": sim.MODEL_VERSION}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        # Hardware throughput settings are not statistical identities. Preserve
        # existing chunk filenames and every semantic identity field, while
        # allowing a large-GPU run to finish on a smaller GPU.
        if (type(stored.get("batch_size")) is not int or stored["batch_size"] < 1 or
                {k: v for k, v in stored.items() if k != "batch_size"} !=
                {k: v for k, v in identity.items() if k != "batch_size"}):
            raise ValueError("Catalogue checkpoint settings/model differ from manifest")
        identity = stored
    else:
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, manifest)
    shape = len(presets), games_per_preset
    scores, steps = np.empty(shape, np.int16), np.empty(shape, np.int16)
    total, completed, cached = scores.size, 0, 0
    own_session = session is None
    if own_session:
        # One-off evaluations have no next model to reuse the ~3 GiB cache.
        # They still overlap CPU preparation with the same GPU batch shapes.
        session = CatalogueEvaluationSession(presets, seed=seed, games_per_preset=games_per_preset,
                                             run_dir=run_dir, cache_mb=0)
    information, statistics = {}, {}
    chunks = []
    disk_batch_size = identity["batch_size"]
    for start in range(0, total, disk_batch_size):
        stop = min(start+disk_batch_size, total)
        path = output_dir / f"chunk_{start:08d}_{stop:08d}.npz"
        chunks.append((start, stop, path, path.exists()))
    missing = iter((substart, min(substart + batch_size, stop))
                   for start, stop, path, exists in chunks if not exists
                   for substart in range(start, stop, batch_size))
    pending = deque()
    ready = False

    def fill_pipeline():
        while len(pending) < session.depth:
            item = next(missing, None)
            if item is None:
                break
            if control:
                control()
            session.submit(*item)
            pending.append(item)

    try:
        for start, stop, path, exists in chunks:
            chunk_started = time.perf_counter()
            batch_seconds = 0.0
            if control:
                control()
            if exists:
                with np.load(path, allow_pickle=False) as data:
                    local_scores, local_steps = _validate_chunk(data["scores"], data["steps"], stop-start)
                    scores.ravel()[start:stop], steps.ravel()[start:stop] = local_scores, local_steps
                cached += stop-start
            else:
                if not ready:
                    information = session.begin_model(model_path, control)
                    ready = True
                    fill_pipeline()
                for substart in range(start, stop, batch_size):
                    substop = min(substart + batch_size, stop)
                    result = session.receive(control)
                    expected = pending.popleft()
                    if (expected != (substart, substop)
                            or (result.get("start"), result.get("stop")) != expected):
                        raise RuntimeError("Catalogue pipeline returned an out-of-order batch")
                    local_scores, local_steps = _validate_chunk(result["scores"], result["steps"], substop-substart)
                    scores.ravel()[substart:substop], steps.ravel()[substart:substop] = local_scores, local_steps
                    statistics = result["statistics"]
                    batch_seconds += result["batch_seconds"]
                    # Refill before persistence; game indices and seeds are
                    # unchanged even when a disk chunk spans several launches.
                    fill_pipeline()
                local_scores, local_steps = scores.ravel()[start:stop], steps.ravel()[start:stop]
                _atomic_npz(path, scores=local_scores, steps=local_steps)
            completed += stop-start
            if progress:
                elapsed = time.perf_counter()-started
                progress({"completed_games": completed, "total_games": total, "cached_games": cached,
                          "seconds": elapsed, "statistics": statistics,
                          "information": information, "batch_start": start, "batch_stop": stop,
                          "batch_games": stop-start, "batch_seconds": batch_seconds,
                          "chunk_seconds": time.perf_counter()-chunk_started,
                          "batch_games_per_second": (stop-start)/batch_seconds if batch_seconds else 0.0})
    except BaseException:
        # A partially drained FIFO cannot safely be reused for a different model.
        session.close()
        raise
    finally:
        if own_session:
            session.close()
    elapsed = time.perf_counter()-started
    summary_path = output_dir / "evaluation_summary.json"
    previous = (json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists() else {})
    summary = {"shape": list(shape), "information": information or previous.get("information", {}),
               "statistics": statistics or previous.get("statistics", {}),
               "cached_games": cached, "wall_seconds": elapsed, "total_games": total,
               "model_sha256": identity["model_sha256"],
               "timing_scope": "this_invocation", "new_games": total-cached,
               "runtime_batch_size": batch_size, "disk_chunk_batch_size": disk_batch_size}
    # Do not replace the original evaluation wall time with a 0.2-second cached
    # reload. Older exports did this, making service/wall ratios misleading.
    if cached == total and previous:
        summary = dict(previous, last_cached_load_seconds=elapsed,
                       last_cached_load_games=cached)
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, summary_path)
    return {"scores": scores, "steps": steps, "identity": identity, "information": information,
            "statistics": statistics, "cached_games": cached,
            "seconds": elapsed, "output_dir": str(output_dir.resolve())}


def select_champions(teacher_scores, candidate_scores, minimum_gain=.05):
    """Vectorized paired, equal-preset statistics and exact integer promotion.

    At 256 games, a 0.05 improvement requires a sum difference >=13.
    Confidence intervals are descriptive and never veto an update.
    """
    teacher, candidate = np.asarray(teacher_scores), np.asarray(candidate_scores)
    if teacher.shape != candidate.shape or teacher.ndim != 2 or teacher.shape[1] < 1:
        raise ValueError("Teacher and candidate must have identical preset-by-game shapes")
    n = teacher.shape[1]
    differences = candidate.astype(np.int64)-teacher.astype(np.int64)
    sums = differences.sum(axis=1)
    required = int((Decimal(str(minimum_gain))*n).to_integral_value(rounding=ROUND_CEILING))
    promote = sums >= required
    means = sums/n
    errors = differences.std(axis=1, ddof=1)/np.sqrt(n) if n > 1 else np.zeros(len(teacher))
    return {"promote": promote, "sum_difference": sums, "required_sum_difference": required,
            "mean_difference": means, "standard_error": errors,
            "ci95_low": means-1.96*errors, "ci95_high": means+1.96*errors,
            "teacher_mean": float(teacher.mean()), "candidate_mean": float(candidate.mean()),
            "equal_preset_mean_difference": float(means.mean()),
            "promoted_count": int(promote.sum())}
