"""One-time CPU worker calibration using REAL complete-return label jobs.

The probe never enters any training or evaluation dataset and never changes
game/model random streams. Timings are measured on the user's own machine.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
from pathlib import Path
import platform
from random import Random
import time

from . import workers
from .cpu_affinity import create_plan
from .environment import Episode, sim
from .io_utils import atomic_json, read_json, digest
from .presets import sample_preset, canonical_seed
from .sampling import label_root


def cpu_probe(task):
    actor = workers.POLICIES["teacher"]
    seed = canonical_seed("hardware_probe_only", task["ordinal"])
    preset = sample_preset(Random(seed), task["ordinal"] % 3, name="timing_only")
    episode = Episode(preset, seed)
    for _ in range(24):
        if episode.done:
            break
        workers.check()
        episode.step(actor.choose(episode.state, episode.actions))
    if episode.done:
        raise RuntimeError("Unexpected short calibration episode")
    state, actions = episode.state, episode.actions
    row = {"preset": preset.to_dict(), "state": sim.to_jsonable(state),
           "action_keys": [sim.action_key(a) for a in actions],
           "rollouts_per_action": 2, "label_seed": seed,
           "family_id": f"timing_{task['ordinal']}", "teacher_id": "timing_only",
           "root_id": f"timing_{task['ordinal']}", "split": "train"}
    result = label_root(row, actor=actor)
    return result["metadata"]["complete_branches"]


def _drain(pool, tasks, paths, control):
    pending = {pool.submit(workers.invoke, cpu_probe, task, paths) for task in tasks}
    branches = 0
    while pending:
        control()
        done, pending = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
        for future in done:
            branches += future.result()
    return branches


def calibrate_workers(run_dir, model_path, default_workers=8, callback=None, control=None):
    run_dir = Path(run_dir)
    control = control or (lambda: None)
    logical = os.cpu_count() or 1
    signature = digest({"version": 2, "affinity": "physical_core_first", "system": platform.system(), "machine": platform.machine(),
                        "processor": platform.processor(), "logical": logical})
    path = run_dir / "cpu_calibration.json"
    if path.exists():
        record = read_json(path)
        if record.get("hardware_signature") == signature and 1 <= record.get("selected_workers", 0) <= logical:
            return record
    candidates = sorted({min(logical, max(1, int(x))) for x in
                         (default_workers, default_workers*1.5, default_workers*2)})
    # Bound the probe on large workstations too; this package targets an 8-core CPU.
    candidates = [n for n in candidates if n <= 16]
    paths = {"teacher": str(model_path)}
    # Identical timed jobs for each worker count; warmup absorbs process import cost.
    tasks = [{"ordinal": i} for i in range(max(24, max(candidates)*2))]
    records = []
    for i, count in enumerate(candidates):
        control()
        if callback:
            callback({"phase_total": len(candidates), "phase_completed": i,
                      "phase_progress": i/len(candidates),
                      "message": f"CPU吞吐校准：实测 {count} 进程；计时数据不参与训练/选模。"})
        pool = ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=workers.initialize,
                                   initargs=(str(run_dir), {}, None, create_plan(count, run_dir)))
        try:
            warm = [{"ordinal": 10000+j} for j in range(count)]
            _drain(pool, warm, paths, control)
            started = time.perf_counter()
            branches = _drain(pool, tasks, paths, control)
            seconds = time.perf_counter()-started
            records.append({"workers": count, "seconds": seconds, "roots": len(tasks),
                            "complete_branches": branches, "roots_per_second": len(tasks)/seconds,
                            "branches_per_second": branches/seconds})
        except BaseException:
            (run_dir / "stop.flag").touch()
            (run_dir / "pause.flag").unlink(missing_ok=True)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    fastest = max(records, key=lambda r: r["roots_per_second"])
    result = {"hardware_signature": signature, "selected_workers": fastest["workers"],
              "records": records, "logical_cores": logical,
              "criterion": "highest measured complete-root label throughput after worker warmup",
              "probe_data_is_training_data": False, "tested_on": platform.platform()}
    atomic_json(path, result)
    return result
