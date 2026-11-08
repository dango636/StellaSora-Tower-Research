"""Opt-in rollout CUDA-event timing, without synchronizing individual stages.

Instantiate only for STELLA_GPU_PROFILE=1. CUDA events add measurement overhead;
CPU wait time can overlap the measured GPU work and must not be added to it.
"""
from __future__ import annotations

import torch


class RolloutProfile:
    def __init__(self):
        self.pending = []
        self.stages = {}
        self.cpu_waits = {}
        self.runs = 0
        self.failed_runs = 0

    def start(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return name, event

    def stop(self, token):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        name, start = token
        self.pending.append((name, start, event))

    @staticmethod
    def _add(summary, name, seconds):
        row = summary.setdefault(name, {"calls": 0, "seconds": 0.0, "max_seconds": 0.0})
        row["calls"] += 1
        row["seconds"] += float(seconds)
        row["max_seconds"] = max(row["max_seconds"], float(seconds))

    def cpu_wait(self, name, seconds):
        self._add(self.cpu_waits, name, seconds)

    def collect_completed_run(self):
        """Call only after the existing blocking scores/steps CPU transfers."""
        for name, start, stop in self.pending:
            self._add(self.stages, name, start.elapsed_time(stop) / 1000.0)
        self.pending.clear()
        self.runs += 1

    def discard_failed_run(self):
        # Control callbacks and rule errors may interrupt a run. The next run
        # must not accidentally mix that partial trace with its own summary.
        if self.pending:
            self.pending.clear()
            self.failed_runs += 1

    def snapshot(self):
        return dict(enabled=True, runs=self.runs, failed_runs=self.failed_runs,
                    scope="generator_lifetime_cumulative",
                    gpu_stages={name: dict(row) for name, row in self.stages.items()},
                    cpu_waits={name: dict(row) for name, row in self.cpu_waits.items()},
                    note="CUDA-event timings include profiling overhead. CPU waits overlap GPU work; do not add the two.")
