"""Bounded background label compression and durable atomic storage."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

from .sampling import save_label


class LabelWriter:
    def __init__(self, max_pending=2048):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="label-ssd-writer")
        self.pending = deque()
        self.max_pending = max_pending
        self.submitted = self.saved = self.peak_pending = 0
        self.write_seconds = self.backpressure_seconds = 0.0

    def _write(self, path, row):
        started = time.perf_counter()
        save_label(path, row)
        return time.perf_counter()-started

    def poll(self):
        while self.pending and self.pending[0].done():
            future = self.pending.popleft()
            self.write_seconds += future.result()
            self.saved += 1

    def submit(self, path, row):
        self.poll()
        if len(self.pending) >= self.max_pending:
            started = time.perf_counter()
            future = self.pending.popleft()
            self.write_seconds += future.result()
            self.saved += 1
            self.backpressure_seconds += time.perf_counter()-started
        self.pending.append(self.executor.submit(self._write, path, row))
        self.submitted += 1
        self.peak_pending = max(self.peak_pending, len(self.pending))

    def snapshot(self):
        self.poll()
        return dict(label_writes_submitted=self.submitted, label_writes_saved=self.saved,
                    label_writes_pending=len(self.pending), label_writes_peak_pending=self.peak_pending,
                    label_write_seconds=self.write_seconds,
                    label_write_backpressure_seconds=self.backpressure_seconds)

    def close(self):
        # Drain even on stop: GPU work already received remains resumable on SSD.
        self.executor.shutdown(wait=True)
        self.poll()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
