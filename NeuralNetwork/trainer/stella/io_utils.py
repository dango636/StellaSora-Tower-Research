"""Small atomic, human-readable run artifacts; no pickle checkpoints."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"), default=json_default)


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers should close handles promptly; retry antivirus races.
        for retry in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if retry == 19:
                    raise
                time.sleep(.05)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_bytes(path, (dumps(value) + "\n").encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_gzip_json(path, value):
    atomic_bytes(path, gzip.compress(dumps(value).encode("utf-8"), compresslevel=1, mtime=0))


def read_gzip_json(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A power failure can leave only the last append incomplete.
            continue
    return rows


def append_jsonl(path, value, limit=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(dumps(value) + "\n")
        stream.flush()
    if limit and path.stat().st_size > limit * 512:
        rows = read_jsonl(path)
        if len(rows) > limit:
            atomic_bytes(path, ("\n".join(dumps(x) for x in rows[-limit:]) + "\n").encode())


def upsert_history(path, record, limit=2000):
    rows = [r for r in read_jsonl(path) if r.get("round") != record["round"]]
    rows.append(record)
    rows.sort(key=lambda r: r["round"])
    atomic_bytes(path, ("\n".join(dumps(r) for r in rows[-limit:]) + "\n").encode("utf-8"))


class StopRequested(Exception):
    """A normal user stop, not an incomplete evaluation result."""


class Control:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.next_check = 0.0

    def check(self, force=False):
        now = time.monotonic()
        if not force and now < self.next_check:
            return
        self.next_check = now + .2
        if (self.run_dir / "stop.flag").exists():
            raise StopRequested("用户要求安全停止；已完成的样本与评测已保存。")
        while (self.run_dir / "pause.flag").exists():
            if (self.run_dir / "stop.flag").exists():
                raise StopRequested("用户要求停止。")
            time.sleep(.2)
