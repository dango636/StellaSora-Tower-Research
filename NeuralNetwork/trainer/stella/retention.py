"""Bounded, reference-aware retention for trainer-owned run artifacts.

Windows retains its recycle-bin behavior. POSIX removes only expired completed
rounds and unreferenced managed weights/scores, so Colab actually releases SSD
space without requiring a desktop trash service. No wildcard shell commands.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil

from .io_utils import read_json
from .run_paths import run_path

_ROUND = re.compile(r"round_([0-9]{6,})\Z")
_MODEL = re.compile(r"M([0-9]{6,})(?:_B[a-zA-Z0-9]+)?(?:_E[0-9]{3,})?\.npz\Z")
_SCORE = re.compile(r"round_([0-9]{6,})\.npz\Z")


def _owned(root: Path, path: Path) -> Path:
    """Reject link traversal and the root itself before planning any deletion."""
    root = root.resolve()
    path = Path(path).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Retention target escapes run directory") from exc
    if not relative.parts:
        raise ValueError("Retention cannot remove the run directory")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Retention refuses symbolic links: " + str(current))
    if not path.resolve().is_relative_to(root):
        raise ValueError("Retention target escapes run directory")
    return path


def _reference(root: Path, value, protected: set[Path]):
    if value is not None:
        if not isinstance(value, str) or not value:
            raise ValueError("Retention encountered an invalid file reference")
        protected.add(_owned(root, run_path(root, value)))


def _protect_record(root: Path, record: dict, protected: set[Path]):
    """Use actual run-relative references; never a lossy basename comparison."""
    for key in ("teacher_checkpoint", "behavior_checkpoint", "teacher_scores", "last_candidate_checkpoint",
                "candidate_checkpoint", "checkpoint"):
        _reference(root, record.get(key), protected)
    for value in record.get("models", {}).values():
        _reference(root, value, protected)
    for value in record.get("candidate_models", {}).values():
        _reference(root, value, protected)
    for value in record.get("teacher_models", {}).values():
        _reference(root, value.get("checkpoint"), protected)
    if "learner_initialization" in record:
        initialization = record["learner_initialization"]
        if not isinstance(initialization, dict):
            raise ValueError("Retention encountered an invalid learner initialization record")
        _reference(root, initialization.get("checkpoint"), protected)


def _protected_ancestors(root, protected):
    """One bounded ancestor expansion replaces repeated cross-product scans."""
    result = set(protected)
    for path in protected:
        for parent in path.parents:
            if parent == root:
                break
            result.add(parent)
    return result


def retention_plan(run_dir, state, *, keep_rounds, keep_checkpoints=0):
    """Build and validate a full plan before mutating anything.

    Incomplete/future rounds are always retained. A corrupt completed summary or
    retained manifest aborts planning, rather than guessing that its data is old.
    """
    if type(keep_rounds) is not int or keep_rounds < 1:
        raise ValueError("keep_rounds must be a positive integer")
    if type(keep_checkpoints) is not int or keep_checkpoints < 0:
        raise ValueError("keep_checkpoints must be a nonnegative integer")
    root = Path(run_dir).resolve()
    next_round = state.get("next_round")
    if type(next_round) is not int or next_round < 1:
        raise ValueError("Retention requires an authoritative next_round")
    for name in ("rounds", "checkpoints", "teacher_scores"):
        _owned(root, root / name)
    directories = []
    completed = []
    for path in (root / "rounds").glob("round_*"):
        match = _ROUND.fullmatch(path.name)
        if match is None:
            continue
        _owned(root, path)
        if not path.is_dir():
            continue
        number = int(match.group(1))
        directories.append((number, path))
        summary = path / "summary.json"
        if number < next_round and summary.is_file():
            _owned(root, summary)
            record = read_json(summary)
            if record.get("round") != number:
                raise ValueError("Completed summary round does not match its directory")
            completed.append((number, path))
    completed.sort()
    expired = {p for _, p in completed[:-keep_rounds]}
    retained = {p for _, p in directories if p not in expired}
    protected: set[Path] = set()
    _protect_record(root, state, protected)
    protected.add(root / "checkpoints" / "M000000.npz")
    for name in ("best_model.json", "champion_bank.json"):
        path = root / name
        if path.is_file():
            _owned(root, path)
            _protect_record(root, read_json(path), protected)
    inspected = set()
    while True:
        for directory in sorted(retained - inspected):
            for name in ("manifest.json", "summary.json"):
                path = directory / name
                if path.is_file():
                    _owned(root, path)
                    _protect_record(root, read_json(path), protected)
            # Official candidates may exist before an incomplete round's summary.
            number = int(_ROUND.fullmatch(directory.name).group(1))
            protected.add(root / "checkpoints" / f"M{number:06d}.npz")
            fit = directory / "fit.json"
            if fit.is_file():
                _owned(root, fit)
                for point in read_json(fit).get("checkpoint_selection", {}).get("candidates", []):
                    _reference(root, point.get("checkpoint"), protected)
            inspected.add(directory)
        # A live model may be stored in a round's fit_checkpoints directory.
        # Retain that entire directory, then inspect its dependencies as well.
        # Iterate to closure so a chain of historical references stays intact.
        required = expired & _protected_ancestors(root, protected)
        if not required:
            break
        retained.update(required)
        expired.difference_update(required)
    retained_numbers = {int(_ROUND.fullmatch(p.name).group(1)) for p in retained}
    checkpoints = []
    for path in (root / "checkpoints").glob("M*.npz"):
        match = _MODEL.fullmatch(path.name)
        if match is None:
            continue
        _owned(root, path)
        if path.is_file():
            checkpoints.append((int(match.group(1)), path.name, path))
    checkpoints.sort()
    # An interrupted old round can have several epoch/branch candidates that
    # have not yet been listed in fit.json. Preserve all of its managed models.
    protected.update(path for number, _, path in checkpoints if number in retained_numbers)
    if keep_checkpoints:
        protected.update(p for _, _, p in checkpoints[-keep_checkpoints:])
    targets = sorted(expired)
    for number, _, path in checkpoints:
        if number < next_round and path not in protected:
            targets.append(path)
            if path.with_suffix(".json").is_file():
                targets.append(_owned(root, path.with_suffix(".json")))
    for path in sorted((root / "teacher_scores").glob("round_*.npz")):
        match = _SCORE.fullmatch(path.name)
        if match is not None:
            _owned(root, path)
            if path.is_file() and int(match.group(1)) < next_round and path not in protected:
                targets.append(path)
    # Capture stat identity to reject replacements between plan and execution.
    records = []
    protected_ancestors = _protected_ancestors(root, protected)
    for path in targets:
        _owned(root, path)
        if path in protected_ancestors:
            raise ValueError("Retention plan would remove a protected reference")
        st = path.stat()
        records.append({"path": str(path.relative_to(root)), "device": st.st_dev,
                        "inode": st.st_ino, "mtime_ns": st.st_mtime_ns,
                        "size": st.st_size, "directory": path.is_dir()})
    return {"schema": "stella_retention_v3", "root": str(root),
            "next_round": next_round, "retained_rounds": [p.name for p in sorted(retained)],
            "protected_files": sorted(str(p.relative_to(root)) for p in protected),
            "targets": records}


def recycle_windows(path):
    """Preserve the existing Windows recycle-bin contract."""
    import ctypes
    from ctypes import wintypes
    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", wintypes.WORD), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", wintypes.LPVOID), ("lpszProgressTitle", wintypes.LPCWSTR)]
    operation = SHFILEOPSTRUCTW(None, 3, str(Path(path).resolve()) + "\0\0", None,
                               0x0040 | 0x0010 | 0x0004 | 0x0400, False, None, None)
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result or operation.fAnyOperationsAborted:
        raise OSError(f"Could not recycle {path}: {result}")


def apply_retention(plan):
    root = Path(plan["root"]).resolve()
    paths = []
    for record in plan["targets"]:
        path = _owned(root, run_path(root, record["path"]))
        st = path.stat()
        if (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size, path.is_dir()) != (
                record["device"], record["inode"], record["mtime_ns"], record["size"], record["directory"]):
            raise ValueError("Retention target changed after planning: " + record["path"])
        paths.append(path)
    removed = []
    for path in paths:
        _owned(root, path)
        if os.name == "nt":
            recycle_windows(path)
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path.relative_to(root)))
    return {"schema": plan["schema"], "next_round": plan["next_round"],
            "mode": "windows_recycle_bin" if os.name == "nt" else "posix_managed_delete",
            "removed": removed, "retained_rounds": plan["retained_rounds"],
            "protected_file_count": len(plan["protected_files"])}
