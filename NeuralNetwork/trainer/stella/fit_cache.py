"""Identify new calibration-stage caches before accepting resumable results."""
from __future__ import annotations

from pathlib import Path

from .io_utils import atomic_json, digest, file_hash, read_json
from .presets import canonical_seed
from .run_paths import run_path

SCHEMA = "stella_calibration_fit_cache_v5"


def implementation_digest():
    # Hash small source modules once per identity construction. This catches a
    # local agent changing loss/sampling code without manually bumping SCHEMA.
    root = Path(__file__).resolve().parent
    names = ("fit_cache.py", "fit.py", "fit_torch.py", "fit_calibration.py",
             "fit_diagnostics.py", "action_calibration.py", "indexing.py",
             "features.py", "network.py", "workflow.py", "learner_initialization.py",
             "engine.py", "catalogue_engine.py")
    return digest({name: file_hash(root / name) for name in names})


def enabled(manifest):
    return bool(manifest.get("learner_initialization")) or bool(manifest.get("catalogue_coverage")) or manifest.get(
        "training_config", {}).get("objective_version") == "action_gap_calibration_v3"


def identity(manifest, base_seed):
    """Only new calibration rounds opt in; historical frozen runs stay readable."""
    if not enabled(manifest):
        return None
    fields = ("round", "rules_version", "preset_version", "teacher_id", "teacher_checkpoint",
              "teacher_sha256", "teacher_models", "teacher_bank", "training_config",
              "compute_config", "catalogue_coverage", "sampling_tasks", "workflow")
    semantic = {key: manifest.get(key) for key in fields}
    if "learner_initialization" in manifest:
        semantic["learner_initialization"] = manifest["learner_initialization"]
    return {"schema": SCHEMA, "manifest_sha256": digest(semantic),
            "implementation_sha256": implementation_digest(),
            "optimizer_seed": manifest.get("workflow", {}).get(
                "optimizer_seed", canonical_seed("optimizer", base_seed, manifest["round"]))}


def prepare(directory, expected):
    if expected is None:
        return None
    directory = Path(directory)
    path = directory / "fit_identity.json"
    if path.exists():
        if read_json(path) != expected:
            raise ValueError("Cached fit belongs to a different frozen calibration/configuration revision")
    else:
        if ((directory / "fit_candidates.json").exists() or
                (directory / "fit.json").exists() or
                any((directory / "fit_checkpoints").glob("*.npz"))):
            raise ValueError("Unidentified calibration fit cache; use a new unfrozen round or explicitly migrate the old stage")
        atomic_json(path, expected)
    return digest(expected)


def validate_pool(directory, pool, expected_digest):
    if expected_digest is None:
        return
    if pool.get("fit_cache_identity") != expected_digest:
        raise ValueError("Cached candidate pool has a different calibration identity")
    points = pool.get("candidates", [])
    if not points:
        raise ValueError("Cached candidate pool is empty")
    for point in points:
        path = run_path(directory, point["checkpoint"])
        if not path.is_file() or file_hash(path) != point.get("sha256"):
            raise ValueError("Cached candidate weights changed after fitting")


def validate_winner(path, summary, expected_digest):
    if expected_digest is not None and (
            summary.get("fit_cache_identity") != expected_digest or
            summary.get("candidate_sha256") != file_hash(path)):
        raise ValueError("Cached winner weights/configuration differ from frozen calibration fit")
