"""Freeze a learning starting point independently of frozen label teachers.

The existing last round's formally selected candidate can carry small updates
across rounds even when it has not replaced the historical best single model.
This does not route labels, change sampling behavior, or select/promote models.
No optimizer state or additional model checkpoint is introduced.
"""
from __future__ import annotations

from .io_utils import file_hash
from .run_paths import run_path

PROTOCOL = "last_candidate_continuation_v1"


def freeze_initialization(training, state, run_dir, number):
    """Return explicit immutable provenance only for an opted-in new round."""
    requested = training.get("learner_initialization")
    if requested is None:
        return None  # Historical manifests and their training semantics survive.
    if requested not in ("best", "last_candidate"):
        raise ValueError("Invalid training.learner_initialization")
    checkpoint = state["teacher_checkpoint"]
    model_id = state["teacher_id"]
    source = "best"
    if requested == "last_candidate":
        previous = state.get("last_candidate_checkpoint")
        if previous is None and number == 1:
            source = "best_bootstrap"
        else:
            if not isinstance(previous, str) or not previous:
                raise ValueError("Last candidate initialization is missing after the first round")
            model_id = f"M{number - 1:06d}"
            registered = state.get("models", {}).get(model_id)
            if (not registered or
                    run_path(run_dir, registered) != run_path(run_dir, previous)):
                raise ValueError("Last candidate initialization disagrees with the committed model registry")
            checkpoint, source = previous, "last_candidate"
    path = run_path(run_dir, checkpoint)
    if not path.is_file():
        raise ValueError("Frozen learner initialization checkpoint is missing: " + checkpoint)
    return {"protocol": PROTOCOL, "requested_source": requested, "source": source,
            "model_id": model_id, "checkpoint": checkpoint, "sha256": file_hash(path),
            "source_round": int(model_id.split("_")[0][1:])}


def fit_initial_model(manifest, run_dir):
    """Validate frozen initialization before accepting either a fit or its cache.

An old manifest without this record retains its original best-model start.
Missing/corrupt continuation files fail closed; resume never silently rewinds.
"""
    record = manifest.get("learner_initialization")
    requested = manifest.get("training_config", {}).get("learner_initialization")
    if record is None:
        if requested == "last_candidate":
            raise ValueError("Continuation requires a frozen learner_initialization record")
        return {"model_id": manifest["teacher_id"], "checkpoint": manifest["teacher_checkpoint"],
                "sha256": manifest["teacher_sha256"]}
    if not isinstance(record, dict) or record.get("protocol") != PROTOCOL:
        raise ValueError("Invalid frozen learner initialization protocol")
    if (requested not in ("best", "last_candidate") or
            record.get("requested_source") != requested):
        raise ValueError("Frozen learner initialization disagrees with the training configuration")
    source = record.get("source")
    valid_source = source == "best" if requested == "best" else source in ("last_candidate", "best_bootstrap")
    if not valid_source:
        raise ValueError("Invalid frozen learner initialization source")
    if (not isinstance(record.get("model_id"), str) or not record["model_id"] or
            not isinstance(record.get("checkpoint"), str) or not record["checkpoint"] or
            not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64 or
            type(record.get("source_round")) is not int or
            not 0 <= record["source_round"] < manifest["round"]):
        raise ValueError("Incomplete frozen learner initialization identity")
    if source == "last_candidate":
        if (record["source_round"] != manifest["round"] - 1 or
                record["model_id"] != f"M{record['source_round']:06d}"):
            raise ValueError("Continuation must use the immediately preceding committed candidate")
        if manifest.get("workflow"):
            raise ValueError("Continuation is separate from the legacy paired workflow")
    else:
        if source == "best_bootstrap" and manifest["round"] != 1:
            raise ValueError("Best-model bootstrap is only valid in the first round")
        if (record["model_id"] != manifest["teacher_id"] or
                record["sha256"] != manifest["teacher_sha256"] or
                run_path(run_dir, record["checkpoint"]) != run_path(run_dir, manifest["teacher_checkpoint"])):
            raise ValueError("Best-model initialization disagrees with its frozen baseline")
    path = run_path(run_dir, record["checkpoint"])
    if not path.is_file() or file_hash(path) != record["sha256"]:
        raise ValueError("Frozen learner initialization checkpoint checksum changed: " + record["checkpoint"])
    return record
