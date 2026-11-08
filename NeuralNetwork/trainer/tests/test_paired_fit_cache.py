"""Exercise paired fitting -> official winner -> resume cache propagation.

Training and games are stubbed; checkpoint bytes, identity validation, paired
pool assembly and the official winner save/resume paths are the actual code.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from stella import catalogue_engine as catalogue
from stella.engine import Engine
from stella.fit_cache import identity
from stella.io_utils import digest, file_hash, read_json
from stella.network import QNetwork
from stella.workflow import PAIRED_ANCHOR_V2


def test_paired_calibration_identity_reaches_winner_and_allows_exact_resume(tmp_path, monkeypatch):
    run = tmp_path / "run"
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    teacher = checkpoint_dir / "M000072.npz"
    QNetwork(input_dim=3, hidden=(2,), seed=1).save(teacher)
    teacher_hash = file_hash(teacher)
    config = {"objective_version": "action_gap_calibration_v3"}
    manifest = {
        "round": 73, "rules_version": "fixed", "preset_version": "fixed",
        "teacher_id": "M000072", "teacher_checkpoint": "checkpoints/M000072.npz",
        "teacher_sha256": teacher_hash,
        "teacher_models": {"M000072": {"checkpoint": "checkpoints/M000072.npz", "sha256": teacher_hash}},
        "teacher_bank": ["M000072", "M000072"],
        "training_config": config, "compute_config": {"backend": "cuda"},
        "evaluation_mode": catalogue.JOINT_EVALUATION, "evaluation_candidates": 6,
        "workflow": {
            "protocol": PAIRED_ANCHOR_V2, "optimizer_seed": 101,
            "evaluation_seed": 202, "development_seed": 303,
            "checkpoint_policy": "three_actual_epochs",
            "branches": [{"tag": tag, "training_config": dict(config, q_anchor_weight=anchor)}
                         for tag, anchor in (("control", .25), ("action", .05))],
        },
    }
    engine = catalogue.CatalogueEngine.__new__(catalogue.CatalogueEngine)
    engine.run_dir = run
    engine.config = {"seed": 11}
    engine.presets = [SimpleNamespace(fingerprint=f"preset-{i}") for i in range(2)]
    engine.control = lambda: None
    engine.publish = lambda *args, **kwargs: None
    calls = {"fit": 0, "evaluation": 0}

    def fake_base_fit(self, branch_manifest, branch_dir):
        calls["fit"] += 1
        candidates = []
        for epoch in (1, 2, 4):
            point = branch_dir / f"epoch_{epoch:03d}.npz"
            model = QNetwork.load(teacher)
            model.params["head_b"][:] = epoch + (10 if branch_manifest["workflow_branch"] == "action" else 0)
            model.save(point)
            candidates.append({"epoch": epoch, "checkpoint": point.name, "sha256": file_hash(point)})
        return {"fit": {"frozen_label_input_sha256": "same-ordered-labels",
                        "initial_parameter_sha256": "same-initial-parameters"},
                "candidates": candidates}

    def fake_evaluation(checkpoint, *args, **kwargs):
        calls["evaluation"] += 1
        value = int(QNetwork.load(checkpoint).params["head_b"][0])
        return np.full((2, 2), value, dtype=np.int16)

    monkeypatch.setattr(Engine, "fit_checkpoints", fake_base_fit)
    monkeypatch.setattr(catalogue, "GAMES", 2)
    monkeypatch.setattr(catalogue, "_enabled", lambda name: False)
    engine.evaluate_matrix = fake_evaluation
    directory = run / "rounds/round_000073"
    directory.mkdir(parents=True)
    winner = checkpoint_dir / "M000073.npz"

    summary = engine.train(manifest, directory, winner)
    expected = digest(identity(manifest, engine.config["seed"]))
    pool = read_json(directory / "fit_candidates.json")
    assert pool["fit_cache_identity"] == expected
    assert len(pool["candidates"]) == 6
    assert summary["fit_cache_identity"] == expected
    assert summary["candidate_sha256"] == file_hash(winner)
    assert summary["selected_branch"] == "action"
    assert calls == {"fit": 2, "evaluation": 6}

    cached = engine.train(manifest, directory, winner)
    assert cached == summary
    assert calls == {"fit": 2, "evaluation": 6}
    changed = copy.deepcopy(manifest)
    changed["training_config"]["new_setting"] = True
    with pytest.raises(ValueError, match="different frozen"):
        engine.train(changed, directory, winner)
    assert calls == {"fit": 2, "evaluation": 6}
