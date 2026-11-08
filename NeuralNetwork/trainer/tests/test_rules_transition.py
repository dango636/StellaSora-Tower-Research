import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from stella.catalogue_engine import CatalogueEngine, GAMES
from stella.environment import sim
from stella.io_utils import StopRequested, atomic_json, read_json
from stella.catalogue_service import CatalogueEvaluationSession
from stella.preset_catalogue import catalogue_presets


class RulesTransitionTests(unittest.TestCase):
    def test_rebind_keeps_worker_and_requires_drained_pipeline(self):
        presets = catalogue_presets()[:2]
        session = CatalogueEvaluationSession(presets, seed=7)
        worker = session.process = Mock()
        session.connection = Mock()
        session.connection.poll.return_value = True
        session.connection.recv.return_value = {"rebound": True}
        session.rebind_presets(presets[:1])
        self.assertIs(session.process, worker)
        self.assertEqual(session.connection.send.call_args.args[0]["kind"], "presets")
        self.assertTrue(session.matches(presets[:1], 7, GAMES, None))
        session.pending = 1
        with self.assertRaises(RuntimeError):
            session.rebind_presets(presets)

    def test_grouped_baseline_resumes_without_changing_weights_or_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = CatalogueEngine.__new__(CatalogueEngine)
            engine.run_dir = Path(folder)
            engine.state_path = engine.run_dir / "run_state.json"
            engine.presets = [SimpleNamespace(name=str(i), fingerprint=str(i), total=100) for i in range(3)]
            engine.config = {"seed": 1}
            engine.control = lambda: None
            engine.publish = lambda *args, **kwargs: None
            engine.state = dict(rules_version="old", next_round=125,
                teacher_bank=["a", "b", "a"], models={"a": "a.npz", "b": "b.npz"},
                teacher_id="a", teacher_checkpoint="a.npz", teacher_scores="old.npz",
                last_candidate_checkpoint="a.npz")
            engine.log = lambda message: None
            calls = []

            def evaluate(checkpoint, presets, directory, message, **kwargs):
                calls.append((checkpoint.name, [p.name for p in presets], kwargs))
                if len(calls) == 2:
                    raise StopRequested()
                return np.full((len(presets), GAMES), 70 if checkpoint.name == "a.npz" else 80,
                               dtype=np.int16)

            engine.evaluate_matrix = evaluate
            with patch("stella.catalogue_engine.CatalogueEvaluationSession"):
                with self.assertRaises(StopRequested):
                    engine.migrate_rules()
            interrupted = read_json(engine.state_path)
            self.assertTrue(interrupted["migration_baseline_pending"])
            self.assertEqual(interrupted["previous_rules_version"], "old")
            engine.state = interrupted
            with patch("stella.catalogue_engine.CatalogueEvaluationSession"):
                engine.migrate_rules()
            self.assertFalse(engine.state["migration_baseline_pending"])
            self.assertEqual(engine.state["teacher_bank"], ["a", "b", "a"])
            self.assertEqual(engine.state["last_candidate_checkpoint"], "a.npz")
            self.assertEqual(engine.state["rules_effective_round"], 125)
            self.assertEqual(engine.state["teacher_scores_rules_version"], sim.MODEL_VERSION)
            self.assertEqual(engine.state["best_candidate_mean"], 70)
            self.assertAlmostEqual(engine.state["composite_teacher_mean"], 220 / 3)
            self.assertEqual(calls[-3][1], ["0", "2"])
            self.assertEqual(calls[-2][1], ["1"])
            self.assertEqual(calls[-1][1], ["0", "1", "2"])

    def test_old_manifest_is_rejected_before_reusing_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = CatalogueEngine.__new__(CatalogueEngine)
            directory = Path(folder)
            atomic_json(directory / "manifest.json", {"rules_version": "old"})
            with self.assertRaisesRegex(ValueError, "旧规则"):
                engine.prepare_manifest(125, directory)


if __name__ == "__main__":
    unittest.main()
