"""Meaningful manifest invariants for complete, independent preset coverage."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import unittest
from unittest.mock import patch

from stella.catalogue_coverage import COVERAGE_PROTOCOL, SEED_FIELDS, transform_sampling_tasks
from stella.preset_catalogue import CATALOGUE_SIZE, catalogue_presets
from stella.presets import canonical_seed


def source_manifest():
    presets = catalogue_presets()
    bank = [f"M{i % 7:06d}" for i in range(CATALOGUE_SIZE)]
    tasks = []
    for index, preset in enumerate(presets):
        split = "val" if index % 4 == 0 else "train"
        student = index % 2 == 0
        tasks.append({
            "family_id": f"r000072_f{index:05d}", "ordinal": index,
            "catalogue_id": index, "preset": preset.to_dict(),
            "teacher_id": bank[index], "split": split,
            "rules_version": "frozen_test_rules", "roots_per_family": 4,
            "hard_sampling": True,
            "rollouts_per_action": 512 if split == "val" else 256,
            "hard_rollouts_per_action": 1024 if split == "val" else 512,
            "epsilon": (0.0, 0.15, 0.5)[index % 3],
            "behavior_source": "best_candidate" if student else "teacher",
            "behavior_model_id": "M000071" if student else bank[index],
            "student_model_id": "M000071",
            **{field: canonical_seed("old_source", index, field) for field in SEED_FIELDS},
        })
    return tasks, bank


class CatalogueCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.bank = source_manifest()
        cls.output, cls.summary = transform_sampling_tasks(cls.tasks, 1357, 73, {}, cls.bank)

    def test_all_presets_have_equal_train_and_validation_quota(self):
        self.assertEqual(len(self.output), 2 * CATALOGUE_SIZE)
        self.assertEqual(self.summary["roots"], 19840)
        self.assertEqual(self.summary["root_count_ratio"], 1.0)
        for split, quota in (("train", 3), ("val", 1)):
            rows = [task for task in self.output if task["split"] == split]
            self.assertEqual(Counter(task["catalogue_id"] for task in rows),
                             Counter(range(CATALOGUE_SIZE)))
            self.assertEqual({task["roots_per_family"] for task in rows}, {quota})
            for task in rows:
                self.assertEqual(task["teacher_id"], self.bank[task["catalogue_id"]])
                self.assertEqual(len(task["root_selection_schedule"]), quota)
        self.assertTrue(all("hard" in task["root_selection_schedule"]
                            for task in self.output if task["split"] == "train"))
        validation_roles = Counter(task["root_selection_schedule"][0]
                                   for task in self.output if task["split"] == "val")
        self.assertEqual(validation_roles, {"natural": CATALOGUE_SIZE})

    def test_independent_families_and_all_four_random_streams(self):
        families = [task["family_id"] for task in self.output]
        self.assertEqual(len(families), len(set(families)))
        seeds = [task[field] for task in self.output for field in SEED_FIELDS]
        self.assertEqual(len(seeds), len(set(seeds)))
        old_seeds = {task[field] for task in self.tasks for field in SEED_FIELDS}
        self.assertFalse(set(seeds) & old_seeds)
        for family in families:
            self.assertEqual(f"{family}_root00".rsplit("_root", 1)[0], family)
        self.assertEqual(self.summary["distinct_sampling_seeds"], 8 * CATALOGUE_SIZE)

    def test_legacy_validation_membership_does_not_change_new_quotas(self):
        for task in self.output:
            expected = (256, 512) if task["split"] == "train" else (512, 1024)
            self.assertEqual((task["rollouts_per_action"], task["hard_rollouts_per_action"]), expected)
            original = self.tasks[task["catalogue_id"]]
            for field in ("teacher_id", "student_model_id", "preset"):
                self.assertEqual(task[field], original[field])
            if task["split"] == "train":
                for field in ("behavior_source", "behavior_model_id", "epsilon"):
                    self.assertEqual(task[field], original[field])
            else:
                self.assertEqual(task["behavior_source"], "best_candidate")
                self.assertEqual(task["behavior_model_id"], original["student_model_id"])
                self.assertEqual(task["epsilon"], 0.0)
        self.assertEqual(self.tasks[0]["split"], "val")
        self.assertNotIn("coverage_protocol", self.tasks[0])

    def test_expanded_budget_and_explicit_rollout_settings(self):
        config = dict(train_roots=4, val_roots=2, train_rollouts_per_action=64,
                      val_rollouts_per_action=128, train_hard_rollouts_per_action=128,
                      val_hard_rollouts_per_action=256,
                      validation_on_policy=False, validation_root_mode="rotating")
        output, summary = transform_sampling_tasks(self.tasks, 1357, 74, config, self.bank)
        self.assertEqual(summary["roots"], 29760)
        self.assertEqual(summary["root_count_ratio"], 1.5)
        for task in output:
            self.assertEqual(task["roots_per_family"], 4 if task["split"] == "train" else 2)
            self.assertEqual(task["rollouts_per_action"], 64 if task["split"] == "train" else 128)
            previous = next(old for old in self.output[task["catalogue_id"] * 2:task["catalogue_id"] * 2 + 2]
                            if old["split"] == task["split"])
            self.assertEqual((task["phase_sampling_index"] - previous["phase_sampling_index"]) % 12, 1)
            if task["split"] == "val":
                original = self.tasks[task["catalogue_id"]]
                for field in ("behavior_source", "behavior_model_id", "epsilon"):
                    self.assertEqual(task[field], original[field])
        self.assertEqual(set(summary["root_role_counts"]["val"]), {"natural", "phase", "hard"})
        prior_seeds = {task[field] for task in self.output for field in SEED_FIELDS}
        new_seeds = {task[field] for task in output for field in SEED_FIELDS}
        self.assertFalse(prior_seeds & new_seeds)

    def test_statistical_identity_does_not_depend_on_task_order_or_model_name(self):
        changed = deepcopy(self.tasks)
        bank = [teacher + "_replacement" for teacher in self.bank]
        for task in changed:
            task["teacher_id"] = bank[task["catalogue_id"]]
            if task["behavior_source"] == "teacher":
                task["behavior_model_id"] = task["teacher_id"]
        output, summary = transform_sampling_tasks(list(reversed(changed)), 1357, 73, {}, bank)
        by_key = {(task["catalogue_id"], task["split"]): task for task in self.output}
        for task in output:
            previous = by_key[task["catalogue_id"], task["split"]]
            for field in (*SEED_FIELDS, "family_id", "phase_sampling_index", "root_selection_schedule"):
                self.assertEqual(task[field], previous[field])
        self.assertNotEqual(summary["teacher_bank_sha256"], self.summary["teacher_bank_sha256"])

    def test_incomplete_duplicate_or_mismatched_catalogue_is_rejected(self):
        with self.assertRaises(ValueError):
            transform_sampling_tasks(self.tasks[:-1], 1, 73, {}, self.bank)
        duplicate = [*self.tasks[:-1], self.tasks[0]]
        with self.assertRaises(ValueError):
            transform_sampling_tasks(duplicate, 1, 73, {}, self.bank)
        mismatch = deepcopy(self.tasks)
        mismatch[0]["preset"] = self.tasks[1]["preset"]
        with self.assertRaisesRegex(ValueError, "catalogue_id"):
            transform_sampling_tasks(mismatch, 1, 73, {}, self.bank)
        mismatch = deepcopy(self.tasks)
        mismatch[0]["teacher_id"] = "unapproved_teacher"
        with self.assertRaisesRegex(ValueError, "Teacher routing"):
            transform_sampling_tasks(mismatch, 1, 73, {}, self.bank)

    def test_bad_config_and_seed_collisions_fail_closed(self):
        for config in ({"train_roots": 0}, {"val_roots": True}, {"typo": 3},
                       {"train_hard_rollouts_per_action": 2}, {"validation_on_policy": 1},
                       {"validation_root_mode": "typo"}):
            with self.assertRaises(ValueError):
                transform_sampling_tasks(self.tasks, 1, 73, config, self.bank)
        with patch("stella.catalogue_coverage.canonical_seed", return_value=42):
            with self.assertRaisesRegex(ValueError, "collision"):
                transform_sampling_tasks(self.tasks, 1, 73, {}, self.bank)

    def test_protocol_and_input_copy_are_explicit(self):
        self.assertEqual(self.summary["protocol"], COVERAGE_PROTOCOL)
        self.assertEqual(self.summary["split_unit"], "independent_full_source_trajectory")
        self.assertTrue(self.summary["same_presets_in_both_splits"])
        self.assertEqual(self.summary["required_sampler_support"], "root_selection_schedule")
        self.assertIsNot(self.output[0]["preset"], self.tasks[0]["preset"])
        self.assertIsNot(self.output[0]["preset"], self.output[1]["preset"])


if __name__ == "__main__":
    unittest.main()
