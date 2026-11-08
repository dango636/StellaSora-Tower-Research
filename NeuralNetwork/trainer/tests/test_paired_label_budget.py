"""Coverage, statistical stream separation, and fixed-budget edge cases."""
from copy import deepcopy
import unittest

import numpy as np

from stella.paired_label_budget import BRANCH_SEED_DOMAIN, plan_paired_label_budget
from stella.presets import canonical_seed


def root(name, differences, *, current=0, split="train", preset=7):
    diffs = np.asarray(differences, dtype=np.float64)
    if diffs.ndim == 1:
        diffs = diffs[None, :]
    values = np.vstack((np.full(diffs.shape[1], 10.0), 10.0+diffs))
    return dict(root_id=name, catalogue_id=preset, split=split, teacher_id="M000071_E004",
                student_action_index=current, legal_action_count=len(values),
                action_keys=[f"action_{i}" for i in range(len(values))],
                action_returns=values, label_seed=canonical_seed("pilot", name))


class PairedLabelBudgetTests(unittest.TestCase):
    def plan(self, roots, budget=1000, **settings):
        return plan_paired_label_budget(roots, budget, seed=77, round_number=73,
                                       min_rollouts=16, max_rollouts=128, **settings)

    def test_uncertain_material_error_receives_more_than_resolved_error(self):
        noisy = root("uncertain", np.tile([2.4, -1.6], 32))
        resolved = root("resolved", np.ones(64))
        output = self.plan([resolved, noisy], budget=300)
        by_id = {row["root_id"]: row for row in output["allocations"]}
        self.assertEqual(by_id["resolved"]["final_n"], 16)
        self.assertGreater(by_id["uncertain"]["final_n"], 16)
        self.assertLessEqual(output["summary"]["final_branches"], 300)

    def test_zero_gap_does_not_absorb_an_unbounded_precision_budget(self):
        tie = root("tie", np.tile([2.0, -2.0], 32))
        small = root("small", np.tile([2.02, -1.98], 32))
        output = self.plan([tie, small], budget=100000, near_tie_extra_rollouts=20)
        by_id = {row["root_id"]: row for row in output["allocations"]}
        self.assertEqual(by_id["tie"]["final_n"], 16)
        self.assertLessEqual(by_id["small"]["final_n"], 36)
        self.assertGreater(output["summary"]["unused_final_branch_budget"], 0)

    def test_tight_budget_reduces_floor_without_dropping_roots_or_actions(self):
        rows = [root("two", np.ones(64)), root("three", np.vstack((np.ones(64), np.full(64, -1.0))))]
        output = self.plan(rows, budget=25)
        self.assertEqual(output["summary"]["effective_min_rollouts"], 5)
        self.assertEqual(output["summary"]["roots_dropped"], 0)
        self.assertEqual([row["final_n"] for row in output["allocations"]], [5, 5])
        self.assertEqual(output["summary"]["final_branches"], 25)
        self.assertEqual([row["action_keys"] for row in output["allocations"]], [row["action_keys"] for row in rows])
        with self.assertRaisesRegex(ValueError, "preserve all roots/actions"):
            self.plan(rows, budget=9)

    def test_final_seed_blocks_are_fresh_and_pilot_cost_is_not_hidden(self):
        rows = [root("a", np.tile([2.4, -1.6], 32)), root("b", np.tile([1.7, -1.3], 32))]
        output = self.plan(rows)
        pilot = {canonical_seed(BRANCH_SEED_DOMAIN, row["label_seed"], k)
                 for row in rows for k in range(row["action_returns"].shape[1])}
        final = []
        for allocation in output["allocations"]:
            self.assertFalse(allocation["reuse_pilot_returns"])
            self.assertEqual(allocation["teacher_id"], rows[0]["teacher_id"])
            self.assertTrue(allocation["all_legal_actions_share_final_n"])
            final.extend(canonical_seed(BRANCH_SEED_DOMAIN, allocation["label_seed"], k)
                         for k in range(allocation["final_n"]))
        self.assertEqual(len(final), len(set(final)))
        self.assertFalse(pilot & set(final))
        summary = output["summary"]
        self.assertEqual(summary["pilot_branches_already_spent"], 256)
        self.assertEqual(summary["combined_pilot_and_final_branches"], summary["final_branches"]+256)
        self.assertEqual(summary["label_estimator"], "mean of fresh final block only; pilot excluded")

    def test_mixed_presets_splits_teachers_and_malformed_returns_are_rejected(self):
        first, second = root("a", np.ones(64)), root("b", np.ones(64))
        for key, value in (("catalogue_id", 8), ("split", "val"), ("teacher_id", "M_other"),
                           ("student_action_index", 3), ("legal_action_count", 3)):
            bad = deepcopy(second)
            bad[key] = value
            with self.assertRaises(ValueError):
                self.plan([first, bad])
        bad = deepcopy(second)
        bad["action_returns"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.plan([first, bad])
        bad = deepcopy(second)
        bad["cap"], bad["root_score"] = 80, 75
        with self.assertRaisesRegex(ValueError, "remaining possible score"):
            self.plan([bad])

    def test_hard_cap_and_budget_for_unequal_action_counts(self):
        rows = [root("a", np.tile([2.4, -1.6], 32)),
                root("b", np.vstack((np.tile([1.9, -1.5], 32), np.full(64, -0.7)))),
                root("c", np.vstack((np.tile([2.6, -1.4], 32), np.full(64, -0.3), np.full(64, -0.9))))]
        original = deepcopy(rows)
        for budget in (18, 37, 73, 197, 1000, 100000):
            output = self.plan(rows, budget=budget, allocation_block=7)
            used = sum(row["legal_action_count"]*row["final_n"] for row in output["allocations"])
            self.assertLessEqual(used, budget)
            self.assertEqual(used, output["summary"]["final_branches"])
            self.assertEqual(len(output["allocations"]), 3)
            self.assertTrue(all(2 <= row["final_n"] <= 128 for row in output["allocations"]))
        for before, after in zip(original, rows):
            self.assertTrue(np.array_equal(before["action_returns"], after["action_returns"]))
            self.assertNotIn("final_n", after)

    def test_order_does_not_change_semantic_seeds_or_allocations(self):
        rows = [root("a", np.tile([2.4, -1.6], 32)), root("b", np.tile([1.9, -1.5], 32))]
        forward, reverse = self.plan(rows, budget=173), self.plan(list(reversed(rows)), budget=173)
        extract = lambda result: {row["root_id"]: (row["final_n"], row["label_seed"], row["teacher_id"])
                                  for row in result["allocations"]}
        self.assertEqual(extract(forward), extract(reverse))

    def test_duplicate_pilot_stream_is_rejected(self):
        rows = [root("a", np.ones(64)), root("b", np.ones(64))]
        rows[1]["label_seed"] = rows[0]["label_seed"]
        with self.assertRaisesRegex(ValueError, "Duplicate pilot"):
            self.plan(rows)


if __name__ == "__main__":
    unittest.main()
