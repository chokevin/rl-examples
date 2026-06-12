from __future__ import annotations

import unittest

from examples.policy_optimizer_catalog import (
    POLICY_OPTIMIZER_TECHNIQUES,
    technique_by_name,
)


class PolicyOptimizerCatalogTest(unittest.TestCase):
    def test_catalog_includes_requested_policy_optimizers(self) -> None:
        self.assertEqual(
            {technique.name for technique in POLICY_OPTIMIZER_TECHNIQUES},
            {"PPO", "TRPO", "GRPO"},
        )

    def test_techniques_are_not_described_as_q_learning_variants(self) -> None:
        for technique in POLICY_OPTIMIZER_TECHNIQUES:
            with self.subTest(technique=technique.name):
                self.assertIn("policy", technique.learns)
                self.assertIn("Q-learning", technique.q_learning_contrast)
                self.assertNotIn("Q-table", technique.mechanism)

    def test_lookup_is_case_insensitive_and_validated(self) -> None:
        self.assertEqual(technique_by_name("ppo").name, "PPO")
        self.assertEqual(technique_by_name("TrPo").name, "TRPO")
        self.assertEqual(technique_by_name("grpo").name, "GRPO")
        with self.assertRaisesRegex(ValueError, "name must be one of"):
            technique_by_name("dqn")


if __name__ == "__main__":
    unittest.main()
