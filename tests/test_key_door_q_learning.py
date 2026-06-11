from __future__ import annotations

import unittest

from examples.envs import KeyDoorGridEnv
from examples.learners import (
    ACTION_NAMES,
    ALGORITHMS,
    compare_techniques,
    greedy_rollout,
    stable_solve_episode,
    train_q_learning,
    train_td_control,
)


class KeyDoorQLearningTest(unittest.TestCase):
    def test_learns_key_then_goal_policy(self) -> None:
        result = train_q_learning(episodes=500, seed=7)
        trace = greedy_rollout(result.q_values, seed=7)

        self.assertTrue(trace[-1].terminated)
        self.assertFalse(trace[-1].truncated)
        self.assertEqual(len(trace), 9)
        self.assertEqual([ACTION_NAMES[step.action] for step in trace[:3]], ["up"] * 3)
        self.assertEqual(trace[2].tile, "key")
        self.assertEqual(trace[-1].tile, "goal")
        self.assertNotIn("locked_goal", [step.tile for step in trace])
        self.assertAlmostEqual(sum(step.reward for step in trace), 1.11)

    def test_rejects_invalid_training_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "episodes must be at least 1"):
            train_q_learning(episodes=0)
        with self.assertRaisesRegex(ValueError, "learning_rate must be"):
            train_q_learning(learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "discount must be"):
            train_q_learning(discount=-0.1)
        with self.assertRaisesRegex(ValueError, "epsilon must be"):
            train_q_learning(epsilon=1.1)
        with self.assertRaisesRegex(ValueError, "min_epsilon must be"):
            train_q_learning(min_epsilon=-0.1)
        with self.assertRaisesRegex(ValueError, "epsilon_decay must be"):
            train_q_learning(epsilon_decay=0.0)

    def test_q_learning_solves_extended_layout(self) -> None:
        result = train_q_learning(
            episodes=800,
            seed=7,
            env=KeyDoorGridEnv(layout="extended"),
        )
        trace = greedy_rollout(
            result.q_values,
            env=KeyDoorGridEnv(layout="extended"),
            seed=7,
        )

        self.assertTrue(trace[-1].terminated)
        self.assertFalse(trace[-1].truncated)
        self.assertEqual(trace[4].tile, "key")
        self.assertEqual(trace[-1].tile, "goal")
        self.assertEqual(len(trace), 15)
        self.assertAlmostEqual(sum(step.reward for step in trace), 1.05)

    def test_td_control_algorithms_solve_extended_layout(self) -> None:
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                result = train_td_control(
                    algorithm=algorithm,
                    episodes=800,
                    seed=3,
                    eval_every=20,
                    env=KeyDoorGridEnv(layout="extended"),
                )
                trace = greedy_rollout(
                    result.q_values,
                    env=KeyDoorGridEnv(layout="extended"),
                    seed=3,
                )

                self.assertIsNotNone(result.episodes_to_solve)
                self.assertTrue(trace[-1].terminated)
                self.assertFalse(trace[-1].truncated)
                self.assertLessEqual(len(trace), 18)
                self.assertGreater(sum(step.reward for step in trace), 0.8)

    def test_compare_techniques_reports_measured_summaries(self) -> None:
        summaries, results = compare_techniques(
            algorithms=("dyna_q", "q_learning"),
            seeds=(1, 2),
            episodes=200,
            eval_every=20,
            planning_steps=15,
        )

        self.assertEqual({summary.algorithm for summary in summaries}, {"dyna_q", "q_learning"})
        for summary in summaries:
            self.assertEqual(summary.solved_runs, 2)
            self.assertEqual(summary.total_runs, 2)
            self.assertIsNotNone(summary.median_episodes_to_solve)
            self.assertGreater(summary.mean_final_return, 0.9)
            self.assertLessEqual(summary.mean_greedy_steps, 18)
            self.assertEqual(len(results[summary.algorithm]), 2)

    def test_stable_solve_episode_requires_remaining_evaluations_to_solve(self) -> None:
        result = train_td_control(
            algorithm="dyna_q",
            episodes=80,
            seed=1,
            eval_every=20,
            env=KeyDoorGridEnv(layout="extended"),
        )

        self.assertEqual(stable_solve_episode(result.evaluations), 20)

    def test_rejects_invalid_td_control_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "algorithm must be one of"):
            train_td_control(algorithm="monte_carlo")
        with self.assertRaisesRegex(ValueError, "planning_steps must be"):
            train_td_control(algorithm="dyna_q", planning_steps=-1)
        with self.assertRaisesRegex(ValueError, "eval_every must be"):
            train_td_control(algorithm="q_learning", eval_every=0)


if __name__ == "__main__":
    unittest.main()
