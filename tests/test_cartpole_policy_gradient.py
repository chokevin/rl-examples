from __future__ import annotations

import math
import unittest

import torch

from examples.learners.cartpole_policy_gradient import (
    CARTPOLE_ENV_ID,
    CARTPOLE_OPTIMIZERS,
    CartPolePolicyNetwork,
    compare_cartpole_optimizers,
    default_learning_rate,
    discounted_returns,
    evaluate_policy,
    make_optimizer,
    sample_policy_action,
    train_cartpole_policy_gradient,
)


class CartPolePolicyGradientTest(unittest.TestCase):
    def test_policy_network_outputs_action_logits(self) -> None:
        network = CartPolePolicyNetwork(
            observation_size=4,
            action_count=2,
            hidden_size=8,
        )

        logits = network(torch.zeros(4))

        self.assertEqual(logits.shape, (2,))

    def test_discounted_returns_match_reward_to_go(self) -> None:
        returns = discounted_returns([1.0, 1.0, 1.0], discount=0.5, normalize=False)

        self.assertTrue(torch.allclose(returns, torch.tensor([1.75, 1.5, 1.0])))

    def test_samples_policy_action_with_log_probability(self) -> None:
        torch.manual_seed(3)
        network = CartPolePolicyNetwork(
            observation_size=4,
            action_count=2,
            hidden_size=4,
        )

        action, log_prob = sample_policy_action(network, [0.0, 0.0, 0.0, 0.0])

        self.assertIn(action, {0, 1})
        self.assertEqual(log_prob.shape, ())
        self.assertTrue(math.isfinite(float(log_prob.item())))

    def test_tiny_training_smoke_reports_shapes_and_metrics(self) -> None:
        result = train_cartpole_policy_gradient(
            optimizer_name="adam",
            episodes=3,
            seed=7,
            eval_every=2,
            eval_episodes=1,
            solve_window=2,
            solved_threshold=10_000.0,
            max_steps_per_episode=6,
        )

        self.assertEqual(result.env_id, CARTPOLE_ENV_ID)
        self.assertEqual(result.optimizer_name, "adam")
        self.assertEqual(result.learning_rate, default_learning_rate("adam"))
        self.assertEqual(result.seed, 7)
        self.assertEqual(len(result.episode_returns), 3)
        self.assertEqual(len(result.episode_lengths), 3)
        self.assertEqual(len(result.episode_losses), 3)
        self.assertEqual([evaluation.episode for evaluation in result.evaluations], [2, 3])
        self.assertEqual(result.total_samples, sum(result.episode_lengths))
        self.assertLessEqual(max(result.episode_lengths), 6)
        self.assertIsNone(result.solve_episode)
        self.assertIsNone(result.estimated_seconds_to_solve)
        self.assertGreaterEqual(result.wall_clock_seconds, 0.0)
        self.assertTrue(all(math.isfinite(loss) for loss in result.episode_losses))

    def test_training_updates_policy_weights(self) -> None:
        torch.manual_seed(5)
        initial = CartPolePolicyNetwork(
            observation_size=4,
            action_count=2,
            hidden_size=8,
        ).state_dict()
        initial_weights = {
            name: parameter.clone() for name, parameter in initial.items()
        }

        result = train_cartpole_policy_gradient(
            optimizer_name="sgd",
            episodes=1,
            seed=5,
            eval_every=1,
            eval_episodes=1,
            hidden_size=8,
            max_steps_per_episode=5,
        )

        self.assertTrue(
            any(
                not torch.allclose(parameter, initial_weights[name])
                for name, parameter in result.policy_network.state_dict().items()
            )
        )

    def test_compare_optimizers_reports_training_summaries(self) -> None:
        summaries, results = compare_cartpole_optimizers(
            optimizers=("sgd", "adam"),
            seeds=(1,),
            episodes=2,
            eval_every=1,
            eval_episodes=1,
            hidden_size=8,
            solved_threshold=10_000.0,
            max_steps_per_episode=5,
        )

        self.assertEqual(
            {summary.optimizer_name for summary in summaries},
            {"sgd", "adam"},
        )
        self.assertEqual(set(results), {"sgd", "adam"})
        for summary in summaries:
            self.assertIn(summary.optimizer_name, CARTPOLE_OPTIMIZERS)
            self.assertEqual(summary.total_runs, 1)
            self.assertEqual(summary.solved_runs, 0)
            self.assertIsNone(summary.median_solve_episode)
            self.assertIsNone(summary.median_seconds_to_solve)
            self.assertGreaterEqual(summary.mean_total_samples, 2.0)
            self.assertGreaterEqual(summary.mean_final_window_return, 1.0)

    def test_evaluate_policy_restores_training_mode(self) -> None:
        network = CartPolePolicyNetwork(
            observation_size=4,
            action_count=2,
            hidden_size=4,
        )
        network.train()

        evaluation = evaluate_policy(
            network,
            episode=1,
            seed=1,
            eval_episodes=1,
            max_steps_per_episode=3,
            solved_threshold=10_000.0,
        )

        self.assertTrue(network.training)
        self.assertEqual(evaluation.episode, 1)
        self.assertEqual(evaluation.eval_episodes, 1)
        self.assertLessEqual(evaluation.mean_length, 3)
        self.assertFalse(evaluation.solved)

    def test_rejects_invalid_cartpole_parameters(self) -> None:
        network = CartPolePolicyNetwork(
            observation_size=4,
            action_count=2,
            hidden_size=4,
        )

        with self.assertRaisesRegex(ValueError, "observation_size"):
            CartPolePolicyNetwork(observation_size=0, action_count=2)
        with self.assertRaisesRegex(ValueError, "action_count"):
            CartPolePolicyNetwork(observation_size=4, action_count=1)
        with self.assertRaisesRegex(ValueError, "optimizer_name must be one of"):
            default_learning_rate("muon")
        with self.assertRaisesRegex(ValueError, "learning_rate must be"):
            make_optimizer("sgd", network.parameters(), learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "rewards must contain"):
            discounted_returns([], discount=0.99)
        with self.assertRaisesRegex(ValueError, "discount must be"):
            discounted_returns([1.0], discount=-0.1)
        with self.assertRaisesRegex(ValueError, "episodes must be"):
            train_cartpole_policy_gradient(episodes=0)
        with self.assertRaisesRegex(ValueError, "discount must be"):
            train_cartpole_policy_gradient(discount=1.1)
        with self.assertRaisesRegex(ValueError, "hidden_size must be"):
            train_cartpole_policy_gradient(hidden_size=0)
        with self.assertRaisesRegex(ValueError, "eval_every must be"):
            train_cartpole_policy_gradient(eval_every=0)
        with self.assertRaisesRegex(ValueError, "eval_episodes must be"):
            train_cartpole_policy_gradient(eval_episodes=0)
        with self.assertRaisesRegex(ValueError, "solve_window must be"):
            train_cartpole_policy_gradient(solve_window=0)
        with self.assertRaisesRegex(ValueError, "solved_threshold must be"):
            train_cartpole_policy_gradient(solved_threshold=0.0)
        with self.assertRaisesRegex(ValueError, "max_steps_per_episode must be"):
            train_cartpole_policy_gradient(max_steps_per_episode=0)
        with self.assertRaisesRegex(ValueError, "unknown optimizers"):
            compare_cartpole_optimizers(optimizers=("muon",))
        with self.assertRaisesRegex(ValueError, "seeds must contain"):
            compare_cartpole_optimizers(seeds=())


if __name__ == "__main__":
    unittest.main()
