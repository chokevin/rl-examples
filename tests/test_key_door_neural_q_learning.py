from __future__ import annotations

import unittest

import torch

from examples.envs import KeyDoorGridEnv
from examples.learners.key_door_neural_q_learning import (
    KeyDoorQNetwork,
    compare_optimizers,
    default_learning_rate,
    encode_state,
    greedy_q_rollout,
    make_optimizer,
    state_vector_size,
    train_neural_q_learning,
)


class KeyDoorNeuralQLearningTest(unittest.TestCase):
    def test_encodes_key_door_state_as_one_hot(self) -> None:
        env = KeyDoorGridEnv(layout="classic")

        start = encode_state((3, 0, 0), env)
        keyed = encode_state((0, 0, 1), env)

        self.assertEqual(state_vector_size(env), 32)
        self.assertEqual(start.shape, (32,))
        self.assertEqual(float(start.sum()), 1.0)
        self.assertEqual(float(keyed.sum()), 1.0)
        self.assertEqual(int(torch.argmax(start).item()), 12)
        self.assertEqual(int(torch.argmax(keyed).item()), 16)

    def test_adam_neural_q_learning_solves_classic_layout(self) -> None:
        result = train_neural_q_learning(
            optimizer_name="adam",
            episodes=200,
            seed=1,
            eval_every=50,
        )
        trace = greedy_q_rollout(
            result.q_network,
            env=KeyDoorGridEnv(layout="classic"),
            seed=1,
        )

        self.assertEqual(result.env_layout, "classic")
        self.assertEqual(result.optimizer_name, "adam")
        self.assertEqual(result.learning_rate, default_learning_rate("adam"))
        self.assertEqual(len(result.episode_returns), 200)
        self.assertEqual(len(result.episode_losses), 200)
        self.assertTrue(result.evaluations[-1].solved)
        self.assertTrue(trace[-1].terminated)
        self.assertFalse(trace[-1].truncated)
        self.assertEqual(trace[2].tile, "key")
        self.assertEqual(trace[-1].tile, "goal")
        self.assertEqual(len(trace), 9)
        self.assertGreater(sum(step.reward for step in trace), 1.0)

    def test_training_updates_network_weights(self) -> None:
        env = KeyDoorGridEnv(layout="classic")
        torch.manual_seed(5)
        initial = KeyDoorQNetwork(state_size=state_vector_size(env)).state_dict()
        initial_weights = {
            name: parameter.clone() for name, parameter in initial.items()
        }

        result = train_neural_q_learning(
            optimizer_name="sgd",
            episodes=1,
            seed=5,
            eval_every=1,
        )

        self.assertTrue(
            any(
                not torch.allclose(parameter, initial_weights[name])
                for name, parameter in result.q_network.state_dict().items()
            )
        )

    def test_compare_optimizers_reports_measured_summaries(self) -> None:
        summaries, results = compare_optimizers(
            optimizers=("sgd", "adam"),
            seeds=(1, 2),
            episodes=40,
            eval_every=20,
        )

        self.assertEqual(
            {summary.optimizer_name for summary in summaries},
            {"sgd", "adam"},
        )
        for summary in summaries:
            self.assertEqual(summary.total_runs, 2)
            self.assertIn(summary.solved_runs, {0, 1, 2})
            self.assertEqual(len(results[summary.optimizer_name]), 2)
            self.assertGreaterEqual(summary.mean_greedy_steps, 1.0)

    def test_rejects_invalid_neural_q_parameters(self) -> None:
        env = KeyDoorGridEnv(layout="classic")
        network = KeyDoorQNetwork(state_size=state_vector_size(env))

        with self.assertRaisesRegex(ValueError, "optimizer_name must be one of"):
            default_learning_rate("muon")
        with self.assertRaisesRegex(ValueError, "learning_rate must be"):
            make_optimizer("sgd", network.parameters(), learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "episodes must be at least 1"):
            train_neural_q_learning(episodes=0)
        with self.assertRaisesRegex(ValueError, "discount must be"):
            train_neural_q_learning(discount=-0.1)
        with self.assertRaisesRegex(ValueError, "epsilon must be"):
            train_neural_q_learning(epsilon=1.1)
        with self.assertRaisesRegex(ValueError, "epsilon_decay must be"):
            train_neural_q_learning(epsilon_decay=0.0)
        with self.assertRaisesRegex(ValueError, "min_epsilon must be"):
            train_neural_q_learning(min_epsilon=-0.1)
        with self.assertRaisesRegex(ValueError, "hidden_size must be"):
            train_neural_q_learning(hidden_size=0)
        with self.assertRaisesRegex(ValueError, "eval_every must be"):
            train_neural_q_learning(eval_every=0)
        with self.assertRaisesRegex(ValueError, "state row is outside"):
            encode_state((4, 0, 0), env)


if __name__ == "__main__":
    unittest.main()
