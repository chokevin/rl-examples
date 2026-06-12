from __future__ import annotations

import unittest

import numpy as np
import torch

from examples.learners.mujoco_batched_cartpole import (
    BATCHED_CARTPOLE_OPTIMIZERS,
    DEFAULT_BATCH_SIZE,
    default_learning_rate,
    make_inverted_pendulum_model,
    make_optimizer,
    make_rollout_data,
    rollout_linear_policies,
    sample_initial_states,
    train_batched_es_cartpole,
)


class MuJoCoBatchedCartPoleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = make_inverted_pendulum_model()
        cls.data = make_rollout_data(cls.model, data_threads=1)

    def test_samples_full_physics_initial_states(self) -> None:
        states = sample_initial_states(self.model, batch_size=4, seed=1)

        self.assertEqual(states.shape, (4, 5))
        self.assertTrue(np.allclose(states[:, 0], 0.0))
        self.assertLessEqual(float(np.abs(states[:, 1:]).max()), 0.01)

    def test_batched_rollout_evaluates_many_linear_policies(self) -> None:
        teacher = np.array([0.0, 10.0, 0.0, 1.0])

        stats, returns = rollout_linear_policies(
            self.model,
            self.data,
            teacher,
            batch_size=16,
            max_steps=20,
            seed=2,
            target_return=20.0,
        )

        self.assertEqual(returns.shape, (16,))
        self.assertEqual(stats.batch_size, 16)
        self.assertEqual(stats.max_steps, 20)
        self.assertEqual(stats.mean_return, 20.0)
        self.assertEqual(stats.solved_fraction, 1.0)
        self.assertGreater(stats.physics_steps_per_second, 0.0)

    def test_tiny_es_training_reports_generation_shapes(self) -> None:
        result = train_batched_es_cartpole(
            optimizer_name="adam",
            batch_size=32,
            eval_batch_size=16,
            generations=1,
            max_steps=8,
            target_return=100.0,
            seed=3,
            data_threads=1,
        )

        self.assertEqual(result.optimizer_name, "adam")
        self.assertEqual(result.learning_rate, default_learning_rate("adam"))
        self.assertEqual(result.batch_size, 32)
        self.assertEqual(result.eval_batch_size, 16)
        self.assertEqual(result.generations, 1)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(len(result.final_policy_parameters), 4)
        self.assertIsNone(result.learned_generation)
        self.assertGreaterEqual(result.total_rollout_seconds, 0.0)

    def test_rejects_invalid_batched_cartpole_parameters(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4))

        with self.assertRaisesRegex(ValueError, "optimizer_name must be one of"):
            default_learning_rate("muon")
        with self.assertRaisesRegex(ValueError, "learning_rate must be"):
            make_optimizer("sgd", [parameter], learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "data_threads must be"):
            make_rollout_data(self.model, data_threads=0)
        with self.assertRaisesRegex(ValueError, "batch_size must be"):
            sample_initial_states(self.model, batch_size=0, seed=1)
        with self.assertRaisesRegex(ValueError, "noise must be"):
            sample_initial_states(self.model, batch_size=1, seed=1, noise=-0.1)
        with self.assertRaisesRegex(ValueError, "policy_parameters must have shape"):
            rollout_linear_policies(
                self.model,
                self.data,
                np.zeros(3),
                batch_size=4,
                max_steps=1,
            )
        with self.assertRaisesRegex(ValueError, "generations must be"):
            train_batched_es_cartpole(generations=0)
        with self.assertRaisesRegex(ValueError, "exploration_sigma must be"):
            train_batched_es_cartpole(exploration_sigma=0.0)
        with self.assertRaisesRegex(ValueError, "sigma_decay must be"):
            train_batched_es_cartpole(sigma_decay=0.0)
        self.assertEqual(DEFAULT_BATCH_SIZE, 8_192)
        self.assertIn("adam", BATCHED_CARTPOLE_OPTIMIZERS)


if __name__ == "__main__":
    unittest.main()
