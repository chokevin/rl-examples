from __future__ import annotations

import unittest

import numpy as np

from examples.envs import LineWorldEnv


class LineWorldEnvTest(unittest.TestCase):
    def test_reset_returns_initial_observation(self) -> None:
        env = LineWorldEnv(size=5, max_steps=8)

        observation, info = env.reset(seed=123)

        np.testing.assert_allclose(observation, np.array([0.0, 0.0], dtype=np.float32))
        self.assertEqual(info, {"position": 0})
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, 2)

    def test_moves_right_to_terminal_goal(self) -> None:
        env = LineWorldEnv(size=3, max_steps=5)
        env.reset(seed=123)

        observation, reward, terminated, truncated, info = env.step(1)

        np.testing.assert_allclose(observation, np.array([0.5, 0.2], dtype=np.float32))
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {"position": 1})

        observation, reward, terminated, truncated, info = env.step(np.array(1))

        np.testing.assert_allclose(observation, np.array([1.0, 0.4], dtype=np.float32))
        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], 2)
        self.assertAlmostEqual(float(info["episode_return"]), 0.99)
        self.assertEqual(info["episode_length"], 2)

    def test_truncates_at_max_steps_before_goal(self) -> None:
        env = LineWorldEnv(size=5, max_steps=2)
        env.reset(seed=123)

        env.step(0)
        observation, reward, terminated, truncated, info = env.step(0)

        np.testing.assert_allclose(observation, np.array([0.0, 1.0], dtype=np.float32))
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["position"], 0)
        self.assertAlmostEqual(float(info["episode_return"]), -0.02)
        self.assertEqual(info["episode_length"], 2)

    def test_rejects_invalid_action(self) -> None:
        env = LineWorldEnv()
        env.reset(seed=123)

        with self.assertRaisesRegex(ValueError, "action must be 0 or 1"):
            env.step(2)


if __name__ == "__main__":
    unittest.main()
