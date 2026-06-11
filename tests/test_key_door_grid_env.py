from __future__ import annotations

import unittest

import numpy as np

from examples.envs import KeyDoorGridEnv


class KeyDoorGridEnvTest(unittest.TestCase):
    def test_reset_returns_deterministic_initial_observation(self) -> None:
        env = KeyDoorGridEnv(height=4, width=4, max_steps=20)

        observation, info = env.reset(seed=123)
        env.step(0)
        repeated_observation, repeated_info = env.reset(seed=123)

        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(observation, expected)
        np.testing.assert_allclose(repeated_observation, expected)
        self.assertEqual(info, {"position": (3, 0), "has_key": False})
        self.assertEqual(repeated_info, info)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, 4)

    def test_collects_key_then_terminates_at_goal(self) -> None:
        env = KeyDoorGridEnv(max_steps=12)
        env.reset(seed=123)

        for action in (0, 0):
            observation, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(reward, -0.01)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["has_key"])

        observation, reward, terminated, truncated, info = env.step(np.array(0))

        np.testing.assert_allclose(
            observation,
            np.array([0.0, 0.0, 1.0, 0.25], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, 0.19)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (0, 0))
        self.assertTrue(info["has_key"])
        self.assertEqual(info["tile"], "key")

        for action in (1, 1, 1, 2, 2):
            observation, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(reward, -0.01)
            self.assertFalse(terminated)
            self.assertFalse(truncated)

        observation, reward, terminated, truncated, info = env.step(2)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 1.0, 1.0, 0.75], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, 0.99)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (3, 3))
        self.assertEqual(info["tile"], "goal")
        self.assertAlmostEqual(float(info["episode_return"]), 1.11)
        self.assertEqual(info["episode_length"], 9)

    def test_locked_goal_blocks_without_key(self) -> None:
        env = KeyDoorGridEnv(max_steps=6)
        env.reset(seed=123)

        env.step(1)
        observation, reward, terminated, truncated, info = env.step(1)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 2 / 3, 0.0, 2 / 6], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, -0.11)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (3, 2))
        self.assertEqual(info["tile"], "mud")

        observation, reward, terminated, truncated, info = env.step(1)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 2 / 3, 0.0, 3 / 6], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, -0.06)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (3, 2))
        self.assertEqual(info["tile"], "locked_goal")
        self.assertTrue(info["blocked"])

    def test_wall_and_boundary_block_movement(self) -> None:
        env = KeyDoorGridEnv()
        env.reset(seed=123)

        observation, reward, terminated, truncated, info = env.step(3)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 0.0, 0.0, 1 / 20], dtype=np.float32),
        )
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (3, 0))
        self.assertEqual(info["tile"], "boundary")
        self.assertTrue(info["blocked"])

        env.reset(seed=123)
        env.step(0)
        env.step(0)
        observation, reward, terminated, truncated, info = env.step(1)

        np.testing.assert_allclose(
            observation,
            np.array([1 / 3, 0.0, 0.0, 3 / 20], dtype=np.float32),
        )
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (1, 0))
        self.assertEqual(info["tile"], "wall")
        self.assertTrue(info["blocked"])

    def test_truncates_at_max_steps_before_goal(self) -> None:
        env = KeyDoorGridEnv(max_steps=2)
        env.reset(seed=123)

        env.step(1)
        observation, reward, terminated, truncated, info = env.step(1)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 2 / 3, 0.0, 1.0], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, -0.11)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["position"], (3, 2))
        self.assertEqual(info["tile"], "mud")
        self.assertAlmostEqual(float(info["episode_return"]), -0.12)
        self.assertEqual(info["episode_length"], 2)

    def test_extended_layout_adds_longer_key_and_goal_route(self) -> None:
        env = KeyDoorGridEnv(layout="extended")
        observation, info = env.reset(seed=123)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(info, {"position": (5, 0), "has_key": False})
        self.assertEqual(env.height, 6)
        self.assertEqual(env.width, 6)
        self.assertEqual(env.max_steps, 45)

        for action in (0, 0, 0, 0):
            observation, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(reward, -0.01)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["has_key"])

        observation, reward, terminated, truncated, info = env.step(0)

        np.testing.assert_allclose(
            observation,
            np.array([0.0, 0.0, 1.0, 5 / 45], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, 0.19)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["has_key"])
        self.assertEqual(info["tile"], "key")

        for action in (1, 1, 1, 1, 1, 2, 2, 2, 2):
            observation, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(reward, -0.01)
            self.assertFalse(terminated)
            self.assertFalse(truncated)

        observation, reward, terminated, truncated, info = env.step(2)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 1.0, 1.0, 15 / 45], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, 0.99)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (5, 5))
        self.assertEqual(info["tile"], "goal")
        self.assertAlmostEqual(float(info["episode_return"]), 1.05)
        self.assertEqual(info["episode_length"], 15)

    def test_extended_layout_has_walls_mud_and_locked_goal(self) -> None:
        env = KeyDoorGridEnv(layout="extended")
        env.reset(seed=123)

        env.step(0)
        env.step(1)
        observation, reward, terminated, truncated, info = env.step(1)

        np.testing.assert_allclose(
            observation,
            np.array([4 / 5, 1 / 5, 0.0, 3 / 45], dtype=np.float32),
        )
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (4, 1))
        self.assertEqual(info["tile"], "wall")
        self.assertTrue(info["blocked"])

        env.reset(seed=123)
        env.step(1)
        env.step(1)
        observation, reward, terminated, truncated, info = env.step(1)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 3 / 5, 0.0, 3 / 45], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, -0.11)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (5, 3))
        self.assertEqual(info["tile"], "mud")

        env.step(1)
        observation, reward, terminated, truncated, info = env.step(1)

        np.testing.assert_allclose(
            observation,
            np.array([1.0, 4 / 5, 0.0, 5 / 45], dtype=np.float32),
        )
        self.assertAlmostEqual(reward, -0.06)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["position"], (5, 4))
        self.assertEqual(info["tile"], "locked_goal")
        self.assertTrue(info["blocked"])

    def test_rejects_invalid_action(self) -> None:
        env = KeyDoorGridEnv()
        env.reset(seed=123)

        with self.assertRaisesRegex(ValueError, "action must be 0, 1, 2, or 3"):
            env.step(4)

    def test_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "height must be at least 3"):
            KeyDoorGridEnv(height=2)
        with self.assertRaisesRegex(ValueError, "width must be at least 3"):
            KeyDoorGridEnv(width=2)
        with self.assertRaisesRegex(ValueError, "max_steps must be at least 1"):
            KeyDoorGridEnv(max_steps=0)
        with self.assertRaisesRegex(ValueError, "layout must be"):
            KeyDoorGridEnv(layout="maze")
        with self.assertRaisesRegex(ValueError, "extended layout must use"):
            KeyDoorGridEnv(height=5, layout="extended")


if __name__ == "__main__":
    unittest.main()
