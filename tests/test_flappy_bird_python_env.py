from __future__ import annotations

import unittest

import numpy as np

from examples.envs import CFlappyBirdEnv


class CFlappyBirdEnvTest(unittest.TestCase):
    def test_reset_returns_observation_in_space(self) -> None:
        env = CFlappyBirdEnv(seed=123)

        observation, info = env.reset(seed=123)

        self.assertEqual(info, {"score": 0})
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, 2)

    def test_flap_moves_bird_up(self) -> None:
        env = CFlappyBirdEnv(seed=123)
        observation, _ = env.reset(seed=123)

        next_observation, reward, terminated, truncated, info = env.step(np.array(1))

        self.assertLess(next_observation[0], observation[0])
        self.assertGreater(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["score"], 0)

    def test_invalid_action_is_rejected(self) -> None:
        env = CFlappyBirdEnv(seed=123)

        with self.assertRaisesRegex(ValueError, "action must be 0 or 1"):
            env.step(2)

    def test_truncates_at_python_max_steps(self) -> None:
        env = CFlappyBirdEnv(max_steps=1, seed=123)

        _, reward, terminated, truncated, info = env.step(0)

        self.assertGreater(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["episode_length"], 1)


if __name__ == "__main__":
    unittest.main()
