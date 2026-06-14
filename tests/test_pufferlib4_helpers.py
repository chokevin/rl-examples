from __future__ import annotations

import unittest

import numpy as np
import torch

from examples.envs import LineWorldEnv
from examples.pufferlib4 import SerialVectorEnv, make_flappy_policy, policy_logits_values


class PufferLib4HelpersTest(unittest.TestCase):
    def test_serial_vector_env_steps_and_auto_resets(self) -> None:
        envs = SerialVectorEnv([lambda: LineWorldEnv(size=3, max_steps=4) for _ in range(2)])
        try:
            observations, infos = envs.reset(seed=10)
            self.assertEqual(observations.shape, (2, 2))
            self.assertEqual(len(infos), 2)

            observations, rewards, terminals, truncations, infos = envs.step(
                np.ones(2, dtype=np.int32)
            )
            self.assertEqual(observations.shape, (2, 2))
            self.assertTrue(np.all(rewards == -0.01))
            self.assertFalse(np.any(terminals))
            self.assertFalse(np.any(truncations))

            _, rewards, terminals, truncations, infos = envs.step(np.ones(2, dtype=np.int32))
            self.assertTrue(np.all(rewards == 1.0))
            self.assertTrue(np.all(terminals))
            self.assertFalse(np.any(truncations))
            self.assertIn("final_info", infos[0])
        finally:
            envs.close()

    def test_flappy_policy_shapes_match_discrete_actions(self) -> None:
        policy = make_flappy_policy(observation_size=4, hidden_size=8)
        observations = torch.zeros((3, 4), dtype=torch.float32)

        logits, values = policy_logits_values(policy, observations)

        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(tuple(values.shape), (3,))


if __name__ == "__main__":
    unittest.main()
