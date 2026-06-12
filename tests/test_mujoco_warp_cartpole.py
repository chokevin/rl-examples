from __future__ import annotations

import unittest

import mujoco
import numpy as np

from examples.learners.mujoco_batched_cartpole import make_inverted_pendulum_model, sample_initial_states
from examples.learners.mujoco_warp_cartpole import (
    DEFAULT_CUDAGRAPH_CONTROLLER_GAINS,
    MuJoCoWarpStatus,
    benchmark_mujoco_warp_cudagraph_steps,
    configure_cudagraph_benchmark_model,
    mujoco_warp_status,
    normalize_cudagraph_controller_gains,
    rollout_linear_policies_mujoco_warp,
    train_mujoco_warp_es_cartpole,
)


class MuJoCoWarpCartPoleTest(unittest.TestCase):
    def test_status_reports_optional_backend_shape(self) -> None:
        status = mujoco_warp_status()

        self.assertIsInstance(status, MuJoCoWarpStatus)
        self.assertIsInstance(status.available, bool)
        if status.available:
            self.assertIsNotNone(status.mujoco_warp_version)
            self.assertIsNotNone(status.warp_version)
            self.assertIsNotNone(status.device)
        else:
            self.assertIsNotNone(status.reason)

    def test_rejects_invalid_warp_parameters_before_backend_setup(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be"):
            rollout_linear_policies_mujoco_warp(np.zeros(4), batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_steps must be"):
            rollout_linear_policies_mujoco_warp(np.zeros(4), batch_size=1, max_steps=0)
        with self.assertRaisesRegex(ValueError, "target_return must be"):
            rollout_linear_policies_mujoco_warp(np.zeros(4), batch_size=1, target_return=0.0)
        with self.assertRaisesRegex(ValueError, "initial_noise must be"):
            rollout_linear_policies_mujoco_warp(np.zeros(4), batch_size=1, initial_noise=-0.1)
        with self.assertRaisesRegex(ValueError, "policy_parameters must have shape"):
            rollout_linear_policies_mujoco_warp(np.zeros(3), batch_size=1)
        with self.assertRaisesRegex(ValueError, "generations must be"):
            train_mujoco_warp_es_cartpole(generations=0)
        with self.assertRaisesRegex(ValueError, "graph_steps must be"):
            benchmark_mujoco_warp_cudagraph_steps(graph_steps=0)
        with self.assertRaisesRegex(ValueError, "graph_replays must be"):
            benchmark_mujoco_warp_cudagraph_steps(graph_replays=0)
        with self.assertRaisesRegex(ValueError, "controller_gains must contain four values"):
            benchmark_mujoco_warp_cudagraph_steps(controller_gains=(1.0, 2.0, 3.0))

    def test_cudagraph_controller_gains_are_normalized(self) -> None:
        self.assertEqual(
            normalize_cudagraph_controller_gains([0, 10, 0, 1]),
            DEFAULT_CUDAGRAPH_CONTROLLER_GAINS,
        )
        self.assertIsNone(normalize_cudagraph_controller_gains(None))

    def test_cudagraph_benchmark_model_options_set_disable_bits(self) -> None:
        model = make_inverted_pendulum_model()

        configure_cudagraph_benchmark_model(
            model,
            disable_contact=True,
            disable_joint_limits=True,
        )

        self.assertTrue(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.assertTrue(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_LIMIT)

    def test_short_horizon_limit_disable_matches_when_limits_are_inactive(self) -> None:
        base_model = make_inverted_pendulum_model()
        optimized_model = make_inverted_pendulum_model()
        configure_cudagraph_benchmark_model(
            optimized_model,
            disable_contact=True,
            disable_joint_limits=True,
        )
        states = sample_initial_states(base_model, batch_size=4, seed=41, noise=0.01)

        for state in states:
            base_data = mujoco.MjData(base_model)
            optimized_data = mujoco.MjData(optimized_model)
            base_data.qpos[:] = state[1:3]
            base_data.qvel[:] = state[3:5]
            optimized_data.qpos[:] = state[1:3]
            optimized_data.qvel[:] = state[3:5]

            for _ in range(20):
                mujoco.mj_step(base_model, base_data)
                mujoco.mj_step(optimized_model, optimized_data)

            self.assertLess(abs(base_data.qpos[0]), 1.0)
            self.assertLess(abs(base_data.qpos[1]), np.pi / 2)
            np.testing.assert_allclose(base_data.qpos, optimized_data.qpos)
            np.testing.assert_allclose(base_data.qvel, optimized_data.qvel)

    def test_cudagraph_controller_keeps_cpu_rollout_away_from_joint_limits(self) -> None:
        model = make_inverted_pendulum_model()
        states = sample_initial_states(model, batch_size=4, seed=41, noise=0.01)

        for state in states:
            data = mujoco.MjData(model)
            data.qpos[:] = state[1:3]
            data.qvel[:] = state[3:5]
            min_limit_margin = float("inf")

            for _ in range(100):
                observation = np.r_[data.qpos.copy(), data.qvel.copy()]
                data.ctrl[0] = float(np.clip(np.dot(observation, DEFAULT_CUDAGRAPH_CONTROLLER_GAINS), -3.0, 3.0))
                mujoco.mj_step(model, data)
                min_limit_margin = min(
                    min_limit_margin,
                    1.0 - abs(float(data.qpos[0])),
                    float(np.pi / 2 - abs(float(data.qpos[1]))),
                )

            self.assertGreater(min_limit_margin, 0.1)

    def test_cudagraph_benchmark_requires_cuda_device(self) -> None:
        status = mujoco_warp_status(device="cpu")
        if not status.available:
            self.skipTest(status.reason or "MuJoCo Warp is not installed")

        with self.assertRaisesRegex(RuntimeError, "requires a CUDA Warp device"):
            benchmark_mujoco_warp_cudagraph_steps(
                batch_size=1,
                graph_steps=1,
                graph_replays=1,
                device="cpu",
            )

    def test_tiny_warp_rollout_smoke_if_backend_available(self) -> None:
        status = mujoco_warp_status()
        if not status.available:
            self.skipTest(status.reason or "MuJoCo Warp is not installed")

        model = make_inverted_pendulum_model()
        stats, returns, rollout_status = rollout_linear_policies_mujoco_warp(
            np.array([0.0, 10.0, 0.0, 1.0]),
            batch_size=4,
            max_steps=4,
            target_return=4.0,
            seed=1,
            model=model,
        )

        self.assertTrue(rollout_status.available)
        self.assertEqual(returns.shape, (4,))
        self.assertEqual(stats.batch_size, 4)
        self.assertEqual(stats.max_steps, 4)
        self.assertGreater(stats.physics_steps_per_second, 0.0)

    def test_tiny_warp_training_smoke_if_backend_available(self) -> None:
        status = mujoco_warp_status()
        if not status.available:
            self.skipTest(status.reason or "MuJoCo Warp is not installed")

        result = train_mujoco_warp_es_cartpole(
            optimizer_name="adam",
            batch_size=4,
            eval_batch_size=4,
            generations=1,
            max_steps=4,
            target_return=100.0,
            seed=2,
        )

        self.assertEqual(result.optimizer_name, "adam")
        self.assertEqual(result.batch_size, 4)
        self.assertEqual(result.eval_batch_size, 4)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(len(result.final_policy_parameters), 4)
        self.assertIsInstance(result.device_is_cuda, bool)


if __name__ == "__main__":
    unittest.main()
