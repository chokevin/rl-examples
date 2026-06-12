from __future__ import annotations

import unittest

import numpy as np
import torch

from examples.envs import MultiCartPoleEnv, TripleCartPoleEnv
from examples.envs.triple_cart_pole import make_stacked_cart_pole_xml
from examples.learners.mujoco_triple_cart_pole import (
    DEMO_MAX_STEPS,
    DEMO_TARGET_RETURN,
    TRIPLE_POLE_OPTIMIZERS,
    TriplePoleLinearPolicy,
    compare_triple_pole_optimizers,
    default_learning_rate,
    default_state_ranges,
    evaluate_policy,
    evaluate_teacher,
    linearize_dynamics,
    lqr_teacher_gain,
    make_optimizer,
    make_teacher_dataset,
    play_policy,
    pole_count_from_state_size,
    solve_discrete_lqr,
    state_size_for_poles,
    teacher_action,
    train_triple_pole_policy,
)


class TripleCartPoleEnvTest(unittest.TestCase):
    def test_env_reset_and_step_expose_three_pole_state(self) -> None:
        env = TripleCartPoleEnv(max_steps=3)
        try:
            observation, info = env.reset(seed=1)
            next_observation, reward, terminated, truncated, next_info = env.step([0.0])

            self.assertEqual(observation.shape, (8,))
            self.assertEqual(next_observation.shape, (8,))
            self.assertEqual(env.action_space.shape, (1,))
            self.assertEqual(len(info["hinge_angles"]), 3)
            self.assertEqual(len(info["global_pole_angles"]), 3)
            self.assertIn(reward, {0.0, 1.0})
            self.assertFalse(truncated)
            self.assertIsInstance(terminated, bool)
            self.assertEqual(next_info["episode_length"], 1)
        finally:
            env.close()

    def test_multi_cart_pole_expands_observation_and_xml(self) -> None:
        xml = make_stacked_cart_pole_xml(5)
        env = MultiCartPoleEnv(pole_count=5, max_steps=3)
        try:
            observation, info = env.reset(seed=1)
            next_observation, _, _, _, next_info = env.step([0.0])

            self.assertIn('name="pole5"', xml)
            self.assertEqual(observation.shape, (12,))
            self.assertEqual(next_observation.shape, (12,))
            self.assertEqual(env.model.nq, 6)
            self.assertEqual(env.model.nv, 6)
            self.assertEqual(info["pole_count"], 5)
            self.assertEqual(len(info["hinge_angles"]), 5)
            self.assertEqual(len(info["global_pole_angles"]), 5)
            self.assertEqual(next_info["episode_length"], 1)
        finally:
            env.close()

    def test_env_rejects_invalid_parameters_and_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "pole_count"):
            MultiCartPoleEnv(pole_count=0)
        with self.assertRaisesRegex(ValueError, "max_steps"):
            TripleCartPoleEnv(max_steps=0)
        with self.assertRaisesRegex(ValueError, "reset_noise"):
            TripleCartPoleEnv(reset_noise=-0.1)
        with self.assertRaisesRegex(ValueError, "render_mode"):
            TripleCartPoleEnv(render_mode="ansi")

        env = TripleCartPoleEnv()
        try:
            env.reset(seed=1)
            with self.assertRaisesRegex(ValueError, "action must have shape"):
                env.step([0.0, 0.0])
        finally:
            env.close()

    def test_rgb_array_rendering_shows_custom_mujoco_model(self) -> None:
        env = TripleCartPoleEnv(max_steps=2, render_mode="rgb_array")
        try:
            env.reset(seed=1)
            frame = env.render()

            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 3)
        finally:
            env.close()


class TripleCartPoleLearnerTest(unittest.TestCase):
    def test_linearized_lqr_teacher_balances_short_rollout(self) -> None:
        gain = lqr_teacher_gain()
        mean_return, min_return = evaluate_teacher(
            gain,
            seed=1,
            eval_episodes=2,
            max_steps=50,
        )

        self.assertEqual(gain.shape, (1, 8))
        self.assertEqual(mean_return, 50.0)
        self.assertEqual(min_return, 50.0)

    def test_teacher_dataset_and_action_shapes(self) -> None:
        gain = np.ones((1, 8), dtype=np.float64)
        observations, actions = make_teacher_dataset(
            gain=gain,
            samples=8,
            seed=1,
        )

        self.assertEqual(observations.shape, (8, 8))
        self.assertEqual(actions.shape, (8, 1))
        self.assertLessEqual(float(actions.max().item()), 1.0)
        self.assertGreaterEqual(float(actions.min().item()), -1.0)
        self.assertIsInstance(teacher_action(np.zeros(8), gain), float)

    def test_five_pole_teacher_and_dataset_shapes(self) -> None:
        gain = lqr_teacher_gain(pole_count=5)
        observations, actions = make_teacher_dataset(
            gain=gain,
            samples=8,
            seed=1,
        )

        self.assertEqual(state_size_for_poles(5), 12)
        self.assertEqual(pole_count_from_state_size(12), 5)
        self.assertEqual(gain.shape, (1, 12))
        self.assertEqual(len(default_state_ranges(5)), 12)
        self.assertEqual(observations.shape, (8, 12))
        self.assertEqual(actions.shape, (8, 1))
        self.assertIsInstance(teacher_action(np.zeros(12), gain), float)

    def test_tiny_training_smoke_reports_metrics(self) -> None:
        result = train_triple_pole_policy(
            optimizer_name="adam",
            epochs=2,
            samples=32,
            batch_size=8,
            seed=3,
            eval_every=1,
            eval_episodes=1,
            max_steps=5,
            target_return=100.0,
        )

        self.assertEqual(result.optimizer_name, "adam")
        self.assertEqual(result.learning_rate, default_learning_rate("adam"))
        self.assertEqual(result.seed, 3)
        self.assertEqual(result.dataset_samples, 32)
        self.assertEqual(len(result.training_losses), 2)
        self.assertEqual([evaluation.epoch for evaluation in result.evaluations], [1, 2])
        self.assertIsNone(result.target_epoch)
        self.assertIsNone(result.estimated_seconds_to_target)
        self.assertGreaterEqual(result.wall_clock_seconds, 0.0)

    def test_tiny_five_pole_training_smoke_reports_shapes(self) -> None:
        result = train_triple_pole_policy(
            optimizer_name="adam",
            pole_count=5,
            epochs=1,
            samples=16,
            batch_size=8,
            seed=3,
            eval_every=1,
            eval_episodes=1,
            max_steps=4,
            target_return=100.0,
        )

        self.assertEqual(result.pole_count, 5)
        self.assertEqual(result.policy_network.observation_size, 12)
        self.assertEqual(result.teacher_gain.shape, (1, 12))
        self.assertEqual(result.evaluations[-1].eval_episodes, 1)

    def test_compare_optimizers_reports_summaries(self) -> None:
        summaries, results = compare_triple_pole_optimizers(
            optimizers=("sgd", "adam"),
            seeds=(1,),
            epochs=1,
            samples=16,
            batch_size=8,
            eval_every=1,
            eval_episodes=1,
            max_steps=4,
            target_return=100.0,
        )

        self.assertEqual(
            {summary.optimizer_name for summary in summaries},
            {"sgd", "adam"},
        )
        self.assertEqual(set(results), {"sgd", "adam"})
        for summary in summaries:
            self.assertIn(summary.optimizer_name, TRIPLE_POLE_OPTIMIZERS)
            self.assertEqual(summary.total_runs, 1)
            self.assertEqual(summary.target_runs, 0)
            self.assertEqual(summary.final_solved_runs, 0)
            self.assertIsNone(summary.median_target_epoch)
            self.assertGreaterEqual(summary.mean_dataset_samples, 16.0)

    def test_rgb_array_playback_renders_stacked_poles(self) -> None:
        policy = TriplePoleLinearPolicy()

        playback = play_policy(
            policy,
            render_mode="rgb_array",
            episodes=1,
            seed=1,
            max_steps=2,
        )

        self.assertEqual(playback.render_mode, "rgb_array")
        self.assertEqual(playback.pole_count, 3)
        self.assertEqual(len(playback.episode_returns), 1)
        self.assertGreaterEqual(playback.rendered_frames, 1)
        self.assertIsNotNone(playback.first_frame_shape)
        self.assertEqual(playback.first_frame_shape[2], 3)

    def test_evaluate_policy_restores_training_mode(self) -> None:
        policy = TriplePoleLinearPolicy()
        policy.train()

        evaluation = evaluate_policy(
            policy,
            epoch=1,
            seed=1,
            eval_episodes=1,
            max_steps=3,
            target_return=100.0,
        )

        self.assertTrue(policy.training)
        self.assertEqual(evaluation.epoch, 1)
        self.assertEqual(evaluation.eval_episodes, 1)
        self.assertLessEqual(evaluation.mean_length, 3)
        self.assertFalse(evaluation.solved)

    def test_rejects_invalid_triple_pole_parameters(self) -> None:
        policy = TriplePoleLinearPolicy()

        with self.assertRaisesRegex(ValueError, "observation_size"):
            TriplePoleLinearPolicy(observation_size=0)
        with self.assertRaisesRegex(ValueError, "pole_count"):
            TriplePoleLinearPolicy(pole_count=0)
        with self.assertRaisesRegex(ValueError, "observation_size"):
            TriplePoleLinearPolicy(pole_count=5, observation_size=8)
        with self.assertRaisesRegex(ValueError, "action_size"):
            TriplePoleLinearPolicy(action_size=2)
        with self.assertRaisesRegex(ValueError, "optimizer_name must be one of"):
            default_learning_rate("muon")
        with self.assertRaisesRegex(ValueError, "learning_rate must be"):
            make_optimizer("sgd", policy.parameters(), learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "epsilon must be"):
            env = TripleCartPoleEnv()
            try:
                linearize_dynamics(env, epsilon=0.0)
            finally:
                env.close()
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            solve_discrete_lqr(np.eye(8), np.ones((8, 1)), max_iterations=0)
        with self.assertRaisesRegex(ValueError, "r_weight"):
            solve_discrete_lqr(np.eye(8), np.ones((8, 1)), r_weight=0.0)
        with self.assertRaisesRegex(ValueError, "observation must have shape"):
            teacher_action(np.zeros(7), np.ones((1, 8)))
        with self.assertRaisesRegex(ValueError, "samples must be"):
            make_teacher_dataset(gain=np.ones((1, 8)), samples=0)
        with self.assertRaisesRegex(ValueError, "state_ranges must contain"):
            make_teacher_dataset(gain=np.ones((1, 8)), samples=1, state_ranges=(1.0,))
        with self.assertRaisesRegex(ValueError, "epochs must be"):
            train_triple_pole_policy(epochs=0)
        with self.assertRaisesRegex(ValueError, "samples must be"):
            train_triple_pole_policy(samples=0)
        with self.assertRaisesRegex(ValueError, "batch_size must be"):
            train_triple_pole_policy(batch_size=0)
        with self.assertRaisesRegex(ValueError, "eval_every must be"):
            train_triple_pole_policy(eval_every=0)
        with self.assertRaisesRegex(ValueError, "eval_episodes must be"):
            train_triple_pole_policy(eval_episodes=0)
        with self.assertRaisesRegex(ValueError, "max_steps must be"):
            train_triple_pole_policy(max_steps=0)
        with self.assertRaisesRegex(ValueError, "target_return must be"):
            train_triple_pole_policy(target_return=0.0)
        with self.assertRaisesRegex(ValueError, "pole_count"):
            train_triple_pole_policy(pole_count=0)
        with self.assertRaisesRegex(ValueError, "unknown optimizers"):
            compare_triple_pole_optimizers(optimizers=("muon",))
        with self.assertRaisesRegex(ValueError, "seeds must contain"):
            compare_triple_pole_optimizers(seeds=())
        with self.assertRaisesRegex(ValueError, "render_mode must be"):
            play_policy(policy, render_mode="ansi")
        with self.assertRaisesRegex(ValueError, "frame_delay_seconds must be"):
            play_policy(policy, frame_delay_seconds=-0.1)
        self.assertEqual(DEMO_MAX_STEPS, 300)
        self.assertEqual(DEMO_TARGET_RETURN, 180.0)


if __name__ == "__main__":
    unittest.main()
