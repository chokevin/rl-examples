from __future__ import annotations

import unittest

import torch

from examples.learners.mujoco_inverted_pendulum import (
    DEMO_MAX_STEPS,
    DEMO_TARGET_RETURN,
    MUJOCO_ENV_ID,
    MUJOCO_OPTIMIZERS,
    InvertedPendulumLinearPolicy,
    compare_mujoco_optimizers,
    default_learning_rate,
    ensure_mujoco_human_viewer_compatibility,
    evaluate_policy,
    evaluate_teacher,
    make_optimizer,
    make_teacher_dataset,
    play_policy,
    solver_iteration_count,
    teacher_action,
    train_inverted_pendulum_policy,
)


class MuJoCoInvertedPendulumTest(unittest.TestCase):
    def test_teacher_action_clips_continuous_action(self) -> None:
        observations = torch.tensor(
            [
                [0.0, 0.10, 0.0, 0.20],
                [0.0, -0.50, 0.0, -0.50],
            ],
            dtype=torch.float32,
        )

        actions = teacher_action(
            observations,
            action_low=[-3.0],
            action_high=[3.0],
        )

        self.assertEqual(actions.shape, (2, 1))
        self.assertTrue(torch.allclose(actions[0], torch.tensor([1.2])))
        self.assertTrue(torch.allclose(actions[1], torch.tensor([-3.0])))

    def test_policy_outputs_bounded_continuous_action(self) -> None:
        policy = InvertedPendulumLinearPolicy(
            observation_size=4,
            action_size=1,
            action_low=[-3.0],
            action_high=[3.0],
        )

        action = policy(torch.ones(4))

        self.assertEqual(action.shape, (1,))
        self.assertGreaterEqual(float(action.item()), -3.0)
        self.assertLessEqual(float(action.item()), 3.0)

    def test_teacher_dataset_has_expected_shapes(self) -> None:
        observations, actions = make_teacher_dataset(
            samples=8,
            seed=1,
            action_low=[-3.0],
            action_high=[3.0],
        )

        self.assertEqual(observations.shape, (8, 4))
        self.assertEqual(actions.shape, (8, 1))
        self.assertLessEqual(float(actions.max().item()), 3.0)
        self.assertGreaterEqual(float(actions.min().item()), -3.0)

    def test_teacher_solves_short_capped_mujoco_run(self) -> None:
        mean_return, min_return = evaluate_teacher(
            seed=1,
            eval_episodes=2,
            max_steps_per_episode=20,
        )

        self.assertEqual(mean_return, 20.0)
        self.assertEqual(min_return, 20.0)

    def test_tiny_training_smoke_reports_metrics(self) -> None:
        result = train_inverted_pendulum_policy(
            optimizer_name="adam",
            epochs=2,
            samples=32,
            batch_size=8,
            seed=3,
            eval_every=1,
            eval_episodes=1,
            max_steps_per_episode=5,
            target_return=100.0,
        )

        self.assertEqual(result.env_id, MUJOCO_ENV_ID)
        self.assertEqual(result.optimizer_name, "adam")
        self.assertEqual(result.learning_rate, default_learning_rate("adam"))
        self.assertEqual(result.seed, 3)
        self.assertEqual(result.dataset_samples, 32)
        self.assertEqual(len(result.training_losses), 2)
        self.assertEqual([evaluation.epoch for evaluation in result.evaluations], [1, 2])
        self.assertIsNone(result.target_epoch)
        self.assertIsNone(result.estimated_seconds_to_target)
        self.assertGreaterEqual(result.wall_clock_seconds, 0.0)

    def test_compare_optimizers_reports_summaries(self) -> None:
        summaries, results = compare_mujoco_optimizers(
            optimizers=("sgd", "adam"),
            seeds=(1,),
            epochs=1,
            samples=16,
            batch_size=8,
            eval_every=1,
            eval_episodes=1,
            max_steps_per_episode=4,
            target_return=100.0,
        )

        self.assertEqual(
            {summary.optimizer_name for summary in summaries},
            {"sgd", "adam"},
        )
        self.assertEqual(set(results), {"sgd", "adam"})
        for summary in summaries:
            self.assertIn(summary.optimizer_name, MUJOCO_OPTIMIZERS)
            self.assertEqual(summary.total_runs, 1)
            self.assertEqual(summary.target_runs, 0)
            self.assertEqual(summary.final_solved_runs, 0)
            self.assertIsNone(summary.median_target_epoch)
            self.assertGreaterEqual(summary.mean_dataset_samples, 16.0)

    def test_evaluate_policy_restores_training_mode(self) -> None:
        policy = InvertedPendulumLinearPolicy(
            observation_size=4,
            action_size=1,
            action_low=[-3.0],
            action_high=[3.0],
        )
        policy.train()

        evaluation = evaluate_policy(
            policy,
            epoch=1,
            seed=1,
            eval_episodes=1,
            max_steps_per_episode=3,
            target_return=100.0,
        )

        self.assertTrue(policy.training)
        self.assertEqual(evaluation.epoch, 1)
        self.assertEqual(evaluation.eval_episodes, 1)
        self.assertLessEqual(evaluation.mean_length, 3)
        self.assertFalse(evaluation.solved)

    def test_rgb_array_playback_renders_mujoco_frames(self) -> None:
        policy = InvertedPendulumLinearPolicy(
            observation_size=4,
            action_size=1,
            action_low=[-3.0],
            action_high=[3.0],
        )

        playback = play_policy(
            policy,
            render_mode="rgb_array",
            episodes=1,
            seed=1,
            max_steps_per_episode=2,
        )

        self.assertEqual(playback.render_mode, "rgb_array")
        self.assertEqual(len(playback.episode_returns), 1)
        self.assertGreaterEqual(playback.rendered_frames, 1)
        self.assertEqual(playback.first_frame_shape, (480, 480, 3))

    def test_human_viewer_compatibility_supports_mujoco_3_solver_field(self) -> None:
        class MuJoCo2Data:
            solver_iter = 4

        class MuJoCo3Data:
            solver_niter = torch.tensor([0, 2, 1])

        self.assertEqual(solver_iteration_count(MuJoCo2Data()), 4)
        self.assertEqual(solver_iteration_count(MuJoCo3Data()), 2)
        ensure_mujoco_human_viewer_compatibility()
        self.assertFalse(ensure_mujoco_human_viewer_compatibility())

    def test_rejects_invalid_mujoco_parameters(self) -> None:
        policy = InvertedPendulumLinearPolicy(
            observation_size=4,
            action_size=1,
            action_low=[-3.0],
            action_high=[3.0],
        )

        with self.assertRaisesRegex(ValueError, "observation_size"):
            InvertedPendulumLinearPolicy(
                observation_size=0,
                action_size=1,
                action_low=[-3.0],
                action_high=[3.0],
            )
        with self.assertRaisesRegex(ValueError, "action_size"):
            InvertedPendulumLinearPolicy(
                observation_size=4,
                action_size=0,
                action_low=[],
                action_high=[],
            )
        with self.assertRaisesRegex(ValueError, "action bounds"):
            InvertedPendulumLinearPolicy(
                observation_size=4,
                action_size=1,
                action_low=[],
                action_high=[3.0],
            )
        with self.assertRaisesRegex(ValueError, "optimizer_name must be one of"):
            default_learning_rate("muon")
        with self.assertRaisesRegex(ValueError, "learning_rate must be"):
            make_optimizer("sgd", policy.parameters(), learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "observation width"):
            teacher_action(torch.zeros(3), action_low=[-3.0], action_high=[3.0])
        with self.assertRaisesRegex(ValueError, "samples must be"):
            make_teacher_dataset(samples=0, action_low=[-3.0], action_high=[3.0])
        with self.assertRaisesRegex(ValueError, "epochs must be"):
            train_inverted_pendulum_policy(epochs=0)
        with self.assertRaisesRegex(ValueError, "samples must be"):
            train_inverted_pendulum_policy(samples=0)
        with self.assertRaisesRegex(ValueError, "batch_size must be"):
            train_inverted_pendulum_policy(batch_size=0)
        with self.assertRaisesRegex(ValueError, "eval_every must be"):
            train_inverted_pendulum_policy(eval_every=0)
        with self.assertRaisesRegex(ValueError, "eval_episodes must be"):
            train_inverted_pendulum_policy(eval_episodes=0)
        with self.assertRaisesRegex(ValueError, "max_steps_per_episode must be"):
            train_inverted_pendulum_policy(max_steps_per_episode=0)
        with self.assertRaisesRegex(ValueError, "target_return must be"):
            train_inverted_pendulum_policy(target_return=0.0)
        with self.assertRaisesRegex(ValueError, "unknown optimizers"):
            compare_mujoco_optimizers(optimizers=("muon",))
        with self.assertRaisesRegex(ValueError, "seeds must contain"):
            compare_mujoco_optimizers(seeds=())
        with self.assertRaisesRegex(ValueError, "render_mode must be"):
            play_policy(policy, render_mode="ansi")
        with self.assertRaisesRegex(ValueError, "frame_delay_seconds must be"):
            play_policy(policy, frame_delay_seconds=-0.1)
        self.assertEqual(DEMO_MAX_STEPS, 200)
        self.assertEqual(DEMO_TARGET_RETURN, 180.0)


if __name__ == "__main__":
    unittest.main()
