"""Compare optimizers on a tiny MuJoCo InvertedPendulum control demo."""

from __future__ import annotations

import argparse

from learners.mujoco_inverted_pendulum import (
    DEMO_MAX_STEPS,
    DEMO_TARGET_RETURN,
    MUJOCO_ENV_ID,
    MUJOCO_OPTIMIZERS,
    TEACHER_GAINS,
    MuJoCoTrainingResult,
    compare_mujoco_optimizers,
    play_policy,
)


def format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_epoch(value: int | None) -> str:
    return "-" if value is None else str(value)


def print_comparison(
    *,
    epochs: int,
    seeds: tuple[int, ...],
    samples: int,
    batch_size: int,
    eval_every: int,
    eval_episodes: int,
    max_steps_per_episode: int,
    target_return: float,
    optimizers: tuple[str, ...],
    render: bool,
    render_mode: str,
    render_optimizer: str | None,
    render_episodes: int,
    render_seed: int,
    render_delay: float,
) -> None:
    summaries, results = compare_mujoco_optimizers(
        optimizers=optimizers,
        seeds=seeds,
        epochs=epochs,
        samples=samples,
        batch_size=batch_size,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        max_steps_per_episode=max_steps_per_episode,
        target_return=target_return,
    )
    print(
        "MuJoCo InvertedPendulum optimizer comparison: "
        f"env={MUJOCO_ENV_ID} epochs={epochs} seeds={list(seeds)} "
        f"samples={samples}"
    )
    print(
        "environment: Gymnasium MuJoCo supplies continuous observations, "
        "continuous Box actions, and real MuJoCo physics"
    )
    print(
        "learner: a tiny linear PyTorch policy distills a stabilizing controller "
        f"with gains={list(TEACHER_GAINS)} using MSE, loss.backward(), and "
        "optimizer.step()"
    )
    print(
        "metric policy: this is a cheap local simulator, so report wall-clock to "
        "target when reached; keep epochs and supervised samples as context"
    )
    print(
        f"target: mean return >= {target_return:.1f} over {eval_episodes} eval "
        f"episodes capped at {max_steps_per_episode} MuJoCo steps"
    )
    print()
    print(
        f"{'rank':<4} {'optimizer':<10} {'lr':<8} {'target':<8} "
        f"{'final':<8} {'wall_to_target':<15} {'target_ep':<10} "
        f"{'run_sec':<8} {'samples':<8} {'final_return':<13} "
        f"{'best_return':<12} {'final_loss'}"
    )
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"{rank:<4} {summary.optimizer_name:<10} "
            f"{summary.learning_rate:<8.3g} "
            f"{summary.target_runs}/{summary.total_runs:<6} "
            f"{summary.final_solved_runs}/{summary.total_runs:<6} "
            f"{format_seconds(summary.median_seconds_to_target):<15} "
            f"{format_epoch(summary.median_target_epoch):<10} "
            f"{summary.mean_wall_clock_seconds:<8.3f} "
            f"{summary.mean_dataset_samples:<8.0f} "
            f"{summary.mean_final_return:<13.1f} "
            f"{summary.mean_best_return:<12.1f} "
            f"{summary.mean_final_loss:.5f}"
        )

    print("\nper-seed evaluation:")
    for optimizer_name in optimizers:
        rows = sorted(results[optimizer_name], key=lambda row: row.seed)
        seed_values = [str(row.seed) for row in rows]
        target_epochs = [format_epoch(row.target_epoch) for row in rows]
        final_returns = [f"{row.evaluations[-1].mean_return:.1f}" for row in rows]
        expert_returns = [f"{row.expert_mean_return:.1f}" for row in rows]
        print(
            f"{optimizer_name:<10} seeds={seed_values} target_ep={target_epochs} "
            f"final_return={final_returns} expert_return={expert_returns}"
        )

    if render:
        optimizer_name = render_optimizer or summaries[0].optimizer_name
        representative = max(
            results[optimizer_name],
            key=lambda row: (
                row.evaluations[-1].mean_return,
                row.evaluations[-1].best_return,
                -row.training_losses[-1],
            ),
        )
        playback = render_trained_policy(
            representative,
            render_mode=render_mode,
            episodes=render_episodes,
            seed=render_seed,
            max_steps_per_episode=max_steps_per_episode,
            frame_delay_seconds=render_delay,
        )
        returns = [f"{value:.1f}" for value in playback.episode_returns]
        print(
            "\nrendered trained policy: "
            f"optimizer={optimizer_name} seed={representative.seed} "
            f"render_mode={playback.render_mode} returns={returns} "
            f"lengths={playback.episode_lengths} "
            f"frames={playback.rendered_frames} "
            f"first_frame_shape={playback.first_frame_shape}"
        )


def render_trained_policy(
    result: MuJoCoTrainingResult,
    *,
    render_mode: str,
    episodes: int,
    seed: int,
    max_steps_per_episode: int,
    frame_delay_seconds: float,
):
    print(
        "\nopening MuJoCo renderer for trained policy "
        f"(render_mode={render_mode}, episodes={episodes})..."
    )
    return play_policy(
        result.policy_network,
        env_id=result.env_id,
        render_mode=render_mode,
        episodes=episodes,
        seed=seed,
        max_steps_per_episode=max_steps_per_episode,
        frame_delay_seconds=frame_delay_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SGD and Adam on a tiny Gymnasium MuJoCo "
            "InvertedPendulum control demo."
        )
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEMO_MAX_STEPS)
    parser.add_argument("--target-return", type=float, default=DEMO_TARGET_RETURN)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open Gymnasium MuJoCo playback after training.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array"),
        default="human",
        help="'human' opens the MuJoCo viewer; 'rgb_array' renders offscreen frames.",
    )
    parser.add_argument(
        "--render-optimizer",
        choices=MUJOCO_OPTIMIZERS,
        default=None,
        help="Optimizer run to render. Defaults to the top-ranked summary.",
    )
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--render-seed", type=int, default=12345)
    parser.add_argument("--render-delay", type=float, default=0.02)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=MUJOCO_OPTIMIZERS,
        default=MUJOCO_OPTIMIZERS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_comparison(
        epochs=args.epochs,
        seeds=tuple(range(1, args.seeds + 1)),
        samples=args.samples,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        target_return=args.target_return,
        optimizers=tuple(args.optimizers),
        render=args.render,
        render_mode=args.render_mode,
        render_optimizer=args.render_optimizer,
        render_episodes=args.render_episodes,
        render_seed=args.render_seed,
        render_delay=args.render_delay,
    )


if __name__ == "__main__":
    main()
