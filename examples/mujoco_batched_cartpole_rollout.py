"""Train MuJoCo cartpole with large batched rollout policy search."""

from __future__ import annotations

import argparse

from learners.mujoco_batched_cartpole import (
    BATCHED_CARTPOLE_ENV_ID,
    BATCHED_CARTPOLE_OPTIMIZERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EXPLORATION_SIGMA,
    DEFAULT_MAX_STEPS,
    DEFAULT_TARGET_RETURN,
    train_batched_es_cartpole,
)
from learners.mujoco_inverted_pendulum import (
    InvertedPendulumLinearPolicy,
    play_policy,
)


def format_generation(value: int | None) -> str:
    return "-" if value is None else str(value)


def print_training(
    *,
    optimizer_name: str,
    learning_rate: float | None,
    batch_size: int,
    eval_batch_size: int,
    generations: int,
    max_steps: int,
    target_return: float,
    exploration_sigma: float,
    sigma_decay: float,
    seed: int,
    data_threads: int | None,
    render: bool,
    render_mode: str,
    render_episodes: int,
    render_seed: int,
    render_delay: float,
) -> None:
    result = train_batched_es_cartpole(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        generations=generations,
        max_steps=max_steps,
        target_return=target_return,
        exploration_sigma=exploration_sigma,
        sigma_decay=sigma_decay,
        seed=seed,
        data_threads=data_threads,
    )
    print(
        "Batched MuJoCo cartpole ES training: "
        f"env={BATCHED_CARTPOLE_ENV_ID} batch={batch_size} "
        f"eval_batch={eval_batch_size} optimizer={result.optimizer_name}"
    )
    print(
        "thread mapping: sample many linear policies, roll them through MuJoCo's "
        "C rollout API one closed-loop step at a time, then use PyTorch "
        "loss.backward() with an evolution-strategy gradient estimate"
    )
    print(
        "note: this mirrors the 8192-agent rollout-batch idea, but Python "
        "closed-loop control is slower than the fully fused 18M steps/s thread"
    )
    print(
        f"target: eval mean return >= {target_return:.1f} over {eval_batch_size} "
        f"agents capped at {max_steps} MuJoCo steps"
    )
    print()
    print(
        f"{'gen':<4} {'sigma':<8} {'train_mean':<12} {'train_best':<11} "
        f"{'eval_mean':<11} {'eval_min':<9} {'eval_solved':<12} "
        f"{'rollout_sec':<12} {'physics_steps/s':<16} {'active_steps/s'}"
    )
    for generation in result.history:
        rollout_seconds = (
            generation.train.wall_clock_seconds
            + generation.evaluation.wall_clock_seconds
        )
        print(
            f"{generation.generation:<4} {generation.sigma:<8.3f} "
            f"{generation.train.mean_return:<12.1f} "
            f"{generation.train.max_return:<11.1f} "
            f"{generation.evaluation.mean_return:<11.1f} "
            f"{generation.evaluation.min_return:<9.1f} "
            f"{generation.evaluation.solved_fraction:<12.2f} "
            f"{rollout_seconds:<12.3f} "
            f"{generation.train.physics_steps_per_second:<16.0f} "
            f"{generation.train.active_steps_per_second:.0f}"
        )

    print(
        "\nlearned_generation="
        f"{format_generation(result.learned_generation)} "
        f"total_rollout_sec={result.total_rollout_seconds:.3f} "
        f"final_policy={['%.3f' % value for value in result.final_policy_parameters]}"
    )

    if render:
        policy = InvertedPendulumLinearPolicy(
            observation_size=4,
            action_size=1,
            action_low=[-3.0],
            action_high=[3.0],
        )
        policy.linear.weight.data[0] = policy.linear.weight.data.new_tensor(
            result.final_policy_parameters
        )
        policy.linear.bias.data.zero_()
        print(
            "\nopening MuJoCo renderer for batched-ES policy "
            f"(render_mode={render_mode}, episodes={render_episodes})..."
        )
        playback = play_policy(
            policy,
            render_mode=render_mode,
            episodes=render_episodes,
            seed=render_seed,
            max_steps_per_episode=max_steps,
            frame_delay_seconds=render_delay,
        )
        returns = [f"{value:.1f}" for value in playback.episode_returns]
        print(
            "rendered batched-ES policy: "
            f"returns={returns} lengths={playback.episode_lengths} "
            f"frames={playback.rendered_frames} "
            f"first_frame_shape={playback.first_frame_shape}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MuJoCo InvertedPendulum cartpole with batched rollouts."
    )
    parser.add_argument("--optimizer", choices=BATCHED_CARTPOLE_OPTIMIZERS, default="adam")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--target-return", type=float, default=DEFAULT_TARGET_RETURN)
    parser.add_argument("--exploration-sigma", type=float, default=DEFAULT_EXPLORATION_SIGMA)
    parser.add_argument("--sigma-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-threads", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-mode", choices=("human", "rgb_array"), default="human")
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--render-seed", type=int, default=12345)
    parser.add_argument("--render-delay", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_training(
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        generations=args.generations,
        max_steps=args.max_steps,
        target_return=args.target_return,
        exploration_sigma=args.exploration_sigma,
        sigma_decay=args.sigma_decay,
        seed=args.seed,
        data_threads=args.data_threads,
        render=args.render,
        render_mode=args.render_mode,
        render_episodes=args.render_episodes,
        render_seed=args.render_seed,
        render_delay=args.render_delay,
    )


if __name__ == "__main__":
    main()
