"""Compare SGD and Adam on Gymnasium CartPole with a tiny policy network."""

from __future__ import annotations

import argparse

from learners.cartpole_policy_gradient import (
    CARTPOLE_ENV_ID,
    CARTPOLE_OPTIMIZERS,
    CARTPOLE_SOLVED_RETURN,
    compare_cartpole_optimizers,
)


def format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_episode(value: int | None) -> str:
    return "-" if value is None else str(value)


def print_comparison(
    *,
    episodes: int,
    seeds: tuple[int, ...],
    eval_every: int,
    eval_episodes: int,
    hidden_size: int,
    solve_window: int,
    solved_threshold: float,
    max_steps_per_episode: int | None,
    optimizers: tuple[str, ...],
) -> None:
    summaries, results = compare_cartpole_optimizers(
        optimizers=optimizers,
        episodes=episodes,
        seeds=seeds,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        hidden_size=hidden_size,
        solve_window=solve_window,
        solved_threshold=solved_threshold,
        max_steps_per_episode=max_steps_per_episode,
    )
    print(
        "CartPole-v1 policy-gradient optimizer comparison: "
        f"episodes={episodes} seeds={list(seeds)} hidden={hidden_size}"
    )
    print(
        "environment: Gymnasium CartPole supplies continuous observations and "
        "rewards; the learner updates policy-network logits with a REINFORCE "
        "loss, loss.backward(), and optimizer.step()"
    )
    print(
        "metric policy: cheap local simulator, so report wall-clock to target "
        "when reached; keep episodes and environment steps as sample context"
    )
    print(
        f"target: rolling {solve_window}-episode mean return >= "
        f"{solved_threshold:.1f}; Gymnasium reward threshold is "
        f"{CARTPOLE_SOLVED_RETURN:.1f}"
    )
    print()
    print(
        f"{'rank':<4} {'optimizer':<10} {'lr':<8} {'solved':<8} "
        f"{'wall_to_target':<15} {'solve_ep':<9} {'run_sec':<8} "
        f"{'env_steps':<10} {'sec/sample':<11} {'final_window':<13} "
        f"{'best_return':<12} {'eval_return'}"
    )
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"{rank:<4} {summary.optimizer_name:<10} "
            f"{summary.learning_rate:<8.3g} "
            f"{summary.solved_runs}/{summary.total_runs:<6} "
            f"{format_seconds(summary.median_seconds_to_solve):<15} "
            f"{format_episode(summary.median_solve_episode):<9} "
            f"{summary.mean_wall_clock_seconds:<8.3f} "
            f"{summary.mean_total_samples:<10.1f} "
            f"{summary.mean_seconds_per_sample:<11.6f} "
            f"{summary.mean_final_window_return:<13.1f} "
            f"{summary.mean_best_return:<12.1f} "
            f"{summary.mean_final_eval_return:.1f}"
        )

    print("\nper-seed training windows and samples:")
    for optimizer_name in optimizers:
        rows = sorted(results[optimizer_name], key=lambda row: row.seed)
        seed_values = [str(row.seed) for row in rows]
        final_windows = [f"{row.final_window_return:.1f}" for row in rows]
        samples = [str(row.total_samples) for row in rows]
        solves = [format_episode(row.solve_episode) for row in rows]
        print(
            f"{optimizer_name:<10} seeds={seed_values} final_window={final_windows} "
            f"env_steps={samples} solve_ep={solves}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SGD and Adam on Gymnasium CartPole with a compact "
            "policy-gradient learner."
        )
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--solve-window", type=int, default=20)
    parser.add_argument("--solved-threshold", type=float, default=CARTPOLE_SOLVED_RETURN)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=CARTPOLE_OPTIMIZERS,
        default=CARTPOLE_OPTIMIZERS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_comparison(
        episodes=args.episodes,
        seeds=tuple(range(1, args.seeds + 1)),
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        hidden_size=args.hidden_size,
        solve_window=args.solve_window,
        solved_threshold=args.solved_threshold,
        max_steps_per_episode=args.max_steps_per_episode,
        optimizers=tuple(args.optimizers),
    )


if __name__ == "__main__":
    main()
