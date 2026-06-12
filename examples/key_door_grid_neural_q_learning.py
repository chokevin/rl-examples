"""Compare neural Q-learning optimizers on classic KeyDoorGrid."""

from __future__ import annotations

import argparse

from envs import KeyDoorGridEnv
from learners.key_door_q_learning import ACTION_NAMES
from learners.key_door_neural_q_learning import (
    NEURAL_OPTIMIZERS,
    compare_optimizers,
    greedy_q_rollout,
)


def format_state(state: tuple[int, int, int]) -> str:
    key = "has_key" if state[2] else "no_key"
    return f"({state[0]}, {state[1]}, {key})"


def print_comparison(
    *,
    episodes: int,
    seeds: tuple[int, ...],
    eval_every: int,
    hidden_size: int,
    optimizers: tuple[str, ...],
) -> None:
    summaries, results = compare_optimizers(
        optimizers=optimizers,
        episodes=episodes,
        seeds=seeds,
        eval_every=eval_every,
        hidden_size=hidden_size,
    )
    print(
        "KeyDoorGrid neural Q-learning optimizer comparison: "
        f"layout=classic episodes={episodes} seeds={list(seeds)}"
    )
    print(
        "environment: KeyDoorGridEnv supplies transitions/rewards; "
        "the learner updates Q-network weights with loss.backward() and optimizer.step()"
    )
    print()
    print(
        f"{'rank':<4} {'optimizer':<10} {'lr':<8} {'solved':<8} "
        f"{'final_return':<13} {'greedy_steps'}"
    )
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"{rank:<4} {summary.optimizer_name:<10} "
            f"{summary.learning_rate:<8.3g} "
            f"{summary.solved_runs}/{summary.total_runs:<6} "
            f"{summary.mean_final_return:<13.2f} "
            f"{summary.mean_greedy_steps:.1f}"
        )

    winner = summaries[0].optimizer_name
    representative = max(
        results[winner],
        key=lambda result: (
            result.evaluations[-1].solved,
            result.evaluations[-1].episode_return,
            -result.evaluations[-1].episode_length,
        ),
    )
    trace = greedy_q_rollout(
        representative.q_network,
        env=KeyDoorGridEnv(layout="classic"),
        seed=seeds[0],
    )
    print(f"\nlearned greedy rollout from best measured optimizer ({winner}):")
    for index, step in enumerate(trace, start=1):
        print(
            f"step={index:>2} "
            f"state={format_state(step.state):<18} "
            f"action={ACTION_NAMES[step.action]:<5} "
            f"reward={step.reward:>5.2f} "
            f"tile={step.tile:<5} "
            f"next={format_state(step.next_state)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare neural Q-learning optimizers on classic KeyDoorGrid."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=NEURAL_OPTIMIZERS,
        default=NEURAL_OPTIMIZERS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_comparison(
        episodes=args.episodes,
        seeds=tuple(range(1, args.seeds + 1)),
        eval_every=args.eval_every,
        hidden_size=args.hidden_size,
        optimizers=tuple(args.optimizers),
    )


if __name__ == "__main__":
    main()
