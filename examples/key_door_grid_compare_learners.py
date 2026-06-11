"""Compare tabular learners on the harder KeyDoorGrid layout."""

from __future__ import annotations

import argparse

from learners import (
    ALGORITHMS,
    ACTION_NAMES,
    compare_techniques,
    greedy_rollout,
    technique_explanation,
)
from envs import KeyDoorGridEnv


def format_state(state: tuple[int, int, int]) -> str:
    key = "has_key" if state[2] else "no_key"
    return f"({state[0]}, {state[1]}, {key})"


def format_solve_episode(value: int | None) -> str:
    return "not stable" if value is None else str(value)


def print_comparison(
    *,
    episodes: int,
    seeds: tuple[int, ...],
    eval_every: int,
    planning_steps: int,
) -> None:
    summaries, results = compare_techniques(
        episodes=episodes,
        seeds=seeds,
        eval_every=eval_every,
        planning_steps=planning_steps,
    )
    print(
        "KeyDoorGrid learner comparison: "
        f"layout=extended episodes={episodes} seeds={list(seeds)}"
    )
    print("convergence: first greedy-evaluation checkpoint solved for all later checks")
    print()
    print(
        f"{'rank':<4} {'technique':<10} {'solved':<8} "
        f"{'median_solve_ep':<16} {'final_return':<13} {'greedy_steps'}"
    )
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"{rank:<4} {summary.algorithm:<10} "
            f"{summary.solved_runs}/{summary.total_runs:<6} "
            f"{format_solve_episode(summary.median_episodes_to_solve):<16} "
            f"{summary.mean_final_return:<13.2f} "
            f"{summary.mean_greedy_steps:.1f}"
        )

    print("\nwhy the observed ranking makes sense:")
    for summary in summaries:
        print(f"- {summary.algorithm}: {technique_explanation(summary)}")

    winner = summaries[0].algorithm
    representative = results[winner][0]
    trace = greedy_rollout(
        representative.q_values,
        env=KeyDoorGridEnv(layout="extended"),
        seed=seeds[0],
    )
    print(f"\nlearned greedy rollout from fastest technique ({winner}, seed={seeds[0]}):")
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
        description="Compare tabular learners on extended KeyDoorGrid."
    )
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--planning-steps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_comparison(
        episodes=args.episodes,
        seeds=tuple(range(1, args.seeds + 1)),
        eval_every=args.eval_every,
        planning_steps=args.planning_steps,
    )


if __name__ == "__main__":
    main()
