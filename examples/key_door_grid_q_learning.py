"""Learn a KeyDoorGrid policy with tabular Q-learning."""

from __future__ import annotations

import argparse

from envs import KeyDoorGridEnv
from learners import ACTION_NAMES, greedy_rollout, train_q_learning


def format_state(state: tuple[int, int, int]) -> str:
    key = "has_key" if state[2] else "no_key"
    return f"({state[0]}, {state[1]}, {key})"


def print_training_summary(
    episodes: int,
    seed: int,
    report_every: int,
    layout: str,
) -> None:
    result = train_q_learning(
        episodes=episodes,
        seed=seed,
        env=KeyDoorGridEnv(layout=layout),
    )
    print(f"Q-learning KeyDoorGrid: layout={layout} episodes={episodes} seed={seed}")
    print("env: examples/envs/key_door_grid.py")
    print("learner: examples/learners/key_door_q_learning.py")

    for end in range(report_every, episodes + 1, report_every):
        window = result.episode_returns[end - report_every : end]
        average_return = sum(window) / len(window)
        successes = sum(value > 0.9 for value in window)
        print(
            f"episodes {end - report_every + 1:>4}-{end:>4}: "
            f"avg_return={average_return:>5.2f} "
            f"successes={successes:>3}/{len(window)}"
        )

    if episodes % report_every:
        start = episodes - (episodes % report_every)
        window = result.episode_returns[start:]
        average_return = sum(window) / len(window)
        successes = sum(value > 0.9 for value in window)
        print(
            f"episodes {start + 1:>4}-{episodes:>4}: "
            f"avg_return={average_return:>5.2f} "
            f"successes={successes:>3}/{len(window)}"
        )

    print("\nlearned greedy rollout:")
    trace = greedy_rollout(result.q_values, env=KeyDoorGridEnv(layout=layout), seed=seed)
    for index, step in enumerate(trace, start=1):
        print(
            f"step={index:>2} "
            f"state={format_state(step.state):<18} "
            f"action={ACTION_NAMES[step.action]:<5} "
            f"reward={step.reward:>5.2f} "
            f"tile={step.tile:<5} "
            f"next={format_state(step.next_state)}"
        )

    total_return = sum(step.reward for step in trace)
    completed = trace[-1].terminated and not trace[-1].truncated
    print(
        f"\nresult: completed={completed} "
        f"steps={len(trace)} "
        f"return={total_return:.2f} "
        f"final_epsilon={result.final_epsilon:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-learning policy for KeyDoorGrid."
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--layout", choices=("classic", "extended"), default="classic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_training_summary(
        episodes=args.episodes,
        seed=args.seed,
        report_every=args.report_every,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
