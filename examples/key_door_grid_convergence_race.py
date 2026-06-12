"""Run implemented KeyDoorGrid learners side by side on a harder layout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from statistics import mean, median

try:
    from examples.envs import KeyDoorGridEnv
    from examples.learners.key_door_neural_q_learning import train_neural_q_learning
    from examples.learners.key_door_td_control import train_td_control
except ModuleNotFoundError as error:
    if error.name not in {"examples", "examples.envs", "examples.learners"}:
        raise
    from envs import KeyDoorGridEnv
    from learners.key_door_neural_q_learning import train_neural_q_learning
    from learners.key_door_td_control import train_td_control


RUNNABLE_LEARNERS = (
    "sarsa",
    "q_learning",
    "dyna_q",
    "neural_q_sgd",
    "neural_q_adam",
)


@dataclass(frozen=True)
class RaceRunResult:
    learner: str
    seed: int
    stable_solve_episode: int | None
    final_solved: bool
    final_return: float
    greedy_steps: int
    wall_clock_seconds: float = 0.0
    estimated_seconds_to_stable: float | None = None


@dataclass(frozen=True)
class RaceSummary:
    learner: str
    stable_runs: int
    total_runs: int
    final_solved_runs: int
    median_stable_solve_episode: int | None
    median_estimated_seconds_to_stable: float | None
    mean_wall_clock_seconds: float
    mean_final_return: float
    mean_greedy_steps: float


def stable_solve_episode(evaluations: list[object]) -> int | None:
    for index, evaluation in enumerate(evaluations):
        if all(point.solved for point in evaluations[index:]):
            return int(evaluation.episode)
    return None


def estimate_seconds_to_stable(
    *,
    wall_clock_seconds: float,
    stable_episode: int | None,
    total_episodes: int,
) -> float | None:
    if stable_episode is None:
        return None
    return wall_clock_seconds * stable_episode / total_episodes


def _run_single_learner(
    *,
    learner: str,
    seed: int,
    layout: str,
    episodes: int,
    eval_every: int,
    planning_steps: int,
) -> RaceRunResult:
    start = time.perf_counter()
    if learner in {"sarsa", "q_learning", "dyna_q"}:
        result = train_td_control(
            algorithm=learner,
            episodes=episodes,
            eval_every=eval_every,
            planning_steps=planning_steps,
            seed=seed,
            env=KeyDoorGridEnv(layout=layout),
        )
        wall_clock_seconds = time.perf_counter() - start
        final = result.evaluations[-1]
        return RaceRunResult(
            learner=learner,
            seed=seed,
            stable_solve_episode=result.episodes_to_solve,
            final_solved=final.solved,
            final_return=final.episode_return,
            greedy_steps=final.episode_length,
            wall_clock_seconds=wall_clock_seconds,
            estimated_seconds_to_stable=estimate_seconds_to_stable(
                wall_clock_seconds=wall_clock_seconds,
                stable_episode=result.episodes_to_solve,
                total_episodes=episodes,
            ),
        )

    optimizer_name = learner.removeprefix("neural_q_")
    result = train_neural_q_learning(
        optimizer_name=optimizer_name,
        episodes=episodes,
        eval_every=eval_every,
        seed=seed,
        env=KeyDoorGridEnv(layout=layout),
    )
    wall_clock_seconds = time.perf_counter() - start
    stable_episode = stable_solve_episode(result.evaluations)
    final = result.evaluations[-1]
    return RaceRunResult(
        learner=learner,
        seed=seed,
        stable_solve_episode=stable_episode,
        final_solved=final.solved,
        final_return=final.episode_return,
        greedy_steps=final.episode_length,
        wall_clock_seconds=wall_clock_seconds,
        estimated_seconds_to_stable=estimate_seconds_to_stable(
            wall_clock_seconds=wall_clock_seconds,
            stable_episode=stable_episode,
            total_episodes=episodes,
        ),
    )


def run_convergence_race(
    *,
    learners: tuple[str, ...] = RUNNABLE_LEARNERS,
    seeds: tuple[int, ...] = tuple(range(1, 6)),
    layout: str = "extended",
    episodes: int = 800,
    eval_every: int = 20,
    planning_steps: int = 15,
) -> dict[str, list[RaceRunResult]]:
    if not learners:
        raise ValueError("learners must contain at least one learner")
    unknown = set(learners) - set(RUNNABLE_LEARNERS)
    if unknown:
        raise ValueError(f"unknown learners: {sorted(unknown)}")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if eval_every < 1:
        raise ValueError("eval_every must be at least 1")
    if planning_steps < 0:
        raise ValueError("planning_steps must be at least 0")

    results = {learner: [] for learner in learners}
    max_workers = min(len(learners), len(learners) * len(seeds))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_single_learner,
                learner=learner,
                seed=seed,
                layout=layout,
                episodes=episodes,
                eval_every=eval_every,
                planning_steps=planning_steps,
            )
            for learner in learners
            for seed in seeds
        ]
        for future in as_completed(futures):
            result = future.result()
            results[result.learner].append(result)

    return results


def summarize_race(
    results: dict[str, list[RaceRunResult]],
) -> list[RaceSummary]:
    summaries: list[RaceSummary] = []
    for learner, rows in results.items():
        stable_episodes = [
            row.stable_solve_episode
            for row in rows
            if row.stable_solve_episode is not None
        ]
        stable_seconds = [
            row.estimated_seconds_to_stable
            for row in rows
            if row.estimated_seconds_to_stable is not None
        ]
        summaries.append(
            RaceSummary(
                learner=learner,
                stable_runs=len(stable_episodes),
                total_runs=len(rows),
                final_solved_runs=sum(1 for row in rows if row.final_solved),
                median_stable_solve_episode=(
                    int(median(stable_episodes)) if stable_episodes else None
                ),
                median_estimated_seconds_to_stable=(
                    median(stable_seconds) if stable_seconds else None
                ),
                mean_wall_clock_seconds=mean(row.wall_clock_seconds for row in rows),
                mean_final_return=mean(row.final_return for row in rows),
                mean_greedy_steps=mean(row.greedy_steps for row in rows),
            )
        )

    summaries.sort(
        key=lambda summary: (
            summary.median_estimated_seconds_to_stable is None,
            summary.median_estimated_seconds_to_stable
            if summary.median_estimated_seconds_to_stable is not None
            else float("inf"),
            summary.median_stable_solve_episode is None,
            summary.median_stable_solve_episode
            if summary.median_stable_solve_episode is not None
            else max(summary.total_runs, 1_000_000),
            -summary.stable_runs,
            -summary.mean_final_return,
        )
    )
    return summaries


def format_solve_episode(value: int | None) -> str:
    return "not stable" if value is None else str(value)


def format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_race(
    *,
    layout: str,
    episodes: int,
    eval_every: int,
    seeds: tuple[int, ...],
    results: dict[str, list[RaceRunResult]],
) -> None:
    summaries = summarize_race(results)
    print(
        "KeyDoorGrid convergence race: "
        f"layout={layout} episodes={episodes} seeds={list(seeds)} "
        f"eval_every={eval_every}"
    )
    print(
        "The extended layout is the better demo environment because its longer "
        "route separates planning, tabular updates, and neural optimizer updates."
    )
    print(
        "Current repo policy: because these are cheap in-process simulators, "
        "rank by estimated wall-clock time to stable solve; keep episode counts "
        "as sample-efficiency context."
    )
    print("Runnable learners only: SARSA, Q-learning, Dyna-Q, neural Q + SGD/Adam.")
    print("PPO/TRPO/GRPO are cataloged, but not implemented trainers yet.")
    print()
    print(
        f"{'rank':<4} {'learner':<15} {'stable':<8} {'final':<8} "
        f"{'wall_to_solve':<14} {'median_solve_ep':<16} "
        f"{'run_sec':<8} {'final_return':<13} {'greedy_steps'}"
    )
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"{rank:<4} {summary.learner:<15} "
            f"{summary.stable_runs}/{summary.total_runs:<6} "
            f"{summary.final_solved_runs}/{summary.total_runs:<6} "
            f"{format_seconds(summary.median_estimated_seconds_to_stable):<14} "
            f"{format_solve_episode(summary.median_stable_solve_episode):<16} "
            f"{summary.mean_wall_clock_seconds:<8.3f} "
            f"{summary.mean_final_return:<13.2f} "
            f"{summary.mean_greedy_steps:.1f}"
        )

    print("\nper-seed stable solve episodes and estimated seconds:")
    for learner in results:
        rows = sorted(results[learner], key=lambda row: row.seed)
        episode_values = [
            "-"
            if row.stable_solve_episode is None
            else str(row.stable_solve_episode)
            for row in rows
        ]
        second_values = [
            format_seconds(row.estimated_seconds_to_stable) for row in rows
        ]
        print(f"{learner:<15} episodes={episode_values} seconds={second_values}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run implemented KeyDoorGrid learners in a convergence race."
    )
    parser.add_argument(
        "--layout",
        choices=("classic", "extended"),
        default="extended",
    )
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--planning-steps", type=int, default=15)
    parser.add_argument(
        "--learners",
        nargs="+",
        choices=RUNNABLE_LEARNERS,
        default=RUNNABLE_LEARNERS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(range(1, args.seeds + 1))
    results = run_convergence_race(
        learners=tuple(args.learners),
        seeds=seeds,
        layout=args.layout,
        episodes=args.episodes,
        eval_every=args.eval_every,
        planning_steps=args.planning_steps,
    )
    print_race(
        layout=args.layout,
        episodes=args.episodes,
        eval_every=args.eval_every,
        seeds=seeds,
        results=results,
    )


if __name__ == "__main__":
    main()
