"""Compare tabular TD-control learners on KeyDoorGrid."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

import numpy as np

try:
    from examples.envs import KeyDoorGridEnv
    from examples.learners.key_door_q_learning import (
        ACTION_NAMES,
        EpisodeStep,
        State,
        greedy_rollout,
        state_from_info,
    )
except ModuleNotFoundError as error:
    if error.name not in {"examples", "examples.envs", "examples.learners"}:
        raise
    from envs import KeyDoorGridEnv
    from learners.key_door_q_learning import (
        ACTION_NAMES,
        EpisodeStep,
        State,
        greedy_rollout,
        state_from_info,
    )


ALGORITHMS = ("sarsa", "q_learning", "dyna_q")


@dataclass(frozen=True)
class EvaluationPoint:
    episode: int
    solved: bool
    episode_return: float
    episode_length: int


@dataclass(frozen=True)
class TDControlResult:
    algorithm: str
    q_values: np.ndarray
    episode_returns: list[float]
    episode_lengths: list[int]
    evaluations: list[EvaluationPoint]
    episodes_to_solve: int | None
    final_epsilon: float


@dataclass(frozen=True)
class TechniqueSummary:
    algorithm: str
    solved_runs: int
    total_runs: int
    median_episodes_to_solve: int | None
    mean_final_return: float
    mean_greedy_steps: float


def clone_env(env: KeyDoorGridEnv) -> KeyDoorGridEnv:
    return KeyDoorGridEnv(
        height=env.height,
        width=env.width,
        max_steps=env.max_steps,
        layout=env.layout,
    )


def evaluate_greedy(
    q_values: np.ndarray,
    env: KeyDoorGridEnv,
    *,
    episode: int,
    seed: int,
) -> EvaluationPoint:
    trace = greedy_rollout(q_values, env=clone_env(env), seed=seed)
    return EvaluationPoint(
        episode=episode,
        solved=trace[-1].terminated and not trace[-1].truncated,
        episode_return=sum(step.reward for step in trace),
        episode_length=len(trace),
    )


def stable_solve_episode(evaluations: list[EvaluationPoint]) -> int | None:
    for index, evaluation in enumerate(evaluations):
        if all(point.solved for point in evaluations[index:]):
            return evaluation.episode
    return None


def _validate_common_parameters(
    *,
    episodes: int,
    learning_rate: float,
    discount: float,
    epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    eval_every: int,
) -> None:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0.0, 1.0]")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0.0, 1.0]")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0.0, 1.0]")
    if not 0.0 <= min_epsilon <= 1.0:
        raise ValueError("min_epsilon must be in [0.0, 1.0]")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be in (0.0, 1.0]")
    if eval_every < 1:
        raise ValueError("eval_every must be at least 1")


def _choose_action(
    q_values: np.ndarray,
    state: State,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[-1]))
    return int(np.argmax(q_values[state]))


def _q_learning_update(
    q_values: np.ndarray,
    state: State,
    action: int,
    reward: float,
    next_state: State,
    done: bool,
    *,
    learning_rate: float,
    discount: float,
) -> None:
    best_next = 0.0 if done else float(np.max(q_values[next_state]))
    target = reward + discount * best_next
    q_values[state + (action,)] += learning_rate * (
        target - q_values[state + (action,)]
    )


def train_td_control(
    *,
    algorithm: str,
    episodes: int = 800,
    learning_rate: float = 0.40,
    discount: float = 0.95,
    epsilon: float = 0.50,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    eval_every: int = 20,
    planning_steps: int = 15,
    seed: int = 0,
    env: KeyDoorGridEnv | None = None,
) -> TDControlResult:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    if planning_steps < 0:
        raise ValueError("planning_steps must be at least 0")
    _validate_common_parameters(
        episodes=episodes,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        eval_every=eval_every,
    )

    training_env = env if env is not None else KeyDoorGridEnv(layout="extended")
    rng = np.random.default_rng(seed)
    q_values = np.zeros(
        (
            training_env.height,
            training_env.width,
            2,
            training_env.action_space.n,
        ),
        dtype=np.float32,
    )
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    evaluations: list[EvaluationPoint] = []
    model: dict[tuple[State, int], tuple[float, State, bool]] = {}
    model_keys: list[tuple[State, int]] = []
    exploration_rate = epsilon

    for episode in range(1, episodes + 1):
        _, info = training_env.reset(seed=seed + episode)
        state = state_from_info(info)
        action = (
            _choose_action(q_values, state, exploration_rate, rng)
            if algorithm == "sarsa"
            else None
        )
        total_reward = 0.0

        while True:
            if action is None:
                action = _choose_action(q_values, state, exploration_rate, rng)

            _, reward, terminated, truncated, info = training_env.step(action)
            next_state = state_from_info(info)
            done = terminated or truncated
            total_reward += reward

            if algorithm == "sarsa":
                next_action = (
                    _choose_action(q_values, next_state, exploration_rate, rng)
                    if not done
                    else None
                )
                next_value = (
                    0.0
                    if done
                    else float(q_values[next_state + (int(next_action),)])
                )
                target = reward + discount * next_value
                q_values[state + (action,)] += learning_rate * (
                    target - q_values[state + (action,)]
                )
                action = next_action
            else:
                _q_learning_update(
                    q_values,
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    learning_rate=learning_rate,
                    discount=discount,
                )
                if algorithm == "dyna_q":
                    model_key = (state, action)
                    if model_key not in model:
                        model_keys.append(model_key)
                    model[model_key] = (reward, next_state, done)
                    for _ in range(planning_steps):
                        replay_state, replay_action = model_keys[
                            int(rng.integers(len(model_keys)))
                        ]
                        replay_reward, replay_next_state, replay_done = model[
                            (replay_state, replay_action)
                        ]
                        _q_learning_update(
                            q_values,
                            replay_state,
                            replay_action,
                            replay_reward,
                            replay_next_state,
                            replay_done,
                            learning_rate=learning_rate,
                            discount=discount,
                        )
                action = None

            state = next_state
            if done:
                episode_returns.append(total_reward)
                episode_lengths.append(int(info["episode_length"]))
                break

        exploration_rate = max(min_epsilon, exploration_rate * epsilon_decay)
        if episode % eval_every == 0 or episode == episodes:
            evaluations.append(
                evaluate_greedy(
                    q_values,
                    training_env,
                    episode=episode,
                    seed=seed + 10_000 + episode,
                )
            )

    return TDControlResult(
        algorithm=algorithm,
        q_values=q_values,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        evaluations=evaluations,
        episodes_to_solve=stable_solve_episode(evaluations),
        final_epsilon=exploration_rate,
    )


def compare_techniques(
    *,
    algorithms: tuple[str, ...] = ALGORITHMS,
    seeds: tuple[int, ...] = tuple(range(1, 11)),
    episodes: int = 800,
    eval_every: int = 20,
    planning_steps: int = 15,
    env_layout: str = "extended",
) -> tuple[list[TechniqueSummary], dict[str, list[TDControlResult]]]:
    results: dict[str, list[TDControlResult]] = {algorithm: [] for algorithm in algorithms}
    summaries: list[TechniqueSummary] = []

    for algorithm in algorithms:
        for seed in seeds:
            results[algorithm].append(
                train_td_control(
                    algorithm=algorithm,
                    episodes=episodes,
                    eval_every=eval_every,
                    planning_steps=planning_steps,
                    seed=seed,
                    env=KeyDoorGridEnv(layout=env_layout),
                )
            )

        solved = [
            result.episodes_to_solve
            for result in results[algorithm]
            if result.episodes_to_solve is not None
        ]
        final_evaluations = [result.evaluations[-1] for result in results[algorithm]]
        summaries.append(
            TechniqueSummary(
                algorithm=algorithm,
                solved_runs=len(solved),
                total_runs=len(seeds),
                median_episodes_to_solve=(
                    int(median(solved)) if len(solved) == len(seeds) else None
                ),
                mean_final_return=mean(
                    evaluation.episode_return for evaluation in final_evaluations
                ),
                mean_greedy_steps=mean(
                    evaluation.episode_length for evaluation in final_evaluations
                ),
            )
        )

    summaries.sort(
        key=lambda summary: (
            summary.median_episodes_to_solve is None,
            summary.median_episodes_to_solve
            if summary.median_episodes_to_solve is not None
            else episodes + 1,
            -summary.mean_final_return,
        )
    )
    return summaries, results


def technique_explanation(summary: TechniqueSummary) -> str:
    if summary.algorithm == "dyna_q":
        return (
            "Dyna-Q learns a model from real transitions and replays simulated "
            "updates, so key/goal rewards propagate through this deterministic "
            "grid with fewer real episodes."
        )
    if summary.algorithm == "q_learning":
        return (
            "Q-learning is off-policy: even while it explores, it backs up the "
            "best next action, which helps it learn the greedy key-to-goal route."
        )
    if summary.algorithm == "sarsa":
        return (
            "SARSA is on-policy: it backs up the action it will actually take "
            "under epsilon-greedy exploration, so it can be more conservative "
            "and slower while exploration is still active."
        )
    return "No explanation registered for this technique."


__all__ = [
    "ACTION_NAMES",
    "ALGORITHMS",
    "EpisodeStep",
    "EvaluationPoint",
    "TDControlResult",
    "TechniqueSummary",
    "compare_techniques",
    "evaluate_greedy",
    "stable_solve_episode",
    "technique_explanation",
    "train_td_control",
]
