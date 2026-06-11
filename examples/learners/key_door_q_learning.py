"""Tabular Q-learning for the KeyDoorGrid environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from examples.envs import KeyDoorGridEnv
except ModuleNotFoundError as error:
    if error.name not in {"examples", "examples.envs"}:
        raise
    from envs import KeyDoorGridEnv


State = tuple[int, int, int]
ACTION_NAMES = {
    0: "up",
    1: "right",
    2: "down",
    3: "left",
}


@dataclass(frozen=True)
class EpisodeStep:
    state: State
    action: int
    reward: float
    next_state: State
    terminated: bool
    truncated: bool
    tile: str


@dataclass(frozen=True)
class QLearningResult:
    q_values: np.ndarray
    episode_returns: list[float]
    episode_lengths: list[int]
    final_epsilon: float


def state_from_info(info: dict[str, Any]) -> State:
    row, column = info["position"]
    return int(row), int(column), int(bool(info["has_key"]))


def _choose_action(
    q_values: np.ndarray,
    state: State,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[-1]))
    return int(np.argmax(q_values[state]))


def train_q_learning(
    *,
    episodes: int = 500,
    learning_rate: float = 0.40,
    discount: float = 0.95,
    epsilon: float = 0.40,
    epsilon_decay: float = 0.99,
    min_epsilon: float = 0.05,
    seed: int = 0,
    env: KeyDoorGridEnv | None = None,
) -> QLearningResult:
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

    training_env = env if env is not None else KeyDoorGridEnv()
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
    exploration_rate = epsilon

    for episode in range(episodes):
        _, info = training_env.reset(seed=seed + episode)
        state = state_from_info(info)
        total_reward = 0.0

        while True:
            action = _choose_action(q_values, state, exploration_rate, rng)
            _, reward, terminated, truncated, info = training_env.step(action)
            next_state = state_from_info(info)
            done = terminated or truncated

            best_next = 0.0 if done else float(np.max(q_values[next_state]))
            target = reward + discount * best_next
            q_values[state + (action,)] += learning_rate * (
                target - q_values[state + (action,)]
            )

            state = next_state
            total_reward += reward
            if done:
                episode_returns.append(total_reward)
                episode_lengths.append(int(info["episode_length"]))
                break

        exploration_rate = max(min_epsilon, exploration_rate * epsilon_decay)

    return QLearningResult(
        q_values=q_values,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        final_epsilon=exploration_rate,
    )


def greedy_rollout(
    q_values: np.ndarray,
    *,
    env: KeyDoorGridEnv | None = None,
    seed: int = 0,
) -> list[EpisodeStep]:
    rollout_env = env if env is not None else KeyDoorGridEnv()
    _, info = rollout_env.reset(seed=seed)
    state = state_from_info(info)
    steps: list[EpisodeStep] = []

    while True:
        action = int(np.argmax(q_values[state]))
        _, reward, terminated, truncated, info = rollout_env.step(action)
        next_state = state_from_info(info)
        steps.append(
            EpisodeStep(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                truncated=truncated,
                tile=str(info["tile"]),
            )
        )
        state = next_state
        if terminated or truncated:
            return steps
