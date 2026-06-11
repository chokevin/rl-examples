"""A tiny one-dimensional Gymnasium environment."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np


class LineWorldEnv(gymnasium.Env):
    """One-dimensional goal-reaching environment for a fast local rollout.

    Observation: [position / (size - 1), elapsed_steps / max_steps].
    Actions: 0 moves left, 1 moves right.
    Reward: +1.0 on reaching the rightmost goal, -0.01 otherwise.
    Done: terminated at the goal; truncated at max_steps before the goal.
    """

    metadata = {"render_modes": []}

    def __init__(self, size: int = 5, max_steps: int = 8) -> None:
        if size < 2:
            raise ValueError("size must be at least 2")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.size = size
        self.max_steps = max_steps
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(2)
        self.position = 0
        self.elapsed_steps = 0
        self.episode_return = 0.0

    def _observation(self) -> np.ndarray:
        return np.array(
            [
                self.position / (self.size - 1),
                self.elapsed_steps / self.max_steps,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        super().reset(seed=seed, options=options)
        self.position = 0
        self.elapsed_steps = 0
        self.episode_return = 0.0
        return self._observation(), {"position": self.position}

    def step(
        self,
        action: int | np.integer[Any] | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float | int]]:
        action_id = int(np.asarray(action).item())
        if action_id not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action_id}")

        move = -1 if action_id == 0 else 1
        self.position = int(np.clip(self.position + move, 0, self.size - 1))
        self.elapsed_steps += 1

        terminated = self.position == self.size - 1
        truncated = self.elapsed_steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else -0.01
        self.episode_return += reward

        info: dict[str, float | int] = {"position": self.position}
        if terminated or truncated:
            info["episode_return"] = self.episode_return
            info["episode_length"] = self.elapsed_steps

        return self._observation(), reward, terminated, truncated, info
