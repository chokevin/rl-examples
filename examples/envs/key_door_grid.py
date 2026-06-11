"""A compact key-and-door grid-world Gymnasium environment."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np


EXTENDED_LAYOUT = {
    "height": 6,
    "width": 6,
    "max_steps": 45,
    "start_position": (5, 0),
    "key_position": (0, 0),
    "goal_position": (5, 5),
    "wall_positions": {(1, 2), (2, 2), (3, 2), (4, 2)},
    "mud_positions": {(5, 3), (5, 4), (1, 4)},
}


class KeyDoorGridEnv(gymnasium.Env):
    """Small deterministic grid world with a key, a locked goal, a wall, and mud.

    Observation: [row / (height - 1), column / (width - 1), has_key, elapsed].
    Actions: 0 up, 1 right, 2 down, 3 left.
    Reward: -0.01 per step, +0.20 for first key pickup, -0.10 on mud,
    -0.05 for bumping into the locked goal without the key, +1.0 at the goal.
    Done: terminated at the goal with the key; truncated at max_steps first.
    """

    metadata = {"render_modes": []}

    ACTIONS: dict[int, tuple[int, int]] = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }
    STEP_REWARD = -0.01
    KEY_REWARD = 0.20
    LOCKED_GOAL_PENALTY = -0.05
    MUD_PENALTY = -0.10
    GOAL_REWARD = 1.0

    def __init__(
        self,
        height: int | None = None,
        width: int | None = None,
        max_steps: int | None = None,
        layout: str = "classic",
    ) -> None:
        if layout not in {"classic", "extended"}:
            raise ValueError("layout must be 'classic' or 'extended'")

        self.layout = layout
        if layout == "extended":
            height = EXTENDED_LAYOUT["height"] if height is None else height
            width = EXTENDED_LAYOUT["width"] if width is None else width
            max_steps = EXTENDED_LAYOUT["max_steps"] if max_steps is None else max_steps
            if height != EXTENDED_LAYOUT["height"] or width != EXTENDED_LAYOUT["width"]:
                raise ValueError("extended layout must use height=6 and width=6")
        else:
            height = 4 if height is None else height
            width = 4 if width is None else width
            max_steps = 20 if max_steps is None else max_steps

        if height < 3:
            raise ValueError("height must be at least 3")
        if width < 3:
            raise ValueError("width must be at least 3")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.height = height
        self.width = width
        self.max_steps = max_steps
        if layout == "extended":
            self.start_position = EXTENDED_LAYOUT["start_position"]
            self.key_position = EXTENDED_LAYOUT["key_position"]
            self.goal_position = EXTENDED_LAYOUT["goal_position"]
            self.wall_positions = set(EXTENDED_LAYOUT["wall_positions"])
            self.mud_positions = set(EXTENDED_LAYOUT["mud_positions"])
        else:
            self.start_position = (height - 1, 0)
            self.key_position = (0, 0)
            self.goal_position = (height - 1, width - 1)
            self.wall_positions = {(1, 1)}
            self.mud_positions = {(height - 1, width - 2)}
        self.observation_space = gymnasium.spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(4)
        self.position = self.start_position
        self.has_key = False
        self.elapsed_steps = 0
        self.episode_return = 0.0

    def _observation(self) -> np.ndarray:
        return np.array(
            [
                self.position[0] / (self.height - 1),
                self.position[1] / (self.width - 1),
                float(self.has_key),
                self.elapsed_steps / self.max_steps,
            ],
            dtype=np.float32,
        )

    def _is_in_bounds(self, position: tuple[int, int]) -> bool:
        row, column = position
        return 0 <= row < self.height and 0 <= column < self.width

    def _info(
        self,
        *,
        tile: str,
        blocked: bool = False,
    ) -> dict[str, bool | float | int | str | tuple[int, int]]:
        return {
            "position": self.position,
            "has_key": self.has_key,
            "tile": tile,
            "blocked": blocked,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, bool | tuple[int, int]]]:
        super().reset(seed=seed, options=options)
        self.position = self.start_position
        self.has_key = False
        self.elapsed_steps = 0
        self.episode_return = 0.0
        return self._observation(), {
            "position": self.position,
            "has_key": self.has_key,
        }

    def step(
        self,
        action: int | np.integer[Any] | np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, bool | float | int | str | tuple[int, int]],
    ]:
        action_id = int(np.asarray(action).item())
        if action_id not in self.ACTIONS:
            raise ValueError(f"action must be 0, 1, 2, or 3, got {action_id}")

        row_delta, column_delta = self.ACTIONS[action_id]
        candidate = (
            self.position[0] + row_delta,
            self.position[1] + column_delta,
        )
        reward = self.STEP_REWARD
        tile = "empty"
        blocked = False

        if not self._is_in_bounds(candidate):
            blocked = True
            tile = "boundary"
        elif candidate in self.wall_positions:
            blocked = True
            tile = "wall"
        elif candidate == self.goal_position and not self.has_key:
            blocked = True
            tile = "locked_goal"
            reward += self.LOCKED_GOAL_PENALTY
        else:
            self.position = candidate
            if self.position == self.key_position:
                tile = "key"
                if not self.has_key:
                    self.has_key = True
                    reward += self.KEY_REWARD
            elif self.position in self.mud_positions:
                tile = "mud"
                reward += self.MUD_PENALTY
            elif self.position == self.goal_position:
                tile = "goal"

        self.elapsed_steps += 1
        terminated = self.position == self.goal_position and self.has_key
        if terminated:
            reward += self.GOAL_REWARD
        truncated = self.elapsed_steps >= self.max_steps and not terminated
        self.episode_return += reward

        info = self._info(tile=tile, blocked=blocked)
        if terminated or truncated:
            info["episode_return"] = self.episode_return
            info["episode_length"] = self.elapsed_steps

        return self._observation(), reward, terminated, truncated, info
