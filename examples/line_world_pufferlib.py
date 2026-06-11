"""Run a tiny Gymnasium environment through PufferLib vectorization."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pufferlib.emulation
import pufferlib.vector
import gymnasium


class LineWorldEnv(gymnasium.Env):
    """One-dimensional goal-reaching environment for a fast local rollout."""

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


def make_env(
    size: int = 5,
    max_steps: int = 8,
    buf: dict[str, np.ndarray] | None = None,
    seed: int | None = 0,
) -> Any:
    return pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=LineWorldEnv,
        env_kwargs={"size": size, "max_steps": max_steps},
        buf=buf,
        seed=seed,
    )


def positions_from_observations(observations: np.ndarray, size: int) -> list[int]:
    return np.rint(observations[:, 0] * (size - 1)).astype(np.int32).tolist()


def rounded_values(values: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in values]


def run_rollout(num_envs: int, steps: int, size: int, max_steps: int, seed: int) -> None:
    envs = pufferlib.vector.make(
        make_env,
        env_kwargs={"size": size, "max_steps": max_steps},
        backend=pufferlib.vector.Serial,
        num_envs=num_envs,
        seed=seed,
    )

    try:
        observations, _ = envs.reset(seed=seed)
        print(
            f"LineWorld via PufferLib: envs={envs.num_envs} "
            f"obs_space={envs.single_observation_space} "
            f"action_space={envs.single_action_space}"
        )
        print(f"reset positions={positions_from_observations(observations, size)}")

        actions = np.ones(envs.num_envs, dtype=np.int32)
        for step_idx in range(1, steps + 1):
            observations, rewards, terminals, truncations, infos = envs.step(actions)
            done = np.logical_or(terminals, truncations)
            episode_returns = [
                round(float(info["episode_return"]), 2)
                for info in infos
                if "episode_return" in info
            ]
            suffix = (
                f" episode_returns={episode_returns}" if episode_returns else ""
            )
            print(
                f"step={step_idx} actions={actions.tolist()} "
                f"positions={positions_from_observations(observations, size)} "
                f"rewards={rounded_values(rewards)} "
                f"done={done.tolist()}{suffix}"
            )
    finally:
        envs.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny vectorized LineWorld rollout with PufferLib."
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rollout(
        num_envs=args.num_envs,
        steps=args.steps,
        size=args.size,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
