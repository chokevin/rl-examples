"""Run a tiny Gymnasium environment through PufferLib vectorization."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pufferlib.emulation
import pufferlib.vector

from envs import LineWorldEnv


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
