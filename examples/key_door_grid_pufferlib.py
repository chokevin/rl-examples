"""Run the key-and-door grid-world environment through PufferLib vectorization."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pufferlib.emulation
import pufferlib.vector

from envs import KeyDoorGridEnv


DEFAULT_POLICY = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)


def make_env(
    height: int = 4,
    width: int = 4,
    max_steps: int = 20,
    buf: dict[str, np.ndarray] | None = None,
    seed: int | None = 0,
) -> Any:
    return pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=KeyDoorGridEnv,
        env_kwargs={"height": height, "width": width, "max_steps": max_steps},
        buf=buf,
        seed=seed,
    )


def states_from_observations(
    observations: np.ndarray,
    height: int,
    width: int,
) -> list[dict[str, int | bool]]:
    positions = np.rint(
        observations[:, :2] * np.array([height - 1, width - 1], dtype=np.float32)
    ).astype(np.int32)
    has_keys = observations[:, 2] > 0.5
    return [
        {"row": int(row), "column": int(column), "has_key": bool(has_key)}
        for (row, column), has_key in zip(positions, has_keys)
    ]


def rounded_values(values: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in values]


def run_rollout(
    num_envs: int,
    steps: int,
    height: int,
    width: int,
    max_steps: int,
    seed: int,
) -> None:
    envs = pufferlib.vector.make(
        make_env,
        env_kwargs={"height": height, "width": width, "max_steps": max_steps},
        backend=pufferlib.vector.Serial,
        num_envs=num_envs,
        seed=seed,
    )

    try:
        observations, _ = envs.reset(seed=seed)
        print(
            f"KeyDoorGrid via PufferLib: envs={envs.num_envs} "
            f"obs_space={envs.single_observation_space} "
            f"action_space={envs.single_action_space}"
        )
        print(f"reset states={states_from_observations(observations, height, width)}")

        for step_idx in range(1, steps + 1):
            action = DEFAULT_POLICY[(step_idx - 1) % len(DEFAULT_POLICY)]
            actions = np.full(envs.num_envs, action, dtype=np.int32)
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
                f"states={states_from_observations(observations, height, width)} "
                f"rewards={rounded_values(rewards)} "
                f"done={done.tolist()}{suffix}"
            )
    finally:
        envs.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a vectorized KeyDoorGrid rollout with PufferLib."
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=len(DEFAULT_POLICY))
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rollout(
        num_envs=args.num_envs,
        steps=args.steps,
        height=args.height,
        width=args.width,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
