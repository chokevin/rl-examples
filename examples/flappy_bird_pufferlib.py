"""Run the C Flappy Bird environment with the PufferLib 4-compatible runner."""

from __future__ import annotations

import argparse

import numpy as np
import pufferlib

from envs.flappy_bird import CFlappyBirdEnv, ensure_flappy_library
from pufferlib4 import SerialVectorEnv


def rounded(values: np.ndarray) -> list[float]:
    return [round(float(value), 3) for value in values]


def run_rollout(num_envs: int, steps: int, max_steps: int, seed: int) -> None:
    ensure_flappy_library()
    envs = SerialVectorEnv(
        [
            lambda env_idx=env_idx: CFlappyBirdEnv(
                max_steps=max_steps,
                seed=seed + env_idx,
            )
            for env_idx in range(num_envs)
        ]
    )

    try:
        observations, _ = envs.reset(seed=seed)
        print(
            f"FlappyBird(C) with PufferLib {pufferlib.__version__}: envs={envs.num_envs} "
            f"obs_space={envs.single_observation_space} "
            f"action_space={envs.single_action_space}"
        )
        print(f"reset first_obs={rounded(observations[0])}")

        actions = np.zeros(envs.num_envs, dtype=np.int32)
        for step_idx in range(1, steps + 1):
            actions[:] = 1 if step_idx % 18 == 1 else 0
            observations, rewards, terminals, truncations, infos = envs.step(actions)
            done = np.logical_or(terminals, truncations)
            scores = [
                int(info["score"])
                for info in infos
                if isinstance(info, dict) and "score" in info
            ]
            print(
                f"step={step_idx} rewards={rounded(rewards)} "
                f"done={done.tolist()} scores={scores[:num_envs]} "
                f"first_obs={rounded(observations[0])}"
            )
    finally:
        envs.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a vectorized rollout of the C Flappy Bird env with PufferLib."
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rollout(
        num_envs=args.num_envs,
        steps=args.steps,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
