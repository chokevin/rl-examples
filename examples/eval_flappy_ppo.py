"""Evaluate a saved Flappy Bird PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from envs.flappy_bird import CFlappyBirdEnv, ensure_flappy_library
from pufferlib4 import load_checkpoint, make_flappy_policy, policy_logits_values


def evaluate(
    checkpoint: Path,
    episodes: int,
    max_steps: int,
    pipe_gap: float | None,
    hidden_size: int,
    seed: int,
    device: str,
) -> None:
    """Run a trained policy without learning and report how well it plays.

    In plain terms: load the saved neural-net weights, start fresh Flappy Bird
    episodes, let the policy pick flap/no-op actions from each observation, and
    print the average score/return/episode length at the end.
    """
    ensure_flappy_library()
    env = CFlappyBirdEnv(
        max_steps=max_steps,
        seed=seed,
        pipe_gap=pipe_gap,
    )
    policy = make_flappy_policy(
        observation_size=int(np.prod(env.single_observation_space.shape)),
        hidden_size=hidden_size,
    ).to(device)
    loaded = load_checkpoint(checkpoint, device)
    policy.load_state_dict(loaded.state_dict)
    policy.eval()

    scores: list[int] = []
    returns: list[float] = []
    lengths: list[int] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        info = {"score": 0}

        while not done:
            obs_tensor = torch.as_tensor(observation[None, :], device=device)
            with torch.no_grad():
                logits, _ = policy_logits_values(policy, obs_tensor)
                action = Categorical(logits=logits).sample()
            observation, reward, terminated, truncated, info = env.step(
                int(action.cpu().numpy().reshape(-1)[0])
            )
            episode_return += reward
            done = terminated or truncated

        scores.append(int(info["score"]))
        returns.append(episode_return)
        lengths.append(int(info.get("episode_length", 0)))

    print(f"checkpoint={checkpoint}")
    print(f"episodes={episodes}")
    print(f"mean_score={np.mean(scores):.2f} max_score={max(scores)}")
    print(f"mean_return={np.mean(returns):.3f} mean_length={np.mean(lengths):.1f}")


def resolve_checkpoint(checkpoint: Path | None, data_dir: Path) -> Path:
    if checkpoint is not None and str(checkpoint) != "latest":
        return checkpoint

    checkpoints = sorted(data_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {data_dir}")
    return checkpoints[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Flappy Bird PPO checkpoint.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments") / "flappy_bird",
    )
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--pipe-gap", type=float, default=220.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        checkpoint=resolve_checkpoint(args.checkpoint, args.data_dir),
        episodes=args.episodes,
        max_steps=args.max_steps,
        pipe_gap=args.pipe_gap,
        hidden_size=args.hidden_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
