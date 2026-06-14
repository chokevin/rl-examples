"""Run PufferLib 4's trainer/TUI against the compiled native Flappy backend."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from . import build_flappy_native
except ImportError:
    import build_flappy_native


def estimate_train_updates(total_timesteps: int, total_agents: int, horizon: int) -> int:
    return total_timesteps // (total_agents * horizon)


def estimate_frames_until_first_score(
    screen_width: float = 800.0,
    pipe_spawn_offset: float = 180.0,
    pipe_width: float = 70.0,
    bird_x: float = 160.0,
    bird_radius: float = 14.0,
    pipe_speed: float = 2.6,
) -> int:
    score_distance = screen_width + pipe_spawn_offset + pipe_width - (bird_x - bird_radius)
    return int(round(score_distance / pipe_speed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--total-agents", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--pipe-gap", type=float, default=220.0)
    parser.add_argument("--no-reward-shaping", action="store_true")
    parser.add_argument("--centering-reward", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["Adam", "Muon"], default="Adam")
    parser.add_argument("--initial-flap-logit-bias", type=float, default=-2.0)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional args passed to `python -m pufferlib.pufferl train flappy_bird`",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        build_flappy_native.build(force=args.force_build)

    runtime = build_flappy_native.BUILD_DIR
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(runtime)
        if not env.get("PYTHONPATH")
        else f"{runtime}{os.pathsep}{env['PYTHONPATH']}"
    )

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    train_updates = estimate_train_updates(
        args.total_timesteps,
        args.total_agents,
        args.horizon,
    )
    if train_updates < 1:
        raise ValueError(
            "total_timesteps must be at least total_agents * horizon "
            f"({args.total_agents * args.horizon})"
        )
    print(
        "Native PPO: "
        f"{train_updates} train updates, "
        f"{args.total_agents * args.horizon} agent steps/update. "
        f"Flappy score stays 0 until an episode survives about "
        f"{estimate_frames_until_first_score()} frames to pass the first pipe.",
        flush=True,
    )

    command = [
        sys.executable,
        "-m",
        "pufferlib.pufferl",
        "train",
        "flappy_bird",
        "--slowly",
        "--train.total-timesteps",
        str(args.total_timesteps),
        "--vec.total-agents",
        str(args.total_agents),
        "--train.horizon",
        str(args.horizon),
        "--train.minibatch-size",
        str(args.minibatch_size),
        "--policy.hidden-size",
        str(args.hidden_size),
        "--policy.num-layers",
        str(args.num_layers),
        "--env.pipe-gap",
        str(args.pipe_gap),
        "--env.reward-shaping",
        "0" if args.no_reward_shaping else "1",
        "--env.centering-reward",
        str(args.centering_reward),
        "--train.learning-rate",
        str(args.learning_rate),
        "--train.optimizer",
        args.optimizer,
        "--train.initial-flap-logit-bias",
        str(args.initial_flap_logit_bias),
        *extra_args,
    ]
    subprocess.run(command, check=True, env=env, cwd=Path.cwd())


if __name__ == "__main__":
    main()
