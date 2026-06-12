"""Benchmark MuJoCo Warp CUDA graph capture for batched CartPole physics."""

from __future__ import annotations

import argparse

from learners.mujoco_batched_cartpole import (
    BATCHED_CARTPOLE_ENV_ID,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_STEPS,
)
from learners.mujoco_warp_cartpole import (
    DEFAULT_CUDAGRAPH_CONTROLLER_GAINS,
    benchmark_mujoco_warp_cudagraph_steps,
    mujoco_warp_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture an unrolled MuJoCo Warp CartPole physics CUDA graph and "
            "measure replay throughput."
        )
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--graph-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--graph-replays", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-noise", type=float, default=0.01)
    parser.add_argument("--control-value", type=float, default=0.0)
    parser.add_argument(
        "--captured-controller",
        action="store_true",
        help=(
            "Set controls inside the captured graph with a tiny linear stabilizing "
            "controller. This keeps joint-limit physics enabled while avoiding "
            "uncontrolled drift into the limits."
        ),
    )
    parser.add_argument(
        "--controller-gains",
        type=float,
        nargs=4,
        metavar=("X", "THETA", "XDOT", "THETADOT"),
        default=DEFAULT_CUDAGRAPH_CONTROLLER_GAINS,
        help=(
            "Linear gains used by --captured-controller for "
            "(x, theta, xdot, thetadot)."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--disable-contact",
        action="store_true",
        help=(
            "Disable MuJoCo contact processing for benchmark-only runs. "
            "Safe for InvertedPendulum's no-contact geometry, but not a general RL default."
        ),
    )
    parser.add_argument(
        "--disable-joint-limits",
        action="store_true",
        help=(
            "Disable joint-limit constraint processing for benchmark-only short-horizon runs. "
            "Only use when the measured trajectory is proven not to approach limits."
        ),
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print MuJoCo Warp device status without requiring CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = mujoco_warp_status(device=args.device)
    if args.status_only:
        print(
            "MuJoCo Warp CUDA graph status: "
            f"available={status.available} device={status.device} cuda={status.is_cuda} "
            f"mujoco_warp={status.mujoco_warp_version} warp_lang={status.warp_version} "
            f"reason={status.reason}"
        )
        return
    if not status.available:
        raise SystemExit(status.reason)
    if not status.is_cuda:
        raise SystemExit(
            f"CUDA graph benchmark requires a CUDA Warp device; got {status.device}. "
            "Run on an NVIDIA machine with --device cuda:0."
        )

    result = benchmark_mujoco_warp_cudagraph_steps(
        batch_size=args.batch_size,
        graph_steps=args.graph_steps,
        graph_replays=args.graph_replays,
        seed=args.seed,
        initial_noise=args.initial_noise,
        control_value=args.control_value,
        controller_gains=tuple(args.controller_gains) if args.captured_controller else None,
        device=args.device,
        disable_contact=args.disable_contact,
        disable_joint_limits=args.disable_joint_limits,
    )
    print(
        "MuJoCo Warp CUDA graph benchmark: "
        f"env={BATCHED_CARTPOLE_ENV_ID} device={result.device} "
        f"batch={result.batch_size} graph_steps={result.graph_steps} "
        f"graph_replays={result.graph_replays}"
    )
    print(
        "capture: one CUDA graph contains the unrolled mujoco_warp.step task; "
        "timing replays the captured graph without a Python per-step loop"
    )
    print(
        f"versions: mujoco_warp={result.mujoco_warp_version} "
        f"warp_lang={result.warp_version}"
    )
    print(f"model_options: disableflags={result.model_disableflags}")
    if result.controller_gains is None:
        print(f"control: constant value={result.control_value:g}")
    else:
        print(f"control: captured_linear_controller gains={result.controller_gains}")
    print(
        f"physics_steps={result.physics_steps} "
        f"wall_clock_sec={result.wall_clock_seconds:.6f} "
        f"physics_steps/s={result.physics_steps_per_second:.0f}"
    )


if __name__ == "__main__":
    main()
