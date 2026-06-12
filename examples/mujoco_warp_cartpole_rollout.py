"""Train MuJoCo cartpole with optional MuJoCo Warp batched simulation."""

from __future__ import annotations

import argparse

from learners.mujoco_batched_cartpole import (
    BATCHED_CARTPOLE_ENV_ID,
    BATCHED_CARTPOLE_OPTIMIZERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EXPLORATION_SIGMA,
    DEFAULT_MAX_STEPS,
    DEFAULT_TARGET_RETURN,
)
from learners.mujoco_warp_cartpole import (
    mujoco_warp_status,
    train_mujoco_warp_es_cartpole,
)


def format_generation(value: int | None) -> str:
    return "-" if value is None else str(value)


def print_training(
    *,
    optimizer_name: str,
    learning_rate: float | None,
    batch_size: int,
    eval_batch_size: int,
    generations: int,
    max_steps: int,
    target_return: float,
    exploration_sigma: float,
    sigma_decay: float,
    seed: int,
    device: str | None,
    require_cuda: bool,
) -> None:
    status = mujoco_warp_status(device=device)
    if not status.available:
        raise SystemExit(status.reason)
    if require_cuda and not status.is_cuda:
        raise SystemExit(
            f"MuJoCo Warp is available on {status.device}, but --require-cuda "
            "was set. Choose a CUDA device or remove --require-cuda for CPU smoke runs."
        )

    result = train_mujoco_warp_es_cartpole(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        generations=generations,
        max_steps=max_steps,
        target_return=target_return,
        exploration_sigma=exploration_sigma,
        sigma_decay=sigma_decay,
        seed=seed,
        device=device,
    )
    print(
        "MuJoCo Warp cartpole ES training: "
        f"env={BATCHED_CARTPOLE_ENV_ID} batch={batch_size} "
        f"eval_batch={eval_batch_size} optimizer={result.optimizer_name} "
        f"device={result.device} cuda={result.device_is_cuda}"
    )
    print(
        "thread mapping: batch many MuJoCo worlds in mujoco_warp.Data, compute "
        "linear-policy controls from the batched state, step with mujoco_warp.step, "
        "then update a PyTorch policy mean through an ES surrogate loss"
    )
    print(
        "note: this is a real MuJoCo Warp backend, but this teaching version still "
        "copies state to Python for closed-loop actions; the 18M steps/s path needs "
        "the policy and rollout loop fused on the accelerator"
    )
    print(
        f"versions: mujoco_warp={result.mujoco_warp_version} "
        f"warp_lang={result.warp_version}"
    )
    print(
        f"target: eval mean return >= {target_return:.1f} over {eval_batch_size} "
        f"agents capped at {max_steps} MuJoCo steps"
    )
    print()
    print(
        f"{'gen':<4} {'sigma':<8} {'train_mean':<12} {'train_best':<11} "
        f"{'eval_mean':<11} {'eval_min':<9} {'eval_solved':<12} "
        f"{'rollout_sec':<12} {'physics_steps/s':<16} {'active_steps/s'}"
    )
    for generation in result.history:
        rollout_seconds = (
            generation.train.wall_clock_seconds
            + generation.evaluation.wall_clock_seconds
        )
        print(
            f"{generation.generation:<4} {generation.sigma:<8.3f} "
            f"{generation.train.mean_return:<12.1f} "
            f"{generation.train.max_return:<11.1f} "
            f"{generation.evaluation.mean_return:<11.1f} "
            f"{generation.evaluation.min_return:<9.1f} "
            f"{generation.evaluation.solved_fraction:<12.2f} "
            f"{rollout_seconds:<12.3f} "
            f"{generation.train.physics_steps_per_second:<16.0f} "
            f"{generation.train.active_steps_per_second:.0f}"
        )

    print(
        "\nlearned_generation="
        f"{format_generation(result.learned_generation)} "
        f"total_rollout_sec={result.total_rollout_seconds:.3f} "
        f"final_policy={['%.3f' % value for value in result.final_policy_parameters]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MuJoCo InvertedPendulum cartpole with MuJoCo Warp batched worlds."
    )
    parser.add_argument("--optimizer", choices=BATCHED_CARTPOLE_OPTIMIZERS, default="adam")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--target-return", type=float, default=DEFAULT_TARGET_RETURN)
    parser.add_argument("--exploration-sigma", type=float, default=DEFAULT_EXPLORATION_SIGMA)
    parser.add_argument("--sigma-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Warp device, for example cpu or cuda:0")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit instead of running the slow CPU development backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_training(
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        generations=args.generations,
        max_steps=args.max_steps,
        target_return=args.target_return,
        exploration_sigma=args.exploration_sigma,
        sigma_decay=args.sigma_decay,
        seed=args.seed,
        device=args.device,
        require_cuda=args.require_cuda,
    )


if __name__ == "__main__":
    main()
