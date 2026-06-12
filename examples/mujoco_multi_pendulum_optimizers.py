"""Compare optimizers on a configurable MuJoCo stacked multi-pendulum cart."""

from __future__ import annotations

from mujoco_triple_cart_pole_optimizers import parse_args, print_comparison


def main() -> None:
    args = parse_args(
        default_poles=5,
        description=(
            "Compare SGD and Adam on a custom MuJoCo cart balancing a configurable "
            "stacked multi-pendulum."
        ),
    )
    print_comparison(
        epochs=args.epochs,
        seeds=tuple(range(1, args.seeds + 1)),
        samples=args.samples,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        target_return=args.target_return,
        optimizers=tuple(args.optimizers),
        render=args.render,
        render_mode=args.render_mode,
        render_optimizer=args.render_optimizer,
        render_episodes=args.render_episodes,
        render_seed=args.render_seed,
        render_delay=args.render_delay,
        pole_count=args.poles,
    )


if __name__ == "__main__":
    main()
