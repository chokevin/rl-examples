# rl-examples
Reinforcement learning examples powered by PufferLib

## PufferLib scaffold

This repo includes tiny local Gymnasium environments plus PufferLib 4 model
scaffolding for policy training.

```bash
uv sync
uv run python examples/smoke_pufferlib.py
uv run python -m unittest discover -s tests
uv run python examples/line_world_pufferlib.py
```

The first repo-owned environment is `LineWorldEnv` in
`examples/envs/line_world.py`. It exposes a two-value observation
`[position_fraction, elapsed_fraction]`, two discrete actions (`0` left,
`1` right), a `+1.0` terminal goal reward, a `-0.01` step cost, termination at
the rightmost goal, and truncation at `max_steps`.

## Basic C/raylib Flappy Bird

`examples/flappy_bird/` contains a small Flappy Bird-style environment written in
C. The environment state and `step` logic live in `flappy_env.c`, while
`main.c` is only the raylib renderer and keyboard loop.

```bash
make -C examples/flappy_bird test
make -C examples/flappy_bird
make -C examples/flappy_bird run
```

Install raylib first to build or run the windowed example. On macOS:

```bash
brew install raylib
```

The C core can also run headless through the PufferLib 4-compatible trainer:

```bash
uv run python examples/flappy_bird_pufferlib.py
uv run python examples/build_flappy_native.py
uv run python examples/train_flappy_native.py --total-timesteps 200000
uv run python examples/train_flappy_ppo.py --total-timesteps 200000
uv run python examples/eval_flappy_ppo.py
uv run python examples/export_flappy_policy.py
make -C examples/flappy_bird run-policy
```

See `examples/README.md` for the PufferLib 4 workflow and upstream native-backend
notes.
