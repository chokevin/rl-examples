# Examples

This directory contains PufferLib-backed reinforcement learning examples.

## Setup

```bash
uv sync
```

The local examples use Gymnasium APIs. The project currently pins the latest
PufferLib release available from PyPI: `pufferlib==3.0.0` with
`gymnasium==0.29.1`.

## Local smoke checks

```bash
uv run python examples/smoke_pufferlib.py
```

The smoke script verifies that the PufferLib package, the modules used by
PufferLib's own examples, Gymnasium, and the `puffer` CLI are available in the
local environment.

## Tiny local rollout

```bash
uv run python examples/line_world_pufferlib.py
```

`line_world_pufferlib.py` defines a small `gymnasium.Env`, wraps it with
`pufferlib.emulation.GymnasiumPufferEnv`, and runs four vectorized environments
through `pufferlib.vector.Serial`. It is intentionally small enough to use as a
starting point for repo-specific examples.

## Running upstream PufferLib examples

PufferLib's upstream examples live in the PufferLib repository:

https://github.com/PufferAI/PufferLib/tree/3.0/examples

Use the smoke check above first. The upstream repository currently defaults to
PufferLib 4.0, while PyPI currently publishes PufferLib 3.0.0, so use the `3.0`
branch examples with this scaffold until the published package catches up or this
project switches to installing PufferLib from Git.

PufferLib 3.0 still installs legacy Gym transitively, but repo-owned examples
should use Gymnasium APIs directly.

Some PufferLib examples materialize a local `resources/` directory at runtime;
that generated directory is ignored by this repo.
