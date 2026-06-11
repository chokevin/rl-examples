# rl-examples
Reinforcement learning examples powered by PufferLib

## PufferLib scaffold

This repo includes a tiny local LineWorld example that wraps a custom Gymnasium
environment with PufferLib and runs it through PufferLib's vector API.

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

See `examples/README.md` for the current PufferLib pin, local validation, and
upstream example workflow.
