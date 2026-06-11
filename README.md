# rl-examples
Reinforcement learning examples powered by PufferLib

## PufferLib scaffold

This repo includes small local Gymnasium environments that can be wrapped with
PufferLib and run through PufferLib's vector API.

```bash
uv sync
uv run python examples/smoke_pufferlib.py
uv run python -m unittest discover -s tests
uv run python examples/line_world_pufferlib.py
uv run python examples/key_door_grid_pufferlib.py
uv run python examples/key_door_grid_q_learning.py
uv run python examples/key_door_grid_compare_learners.py
```

The first repo-owned environment is `LineWorldEnv` in
`examples/envs/line_world.py`. It exposes a two-value observation
`[position_fraction, elapsed_fraction]`, two discrete actions (`0` left,
`1` right), a `+1.0` terminal goal reward, a `-0.01` step cost, termination at
the rightmost goal, and truncation at `max_steps`.

`KeyDoorGridEnv` in `examples/envs/key_door_grid.py` adds a compact 2D grid
with a key, locked goal, wall, mud tile, four movement actions, shaped rewards,
goal termination, and max-step truncation.

Learners live separately under `examples/learners/`. The
`key_door_grid_q_learning.py` script trains a tabular Q-learning policy against
`KeyDoorGridEnv` and then prints the learned greedy rollout.
`key_door_grid_compare_learners.py` runs the harder KeyDoorGrid layout across
SARSA, Q-learning, and Dyna-Q to compare greedy-policy convergence.

See `examples/README.md` for the current PufferLib pin, local validation, and
upstream example workflow.
