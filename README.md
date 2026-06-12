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
uv run python examples/key_door_grid_neural_q_learning.py
uv run python examples/key_door_grid_convergence_race.py
uv run python examples/key_door_policy_optimizers.py
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
`key_door_grid_neural_q_learning.py` keeps `KeyDoorGridEnv` environment-only but
replaces the Q-table with a tiny PyTorch Q-network. The neural learner encodes
`(row, column, has_key)` as a one-hot vector, computes a Bellman MSE loss, and
compares SGD with Adam using `loss.backward()` and `optimizer.step()`.
`key_door_grid_convergence_race.py` runs the implemented learners concurrently
on the harder `extended` layout, which better separates tabular planning,
tabular TD updates, and neural optimizer-driven updates. Because these local
environments are cheap in-process simulators, the race ranks learners by
estimated wall-clock time to stable solve while still reporting episode counts
for sample-efficiency context.
`key_door_policy_optimizers.py` adds PPO, TRPO, and GRPO to the teaching map as
policy-optimization techniques: they update policy networks from rollout data
rather than backing up Q-values.

See `examples/README.md` for the current PufferLib pin, local validation, and
upstream example workflow.
