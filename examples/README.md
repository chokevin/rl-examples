# Examples

This directory contains reinforcement learning examples.

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
uv run python -m unittest discover -s tests
```

The smoke script verifies that the PufferLib package, the modules used by
PufferLib's own examples, Gymnasium, and the `puffer` CLI are available in the
local environment. The unittest suite checks the repo-owned environments'
observation, action, reward, termination, and truncation semantics.

## Layout

`examples/envs/` contains Gymnasium worlds. An environment owns state dynamics:
reset, step, observation/action spaces, rewards, termination, and truncation.

`examples/learners/` contains agents or algorithms that learn from environments.
A learner owns action selection and policy updates, so the same environment can
be reused by hand-written policies, tabular learners, PufferLib, or future neural
training examples.

## Tiny local rollout

```bash
uv run python examples/line_world_pufferlib.py
uv run python examples/key_door_grid_pufferlib.py
uv run python examples/key_door_grid_q_learning.py
uv run python examples/key_door_grid_q_learning.py --layout extended --episodes 800
uv run python examples/key_door_grid_compare_learners.py
```

`examples/envs/line_world.py` defines `LineWorldEnv`, a small `gymnasium.Env`
with a two-value observation `[position_fraction, elapsed_fraction]`, two
actions (`0` left, `1` right), a `+1.0` goal reward, and a `-0.01` step cost.
Episodes terminate at the rightmost goal or truncate at `max_steps`.
`line_world_pufferlib.py` wraps that environment with
`pufferlib.emulation.GymnasiumPufferEnv` and runs four vectorized environments
through `pufferlib.vector.Serial`. It is intentionally small enough to use as a
starting point for repo-specific examples.

`examples/envs/key_door_grid.py` defines `KeyDoorGridEnv`, a deterministic 4x4
grid world with observation `[row_fraction, column_fraction, has_key,
elapsed_fraction]` and four movement actions (`0` up, `1` right, `2` down,
`3` left). The agent starts in the lower-left corner, must collect the key in the
upper-left corner, and can only terminate at the lower-right goal after carrying
the key. A wall blocks one shortcut, a mud tile adds extra cost, locked-goal
bumps are penalized, and episodes truncate at `max_steps`.
`key_door_grid_pufferlib.py` runs a deterministic key-to-goal policy through the
same PufferLib vector wrapper.

The same env also has an `extended` 6x6 layout with a longer key-to-goal route,
a wall barrier, mud penalties, and a tempting locked-goal path. It is still
small enough for tabular methods, but it is large enough to make convergence
differences visible.

`examples/learners/key_door_q_learning.py` trains a tabular Q-learning policy
over discrete states `(row, column, has_key)`. The environment still only applies
world rules; the learner explores, updates Q-values from rewards, and eventually
prints a greedy policy that gets the key before entering the goal.

`examples/key_door_grid_compare_learners.py` compares SARSA, Q-learning, and
Dyna-Q on the extended layout. It measures convergence by periodic greedy
evaluation, not by noisy exploratory training episodes. Dyna-Q usually converges
first here because the environment is deterministic and small: every real step
also teaches a replayable model transition, so rewards propagate with fewer real
episodes. Q-learning is typically next because it backs up the greedy next
action even while exploring. SARSA can be slower because it backs up the action
the exploratory policy will actually take, which makes it more conservative
while epsilon is still nonzero.

## Running upstream PufferLib examples

PufferLib's upstream examples live in the PufferLib repository:

https://github.com/PufferAI/PufferLib/tree/3.0/examples

Use the smoke check above first. PyPI currently publishes PufferLib 3.0.0 as the
latest stable package, so this project pins `pufferlib==3.0.0` and uses the
`3.0` branch examples with this scaffold. The upstream GitHub `experiments` tag
advertises PufferLib 4.0.0, but it is not a drop-in replacement for these
Gymnasium examples: the installed package does not include the v3
`pufferlib.emulation` or `pufferlib.vector` modules, and the `puffer` CLI expects
a compiled `_C` backend.

PufferLib 3.0 still installs legacy Gym transitively, but repo-owned examples
should use Gymnasium APIs directly.

Some PufferLib examples materialize a local `resources/` directory at runtime;
that generated directory is ignored by this repo.
