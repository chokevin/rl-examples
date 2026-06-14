# Examples

This directory contains PufferLib-backed reinforcement learning examples.

## Setup

```bash
uv sync
```

The local examples use Gymnasium APIs for environments and PufferLib 4 model
components for policies. The project installs PufferLib from upstream Git because
PyPI still publishes 3.0.0 while the upstream default branch is 4.0.0.

### PufferLib 4 migration choice

PufferLib 4 is not a drop-in replacement for the old examples:

- `pufferlib.emulation`, `pufferlib.vector`, and `pufferlib.pytorch` are no
  longer packaged.
- `pufferlib.pufferl` expects an environment-specific compiled `_C` backend.
- `pufferlib.models` is packaged and is the stable piece this repo can use
  immediately.

So the migration deliberately breaks the 3.0 wrapper path. There are now two
PufferLib 4 paths:

- `examples/train_flappy_native.py`: builds a compiled `pufferlib._C` backend for
  Flappy and runs PufferLib's `pufferlib.pufferl` trainer/dashboard with
  `--slowly` CPU training. The env stepping is native compiled code.
- `examples/train_flappy_ppo.py`: keeps the simpler repo-owned PPO loop for the
  raylib policy exporter. It uses PufferLib 4 model components but not the
  dashboard trainer.

## Local smoke checks

```bash
uv run python examples/smoke_pufferlib.py
uv run python -m unittest discover -s tests
```

The smoke script verifies that PufferLib 4, `pufferlib.models`, and Gymnasium
import in the local environment. The unittest suite checks the repo-owned
environments and the local PufferLib 4 helper layer.

## Tiny local rollout

```bash
uv run python examples/line_world_pufferlib.py
```

`examples/envs/line_world.py` defines `LineWorldEnv`, a small `gymnasium.Env`
with a two-value observation `[position_fraction, elapsed_fraction]`, two
actions (`0` left, `1` right), a `+1.0` goal reward, and a `-0.01` step cost.
Episodes terminate at the rightmost goal or truncate at `max_steps`.
`line_world_pufferlib.py` runs four copies through the repo-owned
`SerialVectorEnv`. It is intentionally small enough to use as a starting point
for repo-specific examples without depending on PufferLib 3's removed vector
wrapper.

## Basic C/raylib Flappy Bird

```bash
make -C examples/flappy_bird test
make -C examples/flappy_bird run
uv run python examples/flappy_bird_pufferlib.py
```

`examples/flappy_bird/flappy_env.c` defines a deterministic, fixed-size C
environment with two actions (`0` no-op, `1` flap), four observation values
(`[bird_y, bird_velocity, next_pipe_distance, next_gap_delta]` normalized to the
screen), a small alive reward, a pass reward, and a crash penalty. `main.c` keeps
raylib-specific drawing and input separate from the environment step logic.

Building or running the playable renderer requires raylib. The core environment
test target does not require raylib.

`examples/envs/flappy_bird.py` wraps the same C core with `ctypes` and exposes a
Gymnasium environment. It builds `examples/flappy_bird/build/libflappy_env.*` on
first use. PufferLib 4 no longer packages the old Gymnasium vector wrapper, so
the examples batch local env instances with `examples/pufferlib4.py`.

### Gymnasium to training contract

The Gymnasium wrapper is still the small API contract the local trainer needs:

- `observation_space`: shape, dtype, and valid range for what the agent sees.
- `action_space`: legal actions the policy may output (`0` no-op, `1` flap).
- `reset(seed=...)`: starts a fresh episode and returns the first observation.
- `step(action)`: advances one frame and returns `(observation, reward,
  terminated, truncated, info)`.

The local vector runner uses that contract to create many env copies, batch
observations, ask the PufferLib 4 policy for actions, feed those actions back
into `step`, and collect rewards for PPO. The C code owns the game physics; the
Python wrapper owns this training interface.

### Native simulator precedents

This C-core/Python-wrapper split is not unusual for heavier RL environments. Real
examples include:

- Arcade Learning Environment (ALE): Atari simulator in C++ with Python/Gym
  wrappers: https://github.com/Farama-Foundation/Arcade-Learning-Environment
- MuJoCo: physics engine in C with Python/Gymnasium wrappers for robotics tasks:
  https://github.com/google-deepmind/mujoco
- Procgen: procedural game environments with a native backend and Python API:
  https://github.com/openai/procgen
- ViZDoom: Doom-based simulator in C++ with a Python API:
  https://github.com/Farama-Foundation/ViZDoom

PufferLib itself is the training/vectorization layer here. Our installed
PufferLib package has a compiled extension for its advantage kernel, but the
native simulator pattern usually comes from the environment package underneath.

Current PufferLib envs are not all written in C or visualized with raylib. In
the installed package, many `pufferlib.environments.*` modules are Python
adapters around existing environment packages: Classic Control and MuJoCo call
`gymnasium.make(...)`, while Procgen wraps the `procgen` package. PufferLib also
has an `ocean` family of native C-backed envs; those use compiled bindings and
many include raylib for human rendering. Our Flappy Bird example is closest to
that native/Ocean direction, but simpler: C game core, Python training adapter,
and raylib only for the playable viewer.

### Where the current path is fast and where it is not

PufferLib does not make the raylib window faster. The native 4.0 path gets these
wins:

- Headless C stepping: no raylib draw loop during training.
- Compiled `_C` backend: observations, actions, rewards, and done flags live in
  contiguous native buffers exposed directly to PufferLib's trainer.
- PufferLib dashboard: `pufferlib.pufferl` can import the env backend and render
  its Rich TUI again.

On macOS this repo uses PufferLib's `--slowly` PyTorch trainer because upstream's
CUDA backend is not available. The compiled env backend is still native; the PPO
update itself runs through PyTorch on CPU.

## Native PufferLib 4 trainer/TUI

```bash
uv run python examples/build_flappy_native.py
uv run python examples/train_flappy_native.py --total-timesteps 200000
```

`examples/build_flappy_native.py` copies the installed PufferLib 4 Python package
into `examples/flappy_bird/build/native_runtime/`, writes Flappy's config there,
and compiles `examples/flappy_bird/native/binding.c` into `pufferlib/_C*.so`
through the local CPU compatibility wrapper in
`examples/flappy_bird/native/pufferlib_compat/`. `binding.c` is the same seam
used by upstream PufferLib Ocean-style envs: it declares observation/action
metadata, owns env init/step/reset/logging, and lets the `_C` module expose raw
observation/reward/terminal buffers to the trainer.

`examples/train_flappy_native.py` sets `PYTHONPATH` to that native runtime and
runs:

```bash
python -m pufferlib.pufferl train flappy_bird --slowly
```

That is the path that restores PufferLib's Rich TUI. You can pass additional
PufferLib CLI flags after `--`, for example:

```bash
uv run python examples/train_flappy_native.py --total-timesteps 100000 -- --train.ent-coef 0.002
```

The native trainer defaults are intentionally different from PufferLib's large
upstream batch defaults: 32 agents, 64-step horizons, Adam, and a `-2.0` initial
flap-logit bias. With 200k total timesteps that gives 97 PPO updates; the old
1024-agent default gave only 3 updates. The score column is also sparse: a bird
must survive roughly 348 frames to pass the first pipe, while an untrained random
policy usually dies around frame 35. If the TUI shows `score=0` at the start,
watch `episode_length` and `episode_return`; score only moves once episodes are
long enough to clear a pipe.

To watch a native PufferLib checkpoint in the raylib viewer, export the latest
native `.bin` checkpoint to the small text format used by the C app:

```bash
uv run python examples/export_flappy_policy.py \
  --data-dir experiments/native_checkpoints/flappy_bird \
  --output experiments/flappy_bird/policy.txt \
  --pipe-gap 220
make -C examples/flappy_bird run-policy
```

If you trained with a different gap, pass the same `--pipe-gap` value when
exporting. Native checkpoints use PufferLib's `encoder -> MLP -> decoder` model;
the exporter writes `flappy_policy_v2`, which the raylib C viewer can evaluate
without Python.

## PPO training scaffold

```bash
uv run python examples/train_flappy_ppo.py --total-timesteps 200000
uv run python examples/eval_flappy_ppo.py
uv run python examples/export_flappy_policy.py
make -C examples/flappy_bird run-policy
```

`examples/train_flappy_ppo.py` creates a repo-owned serial vector env, builds a
PufferLib 4 policy, and runs a small local PPO loop. The training defaults keep
the C dynamics but use a wider gap, small gap-centering reward, and a no-op-biased
initial policy so PPO sees useful signal before it can clear pipes. The default
checkpoint directory is ignored at `experiments/flappy_bird/`. Increase
`--total-timesteps` for a stronger policy; the short default is meant to prove
the scaffold and produce a first checkpoint quickly.

To watch the trained policy in raylib, `examples/export_flappy_policy.py` converts
the latest local PPO `.pt` checkpoint or native PufferLib `.bin` checkpoint into
`experiments/flappy_bird/policy.txt`. The raylib viewer can load that file with
`--policy`; `make -C examples/flappy_bird run-policy` uses the default exported
path. The C viewer reimplements the small policy forward pass, so the window does
not need Python or PyTorch while it is running.

## Running upstream PufferLib 4 examples

PufferLib's upstream source lives in the PufferLib repository:

https://github.com/PufferAI/PufferLib

The upstream 4.0 trainer path expects a native/Ocean environment build that
produces `pufferlib/_C...so`. Flappy now has the environment-side `binding.c`
contract. The remaining cluster/GPU parity step is packaging it with upstream's
CUDA native toolchain (`bindings.cu`/`build.sh`) instead of this repo's
Mac-friendly CPU compatibility wrapper.

Some PufferLib examples materialize a local `resources/` directory at runtime;
that generated directory is ignored by this repo.
