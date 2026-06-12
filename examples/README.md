# Examples

This directory contains reinforcement learning examples.

## Setup

```bash
uv sync
```

The local examples use Gymnasium APIs. The project currently pins the latest
PufferLib release available from PyPI: `pufferlib==3.0.0` with
`gymnasium[mujoco]==0.29.1`.
MuJoCo Warp is optional because it is aimed at NVIDIA hardware; install it with
`uv sync --extra warp` or run the Warp example with `uv run --extra warp ...`.

## Local smoke checks

```bash
uv run python examples/smoke_pufferlib.py
uv run python -m unittest discover -s tests
```

The smoke script verifies that the PufferLib package, the modules used by
PufferLib's own examples, Gymnasium, and the `puffer` CLI are available in the
local environment. The unittest suite checks repo-owned environment semantics
plus the small neural learner mechanics used by the teaching demos.

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
uv run python examples/key_door_grid_neural_q_learning.py
uv run python examples/key_door_grid_convergence_race.py
uv run python examples/cartpole_neural_optimizers.py
uv run python examples/mujoco_inverted_pendulum_optimizers.py
uv run python examples/mujoco_inverted_pendulum_optimizers.py --render
uv run python examples/mujoco_batched_cartpole_rollout.py
uv run --extra warp python examples/mujoco_warp_cartpole_rollout.py --batch-size 256 --eval-batch-size 64 --generations 1 --max-steps 40
uv run --extra warp python examples/mujoco_warp_cudagraph_benchmark.py --device cuda:0 --batch-size 8192 --graph-steps 200 --graph-replays 10
uv run --extra warp python examples/mujoco_warp_cudagraph_benchmark.py --device cuda:0 --batch-size 3145728 --graph-steps 5 --graph-replays 50 --disable-contact --captured-controller
uv run python examples/mujoco_triple_cart_pole_optimizers.py --render
uv run python examples/mujoco_multi_pendulum_optimizers.py --optimizers adam --poles 5 --epochs 10 --samples 512 --batch-size 64 --eval-every 5 --eval-episodes 2 --max-steps 80 --target-return 40
uv run python examples/key_door_policy_optimizers.py
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

`examples/key_door_grid_neural_q_learning.py` is the optimizer bridge. It uses
the classic `KeyDoorGridEnv` for transitions and rewards, but the environment
still does not know about PyTorch or optimizers. The learner replaces the
Q-table with a tiny Q-network over one-hot states, computes a Bellman MSE loss
for `Q(s, a)`, and updates weights with `loss.backward()` plus either
`torch.optim.SGD` or `torch.optim.Adam`. This is intentionally smaller than DQN:
no replay buffer, target network, Muon, or Shampoo.

`examples/key_door_grid_convergence_race.py` is the better convergence demo. It
defaults to the `extended` layout and runs the implemented learners concurrently:
SARSA, tabular Q-learning, Dyna-Q, neural Q with SGD, and neural Q with Adam.
The longer route and wall barrier make the ranking easier to see than the 4x4
classic grid. For this repo's cheap simulated environments, it ranks by
estimated wall-clock time to stable solve and keeps episode counts as
sample-efficiency context.

`examples/cartpole_neural_optimizers.py` uses Gymnasium's standard
`CartPole-v1` instead of a repo-owned deterministic grid. The learner is a tiny
REINFORCE policy-gradient loop: CartPole supplies continuous observations,
actions, and rewards; a PyTorch policy network emits action logits; and SGD or
Adam updates the network from reward-to-go returns with `loss.backward()` and
`optimizer.step()`. This teaches a different point than KeyDoorGrid: the
environment dynamics are no longer hand-authored or tabular, so the useful
comparison is optimizer behavior on a standard control benchmark. The script
reports wall-clock time, environment steps, rolling episode returns, best return,
and greedy evaluation return so it lines up with the bottleneck table below.

`examples/mujoco_inverted_pendulum_optimizers.py` wires in Gymnasium MuJoCo with
`InvertedPendulum-v4`. It is a continuous-control bridge rather than a full
MuJoCo RL trainer: the environment runs real MuJoCo physics with continuous Box
actions, and a tiny linear PyTorch policy learns to imitate a stabilizing
controller from synthetic states near the upright region. The point is to make
the standard-control stack visibly work in seconds while still showing optimizer
behavior through MSE, `loss.backward()`, and `optimizer.step()`. It reports
wall-clock-to-target when reached, epochs, supervised samples, final MuJoCo
return, best MuJoCo return, and final loss. The default command is headless;
pass `--render` to open Gymnasium's MuJoCo viewer for the best trained policy,
or use `--render --render-mode rgb_array` for offscreen frame rendering. The
human viewer path includes a small compatibility shim for Gymnasium 0.29.1 with
newer MuJoCo releases, where the overlay's old `solver_iter` field is now
reported as `solver_niter`.

`examples/mujoco_batched_cartpole_rollout.py` follows the large-rollout-batch
CartPole pattern from the linked thread. It uses the same MuJoCo
`InvertedPendulum-v4` cartpole, but instead of stepping one Gymnasium env at a
time it samples a batch of linear policies, computes their controls from the
current batched state, and advances all agents through MuJoCo's C
`rollout.rollout` API one closed-loop step at a time. The policy update is an
evolution-strategy gradient estimate expressed as a PyTorch surrogate loss, so
`loss.backward()` and SGD/Adam still drive the policy mean. The default
`--batch-size 8192` mirrors the thread's rollout batch size, but the script
prints measured local throughput instead of claiming the fully fused 18M
steps/sec number.

`examples/mujoco_warp_cartpole_rollout.py` moves that same batched policy-search
lesson onto optional MuJoCo Warp. It copies the `InvertedPendulum-v4` model into
`mujoco_warp.Model`, creates a single `mujoco_warp.Data` with many worlds, and
steps those worlds with `mujoco_warp.step`. The policy update is still the same
PyTorch ES surrogate, so the teaching comparison stays focused: MuJoCo C rollout
shows CPU-side batched stepping, while MuJoCo Warp shows the accelerator-oriented
backend and the cost of copying state back to Python for closed-loop actions.
Use `--device cuda:0 --batch-size 8192` on an NVIDIA machine for the thread-like
path; the documented small command is a CPU smoke run.

`examples/mujoco_warp_cudagraph_benchmark.py` is the speed-first follow-up for
the "APIC/API capture" claim. Instead of copying state back to Python for every
closed-loop action, it captures an unrolled MuJoCo Warp physics task with
`wp.ScopedCapture()` and replays the CUDA graph with `wp.capture_launch()`. This
is deliberately framed as a substrate benchmark, not a solved policy-training
loop: reaching 18M-style throughput requires fusing policy/control into the same
captured device-side task or exposing the captured graph through a C/CUDA API.
For profiler-driven short-horizon experiments, it also accepts `--disable-contact`
and `--disable-joint-limits`; use the joint-limit option only after proving the
measured trajectory stays away from limits. To keep joint-limit physics enabled
while avoiding uncontrolled drift into those limits, pass `--captured-controller`.
That mode captures a tiny linear controller with the physics graph, so controls
are updated on-device before each MuJoCo Warp step; the H200 one-GPU run reached
30M+ physics steps/sec with `disableflags=16` (`mjDSBL_CONTACT` only), not
`mjDSBL_LIMIT`.

`examples/mujoco_triple_cart_pole_optimizers.py` is the "three stacked poles"
variant. Gymnasium's built-in `CartPole-v1` only has one pole, so this example
adds a repo-owned MuJoCo model: one sliding cart, one actuator, and serial
hinge-connected pole links. The demo computes an LQR teacher from a finite-
difference linearization of the MuJoCo dynamics around the upright pose, creates
near-upright supervised state/action samples, then trains a tiny linear PyTorch
policy with SGD or Adam. It defaults to the original three-pole lesson but now
accepts `--poles` for configurable stacked multi-pendulum systems.

`examples/mujoco_multi_pendulum_optimizers.py` is the same learner with a
five-pole default. That is substantially harder than the three-pole render demo,
so the documented command uses a short 80-step target and Adam-only run to show
that the generated model, LQR teacher, PyTorch optimizer update, and metrics all
work without pretending this is a production multi-link RL benchmark. The model
is still plain MuJoCo XML, so it is a natural candidate for future MuJoCo Warp
rollout work; this command trains/evaluates through the small MuJoCo learner path.

`examples/key_door_policy_optimizers.py` adds PPO, TRPO, and GRPO to the map
without pretending they are Q-learning variants. These techniques optimize a
policy from rollout data: PPO clips policy-ratio updates, TRPO constrains KL
movement with a trust region, and GRPO uses group-relative trajectory rewards.
A full trainer for any of them would be a larger follow-up than the tiny neural
Q-learning bridge.

## Choosing the bottleneck metric

Optimize the resource that is actually scarce. In this repo, the environments
are tiny in-process simulations, so wall-clock time is the default optimization
target for convergence demos.

| Bottleneck | Optimize for | Use when | Typical tradeoff |
| --- | --- | --- | --- |
| Cheap simulator, slow Python loop | Wall-clock to target reward | Toy grids, local games, fast unit-test environments | May spend more environment steps because samples are cheap |
| Expensive simulator | Environment steps / sample efficiency | Physics, traffic, CFD, or other slow simulators | More compute per sample can be worthwhile |
| Real-world robot | Real interactions and safety failures | Hardware control, drones, robot arms | Slower compute is acceptable if it avoids bad real actions |
| Human feedback or labels | Labels / preference comparisons | RLHF, reward modeling, annotation-heavy tasks | Extra model training is worthwhile to reduce human work |
| Paid external system | API calls / dollars | Tool-use agents, paid market data, web automation | Cache, replay, or model locally to avoid paid calls |
| GPU training job | GPU-hours / utilization | Large neural RL, PPO/SAC/DQN-style training | Batch/vectorize work even if the code is more complex |
| Memory-constrained deployment | RAM/VRAM footprint | Edge devices, small GPUs, huge replay buffers | Smaller models or online updates may converge slower |
| Final policy quality | Best asymptotic return | Research benchmarks or production agents | More compute and tuning can be justified |
| Inference latency | Fast action selection after training | Control loops, games, embedded agents | Training may be longer to get a cheaper deployed policy |
| Risky exploration | Number or severity of bad actions | Medical, finance, robotics, security | Conservative methods can learn slower but fail safer |

For the current KeyDoorGrid examples, that means the convergence race should
prefer a learner that reaches a stable policy fastest in wall-clock time, not
just the learner that uses the fewest environment episodes.

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
