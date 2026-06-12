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
`cartpole_neural_optimizers.py` moves from repo-owned grid worlds to Gymnasium's
standard `CartPole-v1` control task. It trains a tiny policy network with a
REINFORCE loss, compares SGD with Adam, and reports wall-clock, environment-step,
episode-window, and evaluation-return metrics.
`mujoco_inverted_pendulum_optimizers.py` wires in Gymnasium MuJoCo with
`InvertedPendulum-v4`. It uses real MuJoCo physics and continuous Box actions,
then trains a tiny linear PyTorch policy to imitate a stabilizing controller so
the demo reliably shows SGD and Adam producing a working controller without a
long MuJoCo RL run. Add `--render` to open Gymnasium's MuJoCo viewer for the
best trained policy after the comparison; the default command stays headless for
tests and non-GUI terminals.
`mujoco_batched_cartpole_rollout.py` follows the large-rollout-batch CartPole
pattern: it samples thousands of linear policies, advances MuJoCo's
`InvertedPendulum-v4` through the C rollout API in large batches, and uses a
PyTorch evolution-strategy gradient estimate to update the policy mean. The
default batch is 8192 agents to make throughput and sample parallelism visible.
`mujoco_warp_cartpole_rollout.py` repeats that teaching shape on optional MuJoCo
Warp (`uv run --extra warp`). It creates one `mujoco_warp.Data` with many parallel
worlds and steps it with `mujoco_warp.step`; use a CUDA device for the
interesting throughput path, because CPU Warp is mainly a development backend.
`mujoco_warp_cudagraph_benchmark.py` is the speed-focused substrate check: it
captures an unrolled `mujoco_warp.step` task as a CUDA graph and replays that
graph without a Python per-step loop. It is not a full learning loop yet; it is
the benchmark path to use when chasing APIC/CUDA-graph-style throughput.
`--captured-controller` adds a tiny linear stabilizing controller inside the
captured graph so controls are updated on-device before each physics step; this
keeps joint-limit physics enabled while avoiding uncontrolled drift into the
limits. The H200 profiling command can still use `--disable-contact` because
the Gymnasium `InvertedPendulum-v4` model has no active contacts.
Profiling runs can pass `--disable-contact` and, for verified inactive
short-horizon constraints, `--disable-joint-limits` to separate useful dynamics
work from constraint solver overhead; the joint-limit flag is not a general RL
default.
`mujoco_triple_cart_pole_optimizers.py` adds a custom MuJoCo cart with three
stacked pole links. It computes an LQR teacher from the MuJoCo dynamics
linearized around the upright pose, then trains the same tiny linear policy with
SGD or Adam so the rendered demo shows one cart balancing three poles.
`mujoco_multi_pendulum_optimizers.py` generalizes that same generated MuJoCo
model to N serial pole links. The documented command uses five poles and a short
80-step target so the harder multi-pendulum behavior is visible without turning
the example into a long benchmark; `mujoco_triple_cart_pole_optimizers.py` also
accepts `--poles` for compatibility.
`key_door_policy_optimizers.py` adds PPO, TRPO, and GRPO to the teaching map as
policy-optimization techniques: they update policy networks from rollout data
rather than backing up Q-values.

See `examples/README.md` for the current PufferLib pin, local validation, and
upstream example workflow.
