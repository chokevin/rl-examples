from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest

from examples import build_flappy_native, train_flappy_native


def native_pythonpath(env: dict[str, str]) -> dict[str, str]:
    env = env.copy()
    env["PYTHONPATH"] = (
        str(build_flappy_native.BUILD_DIR)
        if not env.get("PYTHONPATH")
        else f"{build_flappy_native.BUILD_DIR}{os.pathsep}{env['PYTHONPATH']}"
    )
    return env


class FlappyNativeBackendTest(unittest.TestCase):
    def test_default_native_schedule_runs_many_updates(self) -> None:
        self.assertEqual(
            train_flappy_native.estimate_train_updates(
                total_timesteps=200_000,
                total_agents=32,
                horizon=64,
            ),
            97,
        )
        self.assertEqual(
            train_flappy_native.estimate_train_updates(
                total_timesteps=200_000,
                total_agents=1024,
                horizon=64,
            ),
            3,
        )

    def test_native_module_metadata_matches_flappy_binding(self) -> None:
        build_flappy_native.build()
        script = """
from pufferlib import _C

assert _C.env_name == 'flappy_bird'
assert _C.gpu == 0
vec = _C.create_vec({
    'vec': {'total_agents': 1, 'num_buffers': 1},
    'env': {
        'screen_width': 800,
        'screen_height': 450,
        'gravity': 0.35,
        'flap_velocity': -6.5,
        'bird_x': 160.0,
        'bird_radius': 14.0,
        'pipe_width': 70.0,
        'pipe_gap': 220.0,
        'pipe_speed': 2.6,
        'pipe_spacing': 260.0,
        'pipe_count': 3,
        'alive_reward': 0.01,
        'pass_reward': 1.0,
        'crash_reward': -1.0,
        'max_steps': 1800,
        'reward_shaping': 1,
        'centering_reward': 0.05,
        'seed': 1,
    },
}, 0)
assert vec.obs_size == 4
assert vec.num_atns == 1
assert vec.act_sizes == [2]
assert vec.obs_dtype == 'FloatTensor'
vec.close()
"""
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            check=True,
            env=native_pythonpath(os.environ.copy()),
        )

    def test_scripted_policy_can_log_nonzero_native_score(self) -> None:
        build_flappy_native.build()
        script = r"""
import ctypes

from pufferlib import _C

args = {
    'vec': {'total_agents': 1, 'num_buffers': 1},
    'env': {
        'screen_width': 800,
        'screen_height': 450,
        'gravity': 0.35,
        'flap_velocity': -6.5,
        'bird_x': 160.0,
        'bird_radius': 14.0,
        'pipe_width': 70.0,
        'pipe_gap': 220.0,
        'pipe_speed': 2.6,
        'pipe_spacing': 260.0,
        'pipe_count': 3,
        'alive_reward': 0.01,
        'pass_reward': 1.0,
        'crash_reward': -1.0,
        'max_steps': 1800,
        'reward_shaping': 1,
        'centering_reward': 0.05,
        'seed': 1,
    },
}
vec = _C.create_vec(args, 0)
obs = (ctypes.c_float * vec.obs_size).from_address(vec.obs_ptr)
actions = (ctypes.c_float * vec.num_atns)()
max_score = 0.0
for _ in range(2500):
    actions[0] = 1.0 if obs[3] < -0.04 and obs[1] > -0.005 else 0.0
    vec.cpu_step(ctypes.addressof(actions))
    log = vec.log()
    if log:
        max_score = max(max_score, float(log.get('score', 0.0)))
vec.close()
if max_score <= 0.0:
    raise SystemExit(f'expected nonzero score, got {max_score}')
"""
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            check=True,
            env=native_pythonpath(os.environ.copy()),
        )


if __name__ == "__main__":
    unittest.main()
