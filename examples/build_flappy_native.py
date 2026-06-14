"""Build a PufferLib 4 `_C` backend for the native Flappy Bird C env."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLAPPY_DIR = ROOT / "examples" / "flappy_bird"
NATIVE_DIR = FLAPPY_DIR / "native"
BUILD_DIR = FLAPPY_DIR / "build" / "native_runtime"
COMPAT_DIR = NATIVE_DIR / "pufferlib_compat"


DEFAULT_CONFIG = """[base]
env_name = None
rank = 0
world_size = 1
gpu_id = 0
nccl_id = 'None'
profile = False
checkpoint_dir = experiments/native_checkpoints
log_dir = experiments/native_logs
checkpoint_interval = 25
eval_episodes = 32
cudagraphs = -1
seed = 73
reset_state = True

[vec]
total_agents = 32
num_buffers = 1
num_threads = 1

[selfplay]
enabled = 0
max_size = 16
swap_winrate = 0.8
min_games = 2048
elo_init = 0.0
elo_k = 16.0
seed = 42
snapshot_interval = 1_000_000_000
opp_timeout_steps = 500_000_000

[env]

[policy]
hidden_size = 128
num_layers = 1
expansion_factor = 1

[torch]
network = MLP
encoder = DefaultEncoder
decoder = DefaultDecoder

[train]
gpus = 1
seed = 42
total_timesteps = 200_000
learning_rate = 0.001
optimizer = Adam
initial_flap_logit_bias = -2.0
anneal_lr = 1
min_lr_ratio = 0.0
gamma = 0.99
gae_lambda = 0.95
replay_ratio = 1.0
clip_coef = 0.2
vf_coef = 0.5
vf_clip_coef = 0.2
max_grad_norm = 0.5
ent_coef = 0.001
anneal_ent_coef = 0
min_ent_coef_ratio = 0.1
beta1 = 0.95
beta2 = 0.999
eps = 1e-12
minibatch_size = 2048
horizon = 64
vtrace_rho_clip = 1.0
vtrace_c_clip = 1.0
prio_alpha = 0.8
prio_beta0 = 0.2

[sweep]
method = Protein
metric = score
metric_distribution = linear
goal = maximize
max_suggestion_cost = 3600
max_runs = 1200
gpus = 0
downsample = 5
use_gpu = False
prune_pareto = True
early_stop_quantile = 0.3
match_enemy_model_path = ''
match_num_games = 1024
match_enemy_hidden_size = 0
match_enemy_num_layers = 0
"""


OPTIMIZER_PATCH_MARKER = "# flappy-native-optimizer-patch"
ACTION_BIAS_PATCH_MARKER = "# flappy-native-action-bias-patch"


def pufferlib_package_dir() -> Path:
    spec = importlib.util.find_spec("pufferlib")
    if spec is None or spec.origin is None:
        raise RuntimeError("pufferlib is not importable; run `uv sync` first")
    return Path(spec.origin).resolve().parent


def patch_torch_backend(runtime_pkg: Path) -> None:
    torch_backend = runtime_pkg / "torch_pufferl.py"
    text = torch_backend.read_text(encoding="utf-8")

    if OPTIMIZER_PATCH_MARKER not in text:
        old_optimizer = """        self.optimizer = Muon(
            self.policy.parameters(),
            lr=config['learning_rate'],
            momentum=config['beta1'],
            eps=config['eps'],
        )
"""
        new_optimizer = f"""        {OPTIMIZER_PATCH_MARKER}
        optimizer_name = str(config.get('optimizer', 'Muon')).lower()
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config['learning_rate'],
                eps=config.get('eps', 1e-8),
            )
        elif optimizer_name == 'muon':
            self.optimizer = Muon(
                self.policy.parameters(),
                lr=config['learning_rate'],
                momentum=config['beta1'],
                eps=config['eps'],
            )
        else:
            raise ValueError(f"Unsupported optimizer {{config.get('optimizer')!r}}")
"""
        if old_optimizer not in text:
            raise RuntimeError("Could not patch PufferLib optimizer selection")
        text = text.replace(old_optimizer, new_optimizer)

    if ACTION_BIAS_PATCH_MARKER not in text:
        old_return = """    return policy
"""
        new_return = f"""    {ACTION_BIAS_PATCH_MARKER}
    flap_logit_bias = args.get('train', {{}}).get('initial_flap_logit_bias')
    if flap_logit_bias is not None:
        decoder = getattr(policy.decoder, 'decoder', None)
        if decoder is None or decoder.bias is None or decoder.bias.numel() < 2:
            raise ValueError('initial_flap_logit_bias requires a discrete two-action decoder')
        with torch.no_grad():
            decoder.bias[0] = 0.0
            decoder.bias[1] = float(flap_logit_bias)

    return policy
"""
        if old_return not in text:
            raise RuntimeError("Could not patch PufferLib initial action bias")
        text = text.replace(old_return, new_return, 1)

    torch_backend.write_text(text, encoding="utf-8")


def copy_pufferlib_runtime(force: bool) -> Path:
    package_src = pufferlib_package_dir()
    runtime_pkg = BUILD_DIR / "pufferlib"
    if force and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    if not runtime_pkg.exists():
        shutil.copytree(
            package_src,
            runtime_pkg,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_C*.so"),
        )

    config_dir = BUILD_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.ini").write_text(DEFAULT_CONFIG, encoding="utf-8")
    shutil.copy2(NATIVE_DIR / "flappy_bird.ini", config_dir / "flappy_bird.ini")
    patch_torch_backend(runtime_pkg)
    return runtime_pkg


def compile_backend(runtime_pkg: Path) -> Path:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output = runtime_pkg / f"_C{ext_suffix}"
    py_include = sysconfig.get_path("include")
    pybind_include = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"],
        text=True,
    ).strip().split()

    build_objects = FLAPPY_DIR / "build" / "native_objects"
    build_objects.mkdir(parents=True, exist_ok=True)
    env_obj = build_objects / "flappy_env.o"
    binding_obj = build_objects / "binding.o"

    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-fPIC",
            "-I",
            str(FLAPPY_DIR),
            "-c",
            str(FLAPPY_DIR / "flappy_env.c"),
            "-o",
            str(env_obj),
        ],
        check=True,
    )

    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-O2",
            "-fPIC",
            "-I",
            str(FLAPPY_DIR),
            "-I",
            str(COMPAT_DIR),
            "-c",
            str(NATIVE_DIR / "binding.c"),
            "-o",
            str(binding_obj),
        ],
        check=True,
    )

    command = [
        "c++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        str(COMPAT_DIR / "bindings_cpu.cpp"),
        str(env_obj),
        str(binding_obj),
        "-I",
        str(COMPAT_DIR),
        "-I",
        str(py_include),
        *pybind_include,
        "-DENV_NAME=flappy_bird",
        "-o",
        str(output),
    ]
    if platform.system() == "Darwin":
        command.extend(["-undefined", "dynamic_lookup"])
    subprocess.run(command, check=True)
    return output


def build(force: bool = False) -> Path:
    runtime_pkg = copy_pufferlib_runtime(force=force)
    return compile_backend(runtime_pkg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recreate the copied runtime first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build(force=args.force)
    print(f"Built native backend: {output}")
    print(f"Use with PYTHONPATH={BUILD_DIR}")


if __name__ == "__main__":
    main()
