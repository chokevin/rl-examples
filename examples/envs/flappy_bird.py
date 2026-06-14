"""Gymnasium adapter for the C Flappy Bird environment."""

from __future__ import annotations

import ctypes
import platform
import subprocess
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np


OBSERVATION_SIZE = 4
MAX_PIPES = 4

ROOT = Path(__file__).resolve().parents[2]
FLAPPY_DIR = ROOT / "examples" / "flappy_bird"
LIB_NAME = "libflappy_env.dylib" if platform.system() == "Darwin" else "libflappy_env.so"
LIB_PATH = FLAPPY_DIR / "build" / LIB_NAME


class FlappyEnvConfig(ctypes.Structure):
    _fields_ = [
        ("screen_width", ctypes.c_int),
        ("screen_height", ctypes.c_int),
        ("gravity", ctypes.c_float),
        ("flap_velocity", ctypes.c_float),
        ("bird_x", ctypes.c_float),
        ("bird_radius", ctypes.c_float),
        ("pipe_width", ctypes.c_float),
        ("pipe_gap", ctypes.c_float),
        ("pipe_speed", ctypes.c_float),
        ("pipe_spacing", ctypes.c_float),
        ("pipe_count", ctypes.c_int),
        ("alive_reward", ctypes.c_float),
        ("pass_reward", ctypes.c_float),
        ("crash_reward", ctypes.c_float),
    ]


class FlappyPipe(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("gap_y", ctypes.c_float),
        ("scored", ctypes.c_bool),
    ]


class FlappyEnvState(ctypes.Structure):
    _fields_ = [
        ("config", FlappyEnvConfig),
        ("bird_y", ctypes.c_float),
        ("bird_velocity", ctypes.c_float),
        ("pipes", FlappyPipe * MAX_PIPES),
        ("rng_state", ctypes.c_uint32),
        ("score", ctypes.c_int),
        ("frame", ctypes.c_int),
        ("done", ctypes.c_bool),
    ]


class FlappyStepResult(ctypes.Structure):
    _fields_ = [
        ("observation", ctypes.c_float * OBSERVATION_SIZE),
        ("reward", ctypes.c_float),
        ("terminated", ctypes.c_bool),
        ("score", ctypes.c_int),
        ("frame", ctypes.c_int),
    ]


def ensure_flappy_library() -> Path:
    """Build the C core as a shared library when the wrapper needs it."""
    if not LIB_PATH.exists():
        subprocess.run(["make", "-C", str(FLAPPY_DIR), "shared"], check=True)
    return LIB_PATH


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(ensure_flappy_library()))
    lib.flappy_env_default_config.argtypes = []
    lib.flappy_env_default_config.restype = FlappyEnvConfig
    lib.flappy_env_reset.argtypes = [
        ctypes.POINTER(FlappyEnvState),
        FlappyEnvConfig,
        ctypes.c_uint32,
    ]
    lib.flappy_env_reset.restype = None
    lib.flappy_env_step.argtypes = [ctypes.POINTER(FlappyEnvState), ctypes.c_int]
    lib.flappy_env_step.restype = FlappyStepResult
    lib.flappy_env_get_observation.argtypes = [
        ctypes.POINTER(FlappyEnvState),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.flappy_env_get_observation.restype = None
    return lib


_LIB: ctypes.CDLL | None = None


def _library() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_library()
    return _LIB


def default_config() -> FlappyEnvConfig:
    return _library().flappy_env_default_config()


class CFlappyBirdEnv(gymnasium.Env):
    """Single-agent Flappy Bird task backed by the C environment core.

    Observation: [bird_y, bird_velocity, next_pipe_distance, next_gap_delta],
    normalized by screen dimensions.

    Actions: 0 no-op, 1 flap.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 1800,
        seed: int | None = None,
        config: FlappyEnvConfig | None = None,
        pipe_gap: float | None = None,
        reward_shaping: bool = False,
        centering_reward: float = 0.02,
    ) -> None:
        """Create the Python/PufferLib-facing wrapper around the C env.

        The default config preserves the basic raylib game. The optional flags
        below are training knobs used by PPO so early random policies get useful
        signal instead of crashing before the first pipe.

        Args:
            max_steps: Python-side episode cap for PPO rollouts. The C env only
                terminates on crashes, so this prevents endless successful runs.
            seed: Initial C RNG seed for deterministic pipe gaps.
            config: Optional `FlappyEnvConfig` C struct. It contains the actual
                game constants passed into `flappy_env_reset`: screen size,
                gravity, flap velocity, bird size/position, pipe size/spacing,
                and reward values. Most callers leave this as None and use the
                C defaults.
            pipe_gap: Curriculum override for gap size; wider gaps are easier.
            reward_shaping: Add a small non-terminal reward for staying near the
                next pipe gap center. This is PPO-only scaffolding.
            centering_reward: Scale for the optional gap-centering reward.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.lib = _library()
        self.config = config if config is not None else self.lib.flappy_env_default_config()
        if pipe_gap is not None:
            self.config.pipe_gap = pipe_gap
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.centering_reward = centering_reward
        self.state = FlappyEnvState()
        self.episode_return = 0.0
        self.episode_length = 0
        self.render_mode = None

        # Gymnasium/PufferLib need explicit bounds for every observation value.
        # The C env returns:
        #   0: bird_y / screen_height
        #   1: bird_velocity / screen_height
        #   2: distance from bird to next pipe's right edge / screen_width
        #   3: vertical delta from bird to next gap center / screen_height
        # The ranges are intentionally a little wider than normal gameplay so
        # observations remain valid during crashes and curriculum tweaks.
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([-0.5, -0.2, -0.5, -1.0], dtype=np.float32),
            high=np.array([1.5, 0.2, 3.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Two legal actions match the C enum: 0 = no-op, 1 = flap.
        self.action_space = gymnasium.spaces.Discrete(2)
        # PufferLib policies read the per-agent spaces through these names when
        # they are given a single env instead of a vector env.
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        # Initialize the C state immediately so the wrapper is ready for direct
        # Gymnasium use and for PufferLib's driver env inspection.
        self.reset(seed=seed)

    def _observation(self) -> np.ndarray:
        out = (ctypes.c_float * OBSERVATION_SIZE)()
        self.lib.flappy_env_get_observation(ctypes.byref(self.state), out)
        return np.frombuffer(out, dtype=np.float32).copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        super().reset(seed=seed, options=options)
        # Gymnasium lets reset() omit a seed. In that case, derive one from
        # Gymnasium's RNG and pass it into the C RNG so pipe gaps are still
        # controlled by Gymnasium's reproducibility path.
        if seed is None:
            seed = int(self.np_random.integers(1, np.iinfo(np.uint32).max))
        self.lib.flappy_env_reset(ctypes.byref(self.state), self.config, ctypes.c_uint32(seed))
        self.episode_return = 0.0
        self.episode_length = 0
        return self._observation(), {"score": self.state.score}

    def step(
        self,
        action: int | np.integer[Any] | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float | int]]:
        """Advance the game by one frame for the agent.

        In plain terms: the policy chooses either "do nothing" or "flap"; the C
        env applies gravity, moves the bird and pipes, checks for a crash or
        passed pipe, and gives back what the agent sees next plus the reward.
        Gymnasium returns two done flags: terminated means the bird crashed,
        while truncated means Python stopped the episode at max_steps.
        """
        action_id = int(np.asarray(action).item())
        if action_id not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action_id}")

        result = self.lib.flappy_env_step(ctypes.byref(self.state), action_id)
        observation = np.frombuffer(result.observation, dtype=np.float32).copy()
        raw_reward = float(result.reward)
        reward = raw_reward
        if self.reward_shaping and not result.terminated:
            centered = max(0.0, 1.0 - abs(float(observation[3])) / 0.35)
            reward += self.centering_reward * centered
        self.episode_return += reward
        self.episode_length += 1

        terminated = bool(result.terminated)
        truncated = self.episode_length >= self.max_steps and not terminated
        info: dict[str, float | int] = {
            "score": int(result.score),
            "frame": int(result.frame),
            "raw_reward": raw_reward,
        }
        if terminated or truncated:
            info["episode_return"] = self.episode_return
            info["episode_length"] = self.episode_length

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        return None


def make_env(max_steps: int = 1800) -> CFlappyBirdEnv:
    return CFlappyBirdEnv(max_steps=max_steps)
