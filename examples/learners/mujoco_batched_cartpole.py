"""Batched MuJoCo rollout training for InvertedPendulum cartpole."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time

import gymnasium as gym
import mujoco
from mujoco import rollout
import numpy as np
import torch


BATCHED_CARTPOLE_ENV_ID = "InvertedPendulum-v4"
BATCHED_CARTPOLE_OPTIMIZERS = ("sgd", "adam")
DEFAULT_LEARNING_RATES = {"sgd": 1.0, "adam": 0.5}
DEFAULT_BATCH_SIZE = 8_192
DEFAULT_EVAL_BATCH_SIZE = 1_024
DEFAULT_MAX_STEPS = 200
DEFAULT_TARGET_RETURN = 195.0
DEFAULT_EXPLORATION_SIGMA = 3.0


@dataclass(frozen=True)
class BatchedRolloutStats:
    batch_size: int
    max_steps: int
    loop_steps: int
    physics_steps: int
    active_environment_steps: int
    wall_clock_seconds: float
    physics_steps_per_second: float
    active_steps_per_second: float
    mean_return: float
    min_return: float
    max_return: float
    solved_fraction: float


@dataclass(frozen=True)
class BatchedESGeneration:
    generation: int
    sigma: float
    train: BatchedRolloutStats
    evaluation: BatchedRolloutStats
    policy_parameters: tuple[float, float, float, float]


@dataclass(frozen=True)
class BatchedESTrainingResult:
    optimizer_name: str
    learning_rate: float
    batch_size: int
    eval_batch_size: int
    generations: int
    max_steps: int
    target_return: float
    final_policy_parameters: tuple[float, float, float, float]
    learned_generation: int | None
    total_rollout_seconds: float
    history: list[BatchedESGeneration]


def default_learning_rate(optimizer_name: str) -> float:
    if optimizer_name not in BATCHED_CARTPOLE_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {BATCHED_CARTPOLE_OPTIMIZERS}")
    return DEFAULT_LEARNING_RATES[optimizer_name]


def make_optimizer(
    optimizer_name: str,
    parameters: list[torch.nn.Parameter],
    *,
    learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    if learning_rate is None:
        learning_rate = default_learning_rate(optimizer_name)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0.0")
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError(f"optimizer_name must be one of {BATCHED_CARTPOLE_OPTIMIZERS}")


def make_inverted_pendulum_model() -> mujoco.MjModel:
    env = gym.make(BATCHED_CARTPOLE_ENV_ID)
    try:
        return env.unwrapped.model
    finally:
        env.close()


def make_rollout_data(model: mujoco.MjModel, *, data_threads: int | None = None) -> list[mujoco.MjData]:
    if data_threads is None:
        data_threads = min(os.cpu_count() or 1, 8)
    if data_threads < 1:
        raise ValueError("data_threads must be at least 1")
    return [mujoco.MjData(model) for _ in range(data_threads)]


def sample_initial_states(
    model: mujoco.MjModel,
    *,
    batch_size: int,
    seed: int,
    noise: float = 0.01,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if noise < 0.0:
        raise ValueError("noise must be at least 0.0")
    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states = np.zeros((batch_size, state_size), dtype=np.float64)
    rng = np.random.default_rng(seed)
    states[:, 1:3] = rng.uniform(-noise, noise, size=(batch_size, 2))
    states[:, 3:5] = rng.uniform(-noise, noise, size=(batch_size, 2))
    return states


def _normalize_policy_parameters(
    policy_parameters: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    parameters = np.asarray(policy_parameters, dtype=np.float64)
    if parameters.shape == (4,):
        return np.repeat(parameters[None, :], batch_size, axis=0)
    if parameters.shape == (1, 4):
        return np.repeat(parameters, batch_size, axis=0)
    if parameters.shape == (batch_size, 4):
        return parameters
    raise ValueError("policy_parameters must have shape (4,), (1, 4), or (batch_size, 4)")


def rollout_linear_policies(
    model: mujoco.MjModel,
    data: list[mujoco.MjData],
    policy_parameters: np.ndarray,
    *,
    batch_size: int,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 0,
    target_return: float = DEFAULT_TARGET_RETURN,
    initial_noise: float = 0.01,
) -> tuple[BatchedRolloutStats, np.ndarray]:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if target_return <= 0.0:
        raise ValueError("target_return must be greater than 0.0")
    parameters = _normalize_policy_parameters(policy_parameters, batch_size=batch_size)
    states = sample_initial_states(
        model,
        batch_size=batch_size,
        seed=seed,
        noise=initial_noise,
    )
    active = np.ones(batch_size, dtype=bool)
    returns = np.zeros(batch_size, dtype=np.float32)
    controls = np.zeros((batch_size, 1, model.nu), dtype=np.float64)
    next_states = np.empty((batch_size, 1, states.shape[1]), dtype=np.float64)
    active_environment_steps = 0
    loop_steps = 0
    start = time.perf_counter()

    for _ in range(max_steps):
        observations = states[:, 1:5]
        controls[:, 0, 0] = np.clip(np.sum(parameters * observations, axis=1), -3.0, 3.0)
        rollout.rollout(
            model,
            data,
            states,
            controls,
            control_spec=mujoco.mjtState.mjSTATE_CTRL,
            nstep=1,
            state=next_states,
            persistent_pool=True,
        )
        returns[active] += 1.0
        active_environment_steps += int(active.sum())
        loop_steps += 1
        states = next_states[:, 0, :]
        healthy = np.isfinite(states[:, 1:5]).all(axis=1) & (np.abs(states[:, 2]) <= 0.2)
        active &= healthy
        if not active.any():
            break

    wall_clock_seconds = time.perf_counter() - start
    physics_steps = batch_size * loop_steps
    return (
        BatchedRolloutStats(
            batch_size=batch_size,
            max_steps=max_steps,
            loop_steps=loop_steps,
            physics_steps=physics_steps,
            active_environment_steps=active_environment_steps,
            wall_clock_seconds=wall_clock_seconds,
            physics_steps_per_second=physics_steps / max(wall_clock_seconds, 1e-12),
            active_steps_per_second=active_environment_steps / max(wall_clock_seconds, 1e-12),
            mean_return=float(returns.mean()),
            min_return=float(returns.min()),
            max_return=float(returns.max()),
            solved_fraction=float(np.mean(returns >= target_return)),
        ),
        returns,
    )


def train_batched_es_cartpole(
    *,
    optimizer_name: str = "adam",
    learning_rate: float | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    generations: int = 3,
    max_steps: int = DEFAULT_MAX_STEPS,
    target_return: float = DEFAULT_TARGET_RETURN,
    exploration_sigma: float = DEFAULT_EXPLORATION_SIGMA,
    sigma_decay: float = 0.95,
    seed: int = 0,
    data_threads: int | None = None,
) -> BatchedESTrainingResult:
    if optimizer_name not in BATCHED_CARTPOLE_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {BATCHED_CARTPOLE_OPTIMIZERS}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be at least 1")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if target_return <= 0.0:
        raise ValueError("target_return must be greater than 0.0")
    if exploration_sigma <= 0.0:
        raise ValueError("exploration_sigma must be greater than 0.0")
    if not 0.0 < sigma_decay <= 1.0:
        raise ValueError("sigma_decay must be in (0.0, 1.0]")

    model = make_inverted_pendulum_model()
    data = make_rollout_data(model, data_threads=data_threads)
    torch.manual_seed(seed)
    policy_mean = torch.nn.Parameter(torch.zeros(4))
    resolved_learning_rate = (
        learning_rate if learning_rate is not None else default_learning_rate(optimizer_name)
    )
    optimizer = make_optimizer(
        optimizer_name,
        [policy_mean],
        learning_rate=resolved_learning_rate,
    )
    sigma = exploration_sigma
    history: list[BatchedESGeneration] = []
    learned_generation: int | None = None
    total_rollout_seconds = 0.0

    for generation in range(1, generations + 1):
        noise = torch.randn(batch_size, 4)
        candidates = (policy_mean.detach() + sigma * noise).numpy()
        train_stats, train_returns = rollout_linear_policies(
            model,
            data,
            candidates,
            batch_size=batch_size,
            max_steps=max_steps,
            seed=seed + generation,
            target_return=target_return,
        )
        returns = torch.as_tensor(train_returns, dtype=torch.float32)
        advantages = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        surrogate_loss = -(
            advantages * torch.sum(policy_mean * noise, dim=1) / sigma
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        surrogate_loss.backward()
        optimizer.step()
        sigma *= sigma_decay

        evaluation_stats, _ = rollout_linear_policies(
            model,
            data,
            policy_mean.detach().numpy(),
            batch_size=eval_batch_size,
            max_steps=max_steps,
            seed=seed + 10_000 + generation,
            target_return=target_return,
        )
        total_rollout_seconds += (
            train_stats.wall_clock_seconds + evaluation_stats.wall_clock_seconds
        )
        if learned_generation is None and evaluation_stats.mean_return >= target_return:
            learned_generation = generation
        history.append(
            BatchedESGeneration(
                generation=generation,
                sigma=sigma,
                train=train_stats,
                evaluation=evaluation_stats,
                policy_parameters=tuple(float(value) for value in policy_mean.detach().tolist()),
            )
        )
        if learned_generation is not None:
            break

    return BatchedESTrainingResult(
        optimizer_name=optimizer_name,
        learning_rate=resolved_learning_rate,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        generations=len(history),
        max_steps=max_steps,
        target_return=target_return,
        final_policy_parameters=tuple(float(value) for value in policy_mean.detach().tolist()),
        learned_generation=learned_generation,
        total_rollout_seconds=total_rollout_seconds,
        history=history,
    )


__all__ = [
    "BATCHED_CARTPOLE_ENV_ID",
    "BATCHED_CARTPOLE_OPTIMIZERS",
    "BatchedESGeneration",
    "BatchedESTrainingResult",
    "BatchedRolloutStats",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EVAL_BATCH_SIZE",
    "DEFAULT_EXPLORATION_SIGMA",
    "DEFAULT_LEARNING_RATES",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_TARGET_RETURN",
    "default_learning_rate",
    "make_inverted_pendulum_model",
    "make_optimizer",
    "make_rollout_data",
    "rollout_linear_policies",
    "sample_initial_states",
    "train_batched_es_cartpole",
]
