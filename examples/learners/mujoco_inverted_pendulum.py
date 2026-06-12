"""Small MuJoCo inverted-pendulum control demo."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean, median
import time

import gymnasium as gym
import torch

try:
    from examples.mujoco_compat import (
        ensure_mujoco_human_viewer_compatibility,
        solver_iteration_count,
    )
except ModuleNotFoundError as error:
    if error.name not in {"examples", "examples.mujoco_compat"}:
        raise
    from mujoco_compat import (
        ensure_mujoco_human_viewer_compatibility,
        solver_iteration_count,
    )


MUJOCO_ENV_ID = "InvertedPendulum-v4"
MUJOCO_OPTIMIZERS = ("sgd", "adam")
DEFAULT_LEARNING_RATES = {
    "sgd": 0.10,
    "adam": 0.05,
}
DEMO_MAX_STEPS = 200
DEMO_TARGET_RETURN = 180.0
TEACHER_GAINS = (0.0, 10.0, 0.0, 1.0)
SYNTHETIC_STATE_RANGES = (0.18, 0.18, 1.5, 1.5)


class InvertedPendulumLinearPolicy(torch.nn.Module):
    """Tiny neural policy for continuous MuJoCo actions."""

    def __init__(
        self,
        *,
        observation_size: int,
        action_size: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
    ) -> None:
        super().__init__()
        if observation_size < 1:
            raise ValueError("observation_size must be at least 1")
        if action_size < 1:
            raise ValueError("action_size must be at least 1")
        if len(action_low) != action_size or len(action_high) != action_size:
            raise ValueError("action bounds must match action_size")

        self.linear = torch.nn.Linear(observation_size, action_size)
        self.register_buffer(
            "action_low",
            torch.as_tensor(action_low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(action_high, dtype=torch.float32),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.linear(observation), self.action_low, self.action_high)


@dataclass(frozen=True)
class MuJoCoEvaluation:
    epoch: int
    eval_episodes: int
    mean_return: float
    min_return: float
    best_return: float
    mean_length: float
    solved: bool


@dataclass(frozen=True)
class MuJoCoTrainingResult:
    optimizer_name: str
    learning_rate: float
    seed: int
    policy_network: InvertedPendulumLinearPolicy
    training_losses: list[float]
    evaluations: list[MuJoCoEvaluation]
    env_id: str
    dataset_samples: int
    expert_mean_return: float
    expert_min_return: float
    target_epoch: int | None
    wall_clock_seconds: float
    estimated_seconds_to_target: float | None


@dataclass(frozen=True)
class MuJoCoPlayback:
    render_mode: str
    episode_returns: list[float]
    episode_lengths: list[int]
    rendered_frames: int
    first_frame_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class MuJoCoOptimizerSummary:
    optimizer_name: str
    learning_rate: float
    target_runs: int
    final_solved_runs: int
    total_runs: int
    median_target_epoch: int | None
    median_seconds_to_target: float | None
    mean_wall_clock_seconds: float
    mean_dataset_samples: float
    mean_final_return: float
    mean_best_return: float
    mean_final_loss: float


def default_learning_rate(optimizer_name: str) -> float:
    if optimizer_name not in MUJOCO_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {MUJOCO_OPTIMIZERS}")
    return DEFAULT_LEARNING_RATES[optimizer_name]


def make_optimizer(
    optimizer_name: str,
    parameters: Iterable[torch.nn.Parameter],
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
    raise ValueError(f"optimizer_name must be one of {MUJOCO_OPTIMIZERS}")


def _validate_inverted_pendulum_env(env: gym.Env) -> tuple[int, int]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("MuJoCo demo expects a Box observation space")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("MuJoCo demo expects a Box action space")
    if len(env.observation_space.shape) != 1:
        raise ValueError("MuJoCo demo expects a flat observation vector")
    if len(env.action_space.shape) != 1:
        raise ValueError("MuJoCo demo expects a flat action vector")
    if int(env.observation_space.shape[0]) != len(TEACHER_GAINS):
        raise ValueError("teacher gains are tuned for InvertedPendulum-v4")

    return int(env.observation_space.shape[0]), int(env.action_space.shape[0])


def teacher_action(
    observation: torch.Tensor,
    *,
    action_low: Sequence[float],
    action_high: Sequence[float],
    gains: Sequence[float] = TEACHER_GAINS,
) -> torch.Tensor:
    if observation.shape[-1] != len(gains):
        raise ValueError("observation width must match teacher gains")
    low = torch.as_tensor(action_low, dtype=torch.float32)
    high = torch.as_tensor(action_high, dtype=torch.float32)
    gain_tensor = torch.as_tensor(gains, dtype=torch.float32)
    action = (observation * gain_tensor).sum(dim=-1, keepdim=True)
    return torch.clamp(action, low, high)


def make_teacher_dataset(
    *,
    samples: int = 2_048,
    seed: int = 0,
    state_ranges: Sequence[float] = SYNTHETIC_STATE_RANGES,
    action_low: Sequence[float],
    action_high: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if len(state_ranges) != len(TEACHER_GAINS):
        raise ValueError("state_ranges must match teacher gains")

    generator = torch.Generator().manual_seed(seed)
    ranges = torch.as_tensor(state_ranges, dtype=torch.float32)
    observations = (torch.rand((samples, len(state_ranges)), generator=generator) * 2 - 1) * ranges
    actions = teacher_action(
        observations,
        action_low=action_low,
        action_high=action_high,
    )
    return observations, actions


def _policy_action(
    policy_network: InvertedPendulumLinearPolicy,
    observation: object,
) -> torch.Tensor:
    with torch.no_grad():
        return policy_network(torch.as_tensor(observation, dtype=torch.float32))


def evaluate_policy(
    policy_network: InvertedPendulumLinearPolicy,
    *,
    env_id: str = MUJOCO_ENV_ID,
    epoch: int,
    seed: int,
    eval_episodes: int = 5,
    max_steps_per_episode: int = DEMO_MAX_STEPS,
    target_return: float = DEMO_TARGET_RETURN,
) -> MuJoCoEvaluation:
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least 1")
    if target_return <= 0.0:
        raise ValueError("target_return must be greater than 0.0")

    env = gym.make(env_id)
    was_training = policy_network.training
    policy_network.eval()
    try:
        _validate_inverted_pendulum_env(env)
        returns: list[float] = []
        lengths: list[int] = []
        for index in range(eval_episodes):
            observation, _ = env.reset(seed=seed + index)
            episode_return = 0.0
            episode_length = 0
            while True:
                action = _policy_action(policy_network, observation)
                observation, reward, terminated, truncated, _ = env.step(action.numpy())
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated or episode_length >= max_steps_per_episode:
                    returns.append(episode_return)
                    lengths.append(episode_length)
                    break

        mean_return = mean(returns)
        return MuJoCoEvaluation(
            epoch=epoch,
            eval_episodes=eval_episodes,
            mean_return=mean_return,
            min_return=min(returns),
            best_return=max(returns),
            mean_length=mean(lengths),
            solved=mean_return >= target_return,
        )
    finally:
        if was_training:
            policy_network.train()
        env.close()


def evaluate_teacher(
    *,
    env_id: str = MUJOCO_ENV_ID,
    seed: int = 0,
    eval_episodes: int = 5,
    max_steps_per_episode: int = DEMO_MAX_STEPS,
) -> tuple[float, float]:
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least 1")

    env = gym.make(env_id)
    try:
        _validate_inverted_pendulum_env(env)
        action_low = env.action_space.low.tolist()
        action_high = env.action_space.high.tolist()
        returns: list[float] = []
        for index in range(eval_episodes):
            observation, _ = env.reset(seed=seed + index)
            episode_return = 0.0
            episode_length = 0
            while True:
                action = teacher_action(
                    torch.as_tensor(observation, dtype=torch.float32),
                    action_low=action_low,
                    action_high=action_high,
                )
                observation, reward, terminated, truncated, _ = env.step(action.numpy())
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated or episode_length >= max_steps_per_episode:
                    returns.append(episode_return)
                    break
        return mean(returns), min(returns)
    finally:
        env.close()


def play_policy(
    policy_network: InvertedPendulumLinearPolicy,
    *,
    env_id: str = MUJOCO_ENV_ID,
    render_mode: str = "human",
    episodes: int = 1,
    seed: int = 0,
    max_steps_per_episode: int = DEMO_MAX_STEPS,
    frame_delay_seconds: float = 0.0,
) -> MuJoCoPlayback:
    if render_mode not in {"human", "rgb_array"}:
        raise ValueError("render_mode must be 'human' or 'rgb_array'")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least 1")
    if frame_delay_seconds < 0.0:
        raise ValueError("frame_delay_seconds must be at least 0.0")

    if render_mode == "human":
        ensure_mujoco_human_viewer_compatibility()

    env = gym.make(env_id, render_mode=render_mode)
    was_training = policy_network.training
    policy_network.eval()
    try:
        _validate_inverted_pendulum_env(env)
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        rendered_frames = 0
        first_frame_shape: tuple[int, ...] | None = None

        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            episode_length = 0
            frame = env.render()
            rendered_frames += 1
            if first_frame_shape is None and frame is not None:
                first_frame_shape = tuple(int(value) for value in frame.shape)

            while True:
                action = _policy_action(policy_network, observation)
                observation, reward, terminated, truncated, _ = env.step(action.numpy())
                frame = env.render()
                rendered_frames += 1
                if first_frame_shape is None and frame is not None:
                    first_frame_shape = tuple(int(value) for value in frame.shape)
                if frame_delay_seconds:
                    time.sleep(frame_delay_seconds)

                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated or episode_length >= max_steps_per_episode:
                    episode_returns.append(episode_return)
                    episode_lengths.append(episode_length)
                    break

        return MuJoCoPlayback(
            render_mode=render_mode,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            rendered_frames=rendered_frames,
            first_frame_shape=first_frame_shape,
        )
    finally:
        if was_training:
            policy_network.train()
        env.close()


def train_inverted_pendulum_policy(
    *,
    optimizer_name: str = "adam",
    epochs: int = 20,
    learning_rate: float | None = None,
    samples: int = 2_048,
    batch_size: int = 128,
    seed: int = 0,
    env_id: str = MUJOCO_ENV_ID,
    eval_every: int = 5,
    eval_episodes: int = 5,
    max_steps_per_episode: int = DEMO_MAX_STEPS,
    target_return: float = DEMO_TARGET_RETURN,
) -> MuJoCoTrainingResult:
    if optimizer_name not in MUJOCO_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {MUJOCO_OPTIMIZERS}")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if eval_every < 1:
        raise ValueError("eval_every must be at least 1")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")
    if max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least 1")
    if target_return <= 0.0:
        raise ValueError("target_return must be greater than 0.0")

    env = gym.make(env_id)
    try:
        observation_size, action_size = _validate_inverted_pendulum_env(env)
        action_low = env.action_space.low.tolist()
        action_high = env.action_space.high.tolist()
    finally:
        env.close()

    observations, target_actions = make_teacher_dataset(
        samples=samples,
        seed=seed,
        action_low=action_low,
        action_high=action_high,
    )
    torch.manual_seed(seed)
    policy_network = InvertedPendulumLinearPolicy(
        observation_size=observation_size,
        action_size=action_size,
        action_low=action_low,
        action_high=action_high,
    )
    resolved_learning_rate = (
        learning_rate if learning_rate is not None else default_learning_rate(optimizer_name)
    )
    optimizer = make_optimizer(
        optimizer_name,
        policy_network.parameters(),
        learning_rate=resolved_learning_rate,
    )
    expert_mean_return, expert_min_return = evaluate_teacher(
        env_id=env_id,
        seed=seed + 50_000,
        eval_episodes=eval_episodes,
        max_steps_per_episode=max_steps_per_episode,
    )

    generator = torch.Generator().manual_seed(seed)
    training_losses: list[float] = []
    evaluations: list[MuJoCoEvaluation] = []
    target_epoch: int | None = None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(samples, generator=generator)
        batch_losses: list[float] = []
        for offset in range(0, samples, batch_size):
            indexes = permutation[offset : offset + batch_size]
            prediction = policy_network(observations[indexes])
            loss = torch.nn.functional.mse_loss(prediction, target_actions[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))

        training_losses.append(mean(batch_losses))
        if epoch % eval_every == 0 or epoch == epochs:
            evaluation = evaluate_policy(
                policy_network,
                env_id=env_id,
                epoch=epoch,
                seed=seed + 10_000 + epoch,
                eval_episodes=eval_episodes,
                max_steps_per_episode=max_steps_per_episode,
                target_return=target_return,
            )
            evaluations.append(evaluation)
            if target_epoch is None and evaluation.solved:
                target_epoch = epoch

    wall_clock_seconds = time.perf_counter() - start
    estimated_seconds_to_target = (
        wall_clock_seconds * target_epoch / epochs if target_epoch is not None else None
    )
    return MuJoCoTrainingResult(
        optimizer_name=optimizer_name,
        learning_rate=resolved_learning_rate,
        seed=seed,
        policy_network=policy_network,
        training_losses=training_losses,
        evaluations=evaluations,
        env_id=env_id,
        dataset_samples=samples,
        expert_mean_return=expert_mean_return,
        expert_min_return=expert_min_return,
        target_epoch=target_epoch,
        wall_clock_seconds=wall_clock_seconds,
        estimated_seconds_to_target=estimated_seconds_to_target,
    )


def compare_mujoco_optimizers(
    *,
    optimizers: tuple[str, ...] = MUJOCO_OPTIMIZERS,
    seeds: tuple[int, ...] = (1, 2),
    epochs: int = 20,
    samples: int = 2_048,
    batch_size: int = 128,
    eval_every: int = 5,
    eval_episodes: int = 5,
    max_steps_per_episode: int = DEMO_MAX_STEPS,
    target_return: float = DEMO_TARGET_RETURN,
) -> tuple[list[MuJoCoOptimizerSummary], dict[str, list[MuJoCoTrainingResult]]]:
    if not optimizers:
        raise ValueError("optimizers must contain at least one optimizer")
    unknown = set(optimizers) - set(MUJOCO_OPTIMIZERS)
    if unknown:
        raise ValueError(f"unknown optimizers: {sorted(unknown)}")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    results: dict[str, list[MuJoCoTrainingResult]] = {
        optimizer_name: [] for optimizer_name in optimizers
    }
    summaries: list[MuJoCoOptimizerSummary] = []
    for optimizer_name in optimizers:
        for seed in seeds:
            results[optimizer_name].append(
                train_inverted_pendulum_policy(
                    optimizer_name=optimizer_name,
                    epochs=epochs,
                    samples=samples,
                    batch_size=batch_size,
                    seed=seed,
                    eval_every=eval_every,
                    eval_episodes=eval_episodes,
                    max_steps_per_episode=max_steps_per_episode,
                    target_return=target_return,
                )
            )

        rows = results[optimizer_name]
        target_epochs = [
            row.target_epoch for row in rows if row.target_epoch is not None
        ]
        target_seconds = [
            row.estimated_seconds_to_target
            for row in rows
            if row.estimated_seconds_to_target is not None
        ]
        final_evaluations = [row.evaluations[-1] for row in rows]
        summaries.append(
            MuJoCoOptimizerSummary(
                optimizer_name=optimizer_name,
                learning_rate=rows[0].learning_rate,
                target_runs=len(target_epochs),
                final_solved_runs=sum(1 for evaluation in final_evaluations if evaluation.solved),
                total_runs=len(rows),
                median_target_epoch=(
                    int(median(target_epochs)) if target_epochs else None
                ),
                median_seconds_to_target=(
                    median(target_seconds) if target_seconds else None
                ),
                mean_wall_clock_seconds=mean(row.wall_clock_seconds for row in rows),
                mean_dataset_samples=mean(row.dataset_samples for row in rows),
                mean_final_return=mean(
                    evaluation.mean_return for evaluation in final_evaluations
                ),
                mean_best_return=mean(
                    max(evaluation.best_return for evaluation in row.evaluations)
                    for row in rows
                ),
                mean_final_loss=mean(row.training_losses[-1] for row in rows),
            )
        )

    summaries.sort(
        key=lambda summary: (
            -summary.final_solved_runs,
            -summary.target_runs,
            summary.median_seconds_to_target is None,
            summary.median_seconds_to_target
            if summary.median_seconds_to_target is not None
            else float("inf"),
            -summary.mean_final_return,
            summary.mean_wall_clock_seconds,
        )
    )
    return summaries, results


__all__ = [
    "DEFAULT_LEARNING_RATES",
    "DEMO_MAX_STEPS",
    "DEMO_TARGET_RETURN",
    "MUJOCO_ENV_ID",
    "MUJOCO_OPTIMIZERS",
    "SYNTHETIC_STATE_RANGES",
    "TEACHER_GAINS",
    "InvertedPendulumLinearPolicy",
    "MuJoCoEvaluation",
    "MuJoCoOptimizerSummary",
    "MuJoCoPlayback",
    "MuJoCoTrainingResult",
    "compare_mujoco_optimizers",
    "default_learning_rate",
    "ensure_mujoco_human_viewer_compatibility",
    "evaluate_policy",
    "evaluate_teacher",
    "make_optimizer",
    "make_teacher_dataset",
    "play_policy",
    "solver_iteration_count",
    "teacher_action",
    "train_inverted_pendulum_policy",
]
