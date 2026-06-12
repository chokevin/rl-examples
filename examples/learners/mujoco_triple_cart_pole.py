"""Optimizer demo for custom MuJoCo stacked cart-pole systems."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from statistics import mean, median
import time

import mujoco
import numpy as np
import torch

try:
    from examples.envs import DEFAULT_POLE_COUNT, MultiCartPoleEnv, TripleCartPoleEnv
except ModuleNotFoundError as error:
    if error.name not in {"examples", "examples.envs"}:
        raise
    from envs import DEFAULT_POLE_COUNT, MultiCartPoleEnv, TripleCartPoleEnv


TRIPLE_POLE_OPTIMIZERS = ("sgd", "adam")
MULTI_POLE_OPTIMIZERS = TRIPLE_POLE_OPTIMIZERS
DEFAULT_LEARNING_RATES = {"sgd": 0.10, "adam": 0.005}
DEMO_MAX_STEPS = 300
DEMO_TARGET_RETURN = 180.0
DEMO_DATASET_SAMPLES = 4_096


def validate_pole_count(pole_count: int) -> None:
    if pole_count < 1:
        raise ValueError("pole_count must be at least 1")


def state_size_for_poles(pole_count: int) -> int:
    validate_pole_count(pole_count)
    return 2 * (pole_count + 1)


def pole_count_from_state_size(state_size: int) -> int:
    if state_size < 4 or state_size % 2 != 0:
        raise ValueError("state size must be an even value of at least 4")
    return state_size // 2 - 1


def default_state_ranges(pole_count: int = DEFAULT_POLE_COUNT) -> tuple[float, ...]:
    validate_pole_count(pole_count)
    return (0.01, *([0.0015] * pole_count), 0.01, *([0.003] * pole_count))


def default_lqr_q_weights(pole_count: int = DEFAULT_POLE_COUNT) -> tuple[float, ...]:
    validate_pole_count(pole_count)
    if pole_count == 3:
        return (0.1, 50.0, 80.0, 100.0, 0.01, 1.0, 1.0, 1.0)
    angle_weights = tuple(float(value) for value in np.linspace(50.0, 120.0, pole_count))
    return (0.1, *angle_weights, 0.01, *([1.0] * pole_count))


DEMO_STATE_RANGES = default_state_ranges(DEFAULT_POLE_COUNT)


class TriplePoleLinearPolicy(torch.nn.Module):
    """Tiny linear neural policy mapping stacked-pole state to cart force."""

    def __init__(
        self,
        *,
        pole_count: int = DEFAULT_POLE_COUNT,
        observation_size: int | None = None,
        action_size: int = 1,
    ) -> None:
        super().__init__()
        validate_pole_count(pole_count)
        expected_observation_size = state_size_for_poles(pole_count)
        if observation_size is None:
            observation_size = expected_observation_size
        if observation_size < 1:
            raise ValueError("observation_size must be at least 1")
        if observation_size != expected_observation_size:
            raise ValueError("observation_size must match pole_count")
        if action_size != 1:
            raise ValueError("action_size must be 1 for MultiCartPoleEnv")
        self.pole_count = pole_count
        self.observation_size = observation_size
        self.linear = torch.nn.Linear(observation_size, action_size)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def raw_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.linear(observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.raw_action(observation), -1.0, 1.0)


MultiPoleLinearPolicy = TriplePoleLinearPolicy


@dataclass(frozen=True)
class TriplePoleEvaluation:
    epoch: int
    eval_episodes: int
    mean_return: float
    min_return: float
    best_return: float
    mean_length: float
    solved: bool


@dataclass(frozen=True)
class TriplePoleTrainingResult:
    optimizer_name: str
    learning_rate: float
    seed: int
    pole_count: int
    policy_network: TriplePoleLinearPolicy
    teacher_gain: np.ndarray
    training_losses: list[float]
    evaluations: list[TriplePoleEvaluation]
    dataset_samples: int
    teacher_mean_return: float
    teacher_min_return: float
    target_epoch: int | None
    wall_clock_seconds: float
    estimated_seconds_to_target: float | None


@dataclass(frozen=True)
class TriplePoleOptimizerSummary:
    optimizer_name: str
    learning_rate: float
    pole_count: int
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


@dataclass(frozen=True)
class TriplePolePlayback:
    render_mode: str
    pole_count: int
    episode_returns: list[float]
    episode_lengths: list[int]
    rendered_frames: int
    first_frame_shape: tuple[int, ...] | None


def default_learning_rate(optimizer_name: str) -> float:
    if optimizer_name not in TRIPLE_POLE_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {TRIPLE_POLE_OPTIMIZERS}")
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
    raise ValueError(f"optimizer_name must be one of {TRIPLE_POLE_OPTIMIZERS}")


def _make_env(
    *,
    pole_count: int,
    max_steps: int,
    render_mode: str | None = None,
) -> MultiCartPoleEnv:
    if pole_count == DEFAULT_POLE_COUNT:
        return TripleCartPoleEnv(max_steps=max_steps, render_mode=render_mode)
    return MultiCartPoleEnv(
        pole_count=pole_count,
        max_steps=max_steps,
        render_mode=render_mode,
    )


def _next_state(env: MultiCartPoleEnv, state: np.ndarray, action: float) -> np.ndarray:
    env.data.qpos[:] = state[: env.model.nq]
    env.data.qvel[:] = state[env.model.nq :]
    env.data.ctrl[0] = action
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_step(env.model, env.data)
    return np.concatenate([env.data.qpos.copy(), env.data.qvel.copy()])


def linearize_dynamics(env: MultiCartPoleEnv, *, epsilon: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than 0.0")
    state_size = env.model.nq + env.model.nv
    zero = np.zeros(state_size)
    a_matrix = np.zeros((state_size, state_size))
    b_matrix = np.zeros((state_size, 1))
    for index in range(state_size):
        delta = np.zeros(state_size)
        delta[index] = epsilon
        a_matrix[:, index] = (
            _next_state(env, zero + delta, 0.0)
            - _next_state(env, zero - delta, 0.0)
        ) / (2 * epsilon)
    b_matrix[:, 0] = (
        _next_state(env, zero, epsilon) - _next_state(env, zero, -epsilon)
    ) / (2 * epsilon)
    return a_matrix, b_matrix


def solve_discrete_lqr(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    *,
    q_weights: Sequence[float] | None = None,
    r_weight: float = 0.1,
    max_iterations: int = 2_000,
    tolerance: float = 1e-7,
) -> np.ndarray:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be greater than 0.0")
    if r_weight <= 0.0:
        raise ValueError("r_weight must be greater than 0.0")
    if q_weights is None:
        q_weights = default_lqr_q_weights(pole_count_from_state_size(a_matrix.shape[0]))
    if len(q_weights) != a_matrix.shape[0]:
        raise ValueError("q_weights must match state size")

    q_matrix = np.diag(q_weights)
    r_matrix = np.array([[r_weight]])
    p_matrix = q_matrix.copy()
    for _ in range(max_iterations):
        gain = np.linalg.solve(
            r_matrix + b_matrix.T @ p_matrix @ b_matrix,
            b_matrix.T @ p_matrix @ a_matrix,
        )
        next_p = (
            q_matrix
            + a_matrix.T @ p_matrix @ a_matrix
            - a_matrix.T @ p_matrix @ b_matrix @ gain
        )
        if np.max(np.abs(next_p - p_matrix)) < tolerance:
            p_matrix = next_p
            break
        p_matrix = next_p
    return np.linalg.solve(
        r_matrix + b_matrix.T @ p_matrix @ b_matrix,
        b_matrix.T @ p_matrix @ a_matrix,
    )


@lru_cache(maxsize=None)
def _cached_lqr_teacher_gain(pole_count: int) -> tuple[tuple[float, ...], ...]:
    validate_pole_count(pole_count)
    env = _make_env(pole_count=pole_count, max_steps=DEMO_MAX_STEPS)
    try:
        gain = solve_discrete_lqr(
            *linearize_dynamics(env),
            q_weights=default_lqr_q_weights(pole_count),
        )
        return tuple(tuple(float(value) for value in row) for row in gain)
    finally:
        env.close()


def lqr_teacher_gain(pole_count: int = DEFAULT_POLE_COUNT) -> np.ndarray:
    return np.asarray(_cached_lqr_teacher_gain(pole_count), dtype=np.float64)


def teacher_action(
    observation: np.ndarray | torch.Tensor,
    gain: np.ndarray,
    *,
    clip: bool = True,
) -> float:
    values = np.asarray(observation, dtype=np.float64)
    expected_shape = (gain.shape[1],)
    if values.shape != expected_shape:
        raise ValueError(f"observation must have shape {expected_shape}")
    action = float(-(gain @ values)[0])
    return float(np.clip(action, -1.0, 1.0)) if clip else action


def make_teacher_dataset(
    *,
    gain: np.ndarray,
    samples: int = DEMO_DATASET_SAMPLES,
    seed: int = 0,
    state_ranges: Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    state_size = gain.shape[1]
    pole_count = pole_count_from_state_size(state_size)
    if state_ranges is None:
        state_ranges = default_state_ranges(pole_count)
    if len(state_ranges) != state_size:
        raise ValueError(f"state_ranges must contain {state_size} values")
    rng = np.random.default_rng(seed)
    ranges = np.asarray(state_ranges, dtype=np.float32)
    observations = rng.uniform(-ranges, ranges, size=(samples, state_size)).astype(np.float32)
    actions = np.array(
        [[teacher_action(observation, gain)] for observation in observations],
        dtype=np.float32,
    )
    return torch.from_numpy(observations), torch.from_numpy(actions)


def _policy_action(policy_network: TriplePoleLinearPolicy, observation: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        action = policy_network(torch.as_tensor(observation, dtype=torch.float32))
        return action.numpy().astype(np.float32)


def evaluate_policy(
    policy_network: TriplePoleLinearPolicy,
    *,
    epoch: int,
    seed: int,
    eval_episodes: int = 5,
    max_steps: int = DEMO_MAX_STEPS,
    target_return: float = DEMO_TARGET_RETURN,
    pole_count: int | None = None,
) -> TriplePoleEvaluation:
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if target_return <= 0.0:
        raise ValueError("target_return must be greater than 0.0")
    pole_count = policy_network.pole_count if pole_count is None else pole_count
    env = _make_env(pole_count=pole_count, max_steps=max_steps)
    was_training = policy_network.training
    policy_network.eval()
    try:
        returns: list[float] = []
        lengths: list[int] = []
        for index in range(eval_episodes):
            observation, _ = env.reset(seed=seed + index)
            episode_return = 0.0
            while True:
                observation, reward, terminated, truncated, info = env.step(
                    _policy_action(policy_network, observation)
                )
                episode_return += reward
                if terminated or truncated:
                    returns.append(episode_return)
                    lengths.append(int(info["episode_length"]))
                    break
        mean_return = mean(returns)
        return TriplePoleEvaluation(
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
    gain: np.ndarray,
    *,
    seed: int = 0,
    eval_episodes: int = 5,
    max_steps: int = DEMO_MAX_STEPS,
    pole_count: int | None = None,
) -> tuple[float, float]:
    if pole_count is None:
        pole_count = pole_count_from_state_size(gain.shape[1])
    env = _make_env(pole_count=pole_count, max_steps=max_steps)
    try:
        returns: list[float] = []
        for index in range(eval_episodes):
            observation, _ = env.reset(seed=seed + index)
            episode_return = 0.0
            while True:
                observation, reward, terminated, truncated, _ = env.step(
                    [teacher_action(observation, gain)]
                )
                episode_return += reward
                if terminated or truncated:
                    returns.append(episode_return)
                    break
        return mean(returns), min(returns)
    finally:
        env.close()


def train_triple_pole_policy(
    *,
    optimizer_name: str = "adam",
    epochs: int = 60,
    learning_rate: float | None = None,
    samples: int = DEMO_DATASET_SAMPLES,
    batch_size: int = 128,
    seed: int = 0,
    eval_every: int = 5,
    eval_episodes: int = 5,
    max_steps: int = DEMO_MAX_STEPS,
    target_return: float = DEMO_TARGET_RETURN,
    pole_count: int = DEFAULT_POLE_COUNT,
) -> TriplePoleTrainingResult:
    if optimizer_name not in TRIPLE_POLE_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {TRIPLE_POLE_OPTIMIZERS}")
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
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if target_return <= 0.0:
        raise ValueError("target_return must be greater than 0.0")
    validate_pole_count(pole_count)

    gain = lqr_teacher_gain(pole_count)
    observations, target_actions = make_teacher_dataset(
        gain=gain,
        samples=samples,
        seed=seed,
    )
    torch.manual_seed(seed)
    policy_network = TriplePoleLinearPolicy(pole_count=pole_count)
    resolved_learning_rate = (
        learning_rate if learning_rate is not None else default_learning_rate(optimizer_name)
    )
    optimizer = make_optimizer(
        optimizer_name,
        policy_network.parameters(),
        learning_rate=resolved_learning_rate,
    )
    teacher_mean_return, teacher_min_return = evaluate_teacher(
        gain,
        seed=seed + 50_000,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        pole_count=pole_count,
    )

    generator = torch.Generator().manual_seed(seed)
    training_losses: list[float] = []
    evaluations: list[TriplePoleEvaluation] = []
    target_epoch: int | None = None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(samples, generator=generator)
        batch_losses: list[float] = []
        for offset in range(0, samples, batch_size):
            indexes = permutation[offset : offset + batch_size]
            prediction = policy_network.raw_action(observations[indexes])
            loss = torch.nn.functional.mse_loss(prediction, target_actions[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        training_losses.append(mean(batch_losses))

        if epoch % eval_every == 0 or epoch == epochs:
            evaluation = evaluate_policy(
                policy_network,
                epoch=epoch,
                seed=seed + 10_000 + epoch,
                eval_episodes=eval_episodes,
                max_steps=max_steps,
                target_return=target_return,
                pole_count=pole_count,
            )
            evaluations.append(evaluation)
            if target_epoch is None and evaluation.solved:
                target_epoch = epoch

    wall_clock_seconds = time.perf_counter() - start
    estimated_seconds_to_target = (
        wall_clock_seconds * target_epoch / epochs if target_epoch is not None else None
    )
    return TriplePoleTrainingResult(
        optimizer_name=optimizer_name,
        learning_rate=resolved_learning_rate,
        seed=seed,
        pole_count=pole_count,
        policy_network=policy_network,
        teacher_gain=gain,
        training_losses=training_losses,
        evaluations=evaluations,
        dataset_samples=samples,
        teacher_mean_return=teacher_mean_return,
        teacher_min_return=teacher_min_return,
        target_epoch=target_epoch,
        wall_clock_seconds=wall_clock_seconds,
        estimated_seconds_to_target=estimated_seconds_to_target,
    )


def compare_triple_pole_optimizers(
    *,
    optimizers: tuple[str, ...] = TRIPLE_POLE_OPTIMIZERS,
    seeds: tuple[int, ...] = (1, 2),
    epochs: int = 60,
    samples: int = DEMO_DATASET_SAMPLES,
    batch_size: int = 128,
    eval_every: int = 5,
    eval_episodes: int = 5,
    max_steps: int = DEMO_MAX_STEPS,
    target_return: float = DEMO_TARGET_RETURN,
    pole_count: int = DEFAULT_POLE_COUNT,
) -> tuple[list[TriplePoleOptimizerSummary], dict[str, list[TriplePoleTrainingResult]]]:
    if not optimizers:
        raise ValueError("optimizers must contain at least one optimizer")
    unknown = set(optimizers) - set(TRIPLE_POLE_OPTIMIZERS)
    if unknown:
        raise ValueError(f"unknown optimizers: {sorted(unknown)}")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    validate_pole_count(pole_count)

    results: dict[str, list[TriplePoleTrainingResult]] = {
        optimizer_name: [] for optimizer_name in optimizers
    }
    summaries: list[TriplePoleOptimizerSummary] = []
    for optimizer_name in optimizers:
        for seed in seeds:
            results[optimizer_name].append(
                train_triple_pole_policy(
                    optimizer_name=optimizer_name,
                    epochs=epochs,
                    samples=samples,
                    batch_size=batch_size,
                    seed=seed,
                    eval_every=eval_every,
                    eval_episodes=eval_episodes,
                    max_steps=max_steps,
                    target_return=target_return,
                    pole_count=pole_count,
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
            TriplePoleOptimizerSummary(
                optimizer_name=optimizer_name,
                learning_rate=rows[0].learning_rate,
                pole_count=pole_count,
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


def play_policy(
    policy_network: TriplePoleLinearPolicy,
    *,
    render_mode: str = "human",
    episodes: int = 1,
    seed: int = 0,
    max_steps: int = DEMO_MAX_STEPS,
    frame_delay_seconds: float = 0.0,
    pole_count: int | None = None,
) -> TriplePolePlayback:
    if render_mode not in {"human", "rgb_array"}:
        raise ValueError("render_mode must be 'human' or 'rgb_array'")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if frame_delay_seconds < 0.0:
        raise ValueError("frame_delay_seconds must be at least 0.0")
    pole_count = policy_network.pole_count if pole_count is None else pole_count
    env = _make_env(
        pole_count=pole_count,
        max_steps=max_steps,
        render_mode=render_mode,
    )
    was_training = policy_network.training
    policy_network.eval()
    try:
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        rendered_frames = 0
        first_frame_shape: tuple[int, ...] | None = None
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            frame = env.render()
            rendered_frames += 1
            if first_frame_shape is None and frame is not None:
                first_frame_shape = tuple(int(value) for value in frame.shape)
            episode_return = 0.0
            while True:
                observation, reward, terminated, truncated, info = env.step(
                    _policy_action(policy_network, observation)
                )
                frame = env.render()
                rendered_frames += 1
                if first_frame_shape is None and frame is not None:
                    first_frame_shape = tuple(int(value) for value in frame.shape)
                if frame_delay_seconds:
                    time.sleep(frame_delay_seconds)
                episode_return += reward
                if terminated or truncated:
                    episode_returns.append(episode_return)
                    episode_lengths.append(int(info["episode_length"]))
                    break
        return TriplePolePlayback(
            render_mode=render_mode,
            pole_count=pole_count,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            rendered_frames=rendered_frames,
            first_frame_shape=first_frame_shape,
        )
    finally:
        if was_training:
            policy_network.train()
        env.close()


compare_multi_pendulum_optimizers = compare_triple_pole_optimizers
train_multi_pendulum_policy = train_triple_pole_policy


__all__ = [
    "DEFAULT_LEARNING_RATES",
    "DEMO_DATASET_SAMPLES",
    "DEMO_MAX_STEPS",
    "DEMO_STATE_RANGES",
    "DEMO_TARGET_RETURN",
    "MULTI_POLE_OPTIMIZERS",
    "TRIPLE_POLE_OPTIMIZERS",
    "MultiPoleLinearPolicy",
    "TriplePoleEvaluation",
    "TriplePoleLinearPolicy",
    "TriplePoleOptimizerSummary",
    "TriplePolePlayback",
    "TriplePoleTrainingResult",
    "compare_multi_pendulum_optimizers",
    "compare_triple_pole_optimizers",
    "default_learning_rate",
    "default_lqr_q_weights",
    "default_state_ranges",
    "evaluate_policy",
    "evaluate_teacher",
    "linearize_dynamics",
    "lqr_teacher_gain",
    "make_optimizer",
    "make_teacher_dataset",
    "play_policy",
    "pole_count_from_state_size",
    "solve_discrete_lqr",
    "state_size_for_poles",
    "teacher_action",
    "train_multi_pendulum_policy",
    "train_triple_pole_policy",
    "validate_pole_count",
]
