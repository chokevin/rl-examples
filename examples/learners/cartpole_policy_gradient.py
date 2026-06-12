"""Tiny policy-gradient learner for Gymnasium CartPole."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean, median
import time

import gymnasium as gym
import torch


CARTPOLE_ENV_ID = "CartPole-v1"
CARTPOLE_OPTIMIZERS = ("sgd", "adam")
CARTPOLE_SOLVED_RETURN = 475.0
DEFAULT_LEARNING_RATES = {
    "sgd": 0.02,
    "adam": 0.01,
}


class CartPolePolicyNetwork(torch.nn.Module):
    """Small MLP mapping CartPole observations to action logits."""

    def __init__(
        self,
        *,
        observation_size: int,
        action_count: int,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        if observation_size < 1:
            raise ValueError("observation_size must be at least 1")
        if action_count < 2:
            raise ValueError("action_count must be at least 2")
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")

        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, action_count),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


@dataclass(frozen=True)
class CartPoleEvaluation:
    episode: int
    eval_episodes: int
    mean_return: float
    best_return: float
    mean_length: float
    solved: bool


@dataclass(frozen=True)
class CartPoleTrainingResult:
    optimizer_name: str
    learning_rate: float
    seed: int
    policy_network: CartPolePolicyNetwork
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_losses: list[float]
    evaluations: list[CartPoleEvaluation]
    env_id: str
    total_samples: int
    final_window_return: float
    best_return: float
    solve_episode: int | None
    wall_clock_seconds: float
    estimated_seconds_to_solve: float | None


@dataclass(frozen=True)
class CartPoleOptimizerSummary:
    optimizer_name: str
    learning_rate: float
    solved_runs: int
    total_runs: int
    median_solve_episode: int | None
    median_seconds_to_solve: float | None
    mean_wall_clock_seconds: float
    mean_total_samples: float
    mean_seconds_per_sample: float
    mean_final_window_return: float
    mean_best_return: float
    mean_final_eval_return: float


def default_learning_rate(optimizer_name: str) -> float:
    if optimizer_name not in CARTPOLE_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {CARTPOLE_OPTIMIZERS}")
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
    raise ValueError(f"optimizer_name must be one of {CARTPOLE_OPTIMIZERS}")


def discounted_returns(
    rewards: Sequence[float],
    *,
    discount: float,
    normalize: bool = True,
) -> torch.Tensor:
    if not rewards:
        raise ValueError("rewards must contain at least one reward")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0.0, 1.0]")

    running_return = 0.0
    returns: list[float] = []
    for reward in reversed(rewards):
        running_return = float(reward) + discount * running_return
        returns.append(running_return)
    returns.reverse()

    result = torch.tensor(returns, dtype=torch.float32)
    if normalize and result.numel() > 1:
        std = result.std(unbiased=False)
        if float(std.item()) > 1e-8:
            result = (result - result.mean()) / (std + 1e-8)
    return result


def _validate_policy_env(env: gym.Env) -> tuple[int, int]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("CartPole learner expects a Box observation space")
    if len(env.observation_space.shape) != 1:
        raise ValueError("CartPole learner expects a flat observation vector")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("CartPole learner expects a Discrete action space")

    return int(env.observation_space.shape[0]), int(env.action_space.n)


def _as_observation_tensor(observation: object) -> torch.Tensor:
    return torch.as_tensor(observation, dtype=torch.float32)


def sample_policy_action(
    policy_network: CartPolePolicyNetwork,
    observation: object,
) -> tuple[int, torch.Tensor]:
    logits = policy_network(_as_observation_tensor(observation))
    distribution = torch.distributions.Categorical(logits=logits)
    action = distribution.sample()
    return int(action.item()), distribution.log_prob(action)


def greedy_policy_action(
    policy_network: CartPolePolicyNetwork,
    observation: object,
) -> int:
    with torch.no_grad():
        logits = policy_network(_as_observation_tensor(observation))
        return int(torch.argmax(logits).item())


def evaluate_policy(
    policy_network: CartPolePolicyNetwork,
    *,
    env_id: str = CARTPOLE_ENV_ID,
    episode: int,
    seed: int,
    eval_episodes: int = 3,
    max_steps_per_episode: int | None = None,
    solved_threshold: float = CARTPOLE_SOLVED_RETURN,
) -> CartPoleEvaluation:
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")
    if max_steps_per_episode is not None and max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least 1")
    if solved_threshold <= 0.0:
        raise ValueError("solved_threshold must be greater than 0.0")

    env = gym.make(env_id)
    was_training = policy_network.training
    policy_network.eval()
    try:
        _validate_policy_env(env)
        returns: list[float] = []
        lengths: list[int] = []

        for index in range(eval_episodes):
            observation, _ = env.reset(seed=seed + index)
            episode_return = 0.0
            episode_length = 0

            while True:
                action = greedy_policy_action(policy_network, observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                hit_step_cap = (
                    max_steps_per_episode is not None
                    and episode_length >= max_steps_per_episode
                )
                if terminated or truncated or hit_step_cap:
                    returns.append(episode_return)
                    lengths.append(episode_length)
                    break

        mean_return = mean(returns)
        return CartPoleEvaluation(
            episode=episode,
            eval_episodes=eval_episodes,
            mean_return=mean_return,
            best_return=max(returns),
            mean_length=mean(lengths),
            solved=mean_return >= solved_threshold,
        )
    finally:
        if was_training:
            policy_network.train()
        env.close()


def _mean_tail(values: Sequence[float], window: int) -> float:
    return mean(values[-min(window, len(values)) :])


def train_cartpole_policy_gradient(
    *,
    optimizer_name: str = "adam",
    episodes: int = 80,
    learning_rate: float | None = None,
    discount: float = 0.99,
    hidden_size: int = 32,
    seed: int = 0,
    env_id: str = CARTPOLE_ENV_ID,
    eval_every: int = 20,
    eval_episodes: int = 3,
    solve_window: int = 20,
    solved_threshold: float = CARTPOLE_SOLVED_RETURN,
    max_steps_per_episode: int | None = None,
) -> CartPoleTrainingResult:
    if optimizer_name not in CARTPOLE_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {CARTPOLE_OPTIMIZERS}")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0.0, 1.0]")
    if hidden_size < 1:
        raise ValueError("hidden_size must be at least 1")
    if eval_every < 1:
        raise ValueError("eval_every must be at least 1")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")
    if solve_window < 1:
        raise ValueError("solve_window must be at least 1")
    if solved_threshold <= 0.0:
        raise ValueError("solved_threshold must be greater than 0.0")
    if max_steps_per_episode is not None and max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be at least 1")

    env = gym.make(env_id)
    try:
        observation_size, action_count = _validate_policy_env(env)
        torch.manual_seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        policy_network = CartPolePolicyNetwork(
            observation_size=observation_size,
            action_count=action_count,
            hidden_size=hidden_size,
        )
        resolved_learning_rate = (
            learning_rate
            if learning_rate is not None
            else default_learning_rate(optimizer_name)
        )
        optimizer = make_optimizer(
            optimizer_name,
            policy_network.parameters(),
            learning_rate=resolved_learning_rate,
        )

        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        episode_losses: list[float] = []
        evaluations: list[CartPoleEvaluation] = []
        total_samples = 0
        solve_episode: int | None = None
        start = time.perf_counter()

        for episode in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed + episode)
            rewards: list[float] = []
            log_probs: list[torch.Tensor] = []
            episode_return = 0.0
            episode_length = 0

            while True:
                action, log_prob = sample_policy_action(policy_network, observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                log_probs.append(log_prob)
                episode_return += float(reward)
                episode_length += 1

                hit_step_cap = (
                    max_steps_per_episode is not None
                    and episode_length >= max_steps_per_episode
                )
                if terminated or truncated or hit_step_cap:
                    break

            returns = discounted_returns(rewards, discount=discount)
            loss = -(torch.stack(log_probs) * returns).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_losses.append(float(loss.item()))
            total_samples += episode_length

            if (
                solve_episode is None
                and len(episode_returns) >= solve_window
                and _mean_tail(episode_returns, solve_window) >= solved_threshold
            ):
                solve_episode = episode

            if episode % eval_every == 0 or episode == episodes:
                evaluations.append(
                    evaluate_policy(
                        policy_network,
                        env_id=env_id,
                        episode=episode,
                        seed=seed + 10_000 + episode,
                        eval_episodes=eval_episodes,
                        max_steps_per_episode=max_steps_per_episode,
                        solved_threshold=solved_threshold,
                    )
                )

        wall_clock_seconds = time.perf_counter() - start
        estimated_seconds_to_solve = (
            wall_clock_seconds * solve_episode / episodes
            if solve_episode is not None
            else None
        )
        return CartPoleTrainingResult(
            optimizer_name=optimizer_name,
            learning_rate=resolved_learning_rate,
            seed=seed,
            policy_network=policy_network,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            episode_losses=episode_losses,
            evaluations=evaluations,
            env_id=env_id,
            total_samples=total_samples,
            final_window_return=_mean_tail(episode_returns, solve_window),
            best_return=max(episode_returns),
            solve_episode=solve_episode,
            wall_clock_seconds=wall_clock_seconds,
            estimated_seconds_to_solve=estimated_seconds_to_solve,
        )
    finally:
        env.close()


def compare_cartpole_optimizers(
    *,
    optimizers: tuple[str, ...] = CARTPOLE_OPTIMIZERS,
    seeds: tuple[int, ...] = (1, 2),
    episodes: int = 80,
    eval_every: int = 20,
    eval_episodes: int = 3,
    hidden_size: int = 32,
    solve_window: int = 20,
    solved_threshold: float = CARTPOLE_SOLVED_RETURN,
    max_steps_per_episode: int | None = None,
) -> tuple[list[CartPoleOptimizerSummary], dict[str, list[CartPoleTrainingResult]]]:
    if not optimizers:
        raise ValueError("optimizers must contain at least one optimizer")
    unknown = set(optimizers) - set(CARTPOLE_OPTIMIZERS)
    if unknown:
        raise ValueError(f"unknown optimizers: {sorted(unknown)}")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    results: dict[str, list[CartPoleTrainingResult]] = {
        optimizer_name: [] for optimizer_name in optimizers
    }
    summaries: list[CartPoleOptimizerSummary] = []

    for optimizer_name in optimizers:
        for seed in seeds:
            results[optimizer_name].append(
                train_cartpole_policy_gradient(
                    optimizer_name=optimizer_name,
                    episodes=episodes,
                    eval_every=eval_every,
                    eval_episodes=eval_episodes,
                    hidden_size=hidden_size,
                    seed=seed,
                    solve_window=solve_window,
                    solved_threshold=solved_threshold,
                    max_steps_per_episode=max_steps_per_episode,
                )
            )

        rows = results[optimizer_name]
        solve_episodes = [
            row.solve_episode for row in rows if row.solve_episode is not None
        ]
        solve_seconds = [
            row.estimated_seconds_to_solve
            for row in rows
            if row.estimated_seconds_to_solve is not None
        ]
        summaries.append(
            CartPoleOptimizerSummary(
                optimizer_name=optimizer_name,
                learning_rate=rows[0].learning_rate,
                solved_runs=len(solve_episodes),
                total_runs=len(rows),
                median_solve_episode=(
                    int(median(solve_episodes)) if solve_episodes else None
                ),
                median_seconds_to_solve=(
                    median(solve_seconds) if solve_seconds else None
                ),
                mean_wall_clock_seconds=mean(row.wall_clock_seconds for row in rows),
                mean_total_samples=mean(row.total_samples for row in rows),
                mean_seconds_per_sample=mean(
                    row.wall_clock_seconds / max(row.total_samples, 1)
                    for row in rows
                ),
                mean_final_window_return=mean(
                    row.final_window_return for row in rows
                ),
                mean_best_return=mean(row.best_return for row in rows),
                mean_final_eval_return=mean(
                    row.evaluations[-1].mean_return for row in rows
                ),
            )
        )

    summaries.sort(
        key=lambda summary: (
            summary.solved_runs == 0,
            -summary.solved_runs,
            summary.median_seconds_to_solve is None,
            summary.median_seconds_to_solve
            if summary.median_seconds_to_solve is not None
            else float("inf"),
            -summary.mean_final_window_return,
            summary.mean_wall_clock_seconds,
        )
    )
    return summaries, results


__all__ = [
    "CARTPOLE_ENV_ID",
    "CARTPOLE_OPTIMIZERS",
    "CARTPOLE_SOLVED_RETURN",
    "DEFAULT_LEARNING_RATES",
    "CartPoleEvaluation",
    "CartPoleOptimizerSummary",
    "CartPolePolicyNetwork",
    "CartPoleTrainingResult",
    "compare_cartpole_optimizers",
    "default_learning_rate",
    "discounted_returns",
    "evaluate_policy",
    "greedy_policy_action",
    "make_optimizer",
    "sample_policy_action",
    "train_cartpole_policy_gradient",
]
