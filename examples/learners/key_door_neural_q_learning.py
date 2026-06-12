"""Neural Q-learning for the classic KeyDoorGrid environment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import random
from statistics import mean

import torch

try:
    from examples.envs import KeyDoorGridEnv
    from examples.learners.key_door_q_learning import (
        ACTION_NAMES,
        EpisodeStep,
        State,
        state_from_info,
    )
except ModuleNotFoundError as error:
    if error.name not in {"examples", "examples.envs", "examples.learners"}:
        raise
    from envs import KeyDoorGridEnv
    from learners.key_door_q_learning import (
        ACTION_NAMES,
        EpisodeStep,
        State,
        state_from_info,
    )


NEURAL_OPTIMIZERS = ("sgd", "adam")
DEFAULT_LEARNING_RATES = {
    "sgd": 0.10,
    "adam": 0.003,
}


class KeyDoorQNetwork(torch.nn.Module):
    """Tiny MLP mapping one-hot KeyDoorGrid states to action values."""

    def __init__(self, *, state_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        if state_size < 1:
            raise ValueError("state_size must be at least 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(ACTION_NAMES)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


@dataclass(frozen=True)
class NeuralQEvaluation:
    episode: int
    solved: bool
    episode_return: float
    episode_length: int


@dataclass(frozen=True)
class NeuralQResult:
    optimizer_name: str
    learning_rate: float
    q_network: KeyDoorQNetwork
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_losses: list[float]
    evaluations: list[NeuralQEvaluation]
    final_epsilon: float
    env_layout: str
    state_size: int


@dataclass(frozen=True)
class OptimizerSummary:
    optimizer_name: str
    learning_rate: float
    solved_runs: int
    total_runs: int
    mean_final_return: float
    mean_greedy_steps: float


def _clone_env(env: KeyDoorGridEnv) -> KeyDoorGridEnv:
    return KeyDoorGridEnv(
        height=env.height,
        width=env.width,
        max_steps=env.max_steps,
        layout=env.layout,
    )


def state_vector_size(env: KeyDoorGridEnv) -> int:
    return env.height * env.width * 2


def encode_state(state: State, env: KeyDoorGridEnv) -> torch.Tensor:
    row, column, has_key = state
    if not 0 <= row < env.height:
        raise ValueError("state row is outside the environment")
    if not 0 <= column < env.width:
        raise ValueError("state column is outside the environment")
    if has_key not in {0, 1}:
        raise ValueError("state has_key must be 0 or 1")

    state_index = has_key * env.height * env.width + row * env.width + column
    encoded = torch.zeros(state_vector_size(env), dtype=torch.float32)
    encoded[state_index] = 1.0
    return encoded


def default_learning_rate(optimizer_name: str) -> float:
    if optimizer_name not in NEURAL_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {NEURAL_OPTIMIZERS}")
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
    raise ValueError(f"optimizer_name must be one of {NEURAL_OPTIMIZERS}")


def _greedy_action(
    q_network: KeyDoorQNetwork,
    state: State,
    env: KeyDoorGridEnv,
) -> int:
    with torch.no_grad():
        q_values = q_network(encode_state(state, env))
        return int(torch.argmax(q_values).item())


def greedy_q_rollout(
    q_network: KeyDoorQNetwork,
    *,
    env: KeyDoorGridEnv | None = None,
    seed: int = 0,
) -> list[EpisodeStep]:
    rollout_env = env if env is not None else KeyDoorGridEnv(layout="classic")
    was_training = q_network.training
    q_network.eval()
    try:
        _, info = rollout_env.reset(seed=seed)
        state = state_from_info(info)
        steps: list[EpisodeStep] = []

        while True:
            action = _greedy_action(q_network, state, rollout_env)
            _, reward, terminated, truncated, info = rollout_env.step(action)
            next_state = state_from_info(info)
            steps.append(
                EpisodeStep(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                    truncated=truncated,
                    tile=str(info["tile"]),
                )
            )
            state = next_state
            if terminated or truncated:
                return steps
    finally:
        if was_training:
            q_network.train()


def evaluate_q_network(
    q_network: KeyDoorQNetwork,
    env: KeyDoorGridEnv,
    *,
    episode: int,
    seed: int,
) -> NeuralQEvaluation:
    trace = greedy_q_rollout(q_network, env=_clone_env(env), seed=seed)
    return NeuralQEvaluation(
        episode=episode,
        solved=trace[-1].terminated and not trace[-1].truncated,
        episode_return=sum(step.reward for step in trace),
        episode_length=len(trace),
    )


def train_neural_q_learning(
    *,
    optimizer_name: str = "adam",
    episodes: int = 200,
    learning_rate: float | None = None,
    discount: float = 0.95,
    epsilon: float = 0.80,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    hidden_size: int = 16,
    eval_every: int = 25,
    seed: int = 0,
    env: KeyDoorGridEnv | None = None,
) -> NeuralQResult:
    if optimizer_name not in NEURAL_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {NEURAL_OPTIMIZERS}")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0.0, 1.0]")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0.0, 1.0]")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be in (0.0, 1.0]")
    if not 0.0 <= min_epsilon <= 1.0:
        raise ValueError("min_epsilon must be in [0.0, 1.0]")
    if hidden_size < 1:
        raise ValueError("hidden_size must be at least 1")
    if eval_every < 1:
        raise ValueError("eval_every must be at least 1")

    training_env = env if env is not None else KeyDoorGridEnv(layout="classic")
    state_size = state_vector_size(training_env)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    q_network = KeyDoorQNetwork(state_size=state_size, hidden_size=hidden_size)
    resolved_learning_rate = (
        learning_rate
        if learning_rate is not None
        else default_learning_rate(optimizer_name)
    )
    optimizer = make_optimizer(
        optimizer_name,
        q_network.parameters(),
        learning_rate=resolved_learning_rate,
    )

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_losses: list[float] = []
    evaluations: list[NeuralQEvaluation] = []
    exploration_rate = epsilon

    for episode in range(1, episodes + 1):
        _, info = training_env.reset(seed=seed + episode)
        state = state_from_info(info)
        total_reward = 0.0
        losses: list[float] = []

        while True:
            if rng.random() < exploration_rate:
                action = rng.randrange(training_env.action_space.n)
            else:
                action = _greedy_action(q_network, state, training_env)

            _, reward, terminated, truncated, info = training_env.step(action)
            next_state = state_from_info(info)
            done = terminated or truncated
            prediction = q_network(encode_state(state, training_env))[action]
            with torch.no_grad():
                target = torch.tensor(reward, dtype=torch.float32)
                if not done:
                    target = target + discount * torch.max(
                        q_network(encode_state(next_state, training_env))
                    )
            loss = torch.nn.functional.mse_loss(prediction, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_reward += reward
            losses.append(float(loss.item()))
            state = next_state
            if done:
                episode_returns.append(total_reward)
                episode_lengths.append(int(info["episode_length"]))
                episode_losses.append(mean(losses))
                break

        exploration_rate = max(min_epsilon, exploration_rate * epsilon_decay)
        if episode % eval_every == 0 or episode == episodes:
            evaluations.append(
                evaluate_q_network(
                    q_network,
                    training_env,
                    episode=episode,
                    seed=seed + 10_000 + episode,
                )
            )

    return NeuralQResult(
        optimizer_name=optimizer_name,
        learning_rate=resolved_learning_rate,
        q_network=q_network,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_losses=episode_losses,
        evaluations=evaluations,
        final_epsilon=exploration_rate,
        env_layout=training_env.layout,
        state_size=state_size,
    )


def compare_optimizers(
    *,
    optimizers: tuple[str, ...] = NEURAL_OPTIMIZERS,
    seeds: tuple[int, ...] = (1, 2, 3),
    episodes: int = 200,
    eval_every: int = 25,
    hidden_size: int = 16,
) -> tuple[list[OptimizerSummary], dict[str, list[NeuralQResult]]]:
    if not optimizers:
        raise ValueError("optimizers must contain at least one optimizer")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    results: dict[str, list[NeuralQResult]] = {
        optimizer_name: [] for optimizer_name in optimizers
    }
    summaries: list[OptimizerSummary] = []

    for optimizer_name in optimizers:
        for seed in seeds:
            results[optimizer_name].append(
                train_neural_q_learning(
                    optimizer_name=optimizer_name,
                    episodes=episodes,
                    eval_every=eval_every,
                    hidden_size=hidden_size,
                    seed=seed,
                    env=KeyDoorGridEnv(layout="classic"),
                )
            )

        final_evaluations = [
            result.evaluations[-1] for result in results[optimizer_name]
        ]
        summaries.append(
            OptimizerSummary(
                optimizer_name=optimizer_name,
                learning_rate=results[optimizer_name][0].learning_rate,
                solved_runs=sum(
                    1 for evaluation in final_evaluations if evaluation.solved
                ),
                total_runs=len(seeds),
                mean_final_return=mean(
                    evaluation.episode_return for evaluation in final_evaluations
                ),
                mean_greedy_steps=mean(
                    evaluation.episode_length for evaluation in final_evaluations
                ),
            )
        )

    summaries.sort(
        key=lambda summary: (
            -summary.solved_runs,
            -summary.mean_final_return,
            summary.mean_greedy_steps,
        )
    )
    return summaries, results


__all__ = [
    "DEFAULT_LEARNING_RATES",
    "NEURAL_OPTIMIZERS",
    "KeyDoorQNetwork",
    "NeuralQEvaluation",
    "NeuralQResult",
    "OptimizerSummary",
    "compare_optimizers",
    "default_learning_rate",
    "encode_state",
    "evaluate_q_network",
    "greedy_q_rollout",
    "make_optimizer",
    "state_vector_size",
    "train_neural_q_learning",
]
