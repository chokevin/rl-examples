"""Small helpers for using PufferLib 4 model pieces with local Gymnasium envs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
import pufferlib.models
import torch
import torch.nn as nn
import torch.nn.functional as F


class SerialVectorEnv:
    """Minimal serial vector runner for Gymnasium-style envs.

    PufferLib 4's packaged API no longer includes the old Python vector wrapper.
    This keeps the examples runnable while the env core remains a normal
    Gymnasium object and the policy uses PufferLib 4 model components.
    """

    def __init__(self, env_fns: Sequence[Callable[[], gymnasium.Env]]) -> None:
        if not env_fns:
            raise ValueError("SerialVectorEnv needs at least one environment")
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.driver_env = self.envs[0]
        self.single_observation_space = self.driver_env.observation_space
        self.single_action_space = self.driver_env.action_space
        self.observations: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        observations: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        for env_idx, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + env_idx
            observation, info = env.reset(seed=env_seed)
            observations.append(np.asarray(observation, dtype=np.float32))
            infos.append(dict(info))
        self.observations = np.stack(observations)
        return self.observations.copy(), infos

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")

        observations: list[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminals = np.zeros(self.num_envs, dtype=np.bool_)
        truncations = np.zeros(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, Any]] = []

        for env_idx, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            observation, reward, terminated, truncated, info = env.step(action)
            rewards[env_idx] = float(reward)
            terminals[env_idx] = bool(terminated)
            truncations[env_idx] = bool(truncated)

            next_info = dict(info)
            if terminated or truncated:
                final_info = dict(info)
                observation, reset_info = env.reset()
                next_info = dict(reset_info)
                next_info["final_info"] = final_info
                if "score" in final_info:
                    next_info["score"] = final_info["score"]

            observations.append(np.asarray(observation, dtype=np.float32))
            infos.append(next_info)

        self.observations = np.stack(observations)
        return (
            self.observations.copy(),
            rewards,
            terminals,
            truncations,
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


class FlappyGELUNetwork(nn.Module):
    """Activation-only network matching the C raylib policy evaluator."""

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[Any, ...]:
        return ()

    def forward_eval(
        self,
        hidden: torch.Tensor,
        state: tuple[Any, ...],
    ) -> tuple[torch.Tensor, tuple[Any, ...]]:
        return F.gelu(hidden), state

    def forward_train(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.gelu(hidden)


def make_flappy_policy(
    observation_size: int,
    hidden_size: int = 128,
    action_sizes: Sequence[int] = (2,),
) -> pufferlib.models.Policy:
    return pufferlib.models.Policy(
        encoder=pufferlib.models.DefaultEncoder(observation_size, hidden_size),
        decoder=pufferlib.models.DefaultDecoder(action_sizes, hidden_size),
        network=FlappyGELUNetwork(),
    )


def policy_logits_values(
    policy: pufferlib.models.Policy,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = policy.initial_state(observations.shape[0], observations.device)
    logits, values, _ = policy.forward_eval(observations, state)
    return logits, values.squeeze(-1)


@dataclass(frozen=True)
class LoadedCheckpoint:
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def load_checkpoint(path: str | bytes | Any, device: str | torch.device) -> LoadedCheckpoint:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {}
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")

    cleaned = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    return LoadedCheckpoint(state_dict=cleaned, metadata=metadata)
