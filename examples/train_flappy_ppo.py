"""Train a PPO policy on the C Flappy Bird environment with PufferLib 4 models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pufferlib
import torch
from torch.distributions import Categorical

from envs.flappy_bird import CFlappyBirdEnv, ensure_flappy_library
from pufferlib4 import SerialVectorEnv, make_flappy_policy, policy_logits_values


def make_vecenv(args: argparse.Namespace) -> SerialVectorEnv:
    ensure_flappy_library()
    return SerialVectorEnv(
        [
            lambda env_idx=env_idx: CFlappyBirdEnv(
                max_steps=args.max_steps,
                seed=args.seed + env_idx,
                pipe_gap=args.pipe_gap,
                reward_shaping=args.reward_shaping,
                centering_reward=args.centering_reward,
            )
            for env_idx in range(args.num_envs)
        ]
    )


def set_initial_action_bias(policy: torch.nn.Module, flap_logit_bias: float | None) -> None:
    if flap_logit_bias is None:
        return
    decoder = getattr(policy.decoder, "decoder", None)
    if decoder is None:
        return
    with torch.no_grad():
        decoder.bias[0] = 0.0
        decoder.bias[1] = flap_logit_bias


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device)

    for step_idx in reversed(range(rewards.shape[0])):
        next_values = next_value if step_idx == rewards.shape[0] - 1 else values[step_idx + 1]
        next_nonterminal = 1.0 - dones[step_idx]
        delta = rewards[step_idx] + gamma * next_values * next_nonterminal - values[step_idx]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step_idx] = last_gae

    return advantages, advantages + values


def save_checkpoint(
    policy: torch.nn.Module,
    args: argparse.Namespace,
    global_step: int,
    data_dir: Path,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{global_step:016d}.pt"
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "global_step": global_step,
            "hidden_size": args.hidden_size,
            "pipe_gap": args.pipe_gap,
            "pufferlib_version": getattr(pufferlib, "__version__", "unknown"),
            "args": serializable_args,
        },
        path,
    )
    return path


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    vecenv = make_vecenv(args)
    observation_size = int(np.prod(vecenv.single_observation_space.shape))
    policy = make_flappy_policy(observation_size, hidden_size=args.hidden_size).to(args.device)
    set_initial_action_bias(policy, args.initial_flap_logit_bias)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-8)

    observations_np, _ = vecenv.reset(seed=args.seed)
    observations = torch.as_tensor(observations_np, dtype=torch.float32, device=args.device)
    num_updates = max(1, args.total_timesteps // (args.num_envs * args.rollout_steps))
    global_step = 0
    start_time = time.time()
    latest_checkpoint: Path | None = None

    try:
        for update_idx in range(1, num_updates + 1):
            if args.anneal_lr:
                frac = 1.0 - (update_idx - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            obs_buf = torch.zeros(
                (args.rollout_steps, args.num_envs, observation_size),
                dtype=torch.float32,
                device=args.device,
            )
            action_buf = torch.zeros(
                (args.rollout_steps, args.num_envs),
                dtype=torch.long,
                device=args.device,
            )
            logprob_buf = torch.zeros((args.rollout_steps, args.num_envs), device=args.device)
            reward_buf = torch.zeros((args.rollout_steps, args.num_envs), device=args.device)
            done_buf = torch.zeros((args.rollout_steps, args.num_envs), device=args.device)
            value_buf = torch.zeros((args.rollout_steps, args.num_envs), device=args.device)

            completed_scores: list[float] = []
            completed_returns: list[float] = []
            for step_idx in range(args.rollout_steps):
                obs_buf[step_idx] = observations
                with torch.no_grad():
                    logits, values = policy_logits_values(policy, observations)
                    distribution = Categorical(logits=logits)
                    actions = distribution.sample()
                    logprobs = distribution.log_prob(actions)

                action_buf[step_idx] = actions
                logprob_buf[step_idx] = logprobs
                value_buf[step_idx] = values

                next_obs_np, rewards_np, terminals_np, truncations_np, infos = vecenv.step(
                    actions.cpu().numpy()
                )
                dones_np = np.logical_or(terminals_np, truncations_np)
                reward_buf[step_idx] = torch.as_tensor(
                    rewards_np,
                    dtype=torch.float32,
                    device=args.device,
                )
                done_buf[step_idx] = torch.as_tensor(
                    dones_np.astype(np.float32),
                    dtype=torch.float32,
                    device=args.device,
                )
                observations = torch.as_tensor(next_obs_np, dtype=torch.float32, device=args.device)
                global_step += args.num_envs

                for info in infos:
                    final_info = info.get("final_info")
                    if isinstance(final_info, dict):
                        completed_scores.append(float(final_info.get("score", 0.0)))
                        completed_returns.append(float(final_info.get("episode_return", 0.0)))

            with torch.no_grad():
                _, next_value = policy_logits_values(policy, observations)
                advantages, returns = compute_gae(
                    reward_buf,
                    done_buf,
                    value_buf,
                    next_value,
                    args.gamma,
                    args.gae_lambda,
                )

            flat_obs = obs_buf.reshape((-1, observation_size))
            flat_actions = action_buf.reshape(-1)
            flat_logprobs = logprob_buf.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = value_buf.reshape(-1)

            batch_size = flat_obs.shape[0]
            minibatch_size = min(args.minibatch_size, batch_size)
            batch_indices = np.arange(batch_size)
            for _ in range(args.update_epochs):
                np.random.shuffle(batch_indices)
                for start in range(0, batch_size, minibatch_size):
                    indices = torch.as_tensor(
                        batch_indices[start : start + minibatch_size],
                        dtype=torch.long,
                        device=args.device,
                    )
                    logits, new_values = policy_logits_values(policy, flat_obs[indices])
                    distribution = Categorical(logits=logits)
                    new_logprobs = distribution.log_prob(flat_actions[indices])
                    entropy = distribution.entropy().mean()

                    logratio = new_logprobs - flat_logprobs[indices]
                    ratio = logratio.exp()
                    mb_advantages = flat_advantages[indices]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )
                    policy_loss_1 = -mb_advantages * ratio
                    policy_loss_2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - args.clip_coef,
                        1.0 + args.clip_coef,
                    )
                    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                    value_loss_unclipped = (new_values - flat_returns[indices]) ** 2
                    value_clipped = flat_values[indices] + torch.clamp(
                        new_values - flat_values[indices],
                        -args.vf_clip_coef,
                        args.vf_clip_coef,
                    )
                    value_loss_clipped = (value_clipped - flat_returns[indices]) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    loss = policy_loss - args.ent_coef * entropy + args.vf_coef * value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()

            if update_idx % args.checkpoint_interval == 0 or update_idx == num_updates:
                latest_checkpoint = save_checkpoint(policy, args, global_step, args.data_dir)

            steps_per_second = int(global_step / max(time.time() - start_time, 1e-6))
            mean_score = np.mean(completed_scores) if completed_scores else 0.0
            mean_return = np.mean(completed_returns) if completed_returns else 0.0
            print(
                f"update={update_idx}/{num_updates} "
                f"step={global_step} sps={steps_per_second} "
                f"mean_score={mean_score:.2f} mean_return={mean_return:.3f}"
            )
    finally:
        vecenv.close()

    if latest_checkpoint is None:
        latest_checkpoint = save_checkpoint(policy, args, global_step, args.data_dir)
    print(f"Saved PPO checkpoint: {latest_checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on the C Flappy Bird environment with PufferLib 4 models."
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--vf-clip-coef", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--initial-flap-logit-bias", type=float, default=-2.0)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--pipe-gap", type=float, default=220.0)
    parser.add_argument("--reward-shaping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--centering-reward", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments") / "flappy_bird",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
