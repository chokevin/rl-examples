"""Optional MuJoCo Warp batched rollout training for InvertedPendulum cartpole."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
import time

import mujoco
import numpy as np
import torch

try:
    import warp as wp
except ModuleNotFoundError:
    wp = None

try:
    from examples.learners.mujoco_batched_cartpole import (
        BATCHED_CARTPOLE_OPTIMIZERS,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EVAL_BATCH_SIZE,
        DEFAULT_EXPLORATION_SIGMA,
        DEFAULT_MAX_STEPS,
        DEFAULT_TARGET_RETURN,
        BatchedESGeneration,
        BatchedRolloutStats,
        default_learning_rate,
        make_inverted_pendulum_model,
        make_optimizer,
        sample_initial_states,
    )
except ModuleNotFoundError as error:
    if error.name not in {"examples", "examples.learners", "examples.learners.mujoco_batched_cartpole"}:
        raise
    from learners.mujoco_batched_cartpole import (
        BATCHED_CARTPOLE_OPTIMIZERS,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EVAL_BATCH_SIZE,
        DEFAULT_EXPLORATION_SIGMA,
        DEFAULT_MAX_STEPS,
        DEFAULT_TARGET_RETURN,
        BatchedESGeneration,
        BatchedRolloutStats,
        default_learning_rate,
        make_inverted_pendulum_model,
        make_optimizer,
        sample_initial_states,
    )


MUJOCO_WARP_INSTALL_HINT = (
    "MuJoCo Warp is optional. Install it with "
    "`uv sync --extra warp` or run this demo with `uv run --extra warp ...`."
)
DEFAULT_CUDAGRAPH_CONTROLLER_GAINS = (0.0, 10.0, 0.0, 1.0)


if wp is not None:

    @wp.kernel
    def _linear_controller_kernel(
        qpos: wp.array2d(dtype=wp.float32),
        qvel: wp.array2d(dtype=wp.float32),
        ctrl: wp.array2d(dtype=wp.float32),
        k_x: wp.float32,
        k_theta: wp.float32,
        k_xdot: wp.float32,
        k_thetadot: wp.float32,
        action_low: wp.float32,
        action_high: wp.float32,
    ) -> None:
        world = wp.tid()
        action = (
            k_x * qpos[world, 0]
            + k_theta * qpos[world, 1]
            + k_xdot * qvel[world, 0]
            + k_thetadot * qvel[world, 1]
        )
        ctrl[world, 0] = wp.min(wp.max(action, action_low), action_high)

else:
    _linear_controller_kernel = None


@dataclass(frozen=True)
class MuJoCoWarpStatus:
    available: bool
    reason: str | None
    mujoco_warp_version: str | None
    warp_version: str | None
    device: str | None
    is_cuda: bool


@dataclass(frozen=True)
class MuJoCoWarpTrainingResult:
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
    mujoco_warp_version: str
    warp_version: str
    device: str
    device_is_cuda: bool


@dataclass(frozen=True)
class MuJoCoWarpCudaGraphBenchmark:
    batch_size: int
    graph_steps: int
    graph_replays: int
    physics_steps: int
    wall_clock_seconds: float
    physics_steps_per_second: float
    mujoco_warp_version: str
    warp_version: str
    device: str
    model_disableflags: int
    control_value: float
    controller_gains: tuple[float, float, float, float] | None


def configure_cudagraph_benchmark_model(
    model: mujoco.MjModel,
    *,
    disable_contact: bool = False,
    disable_joint_limits: bool = False,
) -> None:
    """Apply optional model simplifications for captured-physics benchmarks."""
    if disable_contact:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    if disable_joint_limits:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT


def normalize_cudagraph_controller_gains(
    controller_gains: Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    if controller_gains is None:
        return None
    gains = tuple(float(value) for value in controller_gains)
    if len(gains) != 4:
        raise ValueError("controller_gains must contain four values")
    return gains


def _launch_linear_controller(
    wp_module: ModuleType,
    *,
    batch_size: int,
    warp_data: object,
    controller_gains: tuple[float, float, float, float],
    action_low: float,
    action_high: float,
) -> None:
    if _linear_controller_kernel is None:
        raise RuntimeError(MUJOCO_WARP_INSTALL_HINT)
    wp_module.launch(
        _linear_controller_kernel,
        dim=batch_size,
        inputs=[
            warp_data.qpos,
            warp_data.qvel,
            warp_data.ctrl,
            np.float32(controller_gains[0]),
            np.float32(controller_gains[1]),
            np.float32(controller_gains[2]),
            np.float32(controller_gains[3]),
            np.float32(action_low),
            np.float32(action_high),
        ],
    )


def _import_mujoco_warp() -> tuple[ModuleType, ModuleType]:
    try:
        import warp as wp
        import mujoco_warp as mjw
    except ModuleNotFoundError as error:
        if error.name in {"warp", "mujoco_warp"}:
            raise RuntimeError(MUJOCO_WARP_INSTALL_HINT) from error
        raise
    return mjw, wp


def _require_mujoco_warp(device: str | None = None) -> tuple[MuJoCoWarpStatus, ModuleType, ModuleType, object]:
    mjw, wp = _import_mujoco_warp()
    if hasattr(wp.config, "log_level") and hasattr(wp, "LOG_WARNING"):
        wp.config.log_level = wp.LOG_WARNING
    else:
        wp.config.quiet = True
    wp.init()
    try:
        device_object = wp.get_device(device) if device is not None else wp.get_device()
    except Exception as error:
        raise RuntimeError(f"MuJoCo Warp device {device!r} is not available") from error
    return (
        MuJoCoWarpStatus(
            available=True,
            reason=None,
            mujoco_warp_version=getattr(mjw, "__version__", "unknown"),
            warp_version=getattr(wp, "__version__", "unknown"),
            device=str(device_object),
            is_cuda=bool(getattr(device_object, "is_cuda", False)),
        ),
        mjw,
        wp,
        device_object,
    )


def mujoco_warp_status(device: str | None = None) -> MuJoCoWarpStatus:
    try:
        status, _, _, _ = _require_mujoco_warp(device=device)
        return status
    except RuntimeError as error:
        return MuJoCoWarpStatus(
            available=False,
            reason=str(error),
            mujoco_warp_version=None,
            warp_version=None,
            device=device,
            is_cuda=False,
        )


def benchmark_mujoco_warp_cudagraph_steps(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    graph_steps: int = DEFAULT_MAX_STEPS,
    graph_replays: int = 10,
    seed: int = 0,
    initial_noise: float = 0.01,
    control_value: float = 0.0,
    controller_gains: Sequence[float] | None = None,
    device: str | None = "cuda:0",
    model: mujoco.MjModel | None = None,
    disable_contact: bool = False,
    disable_joint_limits: bool = False,
) -> MuJoCoWarpCudaGraphBenchmark:
    """Benchmark an unrolled MuJoCo Warp CUDA graph without Python step loops.

    This is intentionally not a learning loop. It measures the high-speed physics
    substrate that an APIC/CUDA-graph path needs before fusing policy inference.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if graph_steps < 1:
        raise ValueError("graph_steps must be at least 1")
    if graph_replays < 1:
        raise ValueError("graph_replays must be at least 1")
    if initial_noise < 0.0:
        raise ValueError("initial_noise must be at least 0.0")
    resolved_controller_gains = normalize_cudagraph_controller_gains(controller_gains)

    status, mjw, wp, device_object = _require_mujoco_warp(device=device)
    if not status.is_cuda:
        raise RuntimeError(
            f"CUDA graph benchmark requires a CUDA Warp device; got {status.device}. "
            "Use --device cuda:0 on an NVIDIA machine."
        )

    model = make_inverted_pendulum_model() if model is None else model
    if resolved_controller_gains is not None and (model.nq, model.nv, model.nu) != (2, 2, 1):
        raise ValueError("controller_gains require an InvertedPendulum-style nq=2, nv=2, nu=1 model")
    configure_cudagraph_benchmark_model(
        model,
        disable_contact=disable_contact,
        disable_joint_limits=disable_joint_limits,
    )
    initial_states = sample_initial_states(
        model,
        batch_size=batch_size,
        seed=seed,
        noise=initial_noise,
    )
    qpos = initial_states[:, 1:3].astype(np.float32)
    qvel = initial_states[:, 3:5].astype(np.float32)
    controls = np.full((batch_size, model.nu), control_value, dtype=np.float32)
    if resolved_controller_gains is not None:
        controls.fill(0.0)
    action_low = float(model.actuator_ctrlrange[0, 0])
    action_high = float(model.actuator_ctrlrange[0, 1])

    with wp.ScopedDevice(device_object):
        mujoco_data = mujoco.MjData(model)
        mujoco.mj_forward(model, mujoco_data)
        warp_model = mjw.put_model(model)
        warp_data = mjw.put_data(model, mujoco_data, nworld=batch_size)
        wp.copy(warp_data.qpos, wp.array(qpos, dtype=wp.float32))
        wp.copy(warp_data.qvel, wp.array(qvel, dtype=wp.float32))
        wp.copy(warp_data.ctrl, wp.array(controls, dtype=wp.float32))
        wp.copy(warp_data.time, wp.array(np.zeros(batch_size, dtype=np.float32)))

        if resolved_controller_gains is not None:
            _launch_linear_controller(
                wp,
                batch_size=batch_size,
                warp_data=warp_data,
                controller_gains=resolved_controller_gains,
                action_low=action_low,
                action_high=action_high,
            )
        mjw.step(warp_model, warp_data)
        wp.copy(warp_data.qpos, wp.array(qpos, dtype=wp.float32))
        wp.copy(warp_data.qvel, wp.array(qvel, dtype=wp.float32))
        wp.copy(warp_data.time, wp.array(np.zeros(batch_size, dtype=np.float32)))

        with wp.ScopedCapture() as capture:
            for _ in range(graph_steps):
                if resolved_controller_gains is not None:
                    _launch_linear_controller(
                        wp,
                        batch_size=batch_size,
                        warp_data=warp_data,
                        controller_gains=resolved_controller_gains,
                        action_low=action_low,
                        action_high=action_high,
                    )
                mjw.step(warp_model, warp_data)
        wp.synchronize()

        start = time.perf_counter()
        for _ in range(graph_replays):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        wall_clock_seconds = time.perf_counter() - start

    physics_steps = batch_size * graph_steps * graph_replays
    return MuJoCoWarpCudaGraphBenchmark(
        batch_size=batch_size,
        graph_steps=graph_steps,
        graph_replays=graph_replays,
        physics_steps=physics_steps,
        wall_clock_seconds=wall_clock_seconds,
        physics_steps_per_second=physics_steps / max(wall_clock_seconds, 1e-12),
        mujoco_warp_version=status.mujoco_warp_version or "unknown",
        warp_version=status.warp_version or "unknown",
        device=status.device or "unknown",
        model_disableflags=int(model.opt.disableflags),
        control_value=control_value,
        controller_gains=resolved_controller_gains,
    )


def rollout_linear_policies_mujoco_warp(
    policy_parameters: np.ndarray,
    *,
    batch_size: int,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 0,
    target_return: float = DEFAULT_TARGET_RETURN,
    initial_noise: float = 0.01,
    device: str | None = None,
    model: mujoco.MjModel | None = None,
    capture_graph: bool = True,
) -> tuple[BatchedRolloutStats, np.ndarray, MuJoCoWarpStatus]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if target_return <= 0.0:
        raise ValueError("target_return must be greater than 0.0")
    if initial_noise < 0.0:
        raise ValueError("initial_noise must be at least 0.0")

    parameters = np.asarray(policy_parameters, dtype=np.float32)
    if parameters.shape == (4,):
        parameters = np.repeat(parameters[None, :], batch_size, axis=0)
    elif parameters.shape == (1, 4):
        parameters = np.repeat(parameters, batch_size, axis=0)
    elif parameters.shape != (batch_size, 4):
        raise ValueError("policy_parameters must have shape (4,), (1, 4), or (batch_size, 4)")

    status, mjw, wp, device_object = _require_mujoco_warp(device=device)
    model = make_inverted_pendulum_model() if model is None else model
    initial_states = sample_initial_states(
        model,
        batch_size=batch_size,
        seed=seed,
        noise=initial_noise,
    )
    qpos = initial_states[:, 1:3].astype(np.float32)
    qvel = initial_states[:, 3:5].astype(np.float32)
    returns = np.zeros(batch_size, dtype=np.float32)
    active = np.ones(batch_size, dtype=bool)
    active_environment_steps = 0
    loop_steps = 0

    with wp.ScopedDevice(device_object):
        mujoco_data = mujoco.MjData(model)
        mujoco.mj_forward(model, mujoco_data)
        warp_model = mjw.put_model(model)
        warp_data = mjw.put_data(model, mujoco_data, nworld=batch_size)
        wp.copy(warp_data.qpos, wp.array(qpos, dtype=wp.float32))
        wp.copy(warp_data.qvel, wp.array(qvel, dtype=wp.float32))
        wp.copy(warp_data.ctrl, wp.array(np.zeros((batch_size, model.nu), dtype=np.float32)))
        wp.copy(warp_data.time, wp.array(np.zeros(batch_size, dtype=np.float32)))

        graph = None
        if capture_graph and status.is_cuda:
            with wp.ScopedCapture() as capture:
                mjw.step(warp_model, warp_data)
            graph = capture.graph

        start = time.perf_counter()
        for _ in range(max_steps):
            observations = np.concatenate((qpos, qvel), axis=1)
            controls = np.clip(np.sum(parameters * observations, axis=1), -3.0, 3.0)
            wp.copy(warp_data.ctrl, wp.array(controls[:, None].astype(np.float32), dtype=wp.float32))
            if graph is None:
                mjw.step(warp_model, warp_data)
            else:
                wp.capture_launch(graph)
                wp.synchronize()

            returns[active] += 1.0
            active_environment_steps += int(active.sum())
            loop_steps += 1
            qpos = warp_data.qpos.numpy()
            qvel = warp_data.qvel.numpy()
            healthy = np.isfinite(qpos).all(axis=1) & np.isfinite(qvel).all(axis=1) & (np.abs(qpos[:, 1]) <= 0.2)
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
        status,
    )


def train_mujoco_warp_es_cartpole(
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
    device: str | None = None,
    capture_graph: bool = True,
) -> MuJoCoWarpTrainingResult:
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
    status = mujoco_warp_status(device=device)
    if not status.available:
        raise RuntimeError(status.reason or MUJOCO_WARP_INSTALL_HINT)

    for generation in range(1, generations + 1):
        noise = torch.randn(batch_size, 4)
        candidates = (policy_mean.detach() + sigma * noise).numpy()
        train_stats, train_returns, status = rollout_linear_policies_mujoco_warp(
            candidates,
            batch_size=batch_size,
            max_steps=max_steps,
            seed=seed + generation,
            target_return=target_return,
            device=device,
            model=model,
            capture_graph=capture_graph,
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

        evaluation_stats, _, _ = rollout_linear_policies_mujoco_warp(
            policy_mean.detach().numpy(),
            batch_size=eval_batch_size,
            max_steps=max_steps,
            seed=seed + 10_000 + generation,
            target_return=target_return,
            device=device,
            model=model,
            capture_graph=capture_graph,
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

    return MuJoCoWarpTrainingResult(
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
        mujoco_warp_version=status.mujoco_warp_version or "unknown",
        warp_version=status.warp_version or "unknown",
        device=status.device or "unknown",
        device_is_cuda=status.is_cuda,
    )


__all__ = [
    "MUJOCO_WARP_INSTALL_HINT",
    "MuJoCoWarpCudaGraphBenchmark",
    "MuJoCoWarpStatus",
    "MuJoCoWarpTrainingResult",
    "DEFAULT_CUDAGRAPH_CONTROLLER_GAINS",
    "benchmark_mujoco_warp_cudagraph_steps",
    "configure_cudagraph_benchmark_model",
    "mujoco_warp_status",
    "normalize_cudagraph_controller_gains",
    "rollout_linear_policies_mujoco_warp",
    "train_mujoco_warp_es_cartpole",
]
