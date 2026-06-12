"""Tiny MuJoCo carts with configurable stacked pendulum links."""

from __future__ import annotations

from typing import Any

import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco
import numpy as np

try:
    from examples.mujoco_compat import ensure_mujoco_human_viewer_compatibility
except ModuleNotFoundError as error:
    if error.name not in {"examples", "examples.mujoco_compat"}:
        raise
    from mujoco_compat import ensure_mujoco_human_viewer_compatibility


DEFAULT_POLE_COUNT = 3
POLE_LENGTH = 0.35
TRIPLE_CART_POLE_XML = ""
_POLE_COLORS = (
    "0.8 0.2 0.2 1",
    "0.2 0.7 0.2 1",
    "0.9 0.7 0.1 1",
    "0.6 0.2 0.8 1",
    "0.1 0.7 0.8 1",
    "0.9 0.4 0.1 1",
)


def _validate_pole_count(pole_count: int) -> None:
    if pole_count < 1:
        raise ValueError("pole_count must be at least 1")


def make_stacked_cart_pole_xml(pole_count: int = DEFAULT_POLE_COUNT) -> str:
    """Build a MuJoCo XML model with one cart and N serial pole links."""
    _validate_pole_count(pole_count)
    nested = ""
    indent = "      "
    for index in range(pole_count, 0, -1):
        body_pos = "0 0 0.08" if index == 1 else f"0 0 {POLE_LENGTH:g}"
        radius = max(0.01, 0.02 - 0.002 * index)
        color = _POLE_COLORS[(index - 1) % len(_POLE_COLORS)]
        child = nested
        nested = (
            f'{indent}<body name="pole{index}" pos="{body_pos}">\n'
            f'{indent}  <joint name="hinge{index}" type="hinge" axis="0 1 0" '
            'damping="0.02" armature="0.02"/>\n'
            f'{indent}  <geom name="pole{index}_geom" type="capsule" '
            f'fromto="0 0 0 0 0 {POLE_LENGTH:g}" size="{radius:g}" '
            f'rgba="{color}" density="20"/>\n'
            f"{child}"
            f"{indent}</body>\n"
        )
        indent += "  "

    return f"""
<mujoco model="stacked_cart_pole_{pole_count}">
  <compiler angle="radian"/>
  <option timestep="0.01" integrator="RK4" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.02" armature="0.02"/>
    <geom friction="0.7 0.1 0.1"/>
  </default>
  <worldbody>
    <light pos="0 -4 6" dir="0 1 -1"/>
    <camera name="track" mode="trackcom" pos="0 -6 2.1" xyaxes="1 0 0 0 0 1"/>
    <geom name="floor" type="plane" pos="0 0 -0.02" size="4 2 0.02" rgba="0.9 0.9 0.9 1"/>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-3 3" damping="0.2"/>
      <geom name="cart_box" type="box" size="0.22 0.14 0.08" rgba="0.1 0.2 0.7 1" density="500"/>
{nested.rstrip()}
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_force" joint="slider" gear="500" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
""".strip()


TRIPLE_CART_POLE_XML = make_stacked_cart_pole_xml(DEFAULT_POLE_COUNT)


class MultiCartPoleEnv(gymnasium.Env):
    """Gymnasium-style MuJoCo env with one cart and N serial pole links."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        *,
        pole_count: int = DEFAULT_POLE_COUNT,
        max_steps: int = 300,
        reset_noise: float = 0.003,
        cart_limit: float = 2.4,
        angle_limit: float = 0.35,
        render_mode: str | None = None,
    ) -> None:
        _validate_pole_count(pole_count)
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if reset_noise < 0.0:
            raise ValueError("reset_noise must be at least 0.0")
        if cart_limit <= 0.0:
            raise ValueError("cart_limit must be greater than 0.0")
        if angle_limit <= 0.0:
            raise ValueError("angle_limit must be greater than 0.0")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")

        self.pole_count = pole_count
        self.model = mujoco.MjModel.from_xml_string(make_stacked_cart_pole_xml(pole_count))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.reset_noise = reset_noise
        self.cart_limit = cart_limit
        self.angle_limit = angle_limit
        self.render_mode = render_mode
        self.elapsed_steps = 0
        self.mujoco_renderer: MujocoRenderer | None = None

        high = np.full(self.model.nq + self.model.nv, np.inf, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def global_pole_angles(self) -> np.ndarray:
        return np.cumsum(self.data.qpos[1 : 1 + self.pole_count]).astype(np.float32)

    def _is_terminated(self) -> bool:
        return bool(
            abs(float(self.data.qpos[0])) > self.cart_limit
            or np.max(np.abs(self.global_pole_angles())) > self.angle_limit
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        noise = self.reset_noise if options is None else options.get("reset_noise", self.reset_noise)
        if noise < 0.0:
            raise ValueError("reset_noise must be at least 0.0")

        self.data.qpos[:] = self.np_random.uniform(-noise, noise, self.model.nq)
        self.data.qpos[0] = self.np_random.uniform(-0.01, 0.01)
        self.data.qvel[:] = self.np_random.uniform(-noise, noise, self.model.nv)
        self.data.ctrl[:] = 0.0
        self.elapsed_steps = 0
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._info()

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, ...],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if clipped_action.shape != (1,):
            raise ValueError("action must have shape (1,)")

        self.data.ctrl[0] = float(clipped_action[0])
        mujoco.mj_step(self.model, self.data)
        self.elapsed_steps += 1
        terminated = self._is_terminated()
        truncated = self.elapsed_steps >= self.max_steps
        reward = 1.0 if not terminated else 0.0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        return {
            "pole_count": self.pole_count,
            "cart_position": float(self.data.qpos[0]),
            "hinge_angles": tuple(float(value) for value in self.data.qpos[1 : 1 + self.pole_count]),
            "global_pole_angles": tuple(float(value) for value in self.global_pole_angles()),
            "episode_length": self.elapsed_steps,
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            ensure_mujoco_human_viewer_compatibility()
        if self.mujoco_renderer is None:
            self.mujoco_renderer = MujocoRenderer(
                self.model,
                self.data,
                default_cam_config={"distance": 4.0, "elevation": -15.0},
            )
        return self.mujoco_renderer.render(self.render_mode)

    def close(self) -> None:
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
            self.mujoco_renderer = None


class TripleCartPoleEnv(MultiCartPoleEnv):
    """Backward-compatible three-link stacked cart-pole environment."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(pole_count=DEFAULT_POLE_COUNT, **kwargs)


__all__ = [
    "DEFAULT_POLE_COUNT",
    "MultiCartPoleEnv",
    "POLE_LENGTH",
    "TRIPLE_CART_POLE_XML",
    "TripleCartPoleEnv",
    "make_stacked_cart_pole_xml",
]
