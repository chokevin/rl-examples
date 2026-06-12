"""Compatibility helpers for Gymnasium's MuJoCo renderer."""

from __future__ import annotations

from collections.abc import Sequence


def solver_iteration_count(data: object) -> int:
    solver_iter = getattr(data, "solver_iter", None)
    if solver_iter is not None:
        return int(solver_iter)

    solver_niter = getattr(data, "solver_niter", None)
    if solver_niter is None:
        raise AttributeError("MuJoCo data has neither solver_iter nor solver_niter")
    if hasattr(solver_niter, "max"):
        return int(solver_niter.max())
    if isinstance(solver_niter, Sequence):
        return int(max(solver_niter)) if solver_niter else 0
    return int(solver_niter)


def ensure_mujoco_human_viewer_compatibility() -> bool:
    from gymnasium.envs.mujoco import mujoco_rendering
    import mujoco

    viewer_class = mujoco_rendering.WindowViewer
    if getattr(viewer_class, "_rl_examples_solver_iter_patch", False):
        return False

    original_create_overlay = viewer_class._create_overlay

    def create_overlay_with_solver_niter(self: object) -> None:
        try:
            original_create_overlay(self)
        except AttributeError as error:
            if "solver_iter" not in str(error) or not hasattr(self.data, "solver_niter"):
                raise
            bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value
            self.add_overlay(
                bottomleft,
                "Solver iterations",
                str(solver_iteration_count(self.data) + 1),
            )
            self.add_overlay(
                bottomleft,
                "Step",
                str(round(self.data.time / self.model.opt.timestep)),
            )
            self.add_overlay(
                bottomleft,
                "timestep",
                "%.5f" % self.model.opt.timestep,
            )

    viewer_class._create_overlay = create_overlay_with_solver_niter
    viewer_class._rl_examples_solver_iter_patch = True
    return True


__all__ = [
    "ensure_mujoco_human_viewer_compatibility",
    "solver_iteration_count",
]
