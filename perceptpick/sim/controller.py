"""GraspSimulatorWithGravityControl — extends the base PyBullet simulator
with the gravity-disabled-during-close-then-enabled-during-lift sequence used
in the BOG benchmark.

Ported from ``core/simulator.py``. Behavior unchanged.
"""
from __future__ import annotations

import logging

from . import robots
from . import simulator as sim
from .viewport import configure_clean_viewport, draw_world_frame

_log = logging.getLogger(__name__)


class GraspSimulatorWithGravityControl(sim.GraspSimulator):
    """GraspSimulator subclass with explicit gravity timing.

    Sequence:
      1. Spawn target + gripper with gravity OFF, no ground plane.
      2. Settle, check pre-grasp collisions, close gripper, settle.
      3. Verify contact established (else NO_CONTACT_ESTABLISHED).
      4. Enable gravity, settle (else SLIPPED_DURING_LIFTING).
      5. Lift to target_pos + lifting_height, settle.
      6. Final contact check → SUCCESS or SLIPPED_DURING_LIFTING.

    GUI is requested by passing ``gui=True``; the underlying ``BulletClient``
    opens (and on ``dismiss()``, closes) one window per simulator instance.
    No global ``pybullet.connect`` is used — that would leak windows across
    sequential grasp evaluations.
    """

    def __init__(self, scene, verbose: bool = False, simulation_config: dict | None = None, gui: bool = False):
        # Route GUI through the parent's BulletClient (verbose=True → p.GUI).
        # Each instance owns its connection and closes it via dismiss().
        super().__init__(scene, verbose=verbose or gui, plane_and_gravity=False)
        self._set_gravity_zero()
        self.simulation_config = simulation_config or {}

    def _reset(self, plane_and_gravity=False):
        """Override so the clean viewport is configured the moment the
        BulletClient is created (eliminates the brief flicker of PyBullet's
        default side panels before our config runs)."""
        super()._reset(plane_and_gravity)
        if self.verbose:
            # Always reapply: resetSimulation can revert visualizer flags.
            self._configure_clean_viewport()
            self._draw_world_frame()

    def _configure_clean_viewport(self) -> None:
        configure_clean_viewport(self._p)

    def _draw_world_frame(self, axis_len: float = 0.1, line_width: float = 2.0) -> None:
        draw_world_frame(self._p, axis_len=axis_len, line_width=line_width)

    def reset_for_next_grasp(self) -> None:
        """Reuse this simulator for another grasp without recreating the
        BulletClient. Resets the scene and re-zeros gravity (PyBullet's
        resetSimulation restores defaults). The clean-viewport reapply
        happens automatically via the overridden ``_reset``."""
        self._reset_scene()
        self._set_gravity_zero()

    def _set_gravity_zero(self) -> None:
        self._p.setGravity(0, 0, 0)

    def execute_grasp_with_gravity_control(
        self,
        gripper_type,
        grasp,
        target,
        max_gripper_width: float = 0.14,
        gripper_scale: float = 1.0,
    ):
        cfg = self.simulation_config

        # Decide gripper opening from per-grasp width if available, else cfg default.
        if hasattr(grasp, "width") and grasp.width > 0:
            desired_width = grasp.width * 1.1  # 10% clearance
            gripper_opening_width = min(1.0, max(0.1, desired_width / max_gripper_width))
        else:
            gripper_opening_width = cfg.get("gripper_opening_width", 0.8)

        _log.debug("loading gripper without gravity")
        gripper = gripper_type(self, gripper_scale)
        gripper.load(grasp.pose)
        gripper.set_open_scale(gripper_opening_width)

        robot = self._load_gripper_mount_and_attach(gripper.body_id)
        self.step(seconds=cfg.get("settle_time_no_gravity", 0.1))

        result = self.check_collisions(gripper, target)
        if result != sim.GraspScores.SUCCESS:
            return result

        _log.debug("closing gripper without gravity")
        gripper.close()
        self.step(seconds=cfg.get("settle_time_after_close", 0.2))

        min_contacts = cfg.get("min_contacts_required", 1)
        if not self._contact_established(gripper, target, min_contacts=min_contacts):
            _log.debug("no contact with target")
            return sim.GraspScores.NO_CONTACT_ESTABLISHED

        _log.debug("enabling gravity")
        self._p.setGravity(0, 0, -9.81)
        self.step(seconds=cfg.get("settle_time_with_gravity", 0.3))

        if not self._contact_established(gripper, target, min_contacts=min_contacts):
            _log.debug("slipped after gravity enabled")
            return sim.GraspScores.SLIPPED_DURING_LIFTING

        _log.debug("lifting object")
        target_pos = robot.end_effector_pos()
        target_pos[2] += cfg.get("lifting_height", 0.1)

        try:
            plan = robots.TrajectoryPlanner(self, robot).lin(target_pos)
            arrived = robot.execute_joint_trajectory(plan)
            if not arrived:
                _log.warning("did not arrive at desired target position")
        except Exception as e:
            _log.warning("lift failed: %s", e)
            return sim.GraspScores.SLIPPED_DURING_LIFTING

        if not self._contact_established(gripper, target, min_contacts=min_contacts):
            _log.debug("slipped during lifting")
            return sim.GraspScores.SLIPPED_DURING_LIFTING

        _log.debug("grasp success")
        return sim.GraspScores.SUCCESS
