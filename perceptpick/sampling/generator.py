"""GraspPoseGenerator — orchestrates antipodal sampling + simulation per
(object, gripper) and writes a categorized JSON of outcomes.

Ported from ``core/grasp_generator.py``. Replaces hardcoded paths with
parameters from ``perceptpick.paths`` / argparse, replaces burg imports
with perceptpick equivalents.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from perceptpick.configs import (
    GRIPPER_MAX_WIDTHS,
    SamplingConfig,
    SimulationConfig,
)
from perceptpick.core import ObjectInstance, ObjectType, Scene
from perceptpick.grippers import get_gripper_class
from perceptpick.sampling import AntipodalGraspSampler
from perceptpick.sim import GraspScores, GraspSimulatorWithGravityControl

_log = logging.getLogger(__name__)


def _categorize(score) -> str:
    """Map a ``GraspScores`` enum value to one of our five JSON buckets.

    Only four physics-driven outcomes can fire in this benchmark's config
    (no ground plane, single physics-enabled target):
      - ``successful``       — gripper closed, lifted against gravity, kept contact
      - ``collision_target`` — gripper body collided with the object during approach
      - ``no_contact``       — gripper closed in free space (typical translation-error failure)
      - ``slipped``          — held briefly, lost grip during gravity-enabled lift

    ``error`` is the catch-all for programmatic exceptions and for any
    GraspScores value the simulator shouldn't emit but did (e.g. the dead
    ``COLLISION_WITH_GROUND``/``COLLISION_WITH_CLUTTER`` paths) — surfacing
    the anomaly instead of silently dropping the grasp.
    """
    if score == GraspScores.SUCCESS:
        return "successful"
    if score == GraspScores.COLLISION_WITH_TARGET:
        return "collision_target"
    if score == GraspScores.NO_CONTACT_ESTABLISHED:
        return "no_contact"
    if score == GraspScores.SLIPPED_DURING_LIFTING:
        return "slipped"
    return "error"


_EMPTY_BUCKETS = (
    "successful",
    "collision_target",
    "no_contact",
    "slipped",
    "error",
)


class GraspPoseGenerator:
    """Sample N antipodal grasps for one (object, gripper), simulate each in
    PyBullet, and write a categorized JSON of poses + statistics.
    """

    def __init__(
        self,
        mesh_path: str | Path,
        urdf_path: str | Path,
        vhacd_path: str | Path,
        object_name: str,
        gripper_type: str,
        output_dir: str | Path,
        sampling_cfg: SamplingConfig | None = None,
        simulation_cfg: SimulationConfig | None = None,
        gui: bool = False,
        simulator: GraspSimulatorWithGravityControl | None = None,
    ) -> None:
        self.mesh_path = Path(mesh_path)
        self.urdf_path = Path(urdf_path)
        self.vhacd_path = Path(vhacd_path)
        self.object_name = object_name
        self.gripper_type = gripper_type
        self.gripper_class = get_gripper_class(gripper_type)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sampling_cfg = sampling_cfg or SamplingConfig()
        self.simulation_cfg = simulation_cfg or SimulationConfig()
        self.gui = gui
        # Optional externally-managed simulator. When provided, the generator
        # reuses it (and does NOT dismiss it) — the caller owns the lifecycle.
        # This is how 02_grasp_sweep.py keeps a single GUI window open across
        # the whole (object × gripper) sweep.
        self.external_simulator = simulator

    # --- pipeline ---------------------------------------------------------
    def make_object_type(self) -> ObjectType:
        for p in (self.mesh_path, self.urdf_path, self.vhacd_path):
            if not p.exists():
                raise FileNotFoundError(p)
        # ObjectType (from _burg_compat) accepts these kwargs.
        return ObjectType(
            identifier=self.object_name,
            name=self.object_name,
            mesh_fn=str(self.vhacd_path),  # use VHACD as collision/sampling proxy
            urdf_fn=str(self.urdf_path),
            vhacd_fn=str(self.vhacd_path),
            mass=0.1,
            friction_coeff=0.5,
        )

    def sample(self, instance: ObjectInstance):
        max_w = GRIPPER_MAX_WIDTHS.get(self.gripper_type, self.sampling_cfg.max_gripper_width)
        sampler = AntipodalGraspSampler(
            mu=self.sampling_cfg.mu,
            n_orientations=self.sampling_cfg.n_orientations,
            n_rays=self.sampling_cfg.n_rays,
            max_targets_per_ref_point=self.sampling_cfg.max_targets_per_ref_point,
            only_grasp_from_above=self.sampling_cfg.only_grasp_from_above,
            no_contact_below_z=self.sampling_cfg.no_contact_below_z,
            min_grasp_width=self.sampling_cfg.min_grasp_width,
            verbose=self.sampling_cfg.verbose,
        )
        graspset, contacts = sampler.sample(
            instance, n=self.sampling_cfg.n_grasps, max_gripper_width=max_w
        )
        _log.info("sampled %d candidate grasps", len(graspset))
        return graspset

    def simulate_sequential(self, graspset, instance: ObjectInstance) -> tuple[dict, dict]:
        """Sequential simulation. Reuses a single PyBullet connection across
        every grasp via ``reset_for_next_grasp``. If an external simulator
        was passed at construction time, it is reused (and not dismissed) —
        otherwise one is created locally for this generator and dismissed
        when done.
        """
        buckets = {k: [] for k in _EMPTY_BUCKETS}
        max_w = GRIPPER_MAX_WIDTHS.get(self.gripper_type, 0.12)

        if self.external_simulator is not None:
            sim = self.external_simulator
            # Replace whatever scene was loaded with this combo's instance and
            # refresh PyBullet so bodies from the previous combo are gone.
            sim.scene.objects = [instance]
            sim.scene.bg_objects = []
            sim.reset_for_next_grasp()
            owns_sim = False
        else:
            scene = Scene()
            scene.objects.append(instance)
            sim = GraspSimulatorWithGravityControl(
                scene, verbose=self.gui, simulation_config=asdict(self.simulation_cfg), gui=self.gui
            )
            owns_sim = True

        try:
            for i, grasp in enumerate(graspset):
                if i > 0:
                    sim.reset_for_next_grasp()
                try:
                    score = sim.execute_grasp_with_gravity_control(
                        self.gripper_class, grasp, instance, max_gripper_width=max_w
                    )
                    buckets[_categorize(score)].append(
                        {"pose": grasp.pose.tolist(), "grasp_id": i}
                    )
                except Exception as e:
                    buckets["error"].append(
                        {"pose": grasp.pose.tolist(), "grasp_id": i, "error_message": str(e)}
                    )
        finally:
            if owns_sim:
                sim.dismiss()
        stats = {k: len(v) for k, v in buckets.items()}
        return buckets, stats

    def simulate_parallel(self, graspset, instance: ObjectInstance, n_workers: int | None = None):
        if n_workers is None:
            n_workers = mp.cpu_count()
        chunk_size = max(1, len(graspset) // n_workers)
        chunks = []
        for i in range(0, len(graspset), chunk_size):
            chunks.append([
                {"grasp_id": i + j, "pose": g.pose.tolist()}
                for j, g in enumerate(graspset[i : i + chunk_size])
            ])

        ot = instance.object_type
        ot_data = dict(
            identifier=ot.identifier, name=ot.name, mesh_fn=ot.mesh_fn,
            urdf_fn=ot.urdf_fn, vhacd_fn=ot.vhacd_fn,
            mass=ot.mass, friction_coeff=ot.friction_coeff,
        )
        sim_cfg = asdict(self.simulation_cfg)
        max_w = GRIPPER_MAX_WIDTHS.get(self.gripper_type, 0.12)

        buckets = {k: [] for k in _EMPTY_BUCKETS}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [
                ex.submit(_simulate_chunk, chunk, ot_data, self.gripper_type, sim_cfg, max_w)
                for chunk in chunks
            ]
            for fut in as_completed(futs):
                worker_buckets = fut.result()
                for k, v in worker_buckets.items():
                    buckets[k].extend(v)
        stats = {k: len(v) for k, v in buckets.items()}
        return buckets, stats

    def save(self, buckets: dict, stats: dict) -> Path:
        out = self.output_dir / f"{self.gripper_type}.json"
        max_w = GRIPPER_MAX_WIDTHS.get(self.gripper_type, 0.12)
        payload = {
            "object_name": self.object_name,
            "gripper_type": self.gripper_type,
            "num_successful_grasps": len(buckets["successful"]),
            "total_grasps_simulated": sum(stats.values()),
            "statistics": stats,
            "gripper_specs": {
                "max_opening_width_m": max_w,
                "gripper_class": self.gripper_class.__name__,
            },
            "configuration": {
                "sampling_config": asdict(self.sampling_cfg),
                "simulation_config": asdict(self.simulation_cfg),
            },
            "metadata": {
                "coordinate_frame": "object",
                "pose_format": "4x4_transformation_matrix",
                "units": "meters_and_radians",
                "has_successful_grasps": len(buckets["successful"]) > 0,
            },
            "categorized_grasps": buckets,
        }
        out.write_text(json.dumps(payload, indent=2))
        _log.info("saved %s", out)
        return out

    def run(self) -> Path:
        ot = self.make_object_type()
        instance = ObjectInstance(ot, pose=np.eye(4))
        graspset = self.sample(instance)
        if len(graspset) == 0:
            _log.error("no grasps were sampled!")
            return self.save({k: [] for k in _EMPTY_BUCKETS}, {k: 0 for k in _EMPTY_BUCKETS})

        # In single-combo GUI debug mode, show every candidate in Open3D first
        # so the user can inspect before running PyBullet. Skip in sweep mode
        # (external_simulator in use): on macOS, opening an Open3D window while
        # a PyBullet GUI window is already alive causes the process to exit
        # silently (both fight for the main-thread OpenGL context).
        if self.gui and self.external_simulator is None:
            try:
                from perceptpick.viz.grasps_viewer import preview_candidates
                preview_candidates(
                    self.mesh_path, graspset,
                    title=f"{self.object_name} × {self.gripper_type}: "
                          f"{len(graspset)} candidate(s) — close to simulate",
                )
            except Exception as e:
                _log.warning("Open3D preview failed (%s); proceeding to simulation", e)

        # Use sequential whenever GUI is on or an external simulator is shared
        # (parallel would spawn worker processes that can't share the GUI conn).
        if self.gui or self.external_simulator is not None:
            buckets, stats = self.simulate_sequential(graspset, instance)
        else:
            buckets, stats = self.simulate_parallel(graspset, instance)
        return self.save(buckets, stats)


# --- module-level helper for ProcessPoolExecutor ---------------------------
def _simulate_chunk(chunk, ot_data, gripper_type, sim_cfg, max_w):
    """Worker function — must be importable at module level for pickling."""
    import numpy as np

    from perceptpick.core import ObjectInstance, ObjectType, Scene
    from perceptpick.grippers import get_gripper_class
    from perceptpick.sim import GraspScores, GraspSimulatorWithGravityControl

    ot = ObjectType(**ot_data)
    instance = ObjectInstance(ot, pose=np.eye(4))
    scene = Scene()
    scene.objects.append(instance)
    gripper_cls = get_gripper_class(gripper_type)

    buckets = {k: [] for k in _EMPTY_BUCKETS}
    for entry in chunk:
        gid = entry["grasp_id"]
        pose = np.asarray(entry["pose"])
        # Build a 1-grasp set so we can call execute_grasp_with_gravity_control.
        from perceptpick.core import Grasp
        g = Grasp()
        g.pose = pose
        try:
            sim = GraspSimulatorWithGravityControl(
                scene, verbose=False, simulation_config=sim_cfg, gui=False
            )
            score = sim.execute_grasp_with_gravity_control(gripper_cls, g, instance, max_gripper_width=max_w)
            buckets[_categorize(score)].append({"pose": pose.tolist(), "grasp_id": gid})
        except Exception as e:
            buckets["error"].append({"pose": pose.tolist(), "grasp_id": gid, "error_message": str(e)})
        finally:
            try:
                sim.dismiss()
            except Exception:
                pass
    return buckets
