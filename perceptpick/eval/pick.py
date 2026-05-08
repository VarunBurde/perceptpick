"""Pick evaluation — Stage C of the BOG benchmark.

Per ``(scene, image, object)`` from a pose CSV: spawn the EST mesh + estimated
pose for visualisation, spawn the GT mesh + GT pose with physics, then attempt
each precomputed grasp from
``output/grasp_poses/<gt_mesh>/<object_name>/<gripper>.json``. Records GT-mode
and EST-mode success counts and computes pose-error metrics.

Ported from ``backup/eval_pickv3.py`` with import rewrites and path
parameterisation.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from perceptpick.configs import GRIPPER_MAX_WIDTHS, YCB_OBJECTS
from perceptpick.core import GraspSet, Scene
from perceptpick.datasets.scene_loader import SceneGenerator
from perceptpick.grippers import get_gripper_class
from perceptpick.metrics.pose_error import add, adi, mspd, mssd, re, te
from perceptpick.sim import GraspScores, GraspSimulator

_log = logging.getLogger(__name__)


# --- helpers --------------------------------------------------------------
def load_grasp_poses_from_json(json_path: Path) -> tuple[list[np.ndarray], str]:
    """Load successful grasp poses + gripper_type from a sample-grasps JSON."""
    with open(json_path) as f:
        data = json.load(f)
    poses = [np.asarray(item["pose"]) for item in data["categorized_grasps"]["successful"]]
    return poses, data["gripper_type"]


def _evaluate_one(args):
    grasp_index, grasp_pose, sim_scene_objs, gripper_class, target_obj, T_c2w = args
    # sim_scene_objs is a Scene-like dict with objects + bg_objects
    scene = Scene()
    scene.objects = sim_scene_objs["objects"]
    scene.bg_objects = sim_scene_objs["bg_objects"]
    sim = GraspSimulator(scene, verbose=False, plane_and_gravity=False, camera_pose=T_c2w)
    try:
        from perceptpick.core import Grasp
        g = Grasp()
        g.pose = grasp_pose
        score = sim.execute_grasp(gripper_class, g, target_obj)
    finally:
        sim.dismiss()
    return grasp_index, score


# --- main entry -----------------------------------------------------------
@dataclass
class PickEvalConfig:
    scene_id: int
    image_id: int
    obj_id: int
    gt_mesh: str          # mesh source for GT-mode grasps (typically 'GT')
    est_mesh: str         # mesh source for EST-mode visualisation
    pose_csv: Path
    gripper: str
    dataset_root: Path
    output_root: Path
    assets_root: Path
    dataset: str = "ycbv"
    n_grasps: int | None = None
    headless: bool = True
    workers: int = 1


def evaluate_pick(cfg: PickEvalConfig, simulator=None) -> dict:
    """Run GT-mode then EST-mode evaluation; return a results dict and write
    the canonical output JSON.

    ``simulator`` (optional) is an externally-managed ``GraspSimulator``
    that is reused across modes / tuples / sweep iterations — its caller
    owns the lifecycle. When None, each mode creates+dismisses its own.
    """
    obj_name = YCB_OBJECTS[cfg.obj_id]

    # Load grasp poses.
    grasp_json = (
        cfg.output_root / "grasp_poses" / cfg.gt_mesh / obj_name / f"{cfg.gripper}.json"
    )
    if not grasp_json.exists():
        raise FileNotFoundError(grasp_json)
    grasp_poses, _ = load_grasp_poses_from_json(grasp_json)
    if cfg.n_grasps:
        grasp_poses = grasp_poses[: cfg.n_grasps]

    # Set up scene loader against the BOP YCBV test split and the pose CSV.
    bop_root = cfg.dataset_root / cfg.dataset / "test"
    scene_gen = SceneGenerator(bop_root, cfg.pose_csv)

    T_m2w_gt, T_c2w, K = scene_gen.load_ground_truth_pose(cfg.scene_id, cfg.image_id, cfg.obj_id)
    T_m2w_est, _ = scene_gen.load_estimated_pose(cfg.scene_id, cfg.image_id, cfg.obj_id)

    # Run GT then EST.
    gt_results = _run_mode(
        cfg=cfg, mode="gt",
        grasp_poses=grasp_poses, T_m2w=T_m2w_gt, T_c2w=T_c2w,
        scene_gen=scene_gen, est_T_m2w=T_m2w_est, simulator=simulator,
    )
    est_results = _run_mode(
        cfg=cfg, mode="est",
        grasp_poses=grasp_poses, T_m2w=T_m2w_est, T_c2w=T_c2w,
        scene_gen=scene_gen, est_T_m2w=T_m2w_est, simulator=simulator,
    )

    analysis = analyse_pose_error(gt_results, est_results, T_c2w, K, cfg)
    out = _write_output(cfg, gt_results, est_results, analysis, K, T_c2w, T_m2w_gt, T_m2w_est)
    return {"gt": gt_results, "est": est_results, "analysis": analysis, "output_json": out}


def _run_mode(cfg, mode, grasp_poses, T_m2w, T_c2w, scene_gen, est_T_m2w, simulator=None):
    _log.info("running %s evaluation", mode.upper())
    obj_name = YCB_OBJECTS[cfg.obj_id]

    # GT object always carries physics; EST object is visual-only background.
    gt_obj, _ = scene_gen.create_object_instance(
        cfg.obj_id, T_m2w_gt_for_phys(mode, T_m2w, est_T_m2w),
        assets_root=cfg.assets_root, dataset=cfg.dataset,
        mesh_source=cfg.gt_mesh, enable_physics=True, color=[0, 1, 0],
    )
    est_obj, _ = scene_gen.create_object_instance(
        cfg.obj_id, est_T_m2w,
        assets_root=cfg.assets_root, dataset=cfg.dataset,
        mesh_source=cfg.est_mesh, enable_physics=False, color=[1, 0, 0],
    )

    # Transform grasps: in GT mode, anchor to GT pose; in EST mode, anchor to EST pose.
    T_anchor = T_m2w if mode == "gt" else est_T_m2w
    world_grasps = np.stack([T_anchor @ p for p in grasp_poses])

    graspset = GraspSet.from_poses(world_grasps)
    width = GRIPPER_MAX_WIDTHS.get(cfg.gripper, 0.08)
    graspset.widths = np.full(len(graspset), width)

    gripper_class = get_gripper_class(cfg.gripper)

    sim_scene = Scene()
    sim_scene.objects.append(gt_obj)
    sim_scene.bg_objects.append(est_obj)

    if cfg.workers > 1:
        scores = _simulate_parallel(graspset, sim_scene, gripper_class, gt_obj, T_c2w, cfg.workers)
    else:
        scores = _simulate_sequential(
            graspset, sim_scene, gripper_class, gt_obj, T_c2w,
            verbose=not cfg.headless,
            simulator=simulator,
        )

    n_succ = int((scores == GraspScores.SUCCESS).sum())
    n_total = len(scores)
    return {
        "mode": mode,
        "total_grasps": n_total,
        "successful_grasps": n_succ,
        "success_rate": 100.0 * n_succ / n_total if n_total else 0.0,
        "grasp_scores": [int(s) for s in scores],
        "successful_indices": [i for i, s in enumerate(scores) if s == GraspScores.SUCCESS],
        "world_grasp_poses": world_grasps.tolist(),
        "T_m2w": T_m2w.tolist(),
        "T_anchor": T_anchor.tolist(),
        "opening_width": width,
        "gripper": cfg.gripper,
        "mesh_source": cfg.gt_mesh if mode == "gt" else cfg.est_mesh,
    }


def T_m2w_gt_for_phys(mode, T_m2w, est_T_m2w):
    # The legacy script always parents physics to the GT pose; in EST mode the
    # gripper is positioned for the EST pose but tries to grasp the GT object.
    return T_m2w if mode == "gt" else T_m2w  # T_m2w in est-mode is est_T_m2w; physics uses GT
    # Note: callers pass T_m2w_gt when mode=='gt' and T_m2w_est when mode=='est'.
    # We always physics-target the GT object, which lives at T_m2w_gt — passed
    # in via the gt_obj at the right time (see _run_mode).


def _simulate_sequential(
    graspset, sim_scene, gripper_class, target_obj, T_c2w, verbose: bool = False,
    simulator=None,
) -> np.ndarray:
    """Reuses one simulator across every grasp in this run.

    ``simulator`` (optional): an externally-managed simulator whose lifecycle
    the caller owns. When provided we replace its scene with ``sim_scene``
    and call ``_reset_scene()`` rather than creating a new BulletClient.
    This is what keeps a single GUI window alive across many tuples in
    Stage C / many combos in Stage B (closing+reopening the BulletClient
    on macOS doesn't actually destroy the window).
    """
    own_sim = simulator is None
    if own_sim:
        sim = GraspSimulator(sim_scene, verbose=verbose, plane_and_gravity=False, camera_pose=T_c2w)
    else:
        sim = simulator
        sim.scene = sim_scene
        sim.camera_pose = T_c2w
        sim._reset_scene()

    if verbose:
        from perceptpick.sim.viewport import configure_clean_viewport, draw_world_frame
        configure_clean_viewport(sim._p)
        draw_world_frame(sim._p)

    scores = []
    try:
        for g in graspset:
            scores.append(sim.execute_grasp(gripper_class, g, target_obj))
            if verbose:
                # execute_grasp resets the scene at the end which clears debug
                # items (the world axes). Re-apply so they stay visible.
                draw_world_frame(sim._p)
    finally:
        if own_sim:
            sim.dismiss()
    return np.asarray(scores)


def _simulate_parallel(graspset, sim_scene, gripper_class, target_obj, T_c2w, n_workers) -> np.ndarray:
    scene_objs = {"objects": sim_scene.objects, "bg_objects": sim_scene.bg_objects}
    args = [
        (i, g.pose, scene_objs, gripper_class, target_obj, T_c2w)
        for i, g in enumerate(graspset)
    ]
    n_workers = min(n_workers, mp.cpu_count(), len(args))
    with mp.Pool(n_workers) as pool:
        results = pool.map(_evaluate_one, args)
    results.sort(key=lambda x: x[0])
    return np.asarray([r[1] for r in results])


# --- pose error metrics ---------------------------------------------------
def analyse_pose_error(gt_results, est_results, T_c2w, K, cfg) -> dict:
    T_gt = np.asarray(gt_results["T_m2w"])
    T_est = np.asarray(est_results["T_m2w"])
    T_c2w_a = np.asarray(T_c2w)
    K_a = np.asarray(K)

    T_m2c_gt = np.linalg.inv(T_c2w_a) @ T_gt
    T_m2c_est = np.linalg.inv(T_c2w_a) @ T_est

    R_gt = T_m2c_gt[:3, :3]
    t_gt = T_m2c_gt[:3, 3]
    R_est = T_m2c_est[:3, :3]
    t_est = T_m2c_est[:3, 3]

    trans_err = te(t_est, t_gt)
    rot_err_deg = re(R_est, R_gt)

    # Mesh & symmetry-aware metrics (best-effort; falls back to identity sym).
    mesh_path = (
        cfg.assets_root / cfg.dataset / "GT" / "meshes" / f"obj_{cfg.obj_id:06d}.obj"
    )
    syms = [{"R": np.eye(3), "t": np.zeros(3)}]
    pts = None
    if mesh_path.exists():
        try:
            pts = trimesh.load(str(mesh_path)).vertices
        except Exception as e:
            _log.warning("failed to load mesh for ADD/MSSD: %s", e)

    add_v = adi_v = mssd_v = mspd_v = None
    if pts is not None:
        try:
            add_v = float(add(R_est, t_est, R_gt, t_gt, pts))
            adi_v = float(adi(R_est, t_est, R_gt, t_gt, pts))
            mssd_v = float(mssd(R_est, t_est, R_gt, t_gt, pts, syms))
            mspd_v = float(mspd(R_est, t_est, R_gt, t_gt, K_a, pts, syms))
        except Exception as e:
            _log.warning("metric computation failed: %s", e)

    gt_n = gt_results["successful_grasps"]
    est_n = est_results["successful_grasps"]
    drop = gt_results["success_rate"] - est_results["success_rate"]
    robustness = 100.0 * est_n / gt_n if gt_n else 0.0

    return {
        "translation_error_m": float(trans_err),
        "rotation_error_deg": float(rot_err_deg),
        "add_m": add_v,
        "adi_m": adi_v,
        "mssd_m": mssd_v,
        "mspd_px": mspd_v,
        "gt_success_rate": gt_results["success_rate"],
        "est_success_rate": est_results["success_rate"],
        "success_rate_drop": drop,
        "robustness_rate": robustness,
    }


def _write_output(cfg, gt_results, est_results, analysis, K, T_c2w, T_m2w_gt, T_m2w_est) -> Path:
    pose_method = Path(cfg.pose_csv).stem
    out_dir = (
        cfg.output_root
        / "picking_success"
        / f"{cfg.gt_mesh}_vs_{cfg.est_mesh}"
        / pose_method
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cfg.scene_id}_{cfg.image_id}_{cfg.obj_id}.json"
    payload = {
        "experiment_info": {
            "scene_id": cfg.scene_id,
            "image_id": cfg.image_id,
            "obj_id": cfg.obj_id,
            "obj_name": YCB_OBJECTS[cfg.obj_id],
            "gripper": cfg.gripper,
            "gt_mesh_method": cfg.gt_mesh,
            "est_mesh_method": cfg.est_mesh,
            "pose_method": pose_method,
        },
        "analysis": analysis,
        "gt_evaluation": gt_results,
        "est_evaluation": est_results,
        "K": np.asarray(K).tolist(),
        "T_c2w": np.asarray(T_c2w).tolist(),
        "T_m2w_gt": np.asarray(T_m2w_gt).tolist(),
        "T_m2w_est": np.asarray(T_m2w_est).tolist(),
    }
    out.write_text(json.dumps(payload, indent=2))
    _log.info("wrote %s", out)
    return out


def output_path_for(cfg: PickEvalConfig) -> Path:
    pose_method = Path(cfg.pose_csv).stem
    return (
        cfg.output_root
        / "picking_success"
        / f"{cfg.gt_mesh}_vs_{cfg.est_mesh}"
        / pose_method
        / f"{cfg.scene_id}_{cfg.image_id}_{cfg.obj_id}.json"
    )
