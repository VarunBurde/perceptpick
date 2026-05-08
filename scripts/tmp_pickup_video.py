"""TMP — PyBullet pickup-simulation videos in the eval visualization style.

Throwaway helper for README / project-website material. For each object,
locates a (scene, image) row in the pose CSV, then renders a short clip
mirroring the Stage C evaluation's visual scheme:

  * **green mesh** at the ground-truth pose (physics target — gripper
    actually picks this one up)
  * **red mesh** at the pose-estimator pose (background, collision-free)
  * world-frame X/Y/Z axes drawn via debug lines
  * camera framed slightly zoomed out so both poses + axes stay in view

Per (object, scene) the script chains a few precomputed grasps through
``execute_grasp_with_gravity_control`` (close → settle → enable gravity →
lift). Grasps are anchored to the GT pose, matching the Stage C "GT mode"
condition in ``perceptpick.eval.pick._run_mode``.

Usage:
    pixi run python scripts/tmp_pickup_video.py \\
        --objects 5,8,11,14 --est-mesh BakedSDF --pose-csv FoundationPose.csv \\
        --n-grasps 3 --fps 30
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pybullet as pb

from perceptpick.configs import GRIPPER_MAX_WIDTHS, SimulationConfig, YCB_OBJECTS
from perceptpick.core import Grasp, ObjectInstance, ObjectType, Scene
from perceptpick.datasets.scene_loader import SceneGenerator
from perceptpick.eval.ranker import best_gripper_for
from perceptpick.grippers import GRIPPER_REGISTRY, get_gripper_class
from perceptpick.paths import resolve_paths
from perceptpick.sim import GraspSimulatorWithGravityControl
from perceptpick.sim.viewport import draw_world_frame

_log = logging.getLogger(__name__)

DEFAULT_OBJECTS = [5, 8, 11, 13, 14, 21]
DEFAULT_GT_MESH = "GT"
DEFAULT_EST_MESH = "BakedSDF"
DEFAULT_POSE_CSV = "FoundationPose.csv"

GT_COLOR = [0.10, 0.85, 0.10, 1.0]
EST_COLOR = [0.95, 0.15, 0.15, 0.55]  # semi-transparent so grippers are still visible


def _parse_int_list(s: str | None, default: list[int]) -> list[int]:
    return [int(x) for x in s.split(",")] if s else list(default)


def _strip_mtl_refs(src_obj: Path, tmp_dir: Path, tag: str) -> Path:
    """Write a copy of ``src_obj`` with ``mtllib`` / ``usemtl`` lines removed.

    PyBullet's ``changeVisualShape(rgbaColor=...)`` only *tints* the diffuse
    texture if one is bound via the OBJ's MTL. Stripping the MTL refs makes
    the rgba fully apply (flat green / flat red) rather than texture-tinted.
    Geometry is unchanged.
    """
    out = tmp_dir / f"{tag}.obj"
    text = src_obj.read_text()
    cleaned = "\n".join(
        line for line in text.splitlines()
        if not (line.lstrip().startswith("mtllib") or line.lstrip().startswith("usemtl"))
    )
    out.write_text(cleaned)
    return out


def _patch_urdf_absolute(
    urdf_src: Path, mesh_visual: Path, vhacd_path: Path, tmp_dir: Path, tag: str
) -> Path:
    """Rewrite ``<visual>`` to point at ``mesh_visual`` and ``<collision>``
    to the VHACD, both as absolute paths so the URDF stays resolvable when
    we move it into a tmpdir.
    """
    text = urdf_src.read_text()
    coll_start, coll_end = text.find("<collision>"), text.find("</collision>")
    if coll_start != -1 and coll_end != -1:
        new_collision = (
            "<collision>\n"
            "      <geometry>\n"
            f'        <mesh filename="{vhacd_path}" scale="1.0 1.0 1.0"/>\n'
            "      </geometry>\n"
            "      <origin xyz=\"0 0 0\"/>\n"
            "      <contact_coefficients mu=\"0.5\" />\n"
            "    "
        )
        text = text[:coll_start] + new_collision + text[coll_end:]

    vis_start, vis_end = text.find("<visual>"), text.find("</visual>")
    if vis_start != -1 and vis_end != -1:
        new_visual = (
            "<visual>\n"
            "      <geometry>\n"
            f'        <mesh filename="{mesh_visual}" scale="1.0 1.0 1.0"/>\n'
            "      </geometry>\n"
            "      <origin xyz=\"0 0 0\"/>\n"
            "    "
        )
        text = text[:vis_start] + new_visual + text[vis_end:]

    out = tmp_dir / f"{tag}.urdf"
    out.write_text(text)
    return out


def _build_object_instance(
    object_id: int,
    pose: np.ndarray,
    asset_dir: Path,
    tmp_dir: Path,
    tag: str,
    enable_physics: bool,
    color: list[float],
) -> ObjectInstance:
    obj_name = YCB_OBJECTS[object_id]
    urdf_src = asset_dir / "urdf" / f"obj_{object_id:06d}.urdf"
    vhacd_path = asset_dir / "vhacd" / f"obj_{object_id:06d}_vhacd.obj"
    mesh_textured = asset_dir / "meshes" / f"obj_{object_id:06d}.obj"
    for p in (urdf_src, vhacd_path, mesh_textured):
        if not p.exists():
            raise FileNotFoundError(p)

    # Strip MTL refs so changeVisualShape(rgbaColor=...) renders flat color,
    # not the texture tinted by it.
    mesh_no_tex = _strip_mtl_refs(mesh_textured, tmp_dir, tag=f"{tag}_visual")
    urdf_patched = _patch_urdf_absolute(urdf_src, mesh_no_tex, vhacd_path, tmp_dir, tag)
    ot = ObjectType(
        # Distinct identifier so scene tracking doesn't dedupe GT and EST.
        identifier=f"{obj_name}__{tag}",
        name=obj_name,
        mesh_fn=str(vhacd_path),
        urdf_fn=str(urdf_patched),
        vhacd_fn=str(vhacd_path),
        mass=0.1 if enable_physics else 0.0,
        friction_coeff=0.5 if enable_physics else 0.0,
    )
    inst = ObjectInstance(ot, pose=pose)
    inst._enable_physics = enable_physics
    inst._color = color
    return inst


def _apply_color(p_client, body_id: int, color: list[float]) -> None:
    """``GraspSimulator.add_object`` only forwards ``_color`` to PyBullet when
    ``verbose=True``. We render headless, so reapply it directly."""
    p_client.changeVisualShape(body_id, -1, rgbaColor=color)


def _setup_camera(p_client, target_world: np.ndarray, width: int, height: int):
    """Camera looks at the GT-pose location, zoomed out enough that both GT
    + EST meshes and the world-frame axes at origin stay in frame."""
    view = p_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[float(target_world[0]),
                              float(target_world[1]),
                              float(target_world[2])],
        distance=0.95,
        yaw=42.0,
        pitch=-28.0,
        roll=0.0,
        upAxisIndex=2,
    )
    proj = p_client.computeProjectionMatrixFOV(
        fov=50.0, aspect=width / height, nearVal=0.01, farVal=10.0
    )
    return view, proj


def _capture_frame(p_client, view, proj, width: int, height: int) -> np.ndarray:
    _, _, rgba, _, _ = p_client.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_TINY_RENDERER,
        flags=pb.ER_NO_SEGMENTATION_MASK,
    )
    arr = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    return arr[..., :3]


def _encode_mp4(frames: list[np.ndarray], mp4_path: Path, fps: int) -> bool:
    if not frames or shutil.which("ffmpeg") is None:
        if not frames:
            _log.warning("no frames to encode")
        else:
            _log.warning("ffmpeg not on PATH; cannot encode mp4")
        return False
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    return proc.wait() == 0


def _resolve_pose_csv(paths, dataset: str, est_mesh: str, pose_csv_name: str) -> Path | None:
    candidates = [
        paths.assets_root / dataset / est_mesh / "pose_estimates" / pose_csv_name,
        paths.dataset_root / dataset / "methods_poses" / pose_csv_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    _log.error("pose CSV %r not found in:\n  %s", pose_csv_name,
               "\n  ".join(str(c) for c in candidates))
    return None


def _pick_scene_image_for_object(scene_gen: SceneGenerator, obj_id: int) -> tuple[int, int] | None:
    df = scene_gen.pose_estimates
    rows = df[df["obj_id"] == obj_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return int(row["scene_id"]), int(row["im_id"])


def _select_grasps(grasp_json_path: Path, n_grasps: int, rng: random.Random) -> list[dict]:
    """Pick up to n_grasps grasps for the video, prioritising successful, then
    padding from other categories so we get a richer demo even when the pool
    of successful grasps is small.
    """
    if not grasp_json_path.exists():
        return []
    cats = json.loads(grasp_json_path.read_text()).get("categorized_grasps", {})
    order = ("successful", "collision_target", "no_contact", "slipped", "error")
    chosen: list[dict] = []
    for cat in order:
        items = list(cats.get(cat, []))
        rng.shuffle(items)
        for it in items:
            chosen.append(it)
            if len(chosen) >= n_grasps:
                return chosen
    return chosen


def render_object_video(
    object_id: int,
    gripper: str,
    gt_mesh: str,
    est_mesh: str,
    pose_csv_path: Path,
    n_grasps: int,
    fps: int,
    width: int,
    height: int,
    paths,
    dataset: str,
    out_dir: Path,
    seed: int,
) -> Path | None:
    obj_name = YCB_OBJECTS.get(object_id)
    if obj_name is None:
        _log.warning("unknown object id %d", object_id)
        return None

    grasp_json_path = paths.output_root / "grasp_poses" / gt_mesh / obj_name / f"{gripper}.json"
    if not grasp_json_path.exists():
        _log.warning("no grasp json at %s", grasp_json_path)
        return None
    rng = random.Random(seed + object_id * 1000 + hash(gripper) % 1000)
    chosen = _select_grasps(grasp_json_path, n_grasps, rng)
    if not chosen:
        _log.warning("no grasps in any category for %s × %s × %s", gt_mesh, obj_name, gripper)
        return None

    bop_root = paths.dataset_root / dataset / "test"
    if not bop_root.is_dir():
        _log.warning("BOP test root missing at %s", bop_root)
        return None
    scene_gen = SceneGenerator(bop_root, pose_csv_path)
    pick = _pick_scene_image_for_object(scene_gen, object_id)
    if pick is None:
        _log.warning("no pose CSV row for obj %d", object_id)
        return None
    scene_id, image_id = pick

    try:
        T_m2w_gt, T_c2w, _ = scene_gen.load_ground_truth_pose(scene_id, image_id, object_id)
        T_m2w_est, _ = scene_gen.load_estimated_pose(scene_id, image_id, object_id)
    except Exception as e:
        _log.warning("pose load failed for obj=%d (scene=%d, image=%d): %s",
                     object_id, scene_id, image_id, e)
        return None

    gt_asset_dir = paths.assets_root / dataset / gt_mesh
    est_asset_dir = paths.assets_root / dataset / est_mesh

    gripper_cls = get_gripper_class(gripper)
    max_w = GRIPPER_MAX_WIDTHS.get(gripper, 0.12)
    sim_cfg = SimulationConfig()
    capture_every_n_steps = max(1, int(240 / fps))  # simulator runs at 240Hz

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        gt_inst = _build_object_instance(
            object_id, T_m2w_gt, gt_asset_dir, tmp_dir,
            tag=f"gt_obj_{object_id:06d}", enable_physics=True, color=GT_COLOR,
        )
        est_inst = _build_object_instance(
            object_id, T_m2w_est, est_asset_dir, tmp_dir,
            tag=f"est_obj_{object_id:06d}", enable_physics=False, color=EST_COLOR,
        )

        scene = Scene()
        scene.objects.append(gt_inst)
        scene.bg_objects.append(est_inst)

        sim = GraspSimulatorWithGravityControl(
            scene, verbose=False, simulation_config=asdict(sim_cfg), gui=False
        )
        try:
            view, proj = _setup_camera(sim._p, T_m2w_gt[:3, 3], width, height)
            frames: list[np.ndarray] = []
            counter = [0]

            def step_capture():
                if counter[0] % capture_every_n_steps == 0:
                    frames.append(_capture_frame(sim._p, view, proj, width, height))
                counter[0] += 1

            for i, grasp_dict in enumerate(chosen):
                if i > 0:
                    sim.scene = scene
                    sim.reset_for_next_grasp()

                # Re-apply colors + redraw the world frame after every reset
                # (resetSimulation clears debug items and PyBullet only forwards
                # _color when verbose=True). Note: body_id 0 is valid, so a
                # plain ``a or b`` fallback drops it as falsy — explicit `in`
                # checks instead.
                for inst, color in ((gt_inst, GT_COLOR), (est_inst, EST_COLOR)):
                    if inst in sim._moving_bodies:
                        body_id = sim._moving_bodies[inst]
                    elif inst in sim._env_bodies:
                        body_id = sim._env_bodies[inst]
                    else:
                        continue
                    _apply_color(sim._p, body_id, color)
                draw_world_frame(sim._p, axis_len=0.15, line_width=3.0)

                sim.register_step_func(step_capture)

                g = Grasp()
                # Anchor grasp to GT pose (matches Stage C "GT mode").
                g.pose = T_m2w_gt @ np.asarray(grasp_dict["pose"])

                sim.step(seconds=0.15)
                try:
                    sim.execute_grasp_with_gravity_control(
                        gripper_cls, g, gt_inst, max_gripper_width=max_w
                    )
                except Exception as e:
                    _log.warning("grasp %d failed mid-execution: %s", i, e)

                try:
                    sim.step(seconds=0.2)
                except Exception:
                    pass

                sim.unregister_step_func(step_capture)
        finally:
            sim.dismiss()

    if not frames:
        _log.warning("no frames captured for %s", obj_name)
        return None

    mp4_path = out_dir / f"{obj_name}__{gripper}__GTvs{est_mesh}__{len(chosen)}grasps.mp4"
    if _encode_mp4(frames, mp4_path, fps):
        print(f"  wrote {mp4_path.name}  "
              f"({len(frames)} frames @ {fps} fps, "
              f"scene={scene_id}/img={image_id}, {gripper})")
        return mp4_path
    return None


def _select_grippers(args, rankings: dict, obj_name: str) -> list[str]:
    if args.all_grippers:
        return sorted(GRIPPER_REGISTRY.keys())
    if args.grippers:
        return [x.strip() for x in args.grippers.split(",") if x.strip()]
    g = best_gripper_for(rankings, obj_name)
    return [g] if g else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--objects", default=None,
                        help=f"comma-separated YCB ids (default: {DEFAULT_OBJECTS})")
    parser.add_argument("--gt-mesh", default=DEFAULT_GT_MESH,
                        help=f"mesh source for GT (green) mesh (default: {DEFAULT_GT_MESH})")
    parser.add_argument("--est-mesh", default=DEFAULT_EST_MESH,
                        help=f"mesh source for EST (red) mesh (default: {DEFAULT_EST_MESH})")
    parser.add_argument("--pose-csv", default=DEFAULT_POSE_CSV,
                        help=f"pose-estimator CSV filename (default: {DEFAULT_POSE_CSV})")
    parser.add_argument("--all-grippers", action="store_true",
                        help="render one video per gripper in GRIPPER_REGISTRY for each object")
    parser.add_argument("--grippers", default=None,
                        help="comma-separated gripper names; overrides best-gripper auto pick")
    parser.add_argument("--n-grasps", type=int, default=20,
                        help="grasps per video; pads from non-successful categories if pool is small")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="ycbv")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--assets-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    objects = _parse_int_list(args.objects, DEFAULT_OBJECTS)
    paths = resolve_paths(args.dataset_root, args.output_root, args.assets_root)
    pose_csv_path = _resolve_pose_csv(paths, args.dataset, args.est_mesh, args.pose_csv)
    if pose_csv_path is None:
        return
    rankings_path = paths.output_root / "grasp_poses" / args.gt_mesh / "gripper_rankings.json"
    rankings = json.loads(rankings_path.read_text()) if rankings_path.exists() else {}
    out_dir = paths.output_root / "visualization" / "videos" / f"{args.gt_mesh}_vs_{args.est_mesh}"

    combos: list[tuple[int, str]] = []
    for obj_id in objects:
        obj_name = YCB_OBJECTS.get(obj_id)
        if obj_name is None:
            continue
        for gripper in _select_grippers(args, rankings, obj_name):
            if gripper not in GRIPPER_REGISTRY:
                _log.warning("unknown gripper %r, skipping", gripper)
                continue
            combos.append((obj_id, gripper))

    print(f"rendering {len(combos)} pickup video(s) — green={args.gt_mesh}, "
          f"red={args.est_mesh}, poses={pose_csv_path.name}, n_grasps={args.n_grasps}")
    n_ok = n_skip = 0
    for obj_id, gripper in combos:
        try:
            p = render_object_video(
                obj_id, gripper, args.gt_mesh, args.est_mesh, pose_csv_path,
                args.n_grasps, args.fps, args.width, args.height,
                paths, args.dataset, out_dir, args.seed,
            )
            if p is None:
                n_skip += 1
            else:
                n_ok += 1
        except Exception as e:
            _log.error("video failed for obj=%d gripper=%s: %s", obj_id, gripper, e)
            n_skip += 1

    print(f"\ndone. ok={n_ok} skipped={n_skip} → {out_dir}")


if __name__ == "__main__":
    main()
