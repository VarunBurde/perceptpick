"""TMP — batch PNG screenshots of objects + categorized grasp poses.

Throwaway helper for README / project-website material. Not part of the
canonical pipeline. Picks the best gripper per (object, mesh-source) from
``output/grasp_poses/<src>/gripper_rankings.json``, then renders grasps
from every outcome category around the object mesh, color-coded:

  * green   — successful
  * yellow  — collision_target  (gripper hit object before close)
  * blue    — no_contact        (closed in free space)
  * magenta — slipped           (lost contact during lift)
  * gray    — error             (programmatic failure)

Usage:
    pixi run python scripts/tmp_screenshot_grasps.py \\
        --objects 5,11 --methods GT --max-per-category 15
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from perceptpick.configs import YCB_OBJECTS
from perceptpick.eval.ranker import best_gripper_for
from perceptpick.paths import resolve_paths

_log = logging.getLogger(__name__)

DEFAULT_OBJECTS = [2, 3, 5, 8, 11, 13, 14, 21]
DEFAULT_METHODS = ["GT"]

# RGB colors per outcome category — mirrors perceptpick.viz.grasps_viewer.CATEGORY_COLORS.
CATEGORY_COLORS = {
    "successful":       [0.10, 0.85, 0.10],
    "collision_target": [1.00, 0.85, 0.00],
    "no_contact":       [0.10, 0.30, 1.00],
    "slipped":          [1.00, 0.10, 0.85],
    "error":            [0.55, 0.55, 0.55],
}
CATEGORY_ORDER = ("successful", "collision_target", "no_contact", "slipped", "error")


def _parse_int_list(s: str | None, default: list[int]) -> list[int]:
    return [int(x) for x in s.split(",")] if s else list(default)


def _parse_str_list(s: str | None, default: list[str]) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else list(default)


def _select_proportional(categorized: dict, target_total: int, seed: int) -> dict[str, list]:
    """Pick up to ``target_total`` grasps across all categories, sampling
    from each in proportion to its available count. Categories with no
    items contribute nothing; if the total available is < target_total,
    we render everything available.
    """
    import random as _r
    rng = _r.Random(seed)
    available = {c: list(categorized.get(c, [])) for c in CATEGORY_ORDER}
    total_avail = sum(len(v) for v in available.values())
    if total_avail == 0:
        return {c: [] for c in CATEGORY_ORDER}
    if total_avail <= target_total:
        return available  # render everything

    # Proportional allocation, then largest-remainder rounding to hit target.
    raw = {c: target_total * len(available[c]) / total_avail for c in CATEGORY_ORDER}
    quota = {c: int(raw[c]) for c in CATEGORY_ORDER}
    remainder = target_total - sum(quota.values())
    by_frac = sorted(CATEGORY_ORDER, key=lambda c: raw[c] - quota[c], reverse=True)
    for c in by_frac[:remainder]:
        quota[c] += 1

    out = {}
    for c in CATEGORY_ORDER:
        items = available[c]
        rng.shuffle(items)
        out[c] = items[: min(quota[c], len(items))]
    return out


def _render_in_subprocess(
    mesh_path: Path,
    grasp_json: Path,
    out_path: Path,
    target_total: int,
    seed: int,
    width: int,
    height: int,
) -> tuple[bool, dict]:
    """Spawn a fresh Python interpreter for each render. Open3D's
    ``Visualizer(visible=False)`` is unstable on macOS across multiple
    create/destroy cycles in one process; isolating each render avoids
    segfaults at cleanup from cascading.

    Returns (ok, counts_per_category).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path = out_path.with_suffix(".counts.json")

    # Pick the grasps in the parent so we can log counts before rendering.
    raw_data = json.loads(grasp_json.read_text())
    selected = _select_proportional(
        raw_data.get("categorized_grasps", {}), target_total, seed
    )
    selected_path = out_path.with_suffix(".selected.json")
    selected_path.write_text(json.dumps(selected))

    code = f"""
import json
import numpy as np
import open3d as o3d
from perceptpick.grippers import TwoFingerGripperVisualisation

CATEGORY_COLORS = {json.dumps(CATEGORY_COLORS)}
CATEGORY_ORDER = {json.dumps(list(CATEGORY_ORDER))}

mesh = o3d.io.read_triangle_mesh({str(mesh_path)!r}, enable_post_processing=True)
mesh.compute_vertex_normals()
if not (len(mesh.textures) or len(mesh.triangle_material_ids)):
    mesh.paint_uniform_color([0.85, 0.85, 0.85])

selected = json.loads(open({str(selected_path)!r}).read())

proto = TwoFingerGripperVisualisation()
geoms = [mesh]
counts = {{}}
for cat in CATEGORY_ORDER:
    items = selected.get(cat, [])
    counts[cat] = len(items)
    color = CATEGORY_COLORS[cat]
    for item in items:
        g = o3d.geometry.TriangleMesh(proto.mesh)
        g.transform(np.asarray(item["pose"]))
        g.paint_uniform_color(color)
        geoms.append(g)

with open({str(counts_path)!r}, "w") as f:
    json.dump(counts, f)

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width={width}, height={height})
for g in geoms:
    vis.add_geometry(g)
opt = vis.get_render_option()
opt.background_color = np.asarray([1.0, 1.0, 1.0])
opt.light_on = True
opt.mesh_show_back_face = True

ctl = vis.get_view_control()
ctl.set_front([0.6, -0.6, 0.4])
ctl.set_lookat([0.0, 0.0, 0.0])
ctl.set_up([0.0, 0.0, 1.0])
ctl.set_zoom(0.65)

vis.poll_events()
vis.update_renderer()
vis.capture_screen_image({str(out_path)!r}, do_render=True)
vis.destroy_window()
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    ok = out_path.exists() and out_path.stat().st_size > 0
    counts = {}
    if counts_path.exists():
        try:
            counts = json.loads(counts_path.read_text())
        except Exception:
            counts = {}
        counts_path.unlink(missing_ok=True)
    selected_path.unlink(missing_ok=True)
    if not ok:
        _log.error("subprocess render failed (rc=%d): %s",
                   res.returncode, res.stderr.strip().splitlines()[-3:] if res.stderr else "")
    return ok, counts


def render_one(
    object_id: int,
    method: str,
    target_total: int,
    seed: int,
    width: int,
    height: int,
    paths,
    dataset: str,
    out_root: Path,
) -> Path | None:
    obj_name = YCB_OBJECTS.get(object_id)
    if obj_name is None:
        _log.warning("unknown object id %d, skipping", object_id)
        return None

    rankings_path = paths.output_root / "grasp_poses" / method / "gripper_rankings.json"
    if not rankings_path.exists():
        _log.warning("no rankings for method=%s at %s, skipping", method, rankings_path)
        return None
    rankings = json.loads(rankings_path.read_text())
    gripper = best_gripper_for(rankings, obj_name)
    if gripper is None:
        _log.warning("no ranked gripper for %s in %s, skipping", obj_name, method)
        return None

    grasp_json = paths.output_root / "grasp_poses" / method / obj_name / f"{gripper}.json"
    if not grasp_json.exists():
        _log.warning("no grasp json at %s, skipping", grasp_json)
        return None
    data = json.loads(grasp_json.read_text())
    total_in_json = sum(len(data.get("categorized_grasps", {}).get(c, [])) for c in CATEGORY_ORDER)
    if total_in_json == 0:
        _log.warning("no categorized grasps for %s × %s × %s", method, obj_name, gripper)
        return None

    mesh_path = paths.assets_root / dataset / method / "meshes" / f"obj_{object_id:06d}.obj"
    if not mesh_path.exists():
        _log.warning("missing mesh %s, skipping", mesh_path)
        return None

    out_path = out_root / method / f"{obj_name}__{gripper}.png"
    ok, counts = _render_in_subprocess(
        mesh_path, grasp_json, out_path, target_total, seed + object_id, width, height
    )
    if ok:
        rendered = sum(counts.values())
        breakdown = " ".join(f"{c}={counts.get(c, 0)}" for c in CATEGORY_ORDER)
        print(f"  wrote {out_path}  ({rendered} grasps, {gripper}, {breakdown})")
        return out_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--objects", default=None,
                        help=f"comma-separated YCB ids (default: {DEFAULT_OBJECTS})")
    parser.add_argument("--methods", default=None,
                        help=f"comma-separated mesh sources (default: {DEFAULT_METHODS})")
    parser.add_argument("--target-total", type=int, default=60,
                        help="target number of grasps to render (default 60). "
                             "Sampled proportionally across the 5 outcome "
                             "categories based on availability.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--dataset", default="ycbv")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--assets-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    objects = _parse_int_list(args.objects, DEFAULT_OBJECTS)
    methods = _parse_str_list(args.methods, DEFAULT_METHODS)
    paths = resolve_paths(args.dataset_root, args.output_root, args.assets_root)
    out_root = paths.output_root / "visualization" / "screenshots"

    print(f"rendering {len(objects)} object(s) × {len(methods)} method(s) "
          f"= {len(objects) * len(methods)} screenshot(s)")
    print("legend:  green=successful  yellow=collision_target  blue=no_contact  "
          "magenta=slipped  gray=error")
    n_ok = n_skip = 0
    for method in methods:
        for obj_id in objects:
            try:
                p = render_one(obj_id, method, args.target_total, args.seed,
                               args.width, args.height, paths, args.dataset, out_root)
                if p is None:
                    n_skip += 1
                else:
                    n_ok += 1
            except Exception as e:
                _log.error("render failed for obj=%d method=%s: %s", obj_id, method, e)
                n_skip += 1

    print(f"\ndone. ok={n_ok} skipped={n_skip} → {out_root}")


if __name__ == "__main__":
    main()
