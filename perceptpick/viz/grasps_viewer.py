"""Open3D viewer for sampled grasp poses on an object mesh.

Ported from ``scripts/visualization/visulize_single.py``. Shows grasps
colour-coded by outcome category (successful=green, collision=red, etc.).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from perceptpick.grippers import TwoFingerGripperVisualisation
from perceptpick.configs import YCB_OBJECTS

_log = logging.getLogger(__name__)

CATEGORY_COLORS = {
    "successful":       [0, 1, 0],
    "collision_target": [1, 1, 0],
    "no_contact":       [0, 0, 1],
    "slipped":          [1, 0, 1],
    "error":            [0.5, 0.5, 0.5],
}


def visualize_grasps(
    object_id: int,
    gripper: str,
    mesh_source: str,
    output_root: Path,
    assets_root: Path,
    dataset: str = "ycbv",
) -> None:
    object_name = YCB_OBJECTS[object_id]
    grasp_json = output_root / "grasp_poses" / mesh_source / object_name / f"{gripper}.json"
    if not grasp_json.exists():
        raise FileNotFoundError(grasp_json)
    data = json.loads(grasp_json.read_text())
    categorized = data.get("categorized_grasps", {})

    mesh_path = assets_root / dataset / mesh_source / "meshes" / f"obj_{object_id:06d}.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    mesh.compute_vertex_normals()
    if not (len(mesh.textures) or len(mesh.triangle_material_ids)):
        mesh.paint_uniform_color([0.8, 0.8, 0.8])

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geoms = [mesh, coord]

    gripper_proto = TwoFingerGripperVisualisation()
    n = 0
    for category, items in categorized.items():
        color = CATEGORY_COLORS.get(category, [0.5, 0.5, 0.5])
        for item in items:
            g = o3d.geometry.TriangleMesh(gripper_proto.mesh)
            g.transform(np.asarray(item["pose"]))
            g.paint_uniform_color(color)
            geoms.append(g)
            n += 1
    _log.info("showing %d grasps over %s", n, object_name)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geoms:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()


def preview_candidates(
    object_mesh_path,
    graspset,
    *,
    gripper_color: tuple[float, float, float] = (0.4, 0.4, 0.9),
    title: str | None = None,
) -> None:
    """Open an Open3D window showing the object mesh + every sampled grasp
    candidate. Blocks until the user closes the window — used as a manual
    "go" gate before launching the PyBullet simulation in GUI mode.

    The grasps are uncategorized (all rendered with the same colour) since
    they haven't been simulated yet. After the window closes, the caller
    proceeds with simulation.
    """
    from pathlib import Path
    p = Path(object_mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(p), enable_post_processing=True)
    mesh.compute_vertex_normals()
    if not (len(mesh.textures) or len(mesh.triangle_material_ids)):
        mesh.paint_uniform_color([0.8, 0.8, 0.8])

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geoms = [mesh, coord]

    gripper_proto = TwoFingerGripperVisualisation()
    n = 0
    for grasp in graspset:
        g = o3d.geometry.TriangleMesh(gripper_proto.mesh)
        g.transform(np.asarray(grasp.pose))
        g.paint_uniform_color(list(gripper_color))
        geoms.append(g)
        n += 1

    label = title or f"{n} candidate grasp(s) — close window to start simulation"
    print(f"[open3d] {label}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=label)
    for g in geoms:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()
