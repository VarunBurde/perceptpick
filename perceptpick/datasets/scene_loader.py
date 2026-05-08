"""Scene loading: GT pose from BOP + estimated pose from a CSV.

Ported from ``core/scene.py``. Produces the (T_m2w, T_c2w, K) tuple used by
the evaluator for both the GT and EST runs of a single (scene, image, object).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from perceptpick.configs import YCB_OBJECTS
from perceptpick.core import ObjectInstance, ObjectType
from perceptpick.datasets.bop_loader import BopLoader

_log = logging.getLogger(__name__)


class SceneGenerator:
    def __init__(self, dataset_path: str | Path, csv_path: str | Path):
        self.dataset_path = str(dataset_path)
        self.csv_path = str(csv_path)
        self.bop_loader = BopLoader(self.dataset_path)
        self.pose_estimates = pd.read_csv(self.csv_path)
        _log.info("loaded %d pose estimates from %s", len(self.pose_estimates), self.csv_path)

    def load_ground_truth_pose(self, scene_id: int, image_id: int, obj_id: int):
        scene_id_str = str(scene_id).zfill(6)
        image_id_str = str(image_id)
        scene_dir = os.path.join(self.dataset_path, scene_id_str)
        with open(os.path.join(scene_dir, "scene_gt.json")) as f:
            gt_data = json.load(f)
        with open(os.path.join(scene_dir, "scene_camera.json")) as f:
            camera_data = json.load(f)

        if image_id_str not in gt_data:
            raise ValueError(f"image {image_id} not in scene {scene_id}")
        scene_info = gt_data[image_id_str]
        camera_info = camera_data[image_id_str]

        target = next((o for o in scene_info if o["obj_id"] == obj_id), None)
        if target is None:
            raise ValueError(f"object {obj_id} not in scene {scene_id}, image {image_id}")

        cam_R_w2c = np.array(camera_info["cam_R_w2c"]).reshape(3, 3)
        cam_t_w2c = np.array(camera_info["cam_t_w2c"]) / 1000.0
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = cam_R_w2c
        T_w2c[:3, 3] = cam_t_w2c
        T_c2w = np.linalg.inv(T_w2c)

        R_m2c = np.array(target["cam_R_m2c"]).reshape(3, 3)
        t_m2c = np.array(target["cam_t_m2c"]) / 1000.0
        T_m2c = np.eye(4)
        T_m2c[:3, :3] = R_m2c
        T_m2c[:3, 3] = t_m2c
        T_m2w_gt = T_c2w @ T_m2c

        K = np.array(camera_info["cam_K"]).reshape(3, 3)
        return T_m2w_gt, T_c2w, K

    def load_estimated_pose(self, scene_id: int, image_id: int, obj_id: int):
        mask = (
            (self.pose_estimates["scene_id"] == scene_id)
            & (self.pose_estimates["im_id"] == image_id)
            & (self.pose_estimates["obj_id"] == obj_id)
        )
        rows = self.pose_estimates[mask]
        if rows.empty:
            raise ValueError(f"no pose estimates for ({scene_id}, {image_id}, {obj_id})")
        if len(rows) > 1:
            _log.warning("multiple pose estimates; using highest score")
            row = rows.loc[rows["score"].idxmax()]
        else:
            row = rows.iloc[0]

        R_m2c_est = np.array([float(x) for x in row["R"].split()]).reshape(3, 3)
        t_m2c_est = np.array([float(x) for x in row["t"].split()]) / 1000.0
        T_m2c_est = np.eye(4)
        T_m2c_est[:3, :3] = R_m2c_est
        T_m2c_est[:3, 3] = t_m2c_est

        _, T_c2w, _ = self.load_ground_truth_pose(scene_id, image_id, obj_id)
        T_m2w_est = T_c2w @ T_m2c_est
        return T_m2w_est, row["score"]

    def create_object_instance(
        self,
        obj_id: int,
        pose: np.ndarray,
        assets_root: Path,
        dataset: str = "ycbv",
        mesh_source: str = "GT",
        enable_physics: bool = True,
        color: list | None = None,
    ) -> tuple[ObjectInstance, str]:
        """Build an ObjectInstance from prepared assets.

        Looks up files at:
          ``<assets_root>/<dataset>/<mesh_source>/{meshes,urdf,vhacd}/obj_NNNNNN.*``
        """
        object_name = YCB_OBJECTS[obj_id]
        asset_dir = Path(assets_root) / dataset / mesh_source
        mesh_path = asset_dir / "meshes" / f"obj_{obj_id:06d}.obj"
        urdf_path = asset_dir / "urdf" / f"obj_{obj_id:06d}.urdf"
        vhacd_path = asset_dir / "vhacd" / f"obj_{obj_id:06d}_vhacd.obj"

        for p in (mesh_path, urdf_path, vhacd_path):
            if not p.exists():
                raise FileNotFoundError(f"missing asset: {p}")

        ot = ObjectType(
            identifier=f"{object_name}_{id(pose)}" if not enable_physics else object_name,
            name=object_name,
            mesh_fn=str(mesh_path),
            urdf_fn=str(urdf_path),
            vhacd_fn=str(vhacd_path),
            mass=0.1 if enable_physics else 0.0,
            friction_coeff=0.5 if enable_physics else 0.0,
        )
        instance = ObjectInstance(ot, pose=pose)
        instance._enable_physics = enable_physics
        instance._color = color
        return instance, object_name
