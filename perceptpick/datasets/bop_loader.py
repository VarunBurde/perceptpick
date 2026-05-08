"""BOP-format dataset loader.

Ported from ``core/bop_loader.py``. Resolves scene IDs, image IDs, and
per-image camera + ground-truth pose data for a BOP-style test split.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from perceptpick.configs import YCB_OBJECTS


class BopLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.ycb_objects = YCB_OBJECTS

    def find_scene_ids(self) -> list[str]:
        return sorted(os.listdir(self.dataset_path))

    def find_image_ids(self, scene_id: int) -> list[str]:
        scene_id_str = str(scene_id).zfill(6)
        scene_gt = os.path.join(self.dataset_path, scene_id_str, "scene_gt.json")
        with open(scene_gt) as f:
            return list(json.load(f).keys())

    def find_scene_info(self, scene_id: int, key: str) -> dict:
        scene_id_str = str(scene_id).zfill(6)
        scene_dir = os.path.join(self.dataset_path, scene_id_str)
        with open(os.path.join(scene_dir, "scene_camera.json")) as f:
            camera_file = json.load(f)
        with open(os.path.join(scene_dir, "scene_gt.json")) as f:
            gt_file = json.load(f)

        scene_info = gt_file[key]
        camera_info = camera_file[key]

        img_id = str(key).zfill(6)
        rgb_file = os.path.join(scene_dir, "rgb", img_id + ".png")
        depth_file = os.path.join(scene_dir, "depth", img_id + ".png")
        img = np.array(Image.open(rgb_file), dtype=np.uint8)
        depth = np.array(Image.open(depth_file), dtype=np.float32) / 1000.0

        cam_K = np.asarray(camera_info["cam_K"]).reshape(3, 3)
        cam_R_w2c = np.asarray(camera_info["cam_R_w2c"]).reshape(3, 3)
        cam_t_w2c = np.asarray(camera_info["cam_t_w2c"]) * 0.001
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = cam_R_w2c
        T_w2c[:3, 3] = cam_t_w2c
        T_c2w = np.linalg.inv(T_w2c)

        object_info: dict[int, np.ndarray] = {}
        for obj in scene_info:
            R_m2c = np.asarray(obj["cam_R_m2c"]).reshape(3, 3)
            t_m2c = np.asarray(obj["cam_t_m2c"]) * 0.001
            T_m2c = np.eye(4)
            T_m2c[:3, :3] = R_m2c
            T_m2c[:3, 3] = t_m2c
            object_info[obj["obj_id"]] = T_c2w @ T_m2c

        return {
            "cam_K": cam_K,
            "T_c2w": T_c2w,
            "T_m2w": object_info,
            "resolution": img.shape,
            "rgb": img,
            "depth": depth,
        }
