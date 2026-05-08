"""Dataclass-based defaults for sampling, simulation, and dataset constants.

These replace the hard-coded dicts in the legacy ``config/config.py``. Scripts
import the dataclasses, optionally override fields via argparse, and pass them
to the sampler / simulator.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:
    """Antipodal grasp sampler parameters."""
    n_grasps: int = 5000
    mu: float = 0.5                         # friction coefficient
    n_orientations: int = 1                 # gripper-roll orientations per contact-pair
    n_rays: int = 30                        # rays cast per reference point
    max_targets_per_ref_point: int = 2
    only_grasp_from_above: bool = False
    no_contact_below_z: float | None = None
    min_grasp_width: float = 0.005          # m
    max_gripper_width: float = 0.12         # m, fallback when gripper-specific not set
    verbose: bool = False


@dataclass(frozen=True)
class SimulationConfig:
    """PyBullet grasp simulation parameters."""
    gripper_opening_width: float = 0.8      # opening scale (0.1-1.0)
    lifting_height: float = 0.1             # m, lift after gripper close
    min_contacts_required: int = 1
    settle_time_no_gravity: float = 0.1     # s, settle before close
    settle_time_after_close: float = 0.2    # s, settle after close, before gravity
    settle_time_with_gravity: float = 0.3   # s, settle after gravity enabled


# YCB object id → BOP folder name (object_NNNNNN convention).
# Aligned with /Users/varunburde/projects/ICRA/dataset/ycbv/models/.
YCB_OBJECTS: dict[int, str] = {
    1:  "002_master_chef_can",
    2:  "003_cracker_box",
    3:  "004_sugar_box",
    4:  "005_tomato_soup_can",
    5:  "006_mustard_bottle",
    6:  "007_tuna_fish_can",
    7:  "008_pudding_box",
    8:  "009_gelatin_box",
    9:  "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


# Per-gripper max opening widths (meters). Used to clamp antipodal sampling
# so candidate grasps fit the physical gripper.
GRIPPER_MAX_WIDTHS: dict[str, float] = {
    "franka":         0.08,
    "robotiq_2f_85":  0.085,
    "robotiq_2f_140": 0.14,
    "wsg_50":         0.11,
    "wsg_32":         0.032,
    "barrett":        0.15,
    "barrett_2f":     0.10,
    "robotiq_3f":     0.15,
    "kinova_3f":      0.14,
    "ezgripper":      0.10,
    "sawyer":         0.08,
}


# Reconstruction-method tags consumed as gt_mesh / est_mesh in scripts/04_evaluate.py.
MESH_SOURCES: tuple[str, ...] = (
    "GT", "BakedSDF", "MonoSDF", "Nerfacto",
    "Neuralangelo", "NGP", "RealCAP", "UniSurf", "VolSDF",
)
