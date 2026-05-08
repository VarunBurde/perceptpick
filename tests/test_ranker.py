"""Test the gripper ranker on synthetic data."""
import json
from pathlib import Path

from perceptpick.eval.ranker import best_gripper_for, rank_grippers


def test_rank_basic(tmp_path: Path) -> None:
    """Sort key is success_rate, not raw success count.

    franka has the highest absolute count (4200) but a smaller candidate pool
    (5000); ezgripper has fewer successes (3500) but found fewer candidates
    that fit (4000), giving a higher rate. ezgripper should win.
    """
    grasp_poses = tmp_path / "grasp_poses"
    obj_dir = grasp_poses / "GT" / "008_pudding_box"
    obj_dir.mkdir(parents=True)

    (obj_dir / "franka.json").write_text(json.dumps({
        "gripper_type": "franka",
        "num_successful_grasps": 4200,
        "total_grasps_simulated": 5000,    # rate = 0.84
        "statistics": {"successful": 4200, "collision_target": 600, "no_contact": 200,
                       "slipped": 0, "error": 0},
    }))
    (obj_dir / "ezgripper.json").write_text(json.dumps({
        "gripper_type": "ezgripper",
        "num_successful_grasps": 3500,
        "total_grasps_simulated": 4000,    # rate = 0.875  <-- highest
        "statistics": {"successful": 3500, "collision_target": 100, "no_contact": 350,
                       "slipped": 50, "error": 0},
    }))
    (obj_dir / "robotiq_2f_140.json").write_text(json.dumps({
        "gripper_type": "robotiq_2f_140",
        "num_successful_grasps": 2000,
        "total_grasps_simulated": 5000,    # rate = 0.40
        "statistics": {"successful": 2000, "collision_target": 1000, "no_contact": 1900,
                       "slipped": 100, "error": 0},
    }))
    (obj_dir / "wsg_32.json").write_text(json.dumps({
        "gripper_type": "wsg_32",
        "num_successful_grasps": 0,
        "total_grasps_simulated": 5000,    # filtered: 0 < min_success_count
        "statistics": {"successful": 0, "collision_target": 0, "no_contact": 5000,
                       "slipped": 0, "error": 0},
    }))

    out = tmp_path / "out" / "gripper_rankings.json"
    payload = rank_grippers(grasp_poses, "GT", out, min_success_count=1)

    assert out.exists()
    rankings = payload["per_object_ranking"]["008_pudding_box"]
    assert [r["gripper"] for r in rankings] == ["ezgripper", "franka", "robotiq_2f_140"]

    # Schema includes success_rate, n_successful, n_total + per-bucket
    # failure-mode breakdown so callers can see *why* each gripper failed.
    top = rankings[0]
    assert top["gripper"] == "ezgripper"
    assert top["success_rate"] == 0.875
    assert top["n_successful"] == 3500
    assert top["n_total"] == 4000
    assert top["collision_target"] == 100
    assert top["no_contact"] == 350
    assert top["slipped"] == 50
    assert top["error"] == 0

    assert best_gripper_for(payload, "008_pudding_box") == "ezgripper"
    assert best_gripper_for(payload, "missing") is None


def test_rank_handles_legacy_jsons_without_statistics(tmp_path: Path) -> None:
    """Old JSONs (pre-schema-change) are missing the 'statistics' field.
    Ranker should default the per-bucket counts to 0 rather than KeyError."""
    grasp_poses = tmp_path / "grasp_poses"
    obj_dir = grasp_poses / "GT" / "001_master_chef_can"
    obj_dir.mkdir(parents=True)
    (obj_dir / "franka.json").write_text(json.dumps({
        "gripper_type": "franka",
        "num_successful_grasps": 100,
        "total_grasps_simulated": 200,
        # no 'statistics' key
    }))

    out = tmp_path / "out" / "gripper_rankings.json"
    payload = rank_grippers(grasp_poses, "GT", out, min_success_count=1)

    top = payload["per_object_ranking"]["001_master_chef_can"][0]
    assert top["success_rate"] == 0.5
    assert top["collision_target"] == 0
    assert top["no_contact"] == 0
    assert top["slipped"] == 0
    assert top["error"] == 0
