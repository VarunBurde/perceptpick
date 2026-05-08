"""Test perceptpick.analysis.grippers on synthetic rankings data."""
import json
from pathlib import Path

from perceptpick.analysis import (
    best_gripper_per_object,
    best_gripper_share,
    gripper_avg_success,
    gripper_failure_totals,
    load_rankings,
    make_markdown,
    success_rate_matrix,
)


def _write_rankings(tmp_path: Path) -> Path:
    """Synthesize a rankings JSON with 2 objects × 3 grippers."""
    payload = {
        "mesh_source": "GT",
        "generated_at": "2026-05-08T00:00:00",
        "min_success_count": 1,
        "per_object_ranking": {
            "002_master_chef_can": [
                {"gripper": "ezgripper", "success_rate": 0.6,
                 "n_successful": 6, "n_total": 10,
                 "collision_target": 2, "no_contact": 1, "slipped": 1, "error": 0},
                {"gripper": "franka", "success_rate": 0.4,
                 "n_successful": 4, "n_total": 10,
                 "collision_target": 5, "no_contact": 1, "slipped": 0, "error": 0},
            ],
            "003_cracker_box": [
                {"gripper": "ezgripper", "success_rate": 0.8,
                 "n_successful": 8, "n_total": 10,
                 "collision_target": 1, "no_contact": 1, "slipped": 0, "error": 0},
                {"gripper": "robotiq_2f_140", "success_rate": 0.5,
                 "n_successful": 5, "n_total": 10,
                 "collision_target": 0, "no_contact": 5, "slipped": 0, "error": 0},
            ],
        },
    }
    p = tmp_path / "gripper_rankings.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_and_best(tmp_path: Path) -> None:
    rankings = load_rankings(_write_rankings(tmp_path))
    best = best_gripper_per_object(rankings)
    assert set(best) == {"002_master_chef_can", "003_cracker_box"}
    assert best["002_master_chef_can"]["gripper"] == "ezgripper"
    assert best["003_cracker_box"]["gripper"] == "ezgripper"


def test_success_rate_matrix(tmp_path: Path) -> None:
    rankings = load_rankings(_write_rankings(tmp_path))
    matrix, objects, grippers = success_rate_matrix(rankings)
    # Order: by YCB id (002 first, 003 second); grippers by GRIPPER_REGISTRY order.
    assert objects == ["002_master_chef_can", "003_cracker_box"]
    assert "ezgripper" in grippers and "franka" in grippers and "robotiq_2f_140" in grippers
    assert matrix.shape == (2, len(grippers))
    # ezgripper appears for both objects with rates 0.6 and 0.8.
    j = grippers.index("ezgripper")
    assert matrix[0, j] == 0.6
    assert matrix[1, j] == 0.8


def test_avg_and_share(tmp_path: Path) -> None:
    rankings = load_rankings(_write_rankings(tmp_path))
    avg = gripper_avg_success(rankings)
    # ezgripper appears on both objects with rates 0.6 and 0.8 → avg 0.7.
    assert abs(avg["ezgripper"]["avg_rate"] - 0.7) < 1e-9
    assert avg["ezgripper"]["n_objects"] == 2
    assert avg["franka"]["n_objects"] == 1

    # ezgripper wins both objects → share = 2.
    share = best_gripper_share(rankings)
    assert share == {"ezgripper": 2}


def test_failure_totals_and_markdown(tmp_path: Path) -> None:
    rankings = load_rankings(_write_rankings(tmp_path))
    totals = gripper_failure_totals(rankings)
    # ezgripper: successful = 6 + 8 = 14, collision_target = 2 + 1 = 3, ...
    assert totals["ezgripper"]["successful"] == 14
    assert totals["ezgripper"]["collision_target"] == 3
    assert totals["ezgripper"]["no_contact"] == 2
    assert totals["ezgripper"]["slipped"] == 1
    assert totals["ezgripper"]["error"] == 0

    md_path = tmp_path / "out.md"
    make_markdown(rankings, md_path)
    text = md_path.read_text()
    assert "Best gripper per object" in text
    assert "ezgripper" in text
    assert "002_master_chef_can" in text
