"""Stage C — benchmark grasp success with GT/reconstructed mesh and GT/estimated pose.

For each ``(scene, image, object)`` in the pose CSV: spawn the object using
``--est-mesh`` mesh and the estimated pose, attempt all precomputed grasps
from ``output/grasp_poses/<gt-mesh>/...``, and record success.

Examples:
    # See one scene visually:
    python scripts/04_evaluate.py --dataset ycbv --gt-mesh GT --est-mesh GT \\
        --pose-csv foundationpose.csv --scenes 48 --gripper robotiq_2f_140

    # Full batch, headless:
    python scripts/04_evaluate.py --dataset ycbv --gt-mesh GT --est-mesh bakedsdf \\
        --pose-csv foundationpose.csv --workers 4 --resume --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from perceptpick.configs import YCB_OBJECTS
from perceptpick.eval.pick import PickEvalConfig, evaluate_pick, output_path_for
from perceptpick.eval.ranker import best_gripper_for
from perceptpick.paths import resolve_paths


def _read_csv_entries(csv_path: Path) -> list[tuple[int, int, int]]:
    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append((int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"])))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="ycbv")
    parser.add_argument("--gt-mesh", required=True)
    parser.add_argument("--est-mesh", required=True)
    parser.add_argument("--pose-csv", required=True,
                        help="pose-estimator CSV filename, e.g. FoundationPose.csv. "
                             "Resolved against either "
                             "<assets_root>/<dataset>/<est-mesh>/pose_estimates/<csv> (new layout) "
                             "or <dataset_root>/<dataset>/methods_poses/<csv> (legacy)")
    parser.add_argument("--gripper", default="auto",
                        help="'auto' (use rankings) or specific gripper name")
    parser.add_argument("--scenes", default=None, help="comma-separated scene ids")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--n-grasps", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None,
                        help="limit number of (scene,image,obj) entries (for testing)")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--assets-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # GUI + multi-worker doesn't make sense — each worker would spawn its own
    # PyBullet window. Force --workers 1 when running with visualisation.
    if not args.headless and args.workers > 1:
        print(f"GUI mode requires --workers 1 (got {args.workers}); forcing 1",
              file=sys.stderr)
        args.workers = 1

    paths = resolve_paths(args.dataset_root, args.output_root, args.assets_root)

    # Prefer the new co-located layout (assets/<dataset>/<est-mesh>/pose_estimates/<csv>);
    # fall back to the legacy flat dataset/<dataset>/methods_poses/<csv>.
    csv_candidates = [
        paths.assets_root / args.dataset / args.est_mesh / "pose_estimates" / args.pose_csv,
        paths.dataset_root / args.dataset / "methods_poses" / args.pose_csv,
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        print(f"pose CSV {args.pose_csv!r} not found; tried:", file=sys.stderr)
        for c in csv_candidates:
            print(f"  {c}", file=sys.stderr)
        sys.exit(1)

    entries = _read_csv_entries(csv_path)
    if args.scenes:
        keep = set(int(x) for x in args.scenes.split(","))
        entries = [e for e in entries if e[0] in keep]
    if args.max_runs:
        entries = entries[: args.max_runs]
    print(f"evaluating {len(entries)} (scene, image, obj) entries")

    rankings = None
    if args.gripper == "auto":
        rank_file = paths.output_root / "grasp_poses" / args.gt_mesh / "gripper_rankings.json"
        if not rank_file.exists():
            print(f"--gripper auto requires {rank_file}; "
                  f"run scripts/02_grasp_sweep.py --mesh-source {args.gt_mesh} first",
                  file=sys.stderr)
            sys.exit(1)
        rankings = json.loads(rank_file.read_text())

    # In GUI mode, keep ONE PyBullet window alive for the whole run — every
    # tuple replaces the scene contents via _reset_scene(). On macOS dismissing
    # a BulletClient(GUI) doesn't actually close the window, so creating a
    # fresh sim per tuple piles up windows.
    shared_sim = None
    if not args.headless:
        from perceptpick.core import Scene
        from perceptpick.sim import GraspSimulator
        from perceptpick.sim.viewport import configure_clean_viewport, draw_world_frame
        shared_sim = GraspSimulator(Scene(), verbose=True, plane_and_gravity=False)
        configure_clean_viewport(shared_sim._p)
        draw_world_frame(shared_sim._p)

    n_ok = n_skip = n_fail = 0
    try:
        for scene_id, image_id, obj_id in entries:
            if args.gripper == "auto":
                gripper = best_gripper_for(rankings, YCB_OBJECTS[obj_id]) or "robotiq_2f_140"
            else:
                gripper = args.gripper

            cfg = PickEvalConfig(
                scene_id=scene_id, image_id=image_id, obj_id=obj_id,
                gt_mesh=args.gt_mesh, est_mesh=args.est_mesh, pose_csv=csv_path,
                gripper=gripper, dataset=args.dataset,
                dataset_root=paths.dataset_root, output_root=paths.output_root,
                assets_root=paths.assets_root,
                n_grasps=args.n_grasps, headless=args.headless, workers=args.workers,
            )

            if args.resume and output_path_for(cfg).exists():
                n_skip += 1
                continue

            try:
                result = evaluate_pick(cfg, simulator=shared_sim)
                n_ok += 1
                print(f"  ({scene_id},{image_id},{obj_id}) gripper={gripper} "
                      f"GT={result['gt']['success_rate']:.1f}% "
                      f"EST={result['est']['success_rate']:.1f}%")
            except Exception as e:
                n_fail += 1
                print(f"  FAILED ({scene_id},{image_id},{obj_id}): {e}", file=sys.stderr)
    finally:
        if shared_sim is not None:
            shared_sim.dismiss()

    print(f"\ndone. ok={n_ok} skipped={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
