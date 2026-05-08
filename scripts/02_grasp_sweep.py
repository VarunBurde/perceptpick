"""Stage B — sample antipodal grasps, simulate, and rank grippers per object.

This is the **whole-Stage-B orchestrator**: one fire-and-forget command runs
antipodal sampling + PyBullet simulation for every (object, gripper)
combination, then aggregates the per-(object, gripper) JSONs into a
``output/grasp_poses/<mesh-source>/gripper_rankings.json`` ready for Stage C
(`scripts/04_evaluate.py`).

Common forms:
    # Whole sweep: 21 YCB objects × 9 grippers (paper's set), GT meshes
    python scripts/02_grasp_sweep.py --dataset ycbv --mesh-source GT --n-grasps 5000

    # Restrict the gripper set
    python scripts/02_grasp_sweep.py --grippers franka,robotiq_2f_140 --n-grasps 5000

    # Restrict to a few objects
    python scripts/02_grasp_sweep.py --objects 1,7,14 --n-grasps 5000

    # Single (object, gripper) — debug mode, GUI on, no ranking step
    python scripts/02_grasp_sweep.py --object 7 --gripper robotiq_2f_140 --n-grasps 50

Output layout:
    output/grasp_poses/<mesh-source>/<object_name>/<gripper>.json   per-combo
    output/grasp_poses/<mesh-source>/gripper_rankings.json           per-object ranking

Re-running is idempotent: combos with an existing JSON are skipped (default).
Pass --no-resume to force re-simulation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, replace

from perceptpick.configs import SamplingConfig, SimulationConfig, YCB_OBJECTS
from perceptpick.core import Scene
from perceptpick.eval.ranker import rank_grippers
from perceptpick.grippers import all_grippers, GRIPPER_REGISTRY
from perceptpick.paths import resolve_paths
from perceptpick.sampling.generator import GraspPoseGenerator
from perceptpick.sim import GraspSimulatorWithGravityControl


def _gripper_short_names_for(classes) -> list[str]:
    """Reverse-lookup the short names in GRIPPER_REGISTRY for a list of classes."""
    cls_to_name = {cls: name for name, cls in GRIPPER_REGISTRY.items()}
    return [cls_to_name[c] for c in classes if c in cls_to_name]


# Paper sweep set: the 9 grippers in perceptpick.grippers.all_grippers.
DEFAULT_GRIPPERS = _gripper_short_names_for(all_grippers)
DEFAULT_OBJECT_IDS = sorted(YCB_OBJECTS.keys())


def _parse_int_list(s: str | None, default: list[int]) -> list[int]:
    if s is None:
        return list(default)
    return [int(x) for x in s.split(",")]


def _parse_str_list(s: str | None, default: list[str]) -> list[str]:
    if s is None:
        return list(default)
    return [x.strip() for x in s.split(",") if x.strip()]


def _summarise_rankings(rankings: dict, top_n: int = 3) -> None:
    """Print a compact per-object top-N summary."""
    per_obj = rankings.get("per_object_ranking", {})
    if not per_obj:
        print("(no rankings — every combo had zero successful grasps)")
        return
    print(f"\nTop-{top_n} grippers per object (success_rate, dominant failure):")
    for obj_name in sorted(per_obj):
        entries = per_obj[obj_name][:top_n]
        parts = []
        for e in entries:
            failures = {k: e.get(k, 0) for k in ("collision_target", "no_contact", "slipped", "error")}
            top_fail = max(failures, key=failures.get) if any(failures.values()) else None
            tag = f" [{top_fail}]" if top_fail else ""
            parts.append(f"{e['gripper']} {e['success_rate']*100:.1f}%{tag}")
        print(f"  {obj_name:<26s} {' | '.join(parts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="ycbv")
    parser.add_argument("--mesh-source", default="GT")
    # Singular flags select a single combo (debug mode).
    parser.add_argument("--object", type=int, default=None,
                        help="single YCB object id (1-21); pair with --gripper for debug mode")
    parser.add_argument("--gripper", default=None,
                        help="single gripper short name; pair with --object for debug mode")
    # Plural flags restrict the sweep set.
    parser.add_argument("--objects", default=None,
                        help=f"comma-separated object ids (default: all 21)")
    parser.add_argument("--grippers", default=None,
                        help=f"comma-separated gripper names (default: paper's 9)")
    parser.add_argument("--n-grasps", type=int, default=5000)
    parser.add_argument("--no-resume", action="store_true",
                        help="force re-simulation of combos with existing JSONs")
    parser.add_argument("--gui", action="store_true",
                        help="force PyBullet GUI even when sweeping")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--assets-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = resolve_paths(args.dataset_root, args.output_root, args.assets_root)

    # Determine the combo set.
    if args.object is not None and args.gripper is not None:
        object_ids = [args.object]
        grippers = [args.gripper]
        single_combo = True
    else:
        if args.object is not None:
            object_ids = [args.object]
        else:
            object_ids = _parse_int_list(args.objects, DEFAULT_OBJECT_IDS)
        if args.gripper is not None:
            grippers = [args.gripper]
        else:
            grippers = _parse_str_list(args.grippers, DEFAULT_GRIPPERS)
        single_combo = False

    n_combos = len(object_ids) * len(grippers)
    print(f"sweep: {len(object_ids)} object(s) × {len(grippers)} gripper(s) = {n_combos} combo(s)")

    # GUI default: on for single combo (debug); off for sweeps unless --gui.
    gui = (single_combo or args.gui) and not (n_combos > 1 and not args.gui)
    if not single_combo and not args.gui:
        print("running headless (sweep > 1 combo); pass --gui to force a window")

    asset_dir = paths.assets_root / args.dataset / args.mesh_source

    # In GUI mode, keep ONE PyBullet window alive for the whole sweep — each
    # combo replaces the scene's object via reset_for_next_grasp(). Avoids
    # the macOS issue where dismissed BulletClient(GUI) windows linger.
    shared_sim: GraspSimulatorWithGravityControl | None = None
    if gui:
        shared_sim = GraspSimulatorWithGravityControl(
            Scene(),
            verbose=True,
            simulation_config=asdict(SimulationConfig()),
            gui=True,
        )

    n_done = n_skip = n_fail = 0
    for obj_id in object_ids:
        if obj_id not in YCB_OBJECTS:
            print(f"  unknown object id {obj_id}, skipping", file=sys.stderr)
            continue
        obj_name = YCB_OBJECTS[obj_id]
        obj_id_str = f"obj_{obj_id:06d}"

        mesh_path = asset_dir / "meshes" / f"{obj_id_str}.obj"
        urdf_path = asset_dir / "urdf" / f"{obj_id_str}.urdf"
        vhacd_path = asset_dir / "vhacd" / f"{obj_id_str}_vhacd.obj"
        if not all(p.exists() for p in (mesh_path, urdf_path, vhacd_path)):
            print(f"  MISSING ASSETS for {obj_id_str}; run scripts/01_prepare_assets.py first",
                  file=sys.stderr)
            for p in (mesh_path, urdf_path, vhacd_path):
                if not p.exists():
                    print(f"    missing: {p}", file=sys.stderr)
            n_fail += 1
            continue

        for gripper in grippers:
            if gripper not in GRIPPER_REGISTRY:
                print(f"  unknown gripper '{gripper}', skipping", file=sys.stderr)
                n_fail += 1
                continue

            output_dir = paths.output_root / "grasp_poses" / args.mesh_source / obj_name
            output_json = output_dir / f"{gripper}.json"

            if not args.no_resume and output_json.exists():
                print(f"  skip {obj_name} × {gripper} (already done: {output_json.name})")
                n_skip += 1
                continue

            print(f"  -> {obj_name} × {gripper}")
            try:
                generator = GraspPoseGenerator(
                    mesh_path=mesh_path,
                    urdf_path=urdf_path,
                    vhacd_path=vhacd_path,
                    object_name=obj_name,
                    gripper_type=gripper,
                    output_dir=output_dir,
                    sampling_cfg=replace(SamplingConfig(), n_grasps=args.n_grasps),
                    simulation_cfg=SimulationConfig(),
                    gui=gui,
                    simulator=shared_sim,
                )
                generator.run()
                n_done += 1
            except Exception as e:
                print(f"     FAILED: {e}", file=sys.stderr)
                n_fail += 1

    print(f"\nsweep complete: {n_done} done, {n_skip} skipped, {n_fail} failed")

    if shared_sim is not None:
        shared_sim.dismiss()

    # Single-combo runs are debug mode — don't touch the rankings file.
    if single_combo:
        return

    rankings_path = paths.output_root / "grasp_poses" / args.mesh_source / "gripper_rankings.json"
    print(f"\naggregating rankings → {rankings_path}")
    rankings = rank_grippers(
        grasp_poses_root=paths.output_root / "grasp_poses",
        mesh_source=args.mesh_source,
        output_path=rankings_path,
        min_success_count=1,
    )
    _summarise_rankings(rankings)


if __name__ == "__main__":
    main()
