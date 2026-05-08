"""Inspect sampled grasps on an object mesh in Open3D.

Usage:
    python scripts/04_visualize.py --kind grasps --object 7 --gripper robotiq_2f_140

The grasps view colour-codes by outcome category:
    successful=green, collision_target=yellow, no_contact=blue,
    slipped=magenta, error=gray.
"""
from __future__ import annotations

import argparse
import sys

from perceptpick.paths import resolve_paths
from perceptpick.viz.grasps_viewer import visualize_grasps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", choices=["grasps"], default="grasps",
                        help="visualization kind (more views coming)")
    parser.add_argument("--dataset", default="ycbv")
    parser.add_argument("--object", type=int, required=True, help="YCB object id (1-21)")
    parser.add_argument("--gripper", required=True)
    parser.add_argument("--mesh-source", default="GT")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--assets-root", default=None)
    args = parser.parse_args()

    paths = resolve_paths(output_root=args.output_root, assets_root=args.assets_root)

    if args.kind == "grasps":
        visualize_grasps(
            object_id=args.object,
            gripper=args.gripper,
            mesh_source=args.mesh_source,
            output_root=paths.output_root,
            assets_root=paths.assets_root,
            dataset=args.dataset,
        )
    else:
        print(f"unknown kind: {args.kind}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
