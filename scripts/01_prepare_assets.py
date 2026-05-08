"""Stage A — generate URDF + VHACD collision decomposition for object meshes.

The whole-dataset invocation: a single command processes every object mesh
present in the BOP dataset (and optionally every reconstructed mesh source).

Common forms:
    # GT meshes for the whole BOP dataset
    python scripts/01_prepare_assets.py --dataset ycbv --mesh-source GT

    # GT + every method under <dataset_root>/reconstructed_mesh/<method>/
    python scripts/01_prepare_assets.py --dataset ycbv --all-mesh-sources

    # Single reconstructed method
    python scripts/01_prepare_assets.py --dataset ycbv --mesh-source bakedsdf

    # Debug: a few specific object ids
    python scripts/01_prepare_assets.py --dataset ycbv --mesh-source GT --objects 1,2,7

Outputs land at ``<assets_root>/<dataset>/<mesh-source>/{meshes,vhacd,urdf}/``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from perceptpick.assets.mesh_prep import (
    discover_bop_models,
    discover_reconstructed_models,
    discover_reconstruction_methods,
    filter_objects,
    prepare_one,
)
from perceptpick.paths import resolve_paths


def _default_scale_for(mesh_source: str) -> float:
    """BOP GT meshes are in mm; reconstructed meshes are typically in m."""
    return 0.001 if mesh_source == "GT" else 1.0


def _prepare_one_source(
    paths,
    dataset: str,
    mesh_source: str,
    object_ids: list[int] | None,
    mass: float,
    friction: float,
    vhacd_resolution: int,
    mesh_scale: float | None,
) -> None:
    if mesh_source == "GT":
        models = discover_bop_models(paths.dataset_root, dataset)
    else:
        models = discover_reconstructed_models(paths.dataset_root, mesh_source)

    if not models:
        print(f"  no meshes for mesh_source={mesh_source}", file=sys.stderr)
        return

    models = filter_objects(models, object_ids)

    scale = _default_scale_for(mesh_source) if mesh_scale is None else mesh_scale

    out_root = paths.assets_root / dataset / mesh_source
    mesh_out = out_root / "meshes"
    vhacd_out = out_root / "vhacd"
    urdf_out = out_root / "urdf"

    print(f"[{mesh_source}] preparing {len(models)} object(s) under {out_root} (mesh_scale={scale})")
    for obj_id, mesh_in in models.items():
        try:
            prepare_one(
                mesh_in=mesh_in,
                mesh_out_dir=mesh_out,
                vhacd_out_dir=vhacd_out,
                urdf_out_dir=urdf_out,
                obj_id=obj_id,
                mass=mass,
                friction=friction,
                vhacd_resolution=vhacd_resolution,
                mesh_scale=scale,
            )
        except Exception as e:
            print(f"  FAILED {obj_id}: {e}", file=sys.stderr)
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="ycbv")
    parser.add_argument("--mesh-source", default="GT",
                        help="GT or a reconstruction method tag (bakedsdf, monosdf, ...)")
    parser.add_argument("--all-mesh-sources", action="store_true",
                        help="prepare GT plus every method under <dataset-root>/reconstructed_mesh/")
    parser.add_argument("--objects", default=None,
                        help="(debug) comma-separated YCB object ids (e.g. '1,2,7'); default = all 21")
    parser.add_argument("--vhacd-resolution", type=int, default=100_000)
    parser.add_argument("--mass", type=float, default=0.3, help="kg")
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--mesh-scale", type=float, default=None,
                        help="vertex scale factor; if omitted, defaults to "
                             "0.001 for GT (BOP mm→m) and 1.0 for reconstructed meshes")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--assets-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    paths = resolve_paths(dataset_root=args.dataset_root, assets_root=args.assets_root)

    object_ids: list[int] | None = None
    if args.objects:
        object_ids = [int(x) for x in args.objects.split(",")]

    if args.all_mesh_sources:
        sources = ["GT"] + discover_reconstruction_methods(paths.dataset_root)
        print(f"--all-mesh-sources: {len(sources)} source(s): {sources}")
    else:
        sources = [args.mesh_source]

    for source in sources:
        _prepare_one_source(
            paths=paths,
            dataset=args.dataset,
            mesh_source=source,
            object_ids=object_ids,
            mass=args.mass,
            friction=args.friction,
            vhacd_resolution=args.vhacd_resolution,
            mesh_scale=args.mesh_scale,
        )

    print("done.")


if __name__ == "__main__":
    main()
