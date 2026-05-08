"""Stage B reporting — paper-style 4-panel summary + markdown table.

Reads ``<output_root>/grasp_poses/<mesh-source>/gripper_rankings.json``
(produced by ``scripts/02_grasp_sweep.py``) and writes:

  * ``<output_root>/reports/gripper_analysis_<mesh-source>.png`` — 4-panel
    figure: per-object success heatmap, best-gripper share pie, average
    success bar, failure-mode breakdown stacked bar.
  * ``<output_root>/reports/gripper_analysis_<mesh-source>.md`` — three
    tables: best gripper per object, per-gripper averages, per-gripper
    failure-mode totals.

Usage:
    python scripts/03_report_grippers.py --mesh-source GT
    python scripts/03_report_grippers.py --mesh-source BakedSDF --no-plot
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from perceptpick.analysis import (
    best_gripper_per_object,
    load_rankings,
    make_figure,
    make_markdown,
)
from perceptpick.paths import resolve_paths


def _print_summary(rankings: dict) -> None:
    best = best_gripper_per_object(rankings)
    if not best:
        print("(no rankings — every combo had zero successful grasps)")
        return
    print("Best gripper per object:")
    print(f"  {'object':<26s} {'gripper':<18s} {'rate':>7s}  dominant_failure")
    for obj_name in sorted(best):
        e = best[obj_name]
        failures = {
            m: e.get(m, 0) for m in ("collision_target", "no_contact", "slipped", "error")
        }
        dom = max(failures, key=lambda k: failures[k]) if any(failures.values()) else "—"
        print(f"  {obj_name:<26s} {e['gripper']:<18s} {e['success_rate']*100:>6.1f}%  {dom}")


def main() -> None:
    desc = (__doc__ or "").split("\n\n")[0]
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--mesh-source", default="GT",
                        help="tag baked into the output filenames (and used to "
                             "verify rankings.json was generated for this mesh source)")
    parser.add_argument("--rankings", default=None,
                        help="explicit path to a gripper_rankings.json "
                             "(default: <output_root>/grasp_poses/<mesh-source>/gripper_rankings.json)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the PNG + md (default: <output_root>/reports/)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the matplotlib figure; only write the markdown")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = resolve_paths(output_root=args.output_root)

    if args.rankings:
        rankings_path = Path(args.rankings)
    else:
        rankings_path = paths.output_root / "grasp_poses" / args.mesh_source / "gripper_rankings.json"
    if not rankings_path.exists():
        print(f"rankings file not found at {rankings_path}", file=sys.stderr)
        print(f"  → run: scripts/02_grasp_sweep.py --mesh-source {args.mesh_source}",
              file=sys.stderr)
        sys.exit(1)

    rankings = load_rankings(rankings_path)
    written_mesh = rankings.get("mesh_source", "GT")
    if written_mesh != args.mesh_source:
        print(
            f"NOTE: rankings file is for mesh_source={written_mesh!r}, "
            f"--mesh-source flag is {args.mesh_source!r}; "
            f"using the file's value for figure title and output filenames.",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir) if args.out_dir else paths.output_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"gripper_analysis_{written_mesh}.md"
    make_markdown(rankings, md_path)

    png_path = None
    if not args.no_plot:
        png_path = out_dir / f"gripper_analysis_{written_mesh}.png"
        make_figure(rankings, png_path)

    print()
    _print_summary(rankings)
    print()
    print(f"wrote {md_path}")
    if png_path is not None:
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
