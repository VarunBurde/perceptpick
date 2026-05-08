"""TMP — aggregate Stage C per-tuple JSONs into a per-object table for
paper reproduction.

Reads ``output/picking_success/<gt-mesh>_vs_<est-mesh>/<pose-method>/*.json``
written by ``perceptpick.eval.pick``, aggregates by object:

  * **GT success rate** (oracle baseline — grasps anchored to GT pose)
  * **EST success rate** (grasps anchored to the pose-estimator's pose)
  * **Drop** = GT − EST (success-rate degradation due to pose error)
  * **Robustness** = EST_n / GT_n (% of GT-successful grasps that survive
    the pose perturbation)
  * Mean translation / rotation pose error per object

Usage:
    pixi run python scripts/tmp_eval_summary.py \\
        --gt-mesh GT --est-mesh BakedSDF --pose-method FoundationPose
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean

from perceptpick.configs import YCB_OBJECTS
from perceptpick.paths import resolve_paths

_log = logging.getLogger(__name__)


def _aggregate(in_dir: Path) -> dict:
    rows = []
    for jf in sorted(in_dir.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception as e:
            _log.warning("skip %s: %s", jf, e)
            continue
        info = d.get("experiment_info", {})
        an = d.get("analysis", {})
        rows.append({
            "obj_id": info.get("obj_id"),
            "obj_name": info.get("obj_name"),
            "gripper": info.get("gripper"),
            "gt_rate": an.get("gt_success_rate"),
            "est_rate": an.get("est_success_rate"),
            "drop": an.get("success_rate_drop"),
            "robustness": an.get("robustness_rate"),
            "trans_err_mm": (an.get("translation_error_m") or 0) * 1000,
            "rot_err_deg": an.get("rotation_error_deg"),
            "add_mm": (an.get("add_m") or 0) * 1000,
        })

    by_obj = defaultdict(list)
    for r in rows:
        by_obj[r["obj_name"]].append(r)

    per_object = {}
    for obj, lst in sorted(by_obj.items()):
        if not lst:
            continue
        per_object[obj] = {
            "n_tuples": len(lst),
            "gripper": lst[0]["gripper"],  # all entries for this obj should share gripper if --gripper auto
            "gt_rate_mean": mean(r["gt_rate"] for r in lst if r["gt_rate"] is not None),
            "est_rate_mean": mean(r["est_rate"] for r in lst if r["est_rate"] is not None),
            "drop_mean": mean(r["drop"] for r in lst if r["drop"] is not None),
            "robustness_mean": mean(r["robustness"] for r in lst if r["robustness"] is not None),
            "trans_err_mm_mean": mean(r["trans_err_mm"] for r in lst if r["trans_err_mm"] is not None),
            "rot_err_deg_mean": mean(r["rot_err_deg"] for r in lst if r["rot_err_deg"] is not None),
            "add_mm_mean": mean(r["add_mm"] for r in lst if r["add_mm"] is not None),
        }

    overall = {
        "n_tuples": len(rows),
        "gt_rate_mean": mean(r["gt_rate"] for r in rows if r["gt_rate"] is not None) if rows else 0.0,
        "est_rate_mean": mean(r["est_rate"] for r in rows if r["est_rate"] is not None) if rows else 0.0,
        "drop_mean": mean(r["drop"] for r in rows if r["drop"] is not None) if rows else 0.0,
        "robustness_mean": mean(r["robustness"] for r in rows if r["robustness"] is not None) if rows else 0.0,
        "trans_err_mm_mean": mean(r["trans_err_mm"] for r in rows if r["trans_err_mm"] is not None) if rows else 0.0,
        "rot_err_deg_mean": mean(r["rot_err_deg"] for r in rows if r["rot_err_deg"] is not None) if rows else 0.0,
    }
    return {"per_object": per_object, "overall": overall}


def _print_table(agg: dict) -> None:
    per_obj = agg["per_object"]
    if not per_obj:
        print("(no Stage C results found)")
        return
    print(f"{'object':<26s} {'gripper':<18s} {'n':>4s} "
          f"{'GT %':>6s} {'EST %':>6s} {'drop %':>7s} {'robust %':>9s} "
          f"{'tErr mm':>8s} {'rErr °':>7s}")
    print("-" * 100)
    for obj in sorted(per_obj):
        e = per_obj[obj]
        print(f"{obj:<26s} {e['gripper']:<18s} {e['n_tuples']:>4d} "
              f"{e['gt_rate_mean']:>6.1f} {e['est_rate_mean']:>6.1f} "
              f"{e['drop_mean']:>7.1f} {e['robustness_mean']:>9.1f} "
              f"{e['trans_err_mm_mean']:>8.1f} {e['rot_err_deg_mean']:>7.2f}")
    print("-" * 100)
    o = agg["overall"]
    print(f"{'OVERALL':<26s} {'':<18s} {o['n_tuples']:>4d} "
          f"{o['gt_rate_mean']:>6.1f} {o['est_rate_mean']:>6.1f} "
          f"{o['drop_mean']:>7.1f} {o['robustness_mean']:>9.1f} "
          f"{o['trans_err_mm_mean']:>8.1f} {o['rot_err_deg_mean']:>7.2f}")


def _write_markdown(agg: dict, md_path: Path, gt_mesh: str, est_mesh: str,
                    pose_method: str) -> None:
    lines = [
        f"# Stage C reproduction — {gt_mesh}/{est_mesh} × {pose_method}",
        "",
        f"Aggregated from {agg['overall']['n_tuples']} per-tuple results in",
        f"`output/picking_success/{gt_mesh}_vs_{est_mesh}/{pose_method}/`.",
        "",
        "**Columns:** GT % = success rate when grasps are anchored to the GT pose "
        "(oracle baseline). EST % = success rate when grasps are anchored to the "
        "pose-estimator's pose. **Drop** = GT − EST. **Robust** = EST_n / GT_n.",
        "",
        "| object | gripper | n | GT % | EST % | Drop | Robust | tErr mm | rErr ° |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for obj in sorted(agg["per_object"]):
        e = agg["per_object"][obj]
        lines.append(
            f"| {obj} | {e['gripper']} | {e['n_tuples']} | "
            f"{e['gt_rate_mean']:.1f} | {e['est_rate_mean']:.1f} | "
            f"{e['drop_mean']:.1f} | {e['robustness_mean']:.1f} | "
            f"{e['trans_err_mm_mean']:.1f} | {e['rot_err_deg_mean']:.2f} |"
        )
    o = agg["overall"]
    lines.append(
        f"| **OVERALL** | | {o['n_tuples']} | "
        f"**{o['gt_rate_mean']:.1f}** | **{o['est_rate_mean']:.1f}** | "
        f"**{o['drop_mean']:.1f}** | **{o['robustness_mean']:.1f}** | "
        f"{o['trans_err_mm_mean']:.1f} | {o['rot_err_deg_mean']:.2f} |"
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gt-mesh", default="GT")
    parser.add_argument("--est-mesh", default="BakedSDF")
    parser.add_argument("--pose-method", default="FoundationPose",
                        help="CSV stem (filename without extension)")
    parser.add_argument("--out-md", default=None,
                        help="default: <output>/picking_success/<...>/summary.md")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = resolve_paths(output_root=args.output_root)
    in_dir = (
        paths.output_root / "picking_success"
        / f"{args.gt_mesh}_vs_{args.est_mesh}" / args.pose_method
    )
    if not in_dir.is_dir():
        print(f"no Stage C output at {in_dir}; run scripts/04_evaluate.py first")
        return

    agg = _aggregate(in_dir)
    _print_table(agg)
    md_path = Path(args.out_md) if args.out_md else (in_dir / "summary.md")
    _write_markdown(agg, md_path, args.gt_mesh, args.est_mesh, args.pose_method)


if __name__ == "__main__":
    main()
