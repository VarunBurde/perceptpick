"""Gripper-analysis report from ``output/grasp_poses/<src>/gripper_rankings.json``.

Consumes the schema produced by ``perceptpick.eval.ranker.rank_grippers``
(per-object lists of grippers sorted by success rate, with failure-mode
breakdown counts) and produces:

  * a 4-panel matplotlib figure modeled on the paper's Fig 2
  * a markdown summary with three tables

Driven by ``scripts/03_report_grippers.py``. Library-only — no side
effects on import; all paths are caller-supplied.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np

from perceptpick.configs import YCB_OBJECTS
from perceptpick.grippers import GRIPPER_REGISTRY

_log = logging.getLogger(__name__)


# Failure-mode keys in the per-(object, gripper) ranking entry, in the
# order we want them to appear in stacked bars / tables.
_FAILURE_MODES = ("collision_target", "no_contact", "slipped", "error")
_BUCKET_ORDER = ("successful", *_FAILURE_MODES)


def load_rankings(path: Path) -> dict:
    """Read a ``gripper_rankings.json`` file."""
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Aggregations over the rankings dict
# --------------------------------------------------------------------------
def best_gripper_per_object(rankings: dict) -> dict[str, dict]:
    """Return ``{obj_name: top_entry}`` for every object that has at least
    one ranked gripper.
    """
    per_obj = rankings.get("per_object_ranking", {})
    return {obj: entries[0] for obj, entries in per_obj.items() if entries}


def _all_grippers_in_rankings(rankings: dict) -> list[str]:
    """Stable list of every gripper that appears in any object's ranking,
    ordered by ``GRIPPER_REGISTRY`` so the figure column order is reproducible.
    """
    seen: set[str] = set()
    for entries in rankings.get("per_object_ranking", {}).values():
        for e in entries:
            seen.add(e["gripper"])
    return [g for g in GRIPPER_REGISTRY if g in seen]


def _all_objects_in_rankings(rankings: dict) -> list[str]:
    """Stable list of every object that appears in the rankings, ordered by
    YCB id (so id 1 → 002_master_chef_can comes first)."""
    seen = set(rankings.get("per_object_ranking", {}).keys())
    return [name for _id, name in sorted(YCB_OBJECTS.items()) if name in seen]


def success_rate_matrix(rankings: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``(matrix, object_names, gripper_names)``.

    ``matrix[i, j]`` is the success rate of ``gripper_names[j]`` on
    ``object_names[i]``, or ``np.nan`` if that gripper had zero successful
    grasps for that object (filtered by the ranker's ``min_success_count``).
    """
    objects = _all_objects_in_rankings(rankings)
    grippers = _all_grippers_in_rankings(rankings)
    obj_to_row = {name: i for i, name in enumerate(objects)}
    grip_to_col = {name: j for j, name in enumerate(grippers)}

    matrix = np.full((len(objects), len(grippers)), np.nan)
    for obj_name, entries in rankings.get("per_object_ranking", {}).items():
        if obj_name not in obj_to_row:
            continue
        i = obj_to_row[obj_name]
        for e in entries:
            j = grip_to_col[e["gripper"]]
            matrix[i, j] = e["success_rate"]
    return matrix, objects, grippers


def gripper_avg_success(rankings: dict) -> dict[str, dict]:
    """Per-gripper aggregate stats over the objects each gripper was ranked on.

    Returns ``{gripper: {avg_rate, best_object, best_rate, worst_object,
    worst_rate, n_objects}}``. Only counts objects where the gripper had
    enough successful grasps to clear ``min_success_count``.
    """
    by_gripper: dict[str, list[tuple[str, float]]] = {}
    for obj, entries in rankings.get("per_object_ranking", {}).items():
        for e in entries:
            by_gripper.setdefault(e["gripper"], []).append((obj, e["success_rate"]))

    out: dict[str, dict] = {}
    for gripper, samples in by_gripper.items():
        rates = [r for _, r in samples]
        best = max(samples, key=lambda x: x[1])
        worst = min(samples, key=lambda x: x[1])
        out[gripper] = {
            "avg_rate": float(np.mean(rates)),
            "best_object": best[0], "best_rate": best[1],
            "worst_object": worst[0], "worst_rate": worst[1],
            "n_objects": len(samples),
        }
    return out


def gripper_failure_totals(rankings: dict) -> dict[str, dict]:
    """Sum each bucket count across every object, per gripper.

    Returns ``{gripper: {successful, collision_target, no_contact, slipped, error}}``.
    """
    out: dict[str, dict[str, int]] = {}
    for entries in rankings.get("per_object_ranking", {}).values():
        for e in entries:
            counts = out.setdefault(e["gripper"], {b: 0 for b in _BUCKET_ORDER})
            counts["successful"] += e.get("n_successful", 0)
            for mode in _FAILURE_MODES:
                counts[mode] += e.get(mode, 0)
    return out


def best_gripper_share(rankings: dict) -> dict[str, int]:
    """``{gripper: number_of_objects_for_which_it_is_top_ranked}``."""
    counts: dict[str, int] = {}
    for top in best_gripper_per_object(rankings).values():
        counts[top["gripper"]] = counts.get(top["gripper"], 0) + 1
    return counts


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def make_figure(rankings: dict, output_path: Path) -> Path:
    """Render the 4-panel summary PNG. Creates the parent directory."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix, objects, grippers = success_rate_matrix(rankings)
    avg = gripper_avg_success(rankings)
    totals = gripper_failure_totals(rankings)
    share = best_gripper_share(rankings)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    mesh_source = rankings.get("mesh_source", "GT")
    fig.suptitle(
        f"PerceptPick — Gripper analysis ({mesh_source} meshes)",
        fontsize=14, fontweight="bold",
    )

    # (a) Per-object × per-gripper success-rate heatmap with rank annotations
    # (1/2/3 marking the top-three grippers in each row).
    ax = axes[0, 0]
    cmap = LinearSegmentedColormap.from_list(
        "rate", ["#f1f1f1", "#a8e0a8", "#1f8a1f"], N=64
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(grippers)))
    ax.set_xticklabels(grippers, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects, fontsize=8)
    ax.set_title("(a) Per-object success rate by gripper "
                 "(1/2/3 = per-object top-3 grippers)", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="success rate")

    # Annotate top-3 cells per row. Skip rows where we have fewer than 3
    # ranked grippers, and skip cells whose value is NaN.
    for i in range(matrix.shape[0]):
        row = matrix[i]
        valid_idx = np.where(~np.isnan(row))[0]
        if valid_idx.size == 0:
            continue
        # Sort the valid columns by success rate descending; tag top-3.
        ranked = sorted(valid_idx, key=lambda j: -row[j])
        for rank, j in enumerate(ranked[:3], start=1):
            # Pick a contrasting text color: white on dark green cells, black
            # on light/empty cells. Threshold ~0.55 matches the cmap midpoint.
            color = "white" if row[j] >= 0.55 else "black"
            ax.text(j, i, str(rank), ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    # (b) Best-gripper share across objects (pie).
    ax = axes[0, 1]
    if share:
        # Order by descending share for readability.
        items = sorted(share.items(), key=lambda kv: -kv[1])
        labels, sizes = zip(*items)
        ax.pie(sizes, labels=[f"{n} ({s})" for n, s in items],
               startangle=90, autopct="%1.0f%%",
               colors=plt.cm.tab10.colors[:len(labels)])
    else:
        ax.text(0.5, 0.5, "no rankings", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"(b) Best-gripper share across {len(objects)} object(s)", fontsize=11, loc="left")

    # (c) Average success rate per gripper.
    ax = axes[1, 0]
    if avg:
        items = sorted(avg.items(), key=lambda kv: kv[1]["avg_rate"], reverse=True)
        names = [k for k, _ in items]
        rates = [v["avg_rate"] for _, v in items]
        ax.barh(range(len(names)), rates, color="#1f8a1f")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("avg success rate")
        ax.set_xlim(0, 1)
        for y, r in enumerate(rates):
            ax.text(r + 0.01, y, f"{r*100:.1f}%", va="center", fontsize=8)
    ax.set_title("(c) Average success rate per gripper", fontsize=11, loc="left")

    # (d) Failure-mode breakdown per gripper (stacked horizontal bar).
    ax = axes[1, 1]
    if totals:
        # Order grippers by total grasps simulated (largest first) for visual stability.
        items = sorted(totals.items(),
                       key=lambda kv: -sum(kv[1].values()))
        names = [k for k, _ in items]
        bucket_colors = {
            "successful":       "#1f8a1f",
            "collision_target": "#d9b441",
            "no_contact":       "#3b6cb8",
            "slipped":          "#a04bc4",
            "error":            "#888888",
        }
        left = np.zeros(len(names))
        for bucket in _BUCKET_ORDER:
            vals = np.array([totals[n][bucket] for n in names])
            ax.barh(range(len(names)), vals, left=left,
                    color=bucket_colors[bucket], label=bucket)
            left = left + vals
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("grasp count (summed across objects)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("(d) Failure-mode breakdown per gripper", fontsize=11, loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    _log.info("wrote %s", output_path)
    return output_path


def make_markdown(rankings: dict, output_path: Path) -> Path:
    """Render the 3-table markdown summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh_source = rankings.get("mesh_source", "GT")
    generated_at = rankings.get("generated_at", dt.datetime.now().isoformat(timespec="seconds"))

    best = best_gripper_per_object(rankings)
    avg = gripper_avg_success(rankings)
    totals = gripper_failure_totals(rankings)

    lines: list[str] = []
    lines.append(f"# Gripper analysis — `{mesh_source}` meshes\n")
    lines.append(f"_Generated {generated_at} · {len(best)} objects · {len(totals)} grippers_\n")

    # Table 1: best gripper per object.
    lines.append("## Best gripper per object\n")
    lines.append("| Object | Best gripper | Success rate | Dominant failure |")
    lines.append("|---|---|---:|---|")
    for obj_name in _all_objects_in_rankings(rankings):
        if obj_name not in best:
            continue
        e = best[obj_name]
        failures = {m: e.get(m, 0) for m in _FAILURE_MODES}
        dom = max(failures, key=failures.get) if any(failures.values()) else "—"
        lines.append(
            f"| {obj_name} | {e['gripper']} | {e['success_rate']*100:.1f}% | {dom} |"
        )

    # Table 2: per-gripper averages.
    lines.append("\n## Average success rate per gripper\n")
    lines.append("| Gripper | Avg success | Best object (rate) | Worst object (rate) | # objects |")
    lines.append("|---|---:|---|---|---:|")
    for g, v in sorted(avg.items(), key=lambda kv: -kv[1]["avg_rate"]):
        lines.append(
            f"| {g} | {v['avg_rate']*100:.1f}% | "
            f"{v['best_object']} ({v['best_rate']*100:.1f}%) | "
            f"{v['worst_object']} ({v['worst_rate']*100:.1f}%) | "
            f"{v['n_objects']} |"
        )

    # Table 3: per-gripper failure totals.
    lines.append("\n## Failure-mode totals per gripper (summed across objects)\n")
    lines.append("| Gripper | successful | collision_target | no_contact | slipped | error |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for g, counts in sorted(totals.items(),
                            key=lambda kv: -kv[1]["successful"]):
        lines.append(
            f"| {g} | {counts['successful']} | {counts['collision_target']} | "
            f"{counts['no_contact']} | {counts['slipped']} | {counts['error']} |"
        )

    output_path.write_text("\n".join(lines) + "\n")
    _log.info("wrote %s", output_path)
    return output_path
