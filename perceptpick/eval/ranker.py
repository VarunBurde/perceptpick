"""Aggregate per-(object, gripper) grasp JSONs into a per-object gripper ranking.

Stage B-2. Produces ``<output_root>/grasp_poses/<mesh-source>/gripper_rankings.json``:
for every object, a list of grippers sorted by success rate, including the failure-mode
breakdown (collision_target / no_contact / slipped / error) so downstream
analysis can see *why* each gripper succeeded or failed, not only how often.

The downstream evaluate stage (``--gripper auto``) reads only the top entry's
``gripper`` field via :func:`best_gripper_for`, so the richer schema is
backward-compatible with that call site.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)


# The five per-(object, gripper) JSON statistics buckets the sampler writes.
# Mirrors ``perceptpick.sampling.generator._EMPTY_BUCKETS``.
_BUCKETS = ("successful", "collision_target", "no_contact", "slipped", "error")


def rank_grippers(
    grasp_poses_root: Path,
    mesh_source: str,
    output_path: Path,
    min_success_count: int = 1,
) -> dict:
    """Walk ``<grasp_poses_root>/<mesh_source>/<object>/<gripper>.json`` and
    write the ranking dict to ``output_path``.

    Each ranked gripper entry has:
      - ``gripper``         : short name (e.g. ``"robotiq_2f_140"``)
      - ``success_rate``    : ``n_successful / n_total`` (primary sort key)
      - ``n_successful``    : count of SUCCESS grasps (tiebreak)
      - ``n_total``         : total grasps simulated for this combo
      - ``collision_target``: count of approach-collision failures
      - ``no_contact``      : count of close-in-free-space failures
      - ``slipped``         : count of grip-lost-during-lift failures
      - ``error``           : count of programmatic errors
    """
    src = grasp_poses_root / mesh_source
    if not src.is_dir():
        raise FileNotFoundError(f"no grasp_poses dir for mesh_source={mesh_source}: {src}")

    per_object: dict[str, list[dict]] = {}
    for obj_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        obj_name = obj_dir.name
        rankings: list[dict] = []
        for json_fn in sorted(obj_dir.glob("*.json")):
            try:
                data = json.loads(json_fn.read_text())
            except json.JSONDecodeError as e:
                _log.warning("skip malformed JSON %s: %s", json_fn, e)
                continue
            n_succ = data.get("num_successful_grasps", 0)
            n_total = data.get("total_grasps_simulated", 0)
            if n_succ < min_success_count:
                continue
            stats = data.get("statistics", {})
            # Avoid division by zero when n_total reads as 0 from a malformed
            # entry; rate becomes 0 in that case so the gripper sorts last.
            denom = n_total if n_total > 0 else 1
            entry = {
                "gripper": data.get("gripper_type", json_fn.stem),
                "success_rate": n_succ / denom,
                "n_successful": n_succ,
                "n_total": n_total,
            }
            # Pull each failure-mode count out of the per-combo statistics.
            # Missing keys default to 0 so this works on both old- and
            # new-schema JSONs without erroring.
            for bucket in _BUCKETS:
                if bucket == "successful":
                    continue
                entry[bucket] = int(stats.get(bucket, 0))
            rankings.append(entry)
        # Primary sort: success_rate descending. Tiebreak: absolute count.
        rankings.sort(key=lambda r: (r["success_rate"], r["n_successful"]), reverse=True)
        if rankings:
            per_object[obj_name] = rankings

    payload = {
        "mesh_source": mesh_source,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "min_success_count": min_success_count,
        "per_object_ranking": per_object,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    _log.info("ranked %d objects → %s", len(per_object), output_path)
    return payload


def best_gripper_for(rankings: dict, object_name: str) -> str | None:
    """Return the top-ranked gripper for an object, or None if not present."""
    entries = rankings.get("per_object_ranking", {}).get(object_name, [])
    return entries[0]["gripper"] if entries else None
