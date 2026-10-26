"""Episode selection plans for training on a subset of the accumulated peg demos and corrections.

A plan is ``{"identities": [[raw, seed], ...], "baseline_datasets": [...], "heldout_cells": [...]}``
and is passed to the flow trainer as ``--selection`` (episodes are read in place).

  fraction  stratified random fraction of unique episodes (per dataset x condition x workspace band)
  spatial   farthest-point spread per category, reserving workspace cells surrounded by training
            placements as interpolation tests (``workspace sample --cells-from <plan>`` makes their eval layouts)
  add       extend a plan with every unused episode of the given datasets; reserved cells that new
            episodes now touch are moved to ``newly_exposed_cells``

  python -m humanoid_mujoco_scenes.peg_insert.curate fraction --dataset D0 --dataset D1 ... --fraction 0.5 --output half.json
  python -m humanoid_mujoco_scenes.peg_insert.curate spatial --dataset ... --map workspace_map.json --output spatial.json
  python -m humanoid_mujoco_scenes.peg_insert.curate add --plan spatial.json --dataset new_corrections --output spatial_plus.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .recording import write_json
from .workspace import NOMINAL

EXCLUSION = 0.008   # m, training placements this close to a reserved cell centre expose it
SURROUND = 0.025    # m, training placements within this radius must surround a reserved cell
# Episodes per category for ``spatial`` (the spatial350 run); corrections are split by kind.
BUDGET = {"nominal": 20, "placement": 70, "broad": 50, "tilt": 40,
          "correction": {"jam": 40, "alignment": 35, "hover": 9, "pre_loss": 1, "other": 15}}
RESERVED = {"0_3": 2, "3_7": 6}  # interpolation cells per moving object and band


def catalogue(roots):
    """One row per unique episode, joined with its raw manifest result (layout, condition, kind)."""
    rows, seen, manifests = [], set(), {}
    for dataset, root in enumerate(map(Path, roots)):
        index = root / "phase_index.json"
        if not index.exists():
            total = json.loads((root / "meta/info.json").read_text())["total_episodes"]
            rows += [{"identity": [str(root), str(e)], "dataset": dataset, "category": "nominal", "seed": e} for e in range(total)]
            continue
        for source in json.loads(index.read_text())["episodes"]:
            identity = (source["raw"], str(source["seed"]))
            if identity in seen:
                continue
            seen.add(identity)
            if source["raw"] not in manifests:
                manifests[source["raw"]] = {r["seed"]: r for r in json.loads((Path(source["raw"]) / "manifest.json").read_text())["results"]}
            result = manifests[source["raw"]][source["seed"]]
            rows.append({**result, "identity": list(identity), "dataset": dataset, "category": category(result), **placement(result)})
    return rows


def category(result):
    if result.get("condition", "").startswith("tilt"):
        return "tilt"
    if "cell" in result and "kind" not in result:
        return "placement"
    if "offset_mm" in result and "clearance_mm" in result:
        return "broad"
    return "correction"


def placement(result):
    layout = result if "cell" in result else result.get("layout", {})
    if layout.get("group") not in ("peg", "hole"):
        return {}
    xy = layout[layout["group"] + "_xy"]
    return {"placement": {"group": layout["group"], "xy": xy, "cell": layout["cell"], "band": layout["band"],
                          "offset_cm": ((np.asarray(xy) - NOMINAL[layout["group"]]) * 100).tolist()}}


def correction_kind(row):
    kind = row.get("kind") or row.get("source", {}).get("kind", "")
    return {"pre_grasp_loss": "pre_loss", "hover_stall": "hover", "early_rim_contact": "jam", "early_rim_jam": "jam",
            "alignment": "alignment"}.get(kind, "alignment" if row.get("takeover_reason") or row.get("condition") in ("aligned", "full") else "other")


def farthest(rows, count, features, mandatory=()):
    """Greedy farthest-point subset in feature space, starting from ``mandatory`` rows."""
    if len(rows) < count:
        raise ValueError(f"only {len(rows)}/{count} eligible episodes")
    position = {tuple(r["identity"]): i for i, r in enumerate(rows)}
    chosen = list(dict.fromkeys(position[tuple(r["identity"])] for r in mandatory))
    values = np.asarray([features(r) for r in rows], dtype=float)
    chosen = chosen or [int(np.argmin(np.sum(values * values, axis=1)))]
    distance = np.min(np.sum((values[:, None] - values[chosen][None]) ** 2, axis=-1), axis=1)
    while len(chosen) < count:
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.sum((values - values[chosen[-1]]) ** 2, axis=1))
    return [rows[i] for i in chosen]


def cell_point(cell):
    return NOMINAL[cell["group"]] + np.asarray(cell["cell"]) * 0.01


def eligible(row, cells):
    """False if the episode's placement is in, or within 8 mm of, a reserved cell of the same object."""
    p = row.get("placement")
    return p is None or not any(p["group"] == c["group"] and (p["cell"] == c["cell"] or math.dist(p["xy"], cell_point(c)) < EXCLUSION)
                                for c in cells)


def surrounded(point, positions):
    """Training placements within 2.5 cm on all sides (largest angular gap < 180 deg)."""
    vectors = np.asarray(positions, dtype=float).reshape(-1, 2) - point
    vectors = vectors[np.linalg.norm(vectors, axis=1) <= SURROUND]
    if len(vectors) < 3:
        return False
    angles = np.sort(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return bool(np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max() < np.pi - 0.05)


def reserve_cells(cells, rows):
    """Pick interpolation cells away from the original +-1 cm reset box and surrounded by placements."""
    chosen = []
    for group in ["peg", "hole"]:
        placements = [r for r in rows if r["category"] == "placement" and r["placement"]["group"] == group]
        candidates = [c for c in cells if c["valid"] and c["group"] == group and _reservable(c, placements)]
        for band, count in RESERVED.items():
            pool = [c for c in candidates if c["band"] == band]
            for _ in range(count):
                existing = [np.asarray(c["cell"]) for c in chosen if c["group"] == group]
                ordered = sorted(pool, key=lambda c: -min((np.linalg.norm(np.asarray(c["cell"]) - v) for v in existing), default=-np.linalg.norm(c["cell"])))
                pick = next((c for c in ordered if _still_surrounded(chosen + [c], placements)), None)
                if pick is None:
                    raise ValueError(f"not enough surrounded {group}/{band} cells to reserve")
                chosen.append(pick)
                pool.remove(pick)
    return chosen


def _reservable(cell, placements):
    offset = cell_point(cell) - NOMINAL[cell["group"]]
    # Older resets jittered each coordinate by up to 1 cm around nominal; stay clear of that box.
    return np.linalg.norm(np.maximum(np.abs(offset) - 0.01, 0)) >= 0.009 and np.linalg.norm(offset) <= 0.06 \
        and surrounded(cell_point(cell), [r["placement"]["xy"] for r in placements])


def _still_surrounded(cells, placements):
    remaining = [r["placement"]["xy"] for r in placements if eligible(r, cells)]
    return all(surrounded(cell_point(c), remaining) for c in cells if c["group"] == placements[0]["placement"]["group"])


def pose_features(row):
    offset = row.get("placement", {}).get("offset_cm", [0, 0])
    return [*np.asarray(offset) / 7, abs(row.get("tilt_degrees", 0)) / 12]


def spatial(args):
    rows = catalogue(args.dataset)
    cells = reserve_cells(json.loads(args.map.read_text())["cells"], rows)
    pool = [r for r in rows if eligible(r, cells)]
    selected = farthest([r for r in pool if r["category"] == "nominal"], BUDGET["nominal"], lambda r: [r["seed"] / 50])
    for group in ["peg", "hole"]:
        group_rows = [r for r in pool if r["category"] == "placement" and r["placement"]["group"] == group]
        selected += farthest(group_rows, BUDGET["placement"], lambda r: np.asarray(r["placement"]["offset_cm"]) / 7)
    selected += farthest([r for r in pool if r["category"] == "broad"], BUDGET["broad"],
                         lambda r: [*np.asarray(r["offset_mm"]) / 80, r["clearance_mm"] / 60])
    tilt = [r for r in pool if r["category"] == "tilt"]
    selected += farthest(tilt, BUDGET["tilt"], lambda r: [r.get("tilt_degrees", 5) / 12, 2 * (r.get("tilt_axis") == "y"), *np.asarray(r.get("offset_mm", [0, 0])) / 30])
    corrections = [r for r in pool if r["category"] == "correction"]
    for kind, count in BUDGET["correction"].items():
        of_kind = [r for r in corrections if correction_kind(r) == kind]
        selected += farthest(of_kind, min(count, len(of_kind)), pose_features)
    plan = {"name": "spatial", "roots": list(map(str, args.dataset)), "identities": [r["identity"] for r in selected],
            "baseline_datasets": sorted({r["dataset"] for r in selected if r["category"] in ("nominal", "placement")}),
            "counts": {c: sum(r["category"] == c for r in selected) for c in ["nominal", "placement", "broad", "tilt", "correction"]},
            "heldout_cells": cells, "exclusion_m": EXCLUSION}
    return plan


def fraction(args):
    rows = catalogue(args.dataset)
    groups = {}
    for row in rows:
        band = row.get("placement", {}).get("band", "")
        groups.setdefault((row["dataset"], row.get("condition", ""), row.get("placement", {}).get("group", ""), band), []).append(row)
    rng = np.random.default_rng(args.seed)
    keys = list(groups)
    rng.shuffle(keys)
    target = math.ceil(len(rows) * args.fraction)
    counts = [math.floor(len(groups[k]) * args.fraction) for k in keys]
    for i in [i for i, k in enumerate(keys) if len(groups[k]) * args.fraction % 1][:target - sum(counts)]:
        counts[i] += 1  # largest-remainder rounding keeps the exact total
    selected = [groups[k][j] for k, n in zip(keys, counts) for j in rng.permutation(len(groups[k]))[:n]]
    return {"name": f"fraction_{args.fraction}", "roots": list(map(str, args.dataset)), "seed": args.seed,
            "identities": [r["identity"] for r in selected], "available": len(rows)}


def add(args):
    base = json.loads(args.plan.read_text())
    existing = {tuple(i) for i in base["identities"]}
    additions = [r for r in catalogue(args.dataset) if tuple(r["identity"]) not in existing]
    exposed = [c for c in base.get("heldout_cells", []) if not all(eligible(r, [c]) for r in additions)]
    offset = len(base["roots"])
    return {**base, "name": base["name"] + f"+{len(additions)}", "roots": base["roots"] + list(map(str, args.dataset)),
            "identities": base["identities"] + [r["identity"] for r in additions],
            "heldout_cells": [c for c in base.get("heldout_cells", []) if c not in exposed],
            "newly_exposed_cells": base.get("newly_exposed_cells", []) + exposed, "added": len(additions), "added_dataset_offset": offset}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["fraction", "spatial", "add"]:
        s = sub.add_parser(name)
        s.add_argument("--dataset", type=Path, action="append", required=True, help="LeRobot roots (in trainer order)")
        s.add_argument("--output", type=Path, required=True)
        if name == "fraction":
            s.add_argument("--fraction", type=float, default=0.5)
            s.add_argument("--seed", type=int, default=1701)
        if name == "spatial":
            s.add_argument("--map", type=Path, required=True, help="workspace_map.json")
        if name == "add":
            s.add_argument("--plan", type=Path, required=True)
    args = p.parse_args()
    plan = {"fraction": fraction, "spatial": spatial, "add": add}[args.command](args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, plan)
    print(json.dumps({k: v for k, v in plan.items() if k not in ("identities", "heldout_cells", "newly_exposed_cells")}), flush=True)


if __name__ == "__main__":
    main()
