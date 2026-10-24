"""Independent peg/hole placement: layout validation, 1 cm workspace map, stratified layout sampling.

Only one object moves per layout (``group`` = peg or hole); the other stays at its nominal
position. Radial bands are ``0_3`` (<=3 cm) and ``3_7`` (3-7 cm). Each mapped cell is either a
training cell or a spatially held-out test cell.

  python -m humanoid_mujoco_scenes.peg_insert.workspace map --output outputs/peg_workspace
  python -m humanoid_mujoco_scenes.peg_insert.workspace sample --map outputs/peg_workspace/workspace_map.json \\
      --split test --counts 20 30 --output outputs/peg_workspace/heldout_layouts.json
  # record expert demos at sampled training layouts
  python -m humanoid_mujoco_scenes.peg_insert.workspace sample --map ... --split train --counts 25 75 \\
      --output datasets/peg_workspace/train_layouts.json --record datasets/peg_workspace/raw
"""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model
from .expert import ExpertFailure, PegExpert
from .recording import CAMERAS, RawDataset, write_json
from .scene import BLOCK_POS, BLOCK_SIZE, PEG_POS, PEG_SIZE, TABLE_HALF_Y, TABLE_X

TABLE_MARGIN = 0.01        # m, object footprint to table edge
SEPARATION_MARGIN = 0.005  # m, peg footprint to block footprint at reset
GRID = 0.01                # m, map cell size
MAX_RADIUS = 0.07          # m
INNER_RADIUS = 0.03        # m, boundary between bands 0_3 and 3_7
NOMINAL = {"peg": np.array(PEG_POS), "hole": np.array(BLOCK_POS)}


def validate_layout(peg_xy, hole_xy):
    peg, hole = np.asarray(peg_xy, dtype=float), np.asarray(hole_xy, dtype=float)
    if peg.shape != (2,) or hole.shape != (2,) or not np.isfinite(np.r_[peg, hole]).all():
        raise ValueError("Invalid XY layout")
    for pos, half in [(peg, PEG_SIZE / 2), (hole, BLOCK_SIZE / 2)]:
        inside_x = TABLE_X[0] + half + TABLE_MARGIN <= pos[0] <= TABLE_X[1] - half - TABLE_MARGIN
        if not inside_x or abs(pos[1]) + half + TABLE_MARGIN > TABLE_HALF_Y:
            raise ValueError("Object outside table margin")
    if np.all(np.abs(peg - hole) < (PEG_SIZE + BLOCK_SIZE) / 2 + SEPARATION_MARGIN):
        raise ValueError("Peg overlaps block footprint or its 5mm margin")


def band(radius):
    return "0_3" if radius <= INNER_RADIUS + 1e-7 else "3_7"


def grid_layouts():
    """All 1 cm cells within 7 cm for each moving object; about a third of outer cells are test cells."""
    layouts = []
    for group in ["peg", "hole"]:
        for ix in range(-7, 8):
            for iy in range(-7, 8):
                radius = np.hypot(ix, iy) * GRID
                if radius > MAX_RADIUS + 1e-7:
                    continue
                peg, hole = NOMINAL["peg"].copy(), NOMINAL["hole"].copy()
                (peg if group == "peg" else hole)[:] += np.array([ix, iy]) * GRID
                test = max(abs(ix), abs(iy)) >= 2 and (ix + 2 * iy + (group == "peg")) % 3 == 0
                layouts.append({"seed": 40000 + len(layouts), "peg_xy": peg.tolist(), "hole_xy": hole.tolist(),
                                "group": group, "band": band(radius), "cell": [ix, iy],
                                "cell_id": f"{group}_{ix}_{iy}", "split": "test" if test else "train"})
    return layouts


def jitter_layout(cell, rng, seed, jitter=0.003):
    """Uniform +-jitter around a cell centre, kept inside the cell's radial band and the table."""
    for _ in range(100):
        offset = np.array(cell["cell"]) * GRID + rng.uniform(-jitter, jitter, 2)
        if band(np.linalg.norm(offset)) != cell["band"] or np.linalg.norm(offset) > MAX_RADIUS:
            continue
        layout = dict(cell, seed=seed, offset_cm=(offset * 100).tolist(),
                      jitter_mm=((offset - np.array(cell["cell"]) * GRID) * 1000).tolist())
        layout[cell["group"] + "_xy"] = (NOMINAL[cell["group"]] + offset).tolist()
        try:
            validate_layout(layout["peg_xy"], layout["hole_xy"])
        except ValueError:
            continue
        return layout
    raise ValueError("No valid in-band jitter for cell")


class WorkspaceCheck:
    """``on_frame`` wrapper rejecting unsafe arm contacts, <1 mm finger/block clearance and occlusion.

    Visibility thresholds are segmentation pixel counts at 120x160 rendering.
    """
    MIN_PIXELS = {"initial": ("ego", "peg", 8), "grasp": (None, "peg", 3), "insertion": (None, "block", 6)}
    MIN_INITIAL_BLOCK = 20
    MIN_CLEARANCE = 0.001  # m

    def __init__(self, expert, renderers=None, record=None):
        self.e, self.renderers, self.record = expert, renderers, record
        self.frame, self.visibility, self.min_finger_block = 0, {}, 0.05
        self.block_geoms = [g for g in range(expert.model.ngeom) if expert.model.geom_bodyid[g] == expert.block]
        self.moving = {expert.model.body(n).id for n in ["link1L", "link2l", "link3l", "link4l", "link5l", "link6l", "link7l", "link8l"]}

    def visible(self, key):
        if self.renderers is None:
            return
        counts = {}
        for name, renderer in self.renderers.items():
            renderer.enable_segmentation_rendering()
            renderer.update_scene(self.e.data, camera=name)
            seg = renderer.render()
            renderer.disable_segmentation_rendering()
            geoms = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            counts[name] = {"peg": int(np.sum(geoms & (seg[:, :, 0] == self.e.peg_geom))),
                            "block": int(np.sum(geoms & np.isin(seg[:, :, 0], self.block_geoms)))}
        self.visibility[key] = counts
        camera, obj, minimum = self.MIN_PIXELS[key]
        seen = counts[camera][obj] if camera else max(c[obj] for c in counts.values())
        if seen < minimum or (key == "initial" and counts["ego"]["block"] < self.MIN_INITIAL_BLOCK):
            raise ExpertFailure(f"{key} visibility")

    def __call__(self, e, action):
        self.check_contacts(e)
        if self.frame % 10 == 0 and e.phase in ("align", "insert", "release"):
            self.check_clearance(e)
        if e.phase == "close" and "grasp" not in self.visibility:
            self.visible("grasp")
        if e.phase == "insert" and "insertion" not in self.visibility:
            self.visible("insertion")
        if self.record:
            self.record(e, action)
        self.frame += 1

    def check_contacts(self, e):
        for c in e.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            ba, bb = int(e.model.geom_bodyid[a]), int(e.model.geom_bodyid[b])
            grasp = (a == e.peg_geom and bb in e.fingers) or (b == e.peg_geom and ba in e.fingers)
            if (ba in self.moving or bb in self.moving) and not grasp and c.dist < -0.001:
                raise ExpertFailure("unsafe arm/finger collision")

    def check_clearance(self, e):
        for finger in e.finger_geoms:
            for block in self.block_geoms:
                distance = float(mujoco.mj_geomDistance(e.model, e.data, finger, block, 0.05, None))
                self.min_finger_block = min(self.min_finger_block, distance)
                if distance < self.MIN_CLEARANCE:
                    raise ExpertFailure("less than 1mm finger/block insertion clearance")


def check_layout(expert, layout, renderers, record=None):
    """Reset to ``layout`` and run the full expert under ``WorkspaceCheck``; returns the check."""
    expert.on_frame = None
    expert.reset(layout["seed"], 0, peg_xy=layout["peg_xy"], hole_xy=layout["hole_xy"])
    check = WorkspaceCheck(expert, renderers, record)
    check.visible("initial")
    expert.on_frame = check
    try:
        expert.run()
    finally:
        expert.on_frame = None
    return check


def map_workspace(args):
    """Validate every grid cell with a physical expert run; resumable."""
    path = args.output / "workspace_map.json"
    rows = json.loads(path.read_text())["cells"] if path.exists() else []
    done = {r["cell_id"] for r in rows}
    model = make_model("peg_insert", cameras=CAMERAS)
    expert = PegExpert(model, mujoco.MjData(model))
    with ExitStack() as stack:
        renderers = {n: stack.enter_context(mujoco.Renderer(model, 120, 160)) for n in CAMERAS}
        for layout in [cell for cell in grid_layouts() if cell["cell_id"] not in done]:
            try:
                check = check_layout(expert, layout, renderers)
                rows.append({**layout, "valid": True, "min_finger_block_m": check.min_finger_block, "visibility": check.visibility})
            except (ExpertFailure, ValueError) as exc:
                rows.append({**layout, "valid": False, "reason": str(exc)})
            write_json(path, {"grid_cm": 1, "max_radius_cm": 7, "cells": rows})
            print("MAP", json.dumps(rows[-1]), flush=True)
    plot_map(rows, args.output / "workspace_map.png")


def plot_map(rows, path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 540), "white")
    draw = ImageDraw.Draw(image)
    for panel, group in enumerate(["peg", "hole"]):
        ox, oy = 250 + 500 * panel, 280
        draw.text((ox - 180, 15), f"{group.upper()} moves; other object fixed", fill="black")
        for k in range(-7, 8):
            draw.line((ox + k * 25, oy - 175, ox + k * 25, oy + 175), fill="#eeeeee")
            draw.line((ox - 175, oy + k * 25, ox + 175, oy + k * 25), fill="#eeeeee")
        for r in (r for r in rows if r["group"] == group):
            x, y = r["cell"]
            color = ("#198c74" if r["split"] == "train" else "#365fcd") if r["valid"] else "#c7c7c7"
            draw.ellipse((ox + x * 25 - 7, oy - y * 25 - 7, ox + x * 25 + 7, oy - y * 25 + 7), fill=color)
        draw.text((ox - 170, 495), "Green: valid train   Blue: held-out   Grey: rejected  (1 cm grid)", fill="black")
    image.save(path)


def sample_layouts(args):
    """Balanced, expert-validated jittered layouts per (group, band); optionally record demos."""
    cells = [c for c in json.loads(args.map.read_text())["cells"] if c["valid"] and args.split in ("any", c["split"])]
    if args.cells_from:  # e.g. a curate plan's reserved interpolation cells
        cells = json.loads(args.cells_from.read_text())["heldout_cells"]
    model = make_model("peg_insert", cameras=CAMERAS)
    expert = PegExpert(model, mujoco.MjData(model))
    raw = RawDataset(args.record, randomize=0, expert_only=True) if args.record else None
    accepted = json.loads(args.output.read_text())["layouts"] if args.output.exists() else []
    with ExitStack() as stack:
        seg = {n: stack.enter_context(mujoco.Renderer(model, 120, 160)) for n in CAMERAS}
        rgb = {n: stack.enter_context(mujoco.Renderer(model, h, w)) for n, (h, w) in CAMERAS.items()} if raw else {}
        for group in ["peg", "hole"]:
            for band_name, target in zip(["0_3", "3_7"], args.counts):
                # A map made for another table may include cells off this one.
                pool = [c for c in cells if (c["group"], c["band"]) == (group, band_name) and on_table(c)]
                rng = np.random.default_rng([args.seed, group == "hole", band_name == "3_7"])
                have = [a for a in accepted if (a["group"], a["band"]) == (group, band_name)]
                for attempt in range(target * 8):
                    if len(have) >= target:
                        break
                    seed = args.seed * 1000 + 100000 * (group == "hole") + 10000 * (band_name == "3_7") + attempt
                    if any(a["seed"] == seed for a in have):
                        continue
                    try:
                        layout = jitter_layout(pool[int(rng.integers(len(pool)))], rng, seed, args.jitter_mm / 1000)
                    except ValueError:  # cell too close to the table edge for any in-band jitter
                        continue
                    if validate_and_record(expert, layout, seg, rgb, raw):
                        layout["split"] = args.split
                        have.append(layout)
                        accepted.append(layout)
                        write_json(args.output, {"layouts": accepted})
                if len(have) < target:
                    raise RuntimeError(f"only {len(have)}/{target} feasible {group}/{band_name} layouts")


def on_table(cell):
    try:
        validate_layout(cell["peg_xy"], cell["hole_xy"])
        return True
    except ValueError:
        return False


def validate_and_record(expert, layout, seg, rgb, raw):
    try:
        if raw is None:
            check_layout(expert, layout, seg)
            return True
        with raw.episode(layout["seed"], rgb) as writer:
            check = check_layout(expert, layout, seg, writer)
            summary = writer.commit()
        raw.add({**layout, "success": True, "condition": f"{layout['group']}_{layout['band']}",
                 "min_finger_block_m": check.min_finger_block, **summary})
        return True
    except (ExpertFailure, ValueError) as exc:
        print("REJECT", layout["seed"], exc, flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    mapping = sub.add_parser("map", help="validate every 1 cm cell with the physical expert")
    mapping.add_argument("--output", type=Path, required=True, help="directory for workspace_map.json/.png")
    sample = sub.add_parser("sample", help="expert-validated jittered layouts, balanced per object and band")
    sample.add_argument("--map", type=Path, required=True)
    sample.add_argument("--split", choices=["train", "test", "any"], required=True, help="which mapped cells to sample")
    sample.add_argument("--counts", type=int, nargs=2, default=[20, 30], metavar=("INNER", "OUTER"), help="layouts per object for bands 0-3 / 3-7 cm")
    sample.add_argument("--jitter-mm", type=float, default=3.0)
    sample.add_argument("--cells-from", type=Path, help="sample only the heldout_cells of this selection plan")
    sample.add_argument("--seed", type=int, default=70)
    sample.add_argument("--output", type=Path, required=True, help="layouts JSON (resumes if present)")
    sample.add_argument("--record", type=Path, help="also record each accepted layout as an expert demo here")
    args = parser.parse_args()
    if args.command == "map":
        args.output.mkdir(parents=True, exist_ok=True)
        map_workspace(args)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sample_layouts(args)


if __name__ == "__main__":
    main()
