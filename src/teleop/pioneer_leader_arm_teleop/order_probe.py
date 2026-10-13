"""Does a policy follow the announced order? (duplo, drawer_stow)

    python order_probe.py --scene duplo --policy <ckpt>/pretrained_model [--episodes 100] [--swap]

The scenes' own success checks do not measure this: drawer_stow accepts the blocks in any order, and in duplo a wrong
first brick just shows up as a failed stack. This logs what the policy actually goes for:
  duplo        first brick displaced 15 mm (and first lifted 3 cm) vs the announced first brick (chance 1/3)
  drawer_stow  the order the blocks enter the drawer vs the announced order: first block (chance 1/3), all three (chance 1/6)
``--swap`` tells the policy the announced colours rotated by one (only for checkpoints with the `order` instruction
encoding) and reports how often it goes for the brick it was told rather than the scene's.
"""
from __future__ import annotations

import argparse
import collections
import importlib

import numpy as np

import qplan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, choices=["duplo", "drawer_stow"])
    p.add_argument("--policy", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max_s", type=float, default=0.0, help="default: 25 s for duplo (first pick only), 90 s for drawer_stow")
    p.add_argument("--swap", action="store_true")
    args = p.parse_args()

    S = importlib.import_module(f"humanoid_mujoco_scenes.{args.scene}.scene")
    prop = qplan.Proposer(args.policy)
    if args.swap:
        encode = prop.encode

        def swapped(state, images, instruction):
            seq, step = instruction.split("|")
            o = seq.split(",")
            cols = [w for w in o if w in S.COLOURS]          # rotate the colours only (drawer_stow also has open / close)
            rot = iter(cols[1:] + cols[:1])
            return encode(state, images, ",".join(next(rot) if w in S.COLOURS else w for w in o) + "|" + step)
        prop.encode = swapped
    body = "brick_{}" if args.scene == "duplo" else "block_{}"
    rows = []
    for seed in range(args.seed, args.seed + args.episodes):
        st = {"order": None, "p0": None, "moved": None, "lifted": None, "entered": []}

        def probe(sim, st=st):
            d = sim.data
            if st["order"] is None:
                st["order"] = list(S._order(d))
                st["p0"] = {c: d.body(body.format(c)).xpos.copy() for c in S.COLOURS}
            for c in S.COLOURS:
                pos = d.body(body.format(c)).xpos
                if st["moved"] is None and np.linalg.norm(pos - st["p0"][c]) > 0.015:
                    st["moved"] = c
                if st["lifted"] is None and pos[2] > st["p0"][c][2] + 0.03:
                    st["lifted"] = c
                if args.scene == "drawer_stow" and c not in st["entered"] and S.in_drawer(sim.model, d, c):
                    st["entered"].append(c)
        qplan.PROBE = probe
        ep = qplan.rollout(args.scene, seed, prop, qplan.pick_first, n=1, z_std=0.0,
                           max_s=args.max_s or (25.0 if args.scene == "duplo" else 90.0))
        rows.append({**st, "success": ep["success"]})
    qplan.PROBE = None

    target = 1 if args.swap else 0            # index in the scene's order of the brick the policy was told comes first
    for key, label in (("moved", "first displaced 15 mm"), ("lifted", "first lifted 3 cm")):
        got = [r for r in rows if r[key]]
        print(f"[ORDER] {args.scene}: {label} = the one it was told: {sum(r[key] == r['order'][target] for r in got)} of {len(got)}"
              + (f"; = the scene's announced first: {sum(r[key] == r['order'][0] for r in got)} of {len(got)}" if args.swap else "")
              + "  (chance 1/3)")
    if args.scene == "drawer_stow":
        first = [r for r in rows if r["entered"]]
        full = [r for r in rows if len(r["entered"]) == 3]
        ok = [r for r in rows if r["success"]]
        print(f"[ORDER] drawer_stow: first block in the drawer = announced first: {sum(r['entered'][0] == r['order'][0] for r in first)} of {len(first)} (chance 1/3)")
        print(f"[ORDER] drawer_stow: all three in the announced order: {sum(r['entered'] == r['order'] for r in full)} of {len(full)} (chance 1/6)")
        print(f"[ORDER] drawer_stow: scene successes {len(ok)} of {len(rows)}, of which followed the order: {sum(r['entered'] == r['order'] for r in ok)}")
    print("[ORDER] first moved, by colour:", dict(collections.Counter(r["moved"] for r in rows if r["moved"])))


if __name__ == "__main__":
    main()
