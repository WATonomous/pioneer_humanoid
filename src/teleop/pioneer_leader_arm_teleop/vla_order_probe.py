"""Which object does a language-conditioned LeRobot policy (pi0.5, SmolVLA, ...) go for first?

    MUJOCO_GL=egl python vla_order_probe.py --scene drawer_stow --policy <pretrained_model> --episodes 100

The same question order_probe.py asks of ACT checkpoints, for policies that take the instruction as text and are
rolled out through sim_policy_eval.py. The policy gets each step's instruction from the scene, as in training.
Reported against the first object the scene announces (chance 1/3): the first one displaced 15 mm, the first one
lifted 3 cm, and for drawer_stow the first one resting in the drawer. An episode ends at the first lift (or entry).
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import sim_policy_eval as E  # noqa: E402
from synthetic_teleop import TeleopSim  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", required=True, choices=["duplo", "drawer_stow"])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_s", type=float, default=0.0, help="default: 25 s for duplo, 60 s for drawer_stow (it opens the drawer first)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    S = importlib.import_module(f"humanoid_mujoco_scenes.{args.scene}.scene")
    body = "brick_{}" if args.scene == "duplo" else "block_{}"
    path = Path(args.policy).resolve()
    policy, pre, post, cfg, cameras = E.load_policy(path, args.device)
    rows = []
    for seed in range(args.seed, args.seed + args.episodes):
        sim = TeleopSim(args.scene, seed, {n: E.RENDER_SIZE for n in cameras})
        model, data = sim.model, sim.data
        policy.reset(), pre.reset(), post.reset()
        order = list(S._order(data))
        start = {c: data.body(body.format(c)).xpos.copy() for c in S.COLOURS}
        moved = lifted = entered = None
        for _ in range(int((args.max_s or (25.0 if args.scene == "duplo" else 60.0)) * 25)):
            _, _, instruction = sim.progress(model, data)
            E.policy_frame(sim, policy, pre, post, torch.device(args.device), 25, instruction, sizes=cameras)
            for c in S.COLOURS:
                pos = data.body(body.format(c)).xpos
                if moved is None and np.linalg.norm(pos - start[c]) > 0.015:
                    moved = c
                if lifted is None and pos[2] > start[c][2] + 0.03:
                    lifted = c
                if args.scene == "drawer_stow" and entered is None and S.in_drawer(model, data, c):
                    entered = c
            if lifted or entered:
                break
        sim.close()
        rows.append({"seed": seed, "order": order, "moved": moved, "lifted": lifted, "entered": entered})
        print(f"[ORDER] seed {seed}: announced {order[0]:6s} moved {moved} lifted {lifted} entered {entered}   "
              f"moved right {sum(r['moved'] == r['order'][0] for r in rows)}/{sum(r['moved'] is not None for r in rows)}", flush=True)
    summary = {"scene": args.scene, "policy": str(path), "episodes": len(rows), "results": rows}
    for key, label in (("moved", "first displaced 15 mm"), ("lifted", "first lifted 3 cm"), ("entered", "first in the drawer")):
        got = [r for r in rows if r[key]]
        if got or key != "entered":
            summary[key] = [sum(r[key] == r["order"][0] for r in got), len(got)]
            print(f"[ORDER] {args.scene}: {label} = the announced first: {summary[key][0]} of {len(got)}  (chance 1/3)")
    print("[ORDER] first moved, by colour:", dict(collections.Counter(r["moved"] for r in rows if r["moved"])))
    out = Path(args.out) if args.out else path.parent / f"vla_order_probe_{args.scene}.json"
    out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
