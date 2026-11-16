"""Re-evaluate a trained BC policy (results*/bc/<tag>.pt) with different execution settings, no retraining:
how overlapping action chunks are combined, and whether the gripper command is snapped to open / closed.

    .venv/bin/python eval_policy.py --tag pioneer_v3 --exec ensemble --ens_m 0.1          # bc.py's default
    .venv/bin/python eval_policy.py --tag pioneer_v3 --grip_snap 0.5
    .venv/bin/python eval_policy.py --tag pioneer_v3 --exec newest --variant light

Same 100 eval layouts (7000+) as bc.py. Writes <results>/bc_exec/<tag>__<setting>.json.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import bc
import common as C
from demo import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exec_mode", "--exec", default="ensemble", choices=("ensemble", "newest"))
    ap.add_argument("--ens_m", type=float, default=0.1)
    ap.add_argument("--grip_snap", type=float, default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=7000,
                    help="first eval layout: 7000 = the test layouts bc.py reports on; 8000 = validation layouts, "
                         "for choosing execution settings without touching the test set")
    ap.add_argument("--eval_batch", type=int, default=25)
    ap.add_argument("--variant", default="clean", choices=("clean", "light", "color", "camera", "fog", "glare", "lens", "shadow", "clutter", "background", "table"))
    args = ap.parse_args()
    C.contract_check()
    t0 = time.time()
    vision, policy, norm = load_policy(args.tag)
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    with torch.no_grad():
        res = bc.evaluate(args, vision, policy, norm, seeds, variant=args.variant)
    s = bc.summarize(res)
    name = f"{args.exec_mode}_m{args.ens_m:g}" + (f"_snap{args.grip_snap:g}" if args.grip_snap is not None else "") + \
        ("" if args.variant == "clean" else f"_{args.variant}") + ("" if args.seed0 == 7000 else f"_val{args.seed0}")
    out = C.OUT / "bc_exec"
    out.mkdir(exist_ok=True)
    json.dump(dict(tag=args.tag, setting=name, robot=C.ROBOT, summary=s, raw=res, secs=time.time() - t0),
              open(out / f"{args.tag}__{name}.json", "w"), indent=1, default=float)
    print(f"[{args.tag} | {name}] seeds {args.seed0}+ success {s['success']:.3f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}] "
          f"grasp {s['grasped']:.2f} align {s['aligned']:.2f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
