"""Demo videos from a trained BC policy (results/bc/<tag>.pt): one video per episode, with a close third-person view
of the peg and hole next to the policy's own two cameras. Re-runs the policy closed-loop; nothing is retrained.

    .venv/bin/python demo.py --tag pioneer_v3 --seeds 7000:7008
    .venv/bin/python demo.py --tag pioneer_v3 --seeds 7000:7004 --variant light      # relit scene
    .venv/bin/python demo.py --tag pioneer_v3 --seeds 7000:7004 --variant camera     # held-out camera move
    .venv/bin/python demo.py --tag pioneer_v3 --seeds 7000:7004 --dark               # colour camera dead

Writes figures/demo/<tag>_<condition>_<seed>.mp4 and prints success per episode.
"""
from __future__ import annotations

import argparse
import json
import types

import torch

import bc
import common as C
import video_util
from probe import parse_config


def load_policy(tag):
    ck = torch.load(C.OUT / "bc" / f"{tag}.pt", map_location="cpu", weights_only=False)   # our own file (holds numpy norm stats)
    streams = parse_config(ck["streams"]) if ck["streams"] != "none" else []
    vision = bc.Vision(streams, finetune=False, pool=ck.get("pool", 1))
    if ck["vision"]:
        for k, sd in ck["vision"].items():
            vision.encs[k].load_state_dict({n: v.float() if v.is_floating_point() else v for n, v in sd.items()})
    policy = bc.make_policy(vision.shapes(), ck["norm"], ck.get("dim", 7), ck.get("cvae", False)).cuda()
    policy.load_state_dict({n: v.float() if v.is_floating_point() else v for n, v in ck["policy"].items()})
    return vision, policy.eval(), ck["norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="7000:7004", help="a:b, eval layouts (bc.py evaluates 7000:7100)")
    ap.add_argument("--variant", default="clean", choices=("clean", "light", "color", "camera", "fog", "glare", "lens", "shadow", "clutter", "background", "table"))
    ap.add_argument("--dark", action="store_true", help="colour camera dead")
    ap.add_argument("--noise", action="store_true", help="sensor-noised depth")
    ap.add_argument("--ens_m", type=float, default=0.6, help="temporal ensembling, as bc.py (0.6; the Nov 11 armv2 runs used 0.1)")
    args = ap.parse_args()
    a, b = map(int, args.seeds.split(":"))
    seeds = list(range(a, b))
    vision, policy, norm = load_policy(args.tag)
    cond = args.variant + ("_dark" if args.dark else "") + ("_noise" if args.noise else "")
    frames = []
    ev = types.SimpleNamespace(ens_m=args.ens_m, eval_batch=len(seeds))
    with torch.no_grad():
        res = bc.evaluate(ev, vision, policy, norm, seeds, test_noise=args.noise, video=frames, variant=args.variant,
                          dark=args.dark, video_n=len(seeds))
    out = C.FIG / "demo"
    out.mkdir(exist_ok=True)
    nice = {"clean": "clean", "light": "lighting changed", "color": "colours changed", "camera": "cameras moved (held-out)", "fog": "fog", "glare": "sun glare", "lens": "wet lens",
            "shadow": "hard shadows", "clutter": "clutter", "background": "new background", "table": "new tablecloth (held out)"}
    for vi, (seed, r) in enumerate(zip(seeds, res)):
        fr = [(0, it, o, st) for v, it, o, st in frames if v == vi]
        label = nice[args.variant] + (", colour camera dead" if args.dark else "") + (", noisy depth" if args.noise else "")
        video_util.rollout_video(fr, out / f"{args.tag}_{cond}_{seed}.mp4", args.tag,
                                 title=lambda _, s=seed, l=label: f"policy: {args.tag}   layout {s}   {l}")
        print(f"seed {seed}: {'SUCCESS' if r['success'] else 'fail'}  (" +
              (f"closest peg-hole {r['min_align_mm']:.1f} mm, " if C.SCENE == "peg_insert" else "") + f"{r['time']:.1f}s)")
    json.dump({str(s): r for s, r in zip(seeds, res)}, open(out / f"{args.tag}_{cond}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
