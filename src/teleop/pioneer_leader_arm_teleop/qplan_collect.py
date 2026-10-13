"""Roll a frozen ACT policy out under a candidate selector and save chunk-level transitions for Q-Planning.

    python qplan_collect.py --scene peg_insert --policy <ckpt>/pretrained_model --out rollouts/r0.pt \
        --seed 10000 --episodes 200 --select random            # exploration data (random candidate per chunk)
    python qplan_collect.py ... --select first                  # the frozen policy itself (baseline)
    python qplan_collect.py ... --select qavg --q q.pt          # the planner (softmax-Q-weighted average)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import qplan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="peg_insert")
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--select", default="first", choices=["first", "random", "qavg", "qmax"])
    p.add_argument("--q", default=None, help="Q-function checkpoint (for qavg / qmax)")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--z_std", type=float, default=1.0)
    p.add_argument("--noise_std", type=float, default=0.0, help="temporally smooth noise on the candidates (normalised action units)")
    p.add_argument("--offset_std", type=float, default=0.0, help="one constant joint offset per candidate (normalised action units)")
    p.add_argument("--oracle", action="store_true", help="pick each chunk by simulator lookahead over the candidates (upper bound; also labels them)")
    p.add_argument("--seeds", default=None, help="comma-separated layout seeds instead of --seed/--episodes")
    p.add_argument("--retry_after_s", type=float, default=0.0, help="no success this long after the start / last retry: rewind and retry (0 = off)")
    p.add_argument("--rewind_s", type=float, default=4.0)
    p.add_argument("--retry_offset_std", type=float, default=0.005)
    p.add_argument("--jitter_scale", type=float, default=1.0, help="peg_insert: scale the peg/block position randomisation (robustness test)")
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--max_s", type=float, default=40.0)
    p.add_argument("--n_exec", type=int, default=0, help="actions executed per chunk (default: the whole chunk)")
    p.add_argument("--probe", action="store_true", help="print the spread of the candidates")
    args = p.parse_args()

    if args.jitter_scale != 1.0:
        from humanoid_mujoco_scenes.peg_insert import scene as peg_scene
        peg_scene.PEG_JITTER *= args.jitter_scale
        peg_scene.BLOCK_JITTER *= args.jitter_scale
    prop = qplan.Proposer(args.policy)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    if args.select == "first":
        select = qplan.pick_first
    elif args.select == "random":
        select = qplan.pick_random(rng)
    else:
        ck = torch.load(args.q, map_location="cuda", weights_only=False)
        qnet = qplan.QNet(**ck["arch"]).cuda().eval()
        qnet.load_state_dict(ck["ema"])
        hl = qplan.HLGauss(device="cuda")
        select = qplan.q_weighted(qnet, hl, args.temperature) if args.select == "qavg" else qplan.best_of_n(qnet, hl)

    spread = []
    if args.probe:
        inner = select

        def select(chunks, obs_feat, info):  # noqa: F811
            real = prop.unnormalise(chunks)
            spread.append(real.std(0).mean(0).cpu().numpy())      # per joint: std over candidates, mean over the chunk
            return inner(chunks, obs_feat, info)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    episodes, t0 = [], time.time()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed + i for i in range(args.episodes)]
    for i, seed in enumerate(seeds):
        if i and i % 50 == 0:
            torch.save({"args": vars(args), "episodes": episodes}, out)      # partial save: a killed run keeps its rollouts
        ep = qplan.rollout(args.scene, seed, prop, select, args.n, args.z_std, args.noise_std, args.max_s,
                           rng=gen, n_exec=args.n_exec or None, offset_std=args.offset_std, oracle=args.oracle,
                           retry_after_s=args.retry_after_s, rewind_s=args.rewind_s, retry_offset_std=args.retry_offset_std)
        episodes.append(ep)
        done = sum(e["success"] for e in episodes)
        print(f"[QPLAN] seed {ep['seed']}: {'ok  ' if ep['success'] else 'FAIL'} {ep['frames'] / qplan.FPS:5.1f} s  {len(ep['steps'])} chunks"
              f"{'' if ep['progress'] is None else '  step %d/%d' % ep['progress']}   running {done}/{len(episodes)} ({time.time() - t0:.0f} s wall)", flush=True)
    torch.save({"args": vars(args), "episodes": episodes}, out)
    rate = sum(e["success"] for e in episodes) / len(episodes)
    print(f"[QPLAN] {args.scene} select={args.select}: success {rate:.1%} ({sum(e['success'] for e in episodes)}/{len(episodes)})  -> {out}")
    if spread:
        s = np.array(spread)
        print("[QPLAN] candidate spread, rad (gripper: closure), mean over planning steps:", np.round(s.mean(0), 4),
              " max:", np.round(s.max(0), 4))


if __name__ == "__main__":
    main()
