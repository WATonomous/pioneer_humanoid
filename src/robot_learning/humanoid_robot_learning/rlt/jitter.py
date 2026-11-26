"""How jerky the commanded arm motion is in the critical phase, and inference-time fixes for it, on a finished RLT run.

Jitter = mean |x[t] - 2 x[t-1] + x[t-2]| of the commanded arm joint targets (rad), per frame, over all episodes; a
chunk boundary where the plan jumps shows up as a spike. Each variant runs every eval start state once and also
reports success, so a smoother setting that stops inserting is visible.

    MUJOCO_GL=egl python -m humanoid_robot_learning.rlt.jitter --policy <pretrained_model> --rlt <rlt out dir>
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from humanoid_robot_learning.rlt.actor_critic import TD3
from humanoid_robot_learning.rlt.base import FrozenBase
from humanoid_robot_learning.rlt.env import VecEnv
from humanoid_robot_learning.rlt.train import FPS, query, summarize


def rollouts(base, token, td3, venv, starts, C, d, max_frames, use_rl, ema=0.0, residual=False, noise=0.0):
    """-> per start (success, seconds), per start (T, A) commanded actions. ``ema``: each executed command is
    ema * previous command + (1 - ema) * the policy's (the arm joints only; the gripper passes through).
    ``residual``: filter only the actor's correction (its action minus the base's), so the base's own motion is
    not delayed. ``noise``: the training exploration (one Gaussian offset per chunk, normalized units) kept on at run
    time, a deliberate wiggle in place of the one an unpenalized actor learns."""
    results, cmds = [], []
    for k in range(0, len(starts), venv.n):
        group = starts[k:k + venv.n]
        idx = list(range(len(group)))
        obs = dict(zip(idx, venv.call("restore", group, idx)))
        plans, last = {i: None for i in idx}, {i: None for i in idx}
        hist = {i: [] for i in idx}
        out, active, frames = {i: (False, None) for i in idx}, list(idx), {i: 0 for i in idx}
        while active:
            context, mask, s, chunk = query(base, [obs[i] for i in active], [plans[i] for i in active], C, d)
            a = chunk[:, :C]
            ref_acts = base.unnormalize(a)
            if use_rl:
                z = token.encode(context.float(), mask)
                ref = a.reshape(len(active), -1)
                a = (td3.act_wiggle(z, s, ref, noise, C) if noise else td3.act(z, s, ref)).view(len(active), C, -1)
            acts = base.unnormalize(a)
            for j, i in enumerate(active):
                plans[i] = chunk[j]
                if ema and residual:
                    for t in range(C):
                        r = acts[j, t, :-1] - ref_acts[j, t, :-1]
                        if last[i] is not None:
                            r = ema * last[i] + (1 - ema) * r
                        last[i] = r.copy()
                        acts[j, t, :-1] = ref_acts[j, t, :-1] + r
                elif ema:
                    for t in range(C):
                        if last[i] is not None:
                            acts[j, t, :-1] = ema * last[i][:-1] + (1 - ema) * acts[j, t, :-1]
                        last[i] = acts[j, t].copy()
            res = venv.call("step", list(acts), active)
            still = []
            for j, (i, (o, info)) in enumerate(zip(active, res)):
                obs[i] = o
                hist[i].append(acts[j, :info["frames"]])
                frames[i] += info["frames"]
                if info["success"]:
                    out[i] = (True, frames[i] / FPS)
                elif not info["dropped"] and frames[i] < max_frames:
                    still.append(i)
            active = still
        results += [out[i] for i in idx]
        cmds += [np.concatenate(hist[i]) for i in idx]
    return results, cmds


def jitter(cmds) -> dict:
    d2 = np.concatenate([np.abs(np.diff(c[:, :-1], 2, axis=0)) for c in cmds if len(c) > 2])
    return {"jitter_mrad": round(float(d2.mean()) * 1000, 3), "jitter_p99_mrad": round(float(np.percentile(d2, 99)) * 1000, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--rlt", required=True)
    ap.add_argument("--actor", default="td3_final.pt")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--variants", default="base:16:6,base:5:2,rl:5:2,rl:5:2:0.5,rl:5:2:0.75,rl:5:5",
                    help="mode:C:prefix[:ema[:res]][:n<std>], comma separated; n<std> keeps exploration noise on")
    ap.add_argument("--report", default="jitter.json", help="file name in --rlt")
    args = ap.parse_args()
    rlt = Path(args.rlt)
    run = json.loads((rlt / "args.json").read_text())
    dev = torch.device("cuda")
    base = FrozenBase(args.policy, "cuda")
    token = torch.load(rlt / "rl_token.pt", weights_only=False).to(dev).eval()
    C0, A = run["chunk"], base.action_dim
    fixed = None
    if not run["free_gripper"]:
        fixed = torch.zeros(C0, A, dtype=torch.bool)
        fixed[:, -1] = True
        fixed = fixed.reshape(-1)
    td3 = TD3(token.to_z[-1].out_features, base.policy.config.robot_state_feature.shape[0], C0 * A, dev, fixed=fixed)
    td3.actor.load_state_dict(torch.load(rlt / args.actor, weights_only=False)["actor"])
    _, starts = pickle.loads((rlt / "starts.pkl").read_bytes())
    max_frames = int(run["episode_s"] * FPS)
    venv = VecEnv(args.envs, base.cameras, FPS)
    report = {}
    try:
        for spec in args.variants.split(","):
            mode, c, p, *rest = spec.split(":")          # mode:C:prefix[:ema[:res]][:n<std>]
            noise = [float(r[1:]) for r in rest if r.startswith("n")]
            rest = [r for r in rest if not r.startswith("n")]
            c, p, ema = int(c), int(p), float(rest[0]) if rest else 0.0
            residual = len(rest) > 1 and rest[1] == "res"
            if noise and mode != "rl":
                raise SystemExit(f"{spec}: noise is on the actor's output")
            if mode == "rl" and c != C0:
                raise SystemExit(f"{spec}: the actor was trained on chunks of {C0}")
            res, cmds = rollouts(base, token, td3, venv, starts, c, p, max_frames, mode == "rl", ema, residual,
                                 noise[0] if noise else 0.0)
            report[spec] = {**summarize(res), **jitter(cmds)}
            print(json.dumps({"variant": spec, **report[spec]}), flush=True)
    finally:
        (rlt / args.report).write_text(json.dumps(report, indent=1))
        venv.close()


if __name__ == "__main__":
    main()
