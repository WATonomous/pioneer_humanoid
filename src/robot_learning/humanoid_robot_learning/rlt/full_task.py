"""Evaluate a finished RLT run, paired against the base alone:

1. Full task from fresh layouts (the Phase 0 eval's seeds): Pioneer-1 runs natively (replan every 16, keep 6) until
   the handoff, then the TD3 actor takes over in the loop it was trained in (C frames, prefix d, a fresh plan at the
   switch, as after a restore). The same seeds are run base-only, same time limit and drop rule.
2. The critical phase again, from the run's eval start states: base (C:d) and actor, per start, to check the
   training run's single final eval (the base samples its chunks, so each eval differs).

    MUJOCO_GL=egl python -m humanoid_robot_learning.rlt.full_task --policy <pretrained_model> --rlt <rlt out dir>
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from humanoid_robot_learning.rlt.actor_critic import TD3
from humanoid_robot_learning.rlt.base import FrozenBase
from humanoid_robot_learning.rlt.env import VecEnv
from humanoid_robot_learning.rlt.train import FPS, query, run_episodes, summarize


def full_task(base, token, act, venv, seeds, C, d, max_s, use_rl, video_seeds=(), video_dir=None, tag=""):
    """-> {seed: {"success", "time_s", "handoff_s"}}; writes an mp4 for each seed in video_seeds. ``act``(z, s, ref)
    -> the actor's flattened chunk."""
    max_frames = int(max_s * FPS)
    out = {}
    for k in range(0, len(seeds), venv.n):
        group = list(seeds[k:k + venv.n])
        idx = list(range(len(group)))
        venv.call("record", [s in video_seeds for s in group], idx)
        obs = dict(zip(idx, venv.call("reset", group, idx)))
        frames, plans, rl = {i: 0 for i in idx}, {i: None for i in idx}, {i: False for i in idx}
        handoff, video = {i: None for i in idx}, {i: [] for i in idx}
        ok, done = {i: False for i in idx}, {i: False for i in idx}
        while not all(done.values()):
            for phase in (False, True):
                grp = [i for i in idx if not done[i] and rl[i] == phase]
                if not grp:
                    continue
                c, p = (C, d) if phase else (16, 6)            # the actor's loop / Pioneer-1's native replanning
                context, mask, s, chunk = query(base, [obs[i] for i in grp], [plans[i] for i in grp], c, p)
                a = chunk[:, :c]
                if phase:
                    z = token.encode(context.float(), mask)
                    a = act(z, s, a.reshape(len(grp), -1)).view(len(grp), c, -1)
                res = venv.call("step", list(base.unnormalize(a)), grp)
                for j, (i, (o, info)) in enumerate(zip(grp, res)):
                    obs[i], plans[i] = o, chunk[j]
                    frames[i] += info["frames"]
                    video[i] += info["video"]
                    if info["handoff"] and handoff[i] is None:
                        handoff[i] = frames[i] / FPS
                        if use_rl:
                            rl[i], plans[i] = True, None         # the actor starts from a fresh plan, as in training
                    if info["success"]:
                        ok[i] = True
                    done[i] = ok[i] or info["dropped"] or frames[i] >= max_frames
        for i, seed in zip(idx, group):
            out[seed] = {"success": ok[i], "time_s": frames[i] / FPS, "handoff_s": handoff[i]}
            if video[i] and video_dir is not None:
                import imageio
                Path(video_dir).mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(Path(video_dir) / f"{tag}_seed{seed}_{'success' if ok[i] else 'fail'}.mp4", video[i], fps=FPS)
        venv.call("record", [False] * len(idx), idx)
    return out


def paired(a: list[bool], b: list[bool]) -> dict:
    """Counts of (base, rl) outcomes per item and the exact two-sided sign-test p value on the discordant ones."""
    from math import comb
    only_b = sum(1 for x, y in zip(a, b) if not x and y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {"both": sum(1 for x, y in zip(a, b) if x and y), "only_base": only_a, "only_rl": only_b,
            "neither": sum(1 for x, y in zip(a, b) if not x and not y), "sign_test_p": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True, help="the frozen base's pretrained_model directory")
    ap.add_argument("--rlt", required=True, help="the RLT run's output directory (rl_token.pt, td3_final.pt, args.json)")
    ap.add_argument("--actor", default="td3_final.pt")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000, help="first layout seed (the Phase 0 eval's)")
    ap.add_argument("--max_s", type=float, default=30.0, help="full-task time limit (the Phase 0 eval's)")
    ap.add_argument("--videos", type=int, default=6, help="record the first N seeds of each mode")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--skip_critical", action="store_true", help="only the full task")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="keep the training exploration on at run time (TD3.act_wiggle); results go to a _n<std> dir")
    args = ap.parse_args()
    rlt = Path(args.rlt)
    run = json.loads((rlt / "args.json").read_text())
    C, d = run["chunk"], run["prefix"]
    dev = torch.device("cuda")

    base = FrozenBase(args.policy, "cuda")
    token = torch.load(rlt / "rl_token.pt", weights_only=False).to(dev).eval()
    A = base.action_dim
    fixed = None
    if not run["free_gripper"]:
        fixed = torch.zeros(C, A, dtype=torch.bool)
        fixed[:, -1] = True
        fixed = fixed.reshape(-1)
    td3 = TD3(token.to_z[-1].out_features, base.policy.config.robot_state_feature.shape[0], C * A, dev, fixed=fixed)
    td3.actor.load_state_dict(torch.load(rlt / args.actor, weights_only=False)["actor"])
    td3.actor.eval()
    import humanoid_mujoco_scenes.peg_insert.scene as peg_scene
    venv = VecEnv(args.envs, base.cameras, FPS)
    clearance_mm = round(peg_scene.CLEARANCE * 1000, 1)      # PEG_CLEARANCE env var; the workers inherit it
    out_dir = rlt / ("full_task" if clearance_mm == round(run["clearance_m"] * 1000, 1) else f"full_task_{clearance_mm:g}mm")
    if args.noise:
        out_dir = out_dir.with_name(f"{out_dir.name}_n{args.noise:g}")
    act = (lambda z, s, ref: td3.act_wiggle(z, s, ref, args.noise, C)) if args.noise else td3.act
    out_dir.mkdir(exist_ok=True)
    report = {"args": vars(args), "chunk": C, "prefix": d, "clearance_mm": clearance_mm,
              "trained_clearance_mm": round(run["clearance_m"] * 1000, 1)}
    try:
        seeds = list(range(args.seed, args.seed + args.episodes))
        vids = set(seeds[:args.videos])
        res = {}
        for tag, use_rl in (("base", False), ("rlt", True)):
            res[tag] = full_task(base, token, act, venv, seeds, C, d, args.max_s, use_rl, vids, out_dir / "videos", tag)
            ok = [res[tag][s]["success"] for s in seeds]
            report[f"full_{tag}"] = summarize([(o, res[tag][s]["time_s"]) for o, s in zip(ok, seeds)])
            report[f"full_{tag}"]["reached_handoff"] = sum(res[tag][s]["handoff_s"] is not None for s in seeds)
            print(json.dumps({"stage": f"full_{tag}", **report[f"full_{tag}"]}), flush=True)
        report["full_paired"] = paired([res["base"][s]["success"] for s in seeds], [res["rlt"][s]["success"] for s in seeds])
        report["full_per_seed"] = {s: {t: res[t][s] for t in res} for s in seeds}
        print(json.dumps({"stage": "full_paired", **report["full_paired"]}), flush=True)

        if not args.skip_critical:
            _, eval_starts = pickle.loads((rlt / "starts.pkl").read_bytes())
            max_frames = int(run["episode_s"] * FPS)
            b = run_episodes(base, token, venv, eval_starts, lambda z, s, ref: ref, C, max_frames, d)
            r = run_episodes(base, token, venv, eval_starts, act, C, max_frames, d)
            report["critical_base"], report["critical_rlt"] = summarize(b), summarize(r)
            report["critical_paired"] = paired([x for x, _ in b], [x for x, _ in r])
            for k in ("critical_base", "critical_rlt", "critical_paired"):
                print(json.dumps({"stage": k, **report[k]}), flush=True)
    finally:
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
        venv.close()


if __name__ == "__main__":
    main()
