"""RLT on a frozen Pioneer-1, peg_insert critical phase. One run, all stages:

    MUJOCO_GL=egl python -m humanoid_robot_learning.rlt.train --policy <run>/checkpoints/last/pretrained_model --out rlt_out

1. Start states: the base runs from fresh layouts (held-out seeds) until the handoff (peg held above the block top,
   within 3 cm of the hole; env.py) and the MuJoCo state is saved. Train and eval start states use disjoint seeds.
   The base's context tokens seen along the way are kept for step 2.
2. RL token: a small encoder/decoder (token.py) trained to reconstruct those context tokens through one vector.
3. Baseline: the base alone from every eval start state, replanning every C frames (as the RL agent does).
4. TD3 (actor_critic.py) on (token, state, base chunk), chunks of C frames, reward 1 when the peg stays inserted
   0.4 s, episode limit 10 s. Replay warmed with base rollouts; UTD ratio 5; evaluated every --eval_every transitions.

Deviations from the paper (arXiv 2604.23073), for a first sim result on a small budget: the token is trained on
the base's own rollouts (not demos) with a parallel (not autoregressive) decoder; no chunk subsampling at stride 2;
the 10 s timeout is treated as terminal; exploration noise is one offset per chunk (consistent motion, mm scale).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from humanoid_robot_learning.rlt.actor_critic import TD3, Replay
from humanoid_robot_learning.rlt.base import FrozenBase
from humanoid_robot_learning.rlt.env import VecEnv
from humanoid_robot_learning.rlt.token import RLToken

FPS = 25


def log(out: Path, **kv) -> None:
    kv["wall"] = round(time.time() - T0, 1)
    print(json.dumps(kv), flush=True)
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(kv) + "\n")


def collect_starts(base, venv, seeds, ctx_keep, ctx_every=2, max_s=12.0, replan=8):
    """Run the base from reset(seed) until the handoff; return snapshots, and append context tokens to ctx_keep."""
    snaps, todo, n = [], list(seeds), venv.n
    stats = {"tried": 0, "no_handoff": 0}
    while todo:
        batch, todo = todo[:n], todo[n:]
        idx = list(range(len(batch)))
        obs = venv.call("reset", batch, idx)
        active, t, q = list(idx), 0, 0
        stats["tried"] += len(batch)
        while active and t < max_s * FPS:
            context, mask, _, chunk = base([obs[i] for i in active])
            if q % ctx_every == 0 and len(ctx_keep) < ctx_keep.cap:
                ctx_keep.add(context, mask)
            q += 1
            acts = base.unnormalize(chunk[:, :replan])
            res = venv.call("step", list(acts), active)
            t += replan
            done = []
            for k, (i, (o, info)) in enumerate(zip(active, res)):
                obs[i] = o
                if info["handoff"]:
                    snaps.append(venv.call("snapshot", [None], [i])[0])
                    done.append(i)
                elif info["success"] or info["dropped"]:
                    done.append(i)        # inserted without a handoff seen, or dropped: no start state from it
            active = [i for i in active if i not in done]
        stats["no_handoff"] = stats["tried"] - len(snaps)
    return snaps, stats


class ContextStore:
    def __init__(self, cap: int):
        self.ctx, self.mask, self.cap = [], [], cap

    def __len__(self):
        return sum(c.shape[0] for c in self.ctx)

    def add(self, context, mask):
        self.ctx.append(context.half().cpu())
        self.mask.append(None if mask is None else mask.cpu())

    def tensors(self):
        ctx = torch.cat(self.ctx)
        mask = None if self.mask[0] is None else torch.cat(self.mask)
        return ctx, mask


def train_token(store, device, steps, batch=64):
    ctx, mask = store.tensors()
    token = RLToken(ctx.shape[-1], ctx.shape[1]).to(device)
    opt = torch.optim.AdamW(token.parameters(), lr=3e-4, weight_decay=1e-4)
    for i in range(steps):
        idx = torch.randint(0, ctx.shape[0], (batch,))
        loss = token.loss(ctx[idx].to(device).float(), None if mask is None else mask[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % 250 == 0 or i == steps - 1:
            yield i, loss.item()
    token.eval()
    for p in token.parameters():
        p.requires_grad_(False)
    train_token.result = token


def query(base, obs, plans, C, d):
    """The base on each observation. An env with a previous plan gets that plan's actions C..C+d as the RTC prefix
    (they continue the motion already under way); a fresh one (just restored) gets none. -> context, mask, s, chunk."""
    n = len(obs)
    has = [i for i in range(n) if d and plans[i] is not None]
    fresh = [i for i in range(n) if i not in has]
    parts = [None] * n
    for group, use in ((has, True), (fresh, False)):
        if group:
            pre = torch.stack([plans[i][C:C + d] for i in group]) if use else None
            c, m, s, ch = base([obs[i] for i in group], prefix=pre)
            for k, i in enumerate(group):
                parts[i] = (c[k], None if m is None else m[k], s[k], ch[k])
    context = torch.stack([p[0] for p in parts])
    mask = None if parts[0][1] is None else torch.stack([p[1] for p in parts])
    return context, mask, torch.stack([p[2] for p in parts]), torch.stack([p[3] for p in parts])


def run_episodes(base, token, venv, starts, policy_fn, C, max_frames, d=0):
    """Each start once with a deterministic policy_fn(z, s, ref) -> a, C frames per step, RTC prefix d, at most
    max_frames frames; returns per-start (success, seconds)."""
    max_chunks = int(np.ceil(max_frames / C))
    results = []
    for k in range(0, len(starts), venv.n):
        group = starts[k:k + venv.n]
        idx = list(range(len(group)))
        obs = venv.call("restore", group, idx)
        active, frames = list(idx), {i: 0 for i in idx}
        out = {i: (False, None) for i in idx}
        plans = {i: None for i in idx}
        for _ in range(max_chunks):
            if not active:
                break
            context, mask, s, chunk = query(base, [obs[i] for i in active], [plans[i] for i in active], C, d)
            for k2, i in enumerate(active):
                plans[i] = chunk[k2]
            z = token.encode(context.float(), mask)
            ref = chunk[:, :C].reshape(len(active), -1)
            a = policy_fn(z, s, ref)
            res = venv.call("step", list(base.unnormalize(a.view(len(active), C, -1))), active)
            still = []
            for i, (o, info) in zip(active, res):
                obs[i], frames[i] = o, frames[i] + info["frames"]
                if info["success"]:
                    out[i] = (True, frames[i] / FPS)
                elif not info["dropped"]:
                    still.append(i)
            active = still
        results += [out[i] for i in idx]
    return results


def summarize(results):
    from humanoid_robot_learning.sim.evaluate import wilson
    k, n = sum(ok for ok, _ in results), len(results)
    times = [t for ok, t in results if ok]
    lo, hi = wilson(k, n)
    return {"success": k, "n": n, "rate": k / n, "ci95": [round(lo, 3), round(hi, 3)],
            "mean_time_s": round(float(np.mean(times)), 2) if times else None,
            "median_time_s": round(float(np.median(times)), 2) if times else None}


def main():
    global T0
    T0 = time.time()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--policy", required=True, help="Pioneer-1 pretrained_model directory (frozen base)")
    p.add_argument("--out", default="rlt_out")
    p.add_argument("--envs", type=int, default=6)
    p.add_argument("--train_starts", type=int, default=240, help="layouts tried for training start states")
    p.add_argument("--eval_starts", type=int, default=48, help="layouts tried for eval start states")
    p.add_argument("--token_steps", type=int, default=2000)
    p.add_argument("--chunk", type=int, default=5, help="C: frames per RL action (0.2 s at 25 fps)")
    p.add_argument("--prefix", type=int, default=5, help="RTC prefix: actions of the base's last plan kept at the start"
                   " of each new chunk (Pioneer-1 is trained with up to 6; 0 = none, which breaks its motion)")
    p.add_argument("--baselines", default="5:0,5:2,5:5,16:6",
                   help="base-only evals before RL, as C:prefix pairs (16:6 = how Pioneer-1 runs natively)")
    p.add_argument("--baseline_only", action="store_true", help="stop after the baselines")
    p.add_argument("--episode_s", type=float, default=10.0)
    p.add_argument("--warmup", type=int, default=2000, help="base-rollout transitions before RL updates")
    p.add_argument("--utd", type=int, default=5)
    p.add_argument("--beta", type=float, default=10.0, help="penalty on ||a - a_ref||^2 (mean over values); 1 let the "
                   "actor drift cm away and knock the peg loose")
    p.add_argument("--free_gripper", action="store_true", help="let the actor change the gripper command (default: "
                   "pinned to the base's, insertion never needs it)")
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--noise", type=float, default=0.01, help="exploration std, normalized action units (~2-3 mm)")
    p.add_argument("--ref_dropout", type=float, default=0.5)
    p.add_argument("--minutes", type=float, default=60.0, help="RL wall-clock budget")
    p.add_argument("--eval_every", type=int, default=3000, help="transitions between evals")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import humanoid_mujoco_scenes.peg_insert.scene as peg_scene
    args.clearance_m = peg_scene.CLEARANCE           # PEG_CLEARANCE env var; the workers inherit it
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dev = torch.device("cuda")

    base = FrozenBase(args.policy, "cuda")
    venv = VecEnv(args.envs, base.cameras, FPS)
    C, A = args.chunk, base.action_dim
    max_frames = int(args.episode_s * FPS)
    max_chunks = max_frames // C
    if args.prefix > base.policy.config.rtc_max_prefix or C + args.prefix > base.horizon:
        raise SystemExit(f"--prefix {args.prefix}: Pioneer-1 takes at most {base.policy.config.rtc_max_prefix} and needs "
                         f"chunk + prefix <= horizon {base.horizon}")
    try:
        # 1. start states (cached: a rerun reuses them)
        cache = out / "starts.pkl"
        store = ContextStore(cap=4000)
        if cache.exists():
            train_starts, eval_starts = pickle.loads(cache.read_bytes())
            log(out, stage="starts", cached=True, train=len(train_starts), eval=len(eval_starts))
        else:
            eval_starts, st = collect_starts(base, venv, range(5000, 5000 + args.eval_starts), store)
            log(out, stage="starts", split="eval", got=len(eval_starts), **st)
            train_starts, st = collect_starts(base, venv, range(6000, 6000 + args.train_starts), store)
            log(out, stage="starts", split="train", got=len(train_starts), contexts=len(store), **st)
            cache.write_bytes(pickle.dumps((train_starts, eval_starts)))
        if not train_starts or not eval_starts:
            raise SystemExit("the base never reached the handoff: no start states")

        # 2. RL token
        tok_path = out / "rl_token.pt"
        if tok_path.exists():
            token = torch.load(tok_path, weights_only=False).to(dev)
        else:
            if not len(store):
                raise SystemExit("token not trained and no contexts cached: delete starts.pkl and rerun")
            for i, loss in train_token(store, dev, args.token_steps):
                log(out, stage="token", step=i, loss=round(loss, 4))
            token = train_token.result
            torch.save(token, tok_path)
        del store

        # 3. baselines: the base alone, at several replanning settings; the one RL uses is C:prefix
        baselines = {}
        for spec in args.baselines.split(","):
            bc, bd = (int(x) for x in spec.split(":"))
            baselines[spec] = summarize(run_episodes(base, token, venv, eval_starts, lambda z, s, ref: ref, bc, max_frames, bd))
            log(out, stage="baseline", setting=spec, **baselines[spec])
        rl_spec = f"{C}:{args.prefix}"
        if rl_spec not in baselines:
            baselines[rl_spec] = summarize(run_episodes(base, token, venv, eval_starts, lambda z, s, ref: ref, C, max_frames,
                                                        args.prefix))
            log(out, stage="baseline", setting=rl_spec, **baselines[rl_spec])
        baseline = baselines[rl_spec]
        (out / "baselines.json").write_text(json.dumps(baselines, indent=1))
        if args.baseline_only:
            return

        # 4. TD3
        zd, sd, ad = token.to_z[-1].out_features, base.policy.config.robot_state_feature.shape[0], C * A
        fixed = None
        if not args.free_gripper:                 # the last value of each step is the gripper command
            fixed = torch.zeros(C, A, dtype=torch.bool)
            fixed[:, -1] = True
            fixed = fixed.reshape(-1)
        td3 = TD3(zd, sd, ad, dev, beta=args.beta, gamma=args.gamma, ref_dropout=args.ref_dropout,
                  policy_noise=args.noise, noise_clip=2 * args.noise, fixed=fixed)
        rb = Replay(200_000, {"z": zd, "s": sd, "ref": ad, "a": ad, "r": 0, "d": 0, "z2": zd, "s2": sd, "ref2": ad}, dev)
        rng = np.random.default_rng(args.seed)
        n = venv.n
        obs = venv.call("restore", [train_starts[i] for i in rng.integers(0, len(train_starts), n)])
        chunks_in_ep = np.zeros(n, int)
        plans = [None] * n                        # each env's last base plan, for the RTC prefix
        pending = [None] * n                      # (z, s, ref, a, r, d) waiting for the next state's z2, s2, ref2
        transitions, episodes, ep_success, ep_dropped, ep_len, next_eval, evals = 0, 0, [], [], [], args.eval_every, []
        deadline = time.time() + args.minutes * 60
        while time.time() < deadline:
            context, mask, s, chunk = query(base, obs, plans, C, args.prefix)
            plans = list(chunk)
            z = token.encode(context.float(), mask)
            ref = chunk[:, :C].reshape(n, -1)
            for i in range(n):                   # complete last step's transitions with this state
                if pending[i] is not None:
                    zz, ss, rr, aa, r, d = pending[i]
                    rb.add(z=zz[None], s=ss[None], ref=rr[None], a=aa[None], r=torch.tensor([r]), d=torch.tensor([d]),
                           z2=z[i:i + 1], s2=s[i:i + 1], ref2=ref[i:i + 1])
                    pending[i] = None
            warm = transitions < args.warmup
            if warm:
                a = ref
            else:
                a = td3.act(z, s, ref)
                noise = torch.randn(n, 1, A, device=dev) * args.noise           # one offset per chunk
                a = td3.fix((a.view(n, C, A) + noise).clamp(-1, 1).reshape(n, -1), ref)
            res = venv.call("step", list(base.unnormalize(a.view(n, C, A))))
            restarts = []
            for i, (o, info) in enumerate(res):
                obs[i] = o
                chunks_in_ep[i] += 1
                r = float(info["success"])
                done = info["success"] or info["dropped"] or chunks_in_ep[i] >= max_chunks
                if done:                          # terminal: next-state fields are unused (d = 1)
                    rb.add(z=z[i:i + 1], s=s[i:i + 1], ref=ref[i:i + 1], a=a[i:i + 1], r=torch.tensor([r]),
                           d=torch.tensor([1.0]), z2=z[i:i + 1], s2=s[i:i + 1], ref2=ref[i:i + 1])
                    episodes += 1
                    ep_success.append(r)
                    ep_dropped.append(float(info["dropped"]))
                    ep_len.append(int(chunks_in_ep[i]))
                    restarts.append(i)
                else:
                    pending[i] = (z[i], s[i], ref[i], a[i], r, 0.0)
                transitions += 1
            if restarts:
                new = venv.call("restore", [train_starts[j] for j in rng.integers(0, len(train_starts), len(restarts))],
                                restarts)
                for i, o in zip(restarts, new):
                    obs[i], chunks_in_ep[i], plans[i] = o, 0, None
            if not warm:
                for _ in range(args.utd * n):
                    stats = td3.update(rb.sample(256))
            if transitions >= next_eval:
                next_eval += args.eval_every
                res_eval = run_episodes(base, token, venv, eval_starts, lambda z_, s_, r_: td3.act(z_, s_, r_), C, max_frames,
                                        args.prefix)
                ev = summarize(res_eval)
                evals.append({"transitions": transitions, **ev})
                recent = ep_success[-200:]
                log(out, stage="rl_eval", transitions=transitions, episodes=episodes,
                    train_success_recent=round(float(np.mean(recent)), 3) if recent else None,
                    train_dropped_recent=round(float(np.mean(ep_dropped[-200:])), 3) if ep_dropped else None,
                    train_ep_chunks_recent=round(float(np.mean(ep_len[-200:])), 1) if ep_len else None,
                    **ev, **({k: round(v, 4) for k, v in stats.items()} if not warm else {}))
                torch.save({"actor": td3.actor.state_dict(), "critic": td3.critic.state_dict()}, out / f"td3_{transitions}.pt")
                # the obs of the training envs are stale after the eval reused the workers: restart them
                obs = venv.call("restore", [train_starts[j] for j in rng.integers(0, len(train_starts), n)])
                chunks_in_ep[:], pending, plans = 0, [None] * n, [None] * n
        final = summarize(run_episodes(base, token, venv, eval_starts, lambda z_, s_, r_: td3.act(z_, s_, r_), C, max_frames,
                                       args.prefix))
        log(out, stage="final", transitions=transitions, episodes=episodes, **final)
        (out / "results.json").write_text(json.dumps({"baseline": baseline, "baselines": baselines, "final": final, "evals": evals,
                                                       "args": vars(args)}, indent=1))
        torch.save({"actor": td3.actor.state_dict(), "critic": td3.critic.state_dict()}, out / "td3_final.pt")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
