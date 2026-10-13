"""Q-Planning (arXiv 2608.21204) on a frozen Diffusion Policy: the setting the paper assumes, a policy whose
samples are genuinely different action chunks.

    # rollouts: --select sample (the policy as is), random (exploration data), qmax / qavg (the planner)
    python qplan_dp.py collect --policy <ckpt>/pretrained_model --out r0.pt --seed 10000 --episodes 400 --select random
    python qplan_dp.py train --data r0.pt --out q0.pt
    python qplan_dp.py collect --policy <ckpt>/pretrained_model --out e1.pt --seed 1000 --episodes 200 --select qavg --q q0.pt

Per planning step the policy's observation encoding is computed once and N chunks are denoised from different noise.
Q(observation encoding, chunk) is a small MLP with an HL-Gauss head, trained on the discounted Monte-Carlo return of
the rollout each chunk came from (reward 1 at success, failures are all-zero). Only Q is ever trained.
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import nn

import qplan
from qplan import FPS, ROBOT, HLGauss, execute
from sim_policy_eval import RENDER_SIZE, load_policy, observe
from synthetic_teleop import TeleopSim


class DPProposer:
    def __init__(self, path, device="cuda"):
        self.policy, self.pre, self.post, self.cfg, self.cameras = load_policy(path, device)
        self.device = torch.device(device)
        self.n_obs, self.H = self.cfg.n_obs_steps, self.cfg.n_action_steps
        zero = torch.zeros(1, 1, 7, device=self.device)
        with torch.inference_mode():
            self.a_mean = self.post(zero.clone()).reshape(7).to(self.device).float()
            self.a_std = self.post(zero.clone() + 1).reshape(7).to(self.device).float() - self.a_mean
        self.queue = deque(maxlen=self.n_obs)

    def reset(self):
        self.queue.clear()

    def unnormalise(self, chunks):
        return chunks * self.a_std + self.a_mean

    @torch.inference_mode()
    def push(self, state, images, instruction):
        """Add one observation (a frame) to the history the policy conditions on."""
        import cv2
        from lerobot.policies.utils import prepare_observation_for_inference
        seen = {n: img if img.shape[:2] == tuple(self.cameras[n]) else
                cv2.resize(img, tuple(self.cameras[n])[::-1], interpolation=cv2.INTER_AREA) for n, img in images.items()}
        obs = {"observation.state": state, **{f"observation.images.{n}": img for n, img in seen.items()}}
        batch = self.pre(prepare_observation_for_inference(obs, self.device, instruction, ROBOT))
        while True:
            self.queue.append(batch)
            if len(self.queue) == self.n_obs:
                break

    @torch.inference_mode()
    def propose(self, n, generator=None):
        """n sampled chunks (n, H, 7), normalised units, and the observation encoding (D,) they were conditioned on."""
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
        batch = {OBS_STATE: torch.stack([b[OBS_STATE] for b in self.queue], dim=1),
                 OBS_IMAGES: torch.stack([torch.stack([b[k] for k in self.cfg.image_features], dim=1) for b in self.queue], dim=1)}
        model = self.policy.diffusion
        cond = model._prepare_global_conditioning(batch)                       # (1, D)
        noise = torch.randn(n, self.cfg.horizon, 7, generator=generator).to(self.device)
        actions = model.conditional_sample(n, global_cond=cond.expand(n, -1), noise=noise)
        start = self.n_obs - 1
        return actions[:, start:start + self.H].float(), cond[0].float().cpu()


class QMLP(nn.Module):
    def __init__(self, cond_dim, H=32, dim=512, bins=51):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim + H * 7, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, bins))

    def forward(self, cond, chunks):
        return self.net(torch.cat([cond, chunks.flatten(1)], -1))


def rollout(scene, seed, prop, select, n, max_s=40.0, hold_s=0.4, generator=None, oracle=False):
    from synthetic_tasks import TASKS
    task = TASKS[scene]
    sim = TeleopSim(scene, seed, {c: RENDER_SIZE for c in prop.cameras})
    prop.reset()
    prop.policy.reset()
    steps, frames, held, ok, limit = [], 0, 0.0, False, int(max_s * FPS)
    prop.push(*observe(sim), task.instruction)
    while frames < limit:
        chunks, cond = prop.propose(n, generator)
        if oracle:
            returns = lookahead(sim, task, prop, chunks, frames, held, max_s, hold_s)
            pick, extra = int(np.argmax(returns)), {"returns": torch.tensor(returns), "cands": chunks.cpu()}
            chunk = chunks[pick]
        else:
            chunk, extra = select(chunks, cond)
        actions = prop.unnormalise(chunk).cpu().numpy().astype(float)[: limit - frames]
        ran, ok, held = run_chunk(sim, task, prop, actions, held, hold_s)
        steps.append({"cond": cond, "chunk": chunk.float().cpu(), "ran": ran, "frame": frames, **extra})
        frames += ran
        if ok:
            break
    sim.close()
    return {"seed": seed, "success": bool(ok), "frames": frames, "steps": steps}


def run_chunk(sim, task, prop, actions, held, hold_s):
    """Execute a chunk, feeding the policy's observation history its last two frames (it conditions on both)."""
    ran, ok, held = execute(sim, task, actions[:-1], held, hold_s)
    if not ok and len(actions) > 1:
        prop.push(*observe(sim), task.instruction)
    if not ok:
        r2, ok, held = execute(sim, task, actions[-1:], held, hold_s)
        ran += r2
        if not ok:
            prop.push(*observe(sim), task.instruction)
    return ran, ok, held


def lookahead(sim, task, prop, chunks, frames, held, max_s, hold_s, gamma=0.997):
    """Counterfactual return of each candidate from this exact sim state (candidate, then policy samples to the end)."""
    mj, model, data = sim.mujoco, sim.model, sim.data
    spec = mj.mjtState.mjSTATE_INTEGRATION
    snap = np.empty(mj.mj_stateSize(model, spec))
    mj.mj_getState(model, data, snap, spec)
    saved, limit, returns = list(prop.queue), int(max_s * FPS), []
    for chunk in chunks:
        mj.mj_setState(model, data, snap, spec)
        mj.mj_forward(model, data)
        prop.queue = deque(saved, maxlen=prop.n_obs)
        f, h, ok, nxt = frames, held, False, chunk
        while f < limit:
            ran, ok, h = run_chunk(sim, task, prop, prop.unnormalise(nxt).cpu().numpy().astype(float)[: limit - f], h, hold_s)
            f += ran
            if ok or f >= limit:
                break
            nxt = prop.propose(1)[0][0]
        returns.append(gamma ** (f - frames) if ok else 0.0)
    mj.mj_setState(model, data, snap, spec)
    mj.mj_forward(model, data)
    prop.queue = deque(saved, maxlen=prop.n_obs)
    return returns


def load_q(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    q = QMLP(**ck["arch"]).to(device).eval()
    q.load_state_dict(ck["ema"])
    return q


def collect(args):
    prop = DPProposer(args.policy)
    rng, gen = np.random.default_rng(args.seed), torch.Generator().manual_seed(args.seed)
    hl = HLGauss(device="cuda")
    spread = []
    if args.select in ("qmax", "qavg"):
        q = load_q(args.q)

    def select(chunks, cond):
        spread.append(prop.unnormalise(chunks).std(0).mean(0).cpu().numpy())
        if args.select == "sample":
            return chunks[0], {}
        if args.select == "random":
            return chunks[int(rng.integers(len(chunks)))], {}
        with torch.inference_mode():
            v = hl.value(q(cond.to("cuda")[None].expand(len(chunks), -1), chunks))
        if args.select == "qmax":
            return chunks[int(v.argmax())], {"q": v.cpu()}
        w = torch.softmax(v / args.temperature, 0)
        return (w[:, None, None] * chunks).sum(0), {"q": v.cpu()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed + i for i in range(args.episodes)]
    episodes, t0 = [], time.time()
    for i, seed in enumerate(seeds):
        if i and i % 100 == 0:
            torch.save({"args": vars(args), "episodes": episodes}, out)
        ep = rollout(args.scene, seed, prop, select, args.n, args.max_s, generator=gen, oracle=args.oracle)
        episodes.append(ep)
        print(f"[QDP] seed {seed}: {'ok  ' if ep['success'] else 'FAIL'} {ep['frames'] / FPS:5.1f} s  {len(ep['steps'])} chunks   "
              f"running {sum(e['success'] for e in episodes)}/{len(episodes)} ({time.time() - t0:.0f} s wall)", flush=True)
    torch.save({"args": vars(args), "episodes": episodes}, out)
    ok = sum(e["success"] for e in episodes)
    print(f"[QDP] {args.scene} select={args.select}{' oracle' if args.oracle else ''}: success {ok / len(episodes):.1%} ({ok}/{len(episodes)})  -> {out}")
    if spread:
        print("[QDP] candidate spread, rad (gripper: closure), mean over planning steps:", np.round(np.mean(spread, 0), 4))


def train(args):
    torch.manual_seed(0)
    gamma, rows, n_ep = 0.997, [], 0
    for path in sorted(sum((glob.glob(d) for d in args.data), [])):
        for ep in torch.load(path, map_location="cpu", weights_only=False)["episodes"]:
            for k, s in enumerate(ep["steps"]):
                rows.append((s["cond"], s["chunk"], gamma ** (ep["frames"] - s["frame"]) if ep["success"] else 0.0, n_ep, float(ep["success"]), k))
            n_ep += 1
    C, A = torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows])
    R, E, S, K = (torch.tensor([r[i] for r in rows]) for i in (2, 3, 4, 5))
    val_ep = torch.randperm(n_ep, generator=torch.Generator().manual_seed(0))[: max(1, n_ep // 10)]
    is_val = torch.isin(E, val_ep)
    tr, va = torch.nonzero(~is_val)[:, 0], torch.nonzero(is_val)[:, 0]
    print(f"[QDP] {n_ep} episodes, {len(rows)} chunk transitions ({len(tr)} train / {len(va)} val), success {S[K == 0].mean():.1%}", flush=True)
    dev, arch = "cuda", dict(cond_dim=C.shape[1], H=A.shape[1])
    q = QMLP(**arch).to(dev)
    if args.init:
        q.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["ema"])
    ema, hl = copy.deepcopy(q).eval(), HLGauss(device=dev)
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    from qplan_train import auc
    for step in range(1, args.steps + 1):
        q.train()
        i = tr[torch.randint(len(tr), (args.batch,))]
        loss = -(hl.probs(R[i].float().to(dev)) * q(C[i].to(dev), A[i].to(dev)).log_softmax(-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pq in zip(ema.parameters(), q.parameters()):
                pe.lerp_(pq, 0.005)
        if step % 1000 == 0 or step == args.steps:
            with torch.inference_mode():
                pred = hl.value(ema(C[va].to(dev), A[va].to(dev))).cpu().numpy()
            s, k = S[va].numpy(), K[va].numpy()
            m = {"auc_all": auc(pred, s), **{f"auc_k{j}": auc(pred[k == j], s[k == j]) for j in (0, 2, 4, 6, 8)}}
            print(f"[QDP] step {step}: loss {loss.item():.3f}  val " + "  ".join(f"{a} {b:.3f}" for a, b in m.items()), flush=True)
    torch.save({"ema": ema.state_dict(), "arch": arch, "val": m}, args.out)
    print(f"[QDP] -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--scene", default="peg_insert")
    c.add_argument("--policy", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=10000)
    c.add_argument("--episodes", type=int, default=100)
    c.add_argument("--seeds", default=None)
    c.add_argument("--select", default="sample", choices=["sample", "random", "qmax", "qavg"])
    c.add_argument("--q", default=None)
    c.add_argument("--n", type=int, default=8)
    c.add_argument("--temperature", type=float, default=0.02)
    c.add_argument("--max_s", type=float, default=40.0)
    c.add_argument("--oracle", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--init", default=None)
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    collect(args) if args.cmd == "collect" else train(args)


if __name__ == "__main__":
    main()
