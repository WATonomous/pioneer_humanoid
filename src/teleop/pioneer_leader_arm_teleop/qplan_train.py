"""Train the Q-Planning value function on chunk-level rollout transitions (qplan_collect.py output).

    python qplan_train.py --data rollouts/explore_*.pt --out q0.pt --steps 6000

Reward is sparse: 1 on the frame the task succeeds, 0 otherwise; a failed rollout is an all-zero trajectory.
Target for a chunk = td_mix * (one-chunk bootstrap from the next executed chunk, target network)
                  + (1 - td_mix) * (discounted Monte-Carlo return of the rollout it came from),
regressed with HL-Gauss cross-entropy. Episodes are split train/validation; validation reports how well Q
separates successful from failed rollouts (AUC), with and without the action offset (does Q see the action?).
"""
from __future__ import annotations

import argparse
import copy
import glob
import time

import numpy as np
import torch

import qplan


def load(paths, gamma, H):
    keys = ("wrist", "ego", "state", "chunk", "base", "mc", "term", "disc", "nxt", "ep", "k", "succ")
    out = {k: [] for k in keys}
    n_ep = 0
    for path in paths:
        for ep in torch.load(path, map_location="cpu", weights_only=False)["episodes"]:
            steps, total = ep["steps"], ep["frames"]
            for k, s in enumerate(steps):
                last = k == len(steps) - 1
                out["wrist"].append(s["obs"]["wrist"]); out["ego"].append(s["obs"]["ego"]); out["state"].append(s["obs"]["state"])
                out["chunk"].append(s["chunk"].float()); out["base"].append(s["base"].float())
                out["mc"].append(gamma ** (total - s["frame"]) if ep["success"] else 0.0)
                # one-chunk target pieces: terminal value (success inside this chunk -> gamma^ran, failure at the end -> 0)
                out["term"].append(float(last)); out["disc"].append(gamma ** s["ran"])
                out["nxt"].append(len(out["mc"]) if not last else len(out["mc"]) - 1)
                out["ep"].append(n_ep); out["k"].append(k); out["succ"].append(float(ep["success"]))
            n_ep += 1
    data = {k: torch.stack(v) if k in ("wrist", "ego", "state", "chunk", "base") else torch.tensor(v) for k, v in out.items()}
    data["nxt"] = data["nxt"].long()
    return data, n_ep


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty(len(score)); rank[order] = np.arange(1, len(score) + 1)
    pos = label > 0.5
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    return float((rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None, help="Q checkpoint to continue from (self-improvement iterations)")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--td_mix", type=float, default=0.0)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--no_delta", action="store_true", help="ablation: hide the action from Q (state-value baseline)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    paths = sorted(sum((glob.glob(d) for d in args.data), []))
    data, n_ep = load(paths, args.gamma, 100)
    n = len(data["mc"])
    val_ep = torch.randperm(n_ep, generator=torch.Generator().manual_seed(0))[: max(1, int(n_ep * args.val_frac))]
    is_val = torch.isin(data["ep"], val_ep)
    tr, va = torch.nonzero(~is_val)[:, 0], torch.nonzero(is_val)[:, 0]
    print(f"[QTRAIN] {len(paths)} files, {n_ep} episodes, {n} chunk transitions ({len(tr)} train / {len(va)} val), "
          f"success {data['succ'][data['k'] == 0].mean():.1%}", flush=True)

    dev = "cuda"
    arch = dict(n_wrist=data["wrist"].shape[2] * data["wrist"].shape[3], n_ego=data["ego"].shape[2] * data["ego"].shape[3])
    q = qplan.QNet(**arch).to(dev)
    if args.init:
        q.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["ema"])
    ema = copy.deepcopy(q).eval()
    hl = qplan.HLGauss(device=dev)
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)

    def batch(idx):
        return {k: data[k][idx].to(dev) for k in ("wrist", "ego", "state", "chunk", "base", "mc", "term", "disc")}

    def logits(net, b):
        base = b["base"]
        chunk = b["base"] if args.no_delta else b["chunk"]
        return net(net.obs_tokens(b["wrist"], b["ego"], b["state"]), chunk, base)

    def evaluate():
        ema.eval()
        pred = []
        with torch.inference_mode():
            for i in range(0, len(va), 512):
                pred.append(hl.value(logits(ema, batch(va[i:i + 512]))).cpu())
        pred = torch.cat(pred).numpy()
        succ, k, mc = data["succ"][va].numpy(), data["k"][va].numpy(), data["mc"][va].numpy()
        return {"mse": float(((pred - mc) ** 2).mean()), "auc_all": auc(pred, succ), "auc_first": auc(pred[k == 0], succ[k == 0]),
                "auc_k1": auc(pred[k == 1], succ[k == 1]), "auc_k2": auc(pred[k == 2], succ[k == 2]), "auc_k3": auc(pred[k == 3], succ[k == 3])}

    t0 = time.time()
    for step in range(1, args.steps + 1):
        q.train()
        idx = tr[torch.randint(len(tr), (args.batch,))]
        b = batch(idx)
        target = b["mc"].float()
        if args.td_mix > 0:
            with torch.inference_mode():
                nb = batch(data["nxt"][idx])
                boot = hl.value(logits(ema, nb))
            succ_term = (data["succ"][idx].to(dev) > 0.5)
            td = torch.where(b["term"] > 0.5, torch.where(succ_term, b["disc"].float(), torch.zeros_like(boot)), b["disc"].float() * boot)
            target = args.td_mix * td + (1 - args.td_mix) * target
        lg = logits(q, b)
        loss = -(hl.probs(target.clone()) * lg.log_softmax(-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pq in zip(ema.parameters(), q.parameters()):
                pe.lerp_(pq, 1 - args.ema)
        if step % 1000 == 0 or step == args.steps:
            m = evaluate()
            print(f"[QTRAIN] step {step}: loss {loss.item():.3f}  val " + "  ".join(f"{k} {v:.3f}" for k, v in m.items())
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
    torch.save({"ema": ema.state_dict(), "arch": arch, "args": vars(args), "val": m}, args.out)
    print(f"[QTRAIN] -> {args.out}")


if __name__ == "__main__":
    main()
