"""Train the Q-function on simulator-lookahead labels (qplan_collect.py --oracle output): at each planning state every
candidate chunk carries its own counterfactual return (candidate, then the frozen policy to the end). Diagnostic for
"can Q learn to pick the candidate at all, given ideal labels"; a real robot cannot produce these.

    python qplan_train_paired.py --data paired_r0.pt --out q_paired.pt

Validation (held-out layouts): success rate of the candidate Q ranks first, against the policy's own chunk
(candidate 0), a random candidate, and the best candidate (oracle); and how often Q orders a succeeding candidate
above a failing one within the same state.
"""
from __future__ import annotations

import argparse
import copy
import glob

import numpy as np
import torch

import qplan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--delta_scale", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    states, n_ep = [], 0
    for path in sorted(sum((glob.glob(d) for d in args.data), [])):
        for ep in torch.load(path, map_location="cpu", weights_only=False)["episodes"]:
            for s in ep["steps"]:
                if "cands" in s:
                    states.append({"obs": s["obs"], "cands": s["cands"], "ret": s["returns"].float(), "ep": n_ep})
            n_ep += 1
    val_ep = set(torch.randperm(n_ep, generator=torch.Generator().manual_seed(0))[: int(n_ep * args.val_frac)].tolist())
    tr = [s for s in states if s["ep"] not in val_ep]
    va = [s for s in states if s["ep"] in val_ep]
    mixed = lambda ss: sum(0 < (s["ret"] > 0).sum() < len(s["ret"]) for s in ss)  # noqa: E731
    print(f"[QPAIR] {n_ep} layouts, {len(states)} planning states x {len(states[0]['ret'])} candidates; train {len(tr)} "
          f"(mixed-outcome {mixed(tr)}), val {len(va)} (mixed-outcome {mixed(va)})", flush=True)

    dev = "cuda"
    stack = lambda ss, f: torch.stack([f(s) for s in ss])  # noqa: E731
    W, E, S = (stack(tr, lambda s: s["obs"][k]) for k in ("wrist", "ego", "state"))
    C, R = stack(tr, lambda s: s["cands"]), stack(tr, lambda s: s["ret"])
    n_c = C.shape[1]
    arch = dict(n_wrist=W.shape[2] * W.shape[3], n_ego=E.shape[2] * E.shape[3], delta_scale=args.delta_scale)
    q = qplan.QNet(**arch).to(dev)
    ema = copy.deepcopy(q).eval()
    hl = qplan.HLGauss(device=dev)
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)

    def evaluate():
        pick, base, rand, best, pair_ok, pair_n = [], [], [], [], 0, 0
        with torch.inference_mode():
            for s in va:
                qv = qplan.q_scores(ema, hl, s["obs"], s["cands"], dev).cpu().numpy()
                ok = (s["ret"] > 0).numpy()
                pick.append(ok[qv.argmax()]); base.append(ok[0]); rand.append(ok.mean()); best.append(ok.any())
                if 0 < ok.sum() < len(ok):
                    pair_ok += (qv[ok][:, None] > qv[~ok][None]).sum(); pair_n += ok.sum() * (~ok).sum()
        return {"q_pick": np.mean(pick), "policy": np.mean(base), "random": np.mean(rand), "oracle": np.mean(best),
                "pair_acc": pair_ok / max(1, pair_n), "pairs": int(pair_n)}

    for step in range(1, args.steps + 1):
        q.train()
        i = torch.randint(len(tr), (args.batch,))
        j = torch.randint(n_c, (args.batch,))
        tok = q.obs_tokens(W[i].to(dev), E[i].to(dev), S[i].to(dev))
        lg = q(tok, C[i, j].to(dev), C[i, 0].to(dev))
        loss = -(hl.probs(R[i, j].to(dev)) * lg.log_softmax(-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pq in zip(ema.parameters(), q.parameters()):
                pe.lerp_(pq, 0.005)
        if step % 1000 == 0 or step == args.steps:
            m = evaluate()
            print(f"[QPAIR] step {step}: loss {loss.item():.3f}  val per planning state: " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
    torch.save({"ema": ema.state_dict(), "arch": arch, "args": vars(args), "val": m}, args.out)
    print(f"[QPAIR] -> {args.out}")


if __name__ == "__main__":
    main()
