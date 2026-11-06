"""Train the action-conditioned predictor in frozen DINOv2 token space, plus the viewing decoder.

Teacher forcing over 4-step rollouts from the last 2 frames, the arm state and the next 0.2 s of actions. Losses: L1
on the tokens, MSE on the next arm state and on the peg height (known only in the random-motion data). The decoder
trains on true and predicted tokens, detached. Held out: the last 20 demo and the last 20 random-motion episodes.

  python train.py <demos .pt> <random .pt> <out dir> --iters 30000
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import HIST, Decoder, Predictor

ROLL = 4
p = argparse.ArgumentParser()
p.add_argument("demos", help="encode.py output")
p.add_argument("random", help="collect_random.py output")
p.add_argument("out", type=Path)
p.add_argument("--iters", type=int, default=30000)
p.add_argument("--batch", type=int, default=48)
p.add_argument("--latents-on-cpu", action="store_true", help="keep the tokens in pinned RAM (16x16 does not fit on a 16 GB GPU)")
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"

data, play = torch.load(args.demos), torch.load(args.random)
if data["tok"] != play["tok"]:
    raise ValueError(f"token grids differ: demos {data['tok']}, random {play['tok']}")
TOK = data["tok"]
N_DEMO = len(data["z"])  # frames with images: the decoder trains on these
data["peg"] = torch.full((N_DEMO,), float("nan"))  # peg height is not recorded in the demos
for k in ("z", "state", "act", "ep", "peg"):
    data[k] = torch.cat([data[k], play[k]])
z = data.pop("z")
# per-channel token statistics, in chunks (a float32 copy of everything does not fit in RAM)
s1, s2, cnt = torch.zeros(384, dtype=torch.float64), torch.zeros(384, dtype=torch.float64), 0
for c in z.split(4096):
    c = c.double()
    s1 += c.sum((0, 1))
    s2 += (c ** 2).sum((0, 1))
    cnt += c.shape[0] * c.shape[1]
zm = (s1 / cnt).float()
zs = (s2 / cnt - (s1 / cnt) ** 2).clamp(min=0).sqrt().float() + 1e-4
zdev = "cpu" if args.latents_on_cpu else dev
Z = torch.cat([((c.to(dev).float() - zm.to(dev)) / zs.to(dev)).half().to(zdev) for c in z.split(4096)])  # (N, NT, 384)
if zdev == "cpu":
    class _Rows:  # Z[gpu ids] -> rows on the GPU
        def __init__(self, t): self.t = t.pin_memory()
        def __getitem__(self, i): return self.t[i.cpu()].to(dev, non_blocking=True)
    Z = _Rows(Z)
zm, zs = zm.to(dev), zs.to(dev)
del z
S, A, EP = data["state"].to(dev), data["act"].to(dev), data["ep"].to(dev)
sm, ss = S.mean(0), S.std(0) + 1e-3
am, as_ = A.mean(0), A.std(0) + 1e-3
IMG, PEG = data["img"].to(dev), data["peg"].to(dev)
N = len(EP)
idx = torch.arange(N, device=dev)
# a start t needs t-1 and t+ROLL in the same episode
ok = (idx >= 1) & (idx + ROLL < N)
ok[ok.clone()] &= (EP[idx[ok] - 1] == EP[idx[ok]]) & (EP[idx[ok] + ROLL] == EP[idx[ok]])
demo_eps, rand_eps = EP[EP < 20000].unique(), EP[EP >= 20000].unique()
if len(demo_eps) <= 20 or len(rand_eps) <= 20:
    raise ValueError(f"need more than 20 episodes of each kind to hold 20 out, got {len(demo_eps)} demo, {len(rand_eps)} random")
val = torch.isin(EP, torch.cat([demo_eps[-20:], rand_eps[-20:]]))
train_idx, val_idx = idx[ok & ~val], idx[ok & val]
demo_train = train_idx[train_idx < N_DEMO]
demo_train = demo_train[EP[demo_train + 1] == EP[demo_train]]  # the decoder also needs the next frame's image


def rollout(P, ids, steps):
    zh = torch.stack([Z[ids - 1], Z[ids]], 1).float()
    s = (S[ids] - sm) / ss
    preds, spreds, ppreds = [], [], []
    for k in range(steps):
        zn, sn, pn = P(zh, s, (A[ids + k] - am) / as_)
        preds.append(zn)
        spreds.append(sn)
        ppreds.append(pn)
        zh = torch.stack([zh[:, -1], zn], 1)
        s = sn
    return torch.stack(preds, 1), torch.stack(spreds, 1), torch.stack(ppreds, 1)


P, D = Predictor(TOK).to(dev), Decoder(TOK).to(dev)
print(f"tok {TOK}x{TOK}, history {HIST}; train {len(train_idx)} val {len(val_idx)}; predictor "
      f"{sum(p.numel() for p in P.parameters()) / 1e6:.1f}M, decoder {sum(p.numel() for p in D.parameters()) / 1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(list(P.parameters()) + list(D.parameters()), 3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.iters, pct_start=0.05)
t0 = time.time()
for it in range(args.iters):
    ids = train_idx[torch.randint(len(train_idx), (args.batch,), device=dev)]
    with torch.autocast("cuda", torch.bfloat16):
        zp, sp, pp = rollout(P, ids, ROLL)
        tgt = torch.stack([Z[ids + 1 + k] for k in range(ROLL)], 1).float()
        stgt = torch.stack([(S[ids + 1 + k] - sm) / ss for k in range(ROLL)], 1)
        lz = F.l1_loss(zp.float(), tgt)
        ls = F.mse_loss(sp.float(), stgt)
        ptgt = torch.stack([PEG[ids + 1 + k] for k in range(ROLL)], 1)
        known = ~ptgt.isnan()
        lp = F.mse_loss(pp.float()[known] * 10, ptgt[known] * 10) if known.any() else torch.zeros((), device=dev)
        di = demo_train[torch.randint(len(demo_train), (16,), device=dev)]
        zd = torch.cat([Z[di].float(), rollout(P, di, 1)[0][:, 0].detach().float()])
        img = torch.cat([IMG[di], IMG[di + 1]]).float() / 255
        ld = F.l1_loss(D(zd).float(), img)
    loss = lz + 0.1 * ls + ld + lp
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(P.parameters(), 1.0)
    opt.step()
    sched.step()
    if it % 500 == 0 or it == args.iters - 1:
        P.eval()
        D.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            vi = val_idx[torch.randperm(len(val_idx), device=dev)[:512]]
            zp, sp, pp = rollout(P, vi, ROLL)
            tgt = torch.stack([Z[vi + 1 + k] for k in range(ROLL)], 1).float()
            err = (zp.float() - tgt).abs().mean((0, 2, 3))
            copy = (Z[vi].float()[:, None] - tgt).abs().mean((0, 2, 3))  # "nothing moves" baseline
            serr = ((sp.float() * ss + sm) - torch.stack([S[vi + 1 + k] for k in range(ROLL)], 1)).abs()[..., :6].mean()
            pt = torch.stack([PEG[vi + 1 + k] for k in range(ROLL)], 1)
            pk = ~pt.isnan()
            perr = (pp.float()[pk] - pt[pk]).abs().mean() if pk.any() else torch.zeros(())
            lifted, fallen = pk & (pt > 0.03), pk & (pt < -0.02)  # fallen: knocked off the table
            perr_l = (pp.float()[lifted] - pt[lifted]).abs().mean() if lifted.any() else torch.full((), float("nan"))
            perr_f = (pp.float()[fallen] - pt[fallen]).abs().mean() if fallen.any() else torch.full((), float("nan"))
        print(f"it {it} loss z {lz.item():.3f} s {ls.item():.3f} dec {ld.item():.3f} | val token L1 per step "
              f"{[round(v, 3) for v in err.tolist()]} copy-last-frame {[round(v, 3) for v in copy.tolist()]} "
              f"arm state err {serr.item():.3f} rad | peg err {perr.item() * 100:.2f} cm (lifted {perr_l.item() * 100:.2f}, "
              f"knocked off {perr_f.item() * 100:.2f}) | {time.time() - t0:.0f} s", flush=True)
        P.train()
        D.train()
torch.save({"P": P.state_dict(), "D": D.state_dict(), "tok": TOK, "zm": zm, "zs": zs, "sm": sm, "ss": ss, "am": am, "as": as_},
           args.out / "wm.pt")
print("saved", args.out / "wm.pt")
