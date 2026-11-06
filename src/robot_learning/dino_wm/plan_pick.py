"""Peg pickup by planning with the world model, on held-out peg_insert layouts.

The goals are three frames of the synthetic take for the same layout: just before the gripper closes for the lift,
0.4 s after it closed, and the peg at its highest. Each goal stays active until the planned token distance drops below
what the take needs 0.4 s before that frame, or a deadline (the take's time for it + 2 s). Closed-loop CEM MPC:
400 action sequences of 8 steps (1.6 s), 8 iterations, the best 40 refit each time; the first 0.2 s is executed.
Compared with holding still and with replaying the take (upper bound). Success: peg 5 cm above the table at the end.

  MUJOCO_GL=egl python plan_pick.py <wm.pt> <out dir> --seeds 9001-9020
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import STRIDE, DinoEncoder, Predictor  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("wm", help="train.py output (wm.pt)")
p.add_argument("out", type=Path)
p.add_argument("--seeds", required=True, help="layout seeds, e.g. 9001-9020")
p.add_argument("--peg-weight", type=float, default=3.0, help="cost weight on the predicted peg height (m) vs the goal's")
p.add_argument("--fall-weight", type=float, default=10.0, help="cost weight on the peg dropping below the table top")
p.add_argument("--synthetic-dir", type=Path,
               default=Path(__file__).resolve().parents[2] / "teleop/pioneer_leader_arm_teleop/synthetic")
args = p.parse_args()
a, b = (int(v) for v in args.seeds.split("-"))
seeds = list(range(a, b + 1))
sys.path.insert(0, str(args.synthetic_dir))
from synthetic_teleop import TeleopSim, run_episode  # noqa: E402

args.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
ck = torch.load(args.wm)
TOK = ck["tok"]
P = Predictor(TOK).to(dev).eval()
P.load_state_dict(ck["P"])
zm, zs, sm, ss, am, as_ = (ck[k].to(dev) for k in ("zm", "zs", "sm", "ss", "am", "as"))
enc = DinoEncoder(TOK)
bounds = json.loads(Path(__file__).with_name("action_bounds.json").read_text())
BOX = {k: torch.tensor(bounds[k], device=dev) for k in ("lo", "hi", "step")}
H, NS, NE, IT, TICKS, SIG = 8, 400, 40, 8, 4, 0.15


def encode(img):
    z = enc(torch.from_numpy(img).to(dev).permute(2, 0, 1)[None].float() / 255).float()
    return (z - zm) / zs


@torch.no_grad()
def imagine(zh, s, acts):
    s = (s - sm) / ss
    zs_, pegs = [], []
    with torch.autocast("cuda", torch.bfloat16):
        for k in range(acts.shape[1]):
            zn, s, pn = P(zh, s, (acts[:, k] - am) / as_)
            zh = torch.stack([zh[:, -1], zn], 1)
            zs_.append(zn.float())
            pegs.append(pn.float())
    return torch.stack(zs_, 1), torch.stack(pegs, 1)


def state_of(sim):
    d = sim.data
    closure = sum((d.qpos[q] - o) / (c - o) for q, o, c in zip(sim.grip_qpos, sim.grip_open, sim.grip_closed)) / len(sim.grip_qpos)
    return np.append(d.qpos[sim.arm_qpos], min(max(closure, 0.0), 1.0)).astype(np.float32)


def apply(sim, act):
    lo, hi = sim.model.jnt_range[[sim.model.joint(j).id for j in sim.joints]].T
    g = min(max(act[6], 0.0), 1.0)
    sim.data.ctrl[sim.arm_acts] = np.clip(act[:6], lo, hi)
    sim.data.ctrl[sim.grip_acts] = [o + g * (c - o) for o, c in zip(sim.grip_open, sim.grip_closed)]
    for _ in range(TICKS):
        if sim.hook is not None:
            sim.hook(sim.model, sim.data)
        sim.mujoco.mj_step(sim.model, sim.data, nstep=sim.substeps)


res = []
for seed in seeds:
    demo = []
    run_episode("peg_insert", seed, frame_sink=lambda a, s, t, e, q: demo.append((a, s, q)), record_every=TICKS)
    sim = TeleopSim("peg_insert", seed, {"ego": (240, 320)})
    adr = sim.model.joint(sim.model.body("peg").jntadr[0]).qposadr[0] + 2
    pz = np.array([d[2][adr] for d in demo]) - demo[0][2][adr]
    g = np.array([d[0][6] for d in demo])
    if not (pz > 0.05).any():
        sim.close()
        raise RuntimeError(f"seed {seed}: the synthetic take never lifts the peg, so it gives no goals")
    lift = int(np.argmax(pz > 0.05))
    close = max(i for i in range(lift) if g[i] > 0.9 and (i == 0 or g[i - 1] <= 0.9))  # the close that lifted it
    peak = lift + int(np.argmax(pz[lift:lift + 75]))
    Ks = [max(close - 10, 5), min(close + 10, lift), peak]
    goal_imgs = [sim.render_at(demo[k][2])["ego"] for k in Ks]
    sim.close()
    K = Ks[-1]
    acts = np.stack([d[0] for d in demo])

    def run(mode):
        sim = TeleopSim("peg_insert", seed, {"ego": (240, 320)})
        sim.data.qpos[:] = demo[0][2]
        sim.data.qvel[:] = 0
        sim.data.ctrl[sim.arm_acts] = demo[0][0][:6]
        sim.mujoco.mj_forward(sim.model, sim.data)
        zgs = [encode(im) for im in goal_imgs]
        thr = []  # switch threshold per goal: the token distance from the take's frame 0.4 s earlier to that goal
        for k, zg in zip(Ks, zgs):
            thr.append(float((encode(sim.render_at(demo[max(k - 10, 0)][2])["ego"]) - zg).abs().mean()))
        sim.data.qpos[:] = demo[0][2]
        sim.mujoco.mj_forward(sim.model, sim.data)
        gi, z_goal, goal_img, peak_z = 0, zgs[0], goal_imgs[0], 0.0
        peg_goal = float(pz[Ks[0]])
        frames, z_prev, mu, last_cost = [], None, None, 1e9
        last = demo[0][0].copy()
        for t in range(K // STRIDE + 20):
            if mode == "plan" and gi < len(Ks) - 1 and (t > Ks[gi] // STRIDE + 10 or (t > 0 and last_cost < thr[gi])):
                gi += 1
                z_goal, goal_img = zgs[gi], goal_imgs[gi]
                peg_goal = float(pz[Ks[gi]])
            img = sim.render_at(sim.data.qpos.copy())["ego"]
            frames.append(np.concatenate([img, goal_img], 1))
            if mode == "replay":
                chunk = acts[np.minimum(np.arange(t * STRIDE, (t + 1) * STRIDE), K)]
            elif mode == "hold":
                chunk = np.repeat(last[None], STRIDE, 0)
            else:
                z = encode(img)
                zh = torch.stack([z_prev if z_prev is not None else z, z], 1).expand(NS, -1, -1, -1)
                s = torch.from_numpy(state_of(sim)).to(dev)[None].expand(NS, -1)
                cur = torch.from_numpy(last).to(dev).float()
                m = cur.repeat(H, 1) if mu is None else torch.cat([mu[1:], mu[-1:]])
                sd = torch.full((H, 7), SIG, device=dev)
                ramp = torch.arange(1, STRIDE + 1, device=dev).float() / STRIDE
                for _ in range(IT):
                    cand = m[None] + sd[None] * torch.randn(NS, H, 7, device=dev)
                    cand[0] = m
                    prev_t = cur.expand(NS, 7)
                    for k in range(H):
                        cand[:, k] = torch.maximum(torch.minimum(cand[:, k], prev_t + BOX["step"]), prev_t - BOX["step"]).clamp(BOX["lo"], BOX["hi"])
                        prev_t = cand[:, k]
                    prev = torch.cat([cur.expand(NS, 1, 7), cand[:, :-1]], 1)
                    full = (prev[:, :, None] + (cand - prev)[:, :, None] * ramp[None, None, :, None]).reshape(NS, H, STRIDE * 7)
                    traj, pegp = imagine(zh, s, full)
                    d = (traj - z_goal[:, None]).abs().mean((2, 3))
                    cost = d[:, -1] + 0.5 * d.mean(1)
                    cost = cost + args.peg_weight * (pegp[:, -1] - peg_goal).abs()
                    cost = cost + args.fall_weight * (-pegp).clamp(min=0).max(1).values
                    last_cost = float(d[0, 0])
                    el = cand[cost.argsort()[:NE]]
                    m, sd = el.mean(0), el.std(0) + 1e-3
                mu, z_prev = m, z
                chunk = (cur[None] + (m[0] - cur)[None] * ramp[:, None]).cpu().numpy()
            for act in chunk:
                apply(sim, act)
            peak_z = max(peak_z, sim.data.qpos[adr] - demo[0][2][adr])
            last = chunk[-1].copy()
        h_end = float(sim.data.qpos[adr] - demo[0][2][adr])
        sim.close()
        return h_end, peak_z, frames

    r = {m_: run(m_) for m_ in ("hold", "plan", "replay")}
    res.append([r[m_][0] for m_ in ("hold", "plan", "replay")] + [r[m_][1] for m_ in ("hold", "plan", "replay")])
    print(f"seed {seed}: peg height at the end (highest) cm: hold {r['hold'][0]*100:.1f} ({r['hold'][1]*100:.1f}) | world-model "
          f"plan {r['plan'][0]*100:.1f} ({r['plan'][1]*100:.1f}) | take replay {r['replay'][0]*100:.1f} ({r['replay'][1]*100:.1f})", flush=True)
    if len(res) <= 3 or r["plan"][1] > 0.05:  # left: camera, right: the active goal
        imageio.mimsave(args.out / f"pick_seed{seed}.gif", [np.repeat(np.repeat(f, 2, 0), 2, 1) for f in r["plan"][2]], duration=0.2, loop=0)
r = np.array(res)
print(f"\n{len(r)} held-out layouts, 3 goals (before the grasp, closed, lifted). Peg lifted > 5 cm at the end / ever:")
for k, n in enumerate(("hold still", "world-model plan", "take replay (upper bound)")):
    print(f"  {n:26s} {np.mean(r[:, k] > 0.05)*100:3.0f}% / {np.mean(r[:, 3 + k] > 0.05)*100:3.0f}%   mean highest {r[:, 3 + k].mean()*100:.1f} cm")
