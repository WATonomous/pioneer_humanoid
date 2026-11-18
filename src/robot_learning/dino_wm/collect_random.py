"""Random-motion data for the world model: the gripper is driven by random keyboard-style moves (+-x, +-y, up/down
or stay for 0.2-0.8 s, 1.5-4 cm per 0.2 s, IK to joint targets), random open/close, and in about half the episodes a
pull towards the peg, so it pokes, pushes, knocks over and sometimes grabs it. Each episode starts from the first
frame of a synthetic peg_insert take. Episodes are numbered 20000 + k.

  MUJOCO_GL=egl python collect_random.py <out .pt> --tok 16 --first-seed 30000 --episodes 800
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
SYNTHETIC = Path(__file__).resolve().parents[2] / "teleop/pioneer_leader_arm_teleop/synthetic"
import numpy as np  # noqa: E402

TICKS, STEPS = 4, 60  # 60 x 0.2 s = 12 s per episode
DIRS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 0, 0)], float)


def ik_target(sim, x_goal, iters=30):
    m, d0 = sim.model, sim.data
    d = sim.mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    dofs = [m.jnt_dofadr[m.joint(j).id] for j in sim.joints]
    lo, hi = m.jnt_range[[m.joint(j).id for j in sim.joints]].T
    for _ in range(iters):
        sim.mujoco.mj_kinematics(m, d)
        sim.mujoco.mj_comPos(m, d)
        e = x_goal - (d.body("link7l").xpos + d.body("link8l").xpos) / 2
        if np.linalg.norm(e) < 1e-3:
            break
        J = np.zeros((2, 3, m.nv))
        for k, b in enumerate(("link7l", "link8l")):
            sim.mujoco.mj_jacBody(m, d, J[k], None, m.body(b).id)
        Ja = J.mean(0)[:, dofs]
        dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + 1e-4 * np.eye(3), e)
        d.qpos[sim.arm_qpos] = np.clip(d.qpos[sim.arm_qpos] + np.clip(dq, -0.2, 0.2), lo, hi)
    return d.qpos[sim.arm_qpos].copy()


def episode(seed):
    sys.path.insert(0, str(SYNTHETIC))
    from model import STRIDE
    from synthetic_teleop import TeleopSim, run_episode
    demo = []
    run_episode("peg_insert", seed, frame_sink=lambda a, s, t, e, q: demo.append((a, q)), record_every=TICKS)
    if not demo:
        raise RuntimeError(f"seed {seed}: the synthetic take recorded no frames")
    rng = np.random.default_rng([seed, 11])
    sim = TeleopSim("peg_insert", seed, {"ego": (240, 320)})
    m, d = sim.model, sim.data
    adr = m.joint(m.body("peg").jntadr[0]).qposadr[0] + 2
    d.qpos[:] = demo[0][1]
    d.qvel[:] = 0
    d.ctrl[sim.arm_acts] = demo[0][0][:6]
    sim.mujoco.mj_forward(m, d)
    z0 = d.qpos[adr]
    x0 = (d.body("link7l").xpos + d.body("link8l").xpos) / 2
    lo_box, hi_box = x0 + [-0.2, -0.2, -0.15], x0 + [0.2, 0.2, 0.15]
    poke = rng.random() < 0.5
    cur = demo[0][0].astype(np.float64).copy()
    goal, dirn, hold, grip = x0.copy(), np.zeros(3), 0, 0.0
    imgs, states, acts, pegs = [], [], [], []
    for _ in range(STEPS):
        closure = sum((d.qpos[q] - o) / (c - o) for q, o, c in zip(sim.grip_qpos, sim.grip_open, sim.grip_closed)) / len(sim.grip_qpos)
        states.append(np.append(d.qpos[sim.arm_qpos], min(max(closure, 0.0), 1.0)))
        pegs.append(d.qpos[adr] - z0)
        imgs.append(sim.render_at(d.qpos.copy())["ego"])
        if hold <= 0:
            if poke and rng.random() < 0.6:  # head for the peg (a little above, beside or at it)
                tgt = d.body("peg").xpos + rng.uniform([-0.04, -0.04, -0.01], [0.04, 0.04, 0.08])
                v = tgt - goal
                dirn = v / (np.linalg.norm(v) + 1e-6)
            else:
                dirn = DIRS[rng.integers(len(DIRS))]
            speed, hold = rng.uniform(0.015, 0.04), int(rng.integers(1, 5))
            if rng.random() < 0.15:
                grip = 1.0 - grip
        hold -= 1
        goal = np.clip(goal + dirn * speed, lo_box, hi_box)
        new = cur.copy()
        new[:6] = ik_target(sim, goal)
        new[6] = grip
        chunk = cur[None] + (new - cur)[None] * (np.arange(1, STRIDE + 1)[:, None] / STRIDE)
        for a in chunk:
            g = min(max(a[6], 0.0), 1.0)
            d.ctrl[sim.arm_acts] = a[:6]
            d.ctrl[sim.grip_acts] = [o + g * (c - o) for o, c in zip(sim.grip_open, sim.grip_closed)]
            for _ in range(TICKS):
                if sim.hook is not None:
                    sim.hook(m, d)
                sim.mujoco.mj_step(m, d, nstep=sim.substeps)
        acts.append(chunk.reshape(-1))
        cur = new
    sim.close()
    return seed, np.stack(imgs), np.array(states, np.float32), np.array(acts, np.float32), np.array(pegs, np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tok", type=int, required=True)
    p.add_argument("--first-seed", type=int, default=30000)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    with mp.get_context("fork").Pool(args.workers) as pool:  # forked before CUDA is touched
        import torch

        from model import DinoEncoder
        enc, dev = DinoEncoder(args.tok), "cuda"
        Z, S, A, EP, PEG = [], [], [], [], []
        moved = fell = 0
        for k, (seed, imgs, st, ac, pg) in enumerate(pool.imap_unordered(episode, range(args.first_seed, args.first_seed + args.episodes))):
            Z.append(enc(torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255).cpu())
            S.append(torch.from_numpy(st))
            A.append(torch.from_numpy(ac))
            PEG.append(torch.from_numpy(pg))
            EP.append(torch.full((len(st),), 20000 + k))
            moved += np.abs(pg).max() > 0.005
            fell += pg.min() < -0.02
            if k % 20 == 0:
                print(f"{k + 1}/{args.episodes} episodes, peg moved in {moved}, knocked off in {fell}", flush=True)
    torch.save({"z": torch.cat(Z), "state": torch.cat(S), "act": torch.cat(A), "ep": torch.cat(EP), "peg": torch.cat(PEG),
                "tok": args.tok}, args.out)
    print("saved", args.out, tuple(torch.cat(Z).shape), f"peg moved {moved}/{args.episodes}, knocked off {fell}/{args.episodes}")
