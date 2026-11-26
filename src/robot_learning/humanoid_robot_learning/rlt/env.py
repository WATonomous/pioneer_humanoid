"""peg_insert in N worker processes, with MuJoCo state snapshots for the RLT critical phase.

Each worker owns one SceneEnv (evaluate.py: same control loop, cameras and success test as the Phase 0 eval). The
critical phase starts at the *handoff*: peg held above the block top, its centre within HANDOFF_XY of the hole's.
A snapshot is everything needed to put a fresh worker back in that state, including the block's randomized position
(a model parameter, not part of qpos).
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

HANDOFF_XY = 0.03        # m, peg centre to hole centre
GRASPED = 0.3            # gripper closure (0 open .. 1 closed) above which the peg counts as held
HOLD_FRAMES = 10         # success = is_inserted for 0.4 s at 25 fps (the Phase 0 eval's HOLD_S)


def _peg_info(env):
    import humanoid_mujoco_scenes.peg_insert.scene as S
    m, d = env.model, env.data
    peg, block = d.xpos[m.body("peg").id], d.xpos[m.body("hole_block").id]
    bottom = peg[2] - S.PEG_HEIGHT / 2
    block_top = S.TABLE_TOP_Z + S.BLOCK_HEIGHT
    return peg, block, bottom, block_top, S.TABLE_TOP_Z


def _worker(conn, cameras, fps):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from humanoid_robot_learning.sim.evaluate import CONTROL_DT, SceneEnv

    env = SceneEnv("peg_insert", cameras, "8dof")
    control_steps = round(1 / (fps * CONTROL_DT))
    block_id = env.model.body("hole_block").id
    held, lifted = 0, False                      # a drop only counts once the peg has been off the table
    record = False                               # also return every frame's cameras (half size) from "step"
    while True:
        cmd, arg = conn.recv()
        if cmd == "record":
            record = bool(arg)
            conn.send(None)
        elif cmd == "reset":                     # fresh layout from a seed (start-state collection)
            env.reset(arg)
            held, lifted = 0, False
            conn.send(env.observe())
        elif cmd == "restore":                   # a handoff snapshot (RL episodes)
            mujoco.mj_resetData(env.model, env.data)
            env.model.body_pos[block_id] = arg["block_pos"]
            env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = arg["qpos"], arg["qvel"], arg["ctrl"]
            if env.model.na:
                env.data.act[:] = arg["act"]
            env.data.time = arg["time"]
            mujoco.mj_forward(env.model, env.data)
            held, lifted = 0, True               # a handoff state: the peg is held above the block
            conn.send(env.observe())
        elif cmd == "snapshot":
            conn.send({"qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(), "ctrl": env.data.ctrl.copy(),
                       "act": env.data.act.copy(), "time": float(env.data.time),
                       "block_pos": env.model.body_pos[block_id].copy()})
        elif cmd == "step":                      # (k, A) actions, one per frame; stop early on success or a drop
            success = dropped = False
            frames, video = 0, []
            for a in arg:
                env.step(np.asarray(a, dtype=np.float64), control_steps)
                frames += 1
                if record:
                    o = env.observe()
                    video.append(np.concatenate([v[::2, ::2] for k, v in o.items() if k.startswith("observation.images")], 1))
                held = held + 1 if env.success(env.model, env.data) else 0
                if held >= HOLD_FRAMES:
                    success = True
                    break
                peg, block, bottom, block_top, table = _peg_info(env)
                lifted = lifted or bottom > table + 0.02
                if lifted and bottom < table + 0.005 and np.max(np.abs(peg[:2] - block[:2])) > 0.03:
                    dropped = True               # peg back on the table, away from the hole: this try is over
                    break
            peg, block, bottom, block_top, _ = _peg_info(env)
            obs = env.observe()
            handoff = (bool(obs["observation.state"][-1] > GRASPED) and bottom > block_top + 0.002
                       and float(np.linalg.norm(peg[:2] - block[:2])) < HANDOFF_XY)
            conn.send((obs, {"success": success, "dropped": dropped, "handoff": handoff, "frames": frames,
                             "time": float(env.data.time), "video": video}))
        elif cmd == "close":
            env.close()
            conn.close()
            return


class VecEnv:
    """N peg_insert workers driven in lockstep: send to all, then gather (so physics and rendering run in parallel)."""

    def __init__(self, n: int, cameras: dict[str, tuple[int, int]], fps: int = 25):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, cameras, fps), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.n = n

    def call(self, cmd: str, args: list, idx: list[int] | None = None):
        idx = list(range(self.n)) if idx is None else idx
        for i, a in zip(idx, args):
            self.conns[i].send((cmd, a))
        return [self.conns[i].recv() for i in idx]

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=10)
