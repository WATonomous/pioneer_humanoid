"""tabletop_pick environments in worker processes, stepped one action chunk at a time.

A worker renders only when asked for an observation (once per chunk: Pioneer-1 sees only the current frame), so the
physics between replans runs without cameras.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

FPS = 25


def _worker(conn, scene, cameras, robot, max_s):
    from humanoid_robot_learning.sim.evaluate import CONTROL_DT, HOLD_S, SceneEnv

    env = SceneEnv(scene, cameras, robot)
    control_steps = round(1 / (FPS * CONTROL_DT))
    frames, held = 0, 0.0
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            env.reset(arg)
            frames, held = 0, 0.0
            conn.send((env.observe(), env.instruction()))
        elif cmd == "chunk":        # execute actions (T, A) frame by frame; stop early on success
            success = False
            for a in arg:
                env.step(a, control_steps)
                frames += 1
                held = held + 1 / FPS if env.success(env.model, env.data) else 0.0
                if held >= HOLD_S:
                    success = True
                    break
                if frames >= max_s * FPS:
                    break
            timeout = not success and frames >= max_s * FPS
            obs = None if (success or timeout) else env.observe()
            conn.send((obs, env.instruction() if obs is not None else None, success, timeout, frames))
        elif cmd == "close":
            env.close()
            conn.close()
            return
        else:
            raise ValueError(cmd)


class Envs:
    def __init__(self, n: int, cameras: dict, robot: str, scene: str = "tabletop_pick", max_s: float = 30.0):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, scene, cameras, robot, max_s), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.n = n

    def reset(self, ids, seeds):
        for i, s in zip(ids, seeds):
            self.conns[i].send(("reset", int(s)))
        return [self.conns[i].recv() for i in ids]

    def chunk(self, ids, actions: np.ndarray):
        for i, a in zip(ids, actions):
            self.conns[i].send(("chunk", a))
        return [self.conns[i].recv() for i in ids]

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=30)
