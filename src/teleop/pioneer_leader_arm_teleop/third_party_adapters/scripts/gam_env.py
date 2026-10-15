"""Our MuJoCo scenes as a GAM.eval.run environment (the adapter protocol of GAM/eval/envs.py).

    python -m GAM.eval.run --config configs/pioneer_peg.yaml --ckpt runs/.../latest.pt \
        --env /abs/path/gam_env.py:PegInsertEnv --episodes 200

Episode i is layout PIONEER_SEED (default 1000) + i, as in sim_policy_eval.py; one action per 1/25 s frame, the
same actuation (joint targets clipped to the limits, gripper closure 0..1) and the same success rule (the scene's
check held for 0.4 s, within PIONEER_MAX_S seconds, default 40). Calibration comes from the yaml, which holds the
simulator's exact values.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/teleop/pioneer_leader_arm_teleop"))

import numpy as np  # noqa: E402


class PioneerEnv:
    scene, n_episodes, fps, hold_s = "peg_insert", 200, 25, 0.4

    def __init__(self, cfg):
        import synthetic_teleop  # noqa: F401  (puts the repo packages on sys.path)
        from synthetic_tasks import TASKS

        self.task = TASKS[self.scene]
        self.K = np.array(cfg["data"]["camera_intrinsics"], dtype=np.float64)
        self.E = np.array(cfg["data"]["camera_extrinsics"], dtype=np.float64)[:3]
        self.seed0, self.max_s = int(os.environ.get("PIONEER_SEED", 1000)), float(os.environ.get("PIONEER_MAX_S", 40))
        self.depth = cfg.get("perception", {}).get("detector_depth") == "sensor" or cfg.get("perception", {}).get("geometry_depth") == "sensor"
        self.sim = self.depth_renderer = None

    def reset(self, episode):
        from synthetic_teleop import TeleopSim

        self.close()
        self.sim = TeleopSim(self.scene, self.seed0 + int(episode), {"ego": (480, 640)})
        self.held, self.frames = 0.0, 0
        if self.depth:                                  # the simulator's exact depth image, as an RGB-D camera gives it
            import mujoco
            self.depth_renderer = mujoco.Renderer(self.sim.model, 480, 640)
            self.depth_renderer.enable_depth_rendering()
        return self.instruction()

    def instruction(self):
        sim = self.sim
        return sim.progress(sim.model, sim.data)[2] if sim.progress is not None else self.task.instruction

    def observe(self):
        sim, d = self.sim, self.sim.data
        closure = sum((d.qpos[q] - o) / (c - o) for q, o, c in zip(sim.grip_qpos, sim.grip_open, sim.grip_closed)) / len(sim.grip_qpos)
        state = np.append(d.qpos[sim.arm_qpos], min(max(closure, 0.0), 1.0)).astype(np.float32)
        rgb = sim.render_at(d.qpos.copy())["ego"]
        depth = np.zeros(rgb.shape[:2], np.float32)
        if self.depth_renderer is not None:
            self.depth_renderer.update_scene(d, camera="ego")
            depth = self.depth_renderer.render().astype(np.float32)
        return {"rgb": rgb, "depth": depth, "K": self.K, "E": self.E, "state": state}

    def step(self, action):
        sim, m, d = self.sim, self.sim.model, self.sim.data
        action = np.asarray(action, dtype=float)
        lo, hi = m.jnt_range[[m.joint(j).id for j in sim.joints]].T
        g = min(max(action[6], 0.0), 1.0)
        d.ctrl[sim.arm_acts] = np.clip(action[:6], lo, hi)
        d.ctrl[sim.grip_acts] = [o + g * (c - o) for o, c in zip(sim.grip_open, sim.grip_closed)]
        for _ in range(round(1.0 / self.fps / sim.control_dt)):
            if sim.hook is not None:
                sim.hook(m, d)
            sim.mujoco.mj_step(m, d, nstep=sim.substeps)
        self.frames += 1
        self.held = self.held + 1.0 / self.fps if self.task.success(m, d) else 0.0
        success = self.held >= self.hold_s
        return success or self.frames >= self.max_s * self.fps, success

    def close(self):
        if self.depth_renderer is not None:
            self.depth_renderer.close()
            self.depth_renderer = None
        if self.sim is not None:
            self.sim.close()
            self.sim = None


class PegInsertEnv(PioneerEnv):
    scene = "peg_insert"


class DuploEnv(PioneerEnv):
    scene, n_episodes = "duplo", 100
