"""tidy_table in plain MuJoCo with the leader teleop's observation / action contract, for scripted demos and policy
evaluation. One place, so the data a policy trains on and the sim it is evaluated in can't drift apart.

Contract (pioneer_leader_arm_teleop/mujoco_sim.py --record, schema dataset_schema_pioneer_v1.yaml):
  control       every CONTROL_DT = 10 ms of sim time; frames at the schema's 25 fps (every 4th control step)
  state  (7)    left arm joints (rad) + mean finger closure (0 open .. 1 closed)
  action (7)    left arm joint targets (rad) + gripper command (0 open .. 1 closed)
  images        arm cameras (ego, wrist_left, ...) and the scene's own cameras (top), RGB uint8 HxWx3
  observation.environment_state   the scene's condition vector (tidy_table: one-hot target + done)
  task          the current step's instruction; subtask_index its index

Needs mujoco and the repo's pioneer_humanoid + humanoid_mujoco_scenes packages (installed, or on PYTHONPATH).
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model, scene_condition, scene_progress, scene_reset, scene_step
from pioneer_humanoid.arm_params import (
    CAMERA_NAMES,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CLOSED,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
)
from pioneer_humanoid.mujoco_bimanual_arm import set_home

SCENE = "tidy_table"
CONTROL_DT = 0.01        # leader_mapping.CONTROL_DT
WRIST_DAMPING = 2.5      # leader_mapping.WRIST_DAMPING: the teleop lowers joint6l's kv to this
FPS = 25                 # dataset_schema_pioneer_v1.yaml
RECORD_EVERY = round(1 / (CONTROL_DT * FPS))


class TidySim:
    """The scene, the left arm and its cameras. ``cameras``: {name: (height, width)}."""

    def __init__(self, cameras: dict[str, tuple[int, int]], scene: str = SCENE):
        self.scene = scene
        self.model = make_model(scene, cameras={n: hw for n, hw in cameras.items() if n in CAMERA_NAMES})
        m = self.model
        unknown = [n for n in cameras if n not in CAMERA_NAMES and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n) < 0]
        if unknown:
            raise ValueError(f"cameras {unknown} are neither arm cameras {list(CAMERA_NAMES)} nor cameras of {scene!r}")
        m.actuator_biasprm[m.actuator("joint6l").id, 2] = -WRIST_DAMPING
        self.data = mujoco.MjData(m)
        self.hook, self.randomise = scene_step(scene), scene_reset(scene)
        self.progress = scene_progress(scene)
        cond = scene_condition(scene)
        self.condition_names, self.condition = cond if cond is not None else (None, None)
        self.arm_act = [m.actuator(j).id for j in LEFT_ARM_JOINTS]
        self.grip_act = [m.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
        self.arm_qpos = [m.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
        self.grip_qpos = [m.joint(j).qposadr[0] for j in LEFT_GRIPPER_JOINTS]
        self.grip_open = np.array([LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS])
        self.grip_closed = np.array([LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS])
        self.substeps = max(1, round(CONTROL_DT / m.opt.timestep))
        self.renderers = {n: mujoco.Renderer(m, h, w) for n, (h, w) in cameras.items()}
        self.target = np.zeros(len(LEFT_ARM_JOINTS))
        self.grip = 0.0

    def reset(self, seed: int) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        set_home(m, d)
        if self.randomise is not None:
            self.randomise(m, d, np.random.default_rng(seed))
        mujoco.mj_forward(m, d)
        self.target = d.qpos[self.arm_qpos].copy()
        self.grip = 0.0

    @property
    def home(self) -> np.ndarray:
        from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS

        return np.array([DEFAULT_JOINT_POS[j] for j in LEFT_ARM_JOINTS])

    def act(self, action) -> None:
        """Set the 7-D action (6 joint targets + gripper 0..1) and advance one control step."""
        m, d = self.model, self.data
        lo, hi = m.jnt_range[[m.joint(j).id for j in LEFT_ARM_JOINTS]].T
        self.target = np.clip(np.asarray(action[:6], dtype=float), lo, hi)
        self.grip = float(np.clip(action[6], 0.0, 1.0))
        d.ctrl[self.arm_act] = self.target
        d.ctrl[self.grip_act] = self.grip_open + self.grip * (self.grip_closed - self.grip_open)
        if self.hook is not None:
            self.hook(m, d)
        mujoco.mj_step(m, d, nstep=self.substeps)

    def state(self) -> np.ndarray:
        d = self.data
        closure = np.mean((d.qpos[self.grip_qpos] - self.grip_open) / (self.grip_closed - self.grip_open))
        return np.append(d.qpos[self.arm_qpos], np.clip(closure, 0.0, 1.0)).astype(np.float32)

    def action(self) -> np.ndarray:
        return np.append(self.target, self.grip).astype(np.float32)

    def images(self) -> dict[str, np.ndarray]:
        out = {}
        for name, r in self.renderers.items():
            r.update_scene(self.data, name)
            out[name] = r.render()
        return out

    def env_state(self) -> np.ndarray | None:
        return None if self.condition is None else self.condition(self.model, self.data).astype(np.float32)

    def step_info(self) -> tuple[int, int, str]:
        return self.progress(self.model, self.data)

    def close(self) -> None:
        for r in self.renderers.values():
            r.close()
