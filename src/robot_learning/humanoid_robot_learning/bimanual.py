"""Pioneer data contract and MuJoCo mapping shared by recording and rollout.

Order: left six joints, left closure, right six joints, right closure.
Arm units are radians; closures are 0 (open) to 1 (closed).
"""
from __future__ import annotations

import numpy as np
from pioneer_humanoid import arm_params as params

ARM_FIELDS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
              "elbow_roll", "wrist_pitch", "gripper")
BIMANUAL_RECORD_JOINTS = [f"{side}_{name}" for side in ("left", "right") for name in ARM_FIELDS]


class BimanualMapping:
    """Map fourteen command values to sixteen MuJoCo actuator targets, by name."""

    def __init__(self, model):
        self.arms = []
        for side in ("LEFT", "RIGHT"):
            joints = getattr(params, f"{side}_ARM_JOINTS")
            fingers = getattr(params, f"{side}_GRIPPER_JOINTS")
            opened = getattr(params, f"{side}_GRIPPER_OPEN")
            closed = getattr(params, f"{side}_GRIPPER_CLOSED")
            self.arms.append(dict(
                qpos=[int(model.joint(j).qposadr[0]) for j in joints],
                actuators=[model.actuator(j).id for j in joints],
                finger_qpos=[int(model.joint(j).qposadr[0]) for j in fingers],
                finger_actuators=[model.actuator(j).id for j in fingers],
                low=np.array([model.joint(j).range[0] for j in joints]),
                high=np.array([model.joint(j).range[1] for j in joints]),
                opened=np.array([opened[j] for j in fingers]),
                closed=np.array([closed[j] for j in fingers]),
            ))

    def state(self, data):
        values = []
        for arm in self.arms:
            closure = (data.qpos[arm["finger_qpos"]] - arm["opened"]) / (arm["closed"] - arm["opened"])
            values.extend(data.qpos[arm["qpos"]])
            values.append(np.clip(closure.mean(), 0, 1))
        return np.asarray(values, dtype=np.float32)

    def apply(self, data, action):
        """Validate the entire action before writing either arm. Return bounded targets."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected fourteen finite action values")
        bounded = action.copy()
        for offset, arm in zip((0, 7), self.arms):
            bounded[offset:offset + 6] = np.clip(action[offset:offset + 6], arm["low"], arm["high"])
            bounded[offset + 6] = np.clip(action[offset + 6], 0, 1)
        for offset, arm in zip((0, 7), self.arms):
            data.ctrl[arm["actuators"]] = bounded[offset:offset + 6]
            data.ctrl[arm["finger_actuators"]] = arm["opened"] + bounded[offset + 6] * (arm["closed"] - arm["opened"])
        return bounded.astype(np.float32)
