"""Extra per-frame dataset fields for the MuJoCo arm teleops (8dof rig): hand pose, joint velocity, joint effort.

Recorded next to the joint-space observation.state / action, so the same takes can train joint-space policies (today)
and hand-pose policies (Pixel / UMI style, which also lets human demos and a changed arm share data):

  observation.ee_pose   per arm: hand position (m) + 6D rotation, measured (FK of the measured joints)
  action.ee_pose        per arm: the same for the commanded joints (FK of the action's joint targets)
  observation.velocity  per arm: joint velocities (rad/s), then the gripper's closure rate (1/s, 0-1 closure per s)
  observation.effort    per arm: actuator torque on each joint (N m), then the gripper's mean finger force (N)

Hand frame: the arm's last link (``<side>_link7``, the wrist-yaw link that carries the gripper), expressed in
``base_link``. 6D rotation: the first two columns of the rotation matrix, column by column (r11 r21 r31 r12 r22 r32;
Zhou et al. 2019, as Pixel, UMI and Diffusion Policy use). Arms in the frame's order: left first, as the schema.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSE_COMPONENTS = ("x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32")


@dataclass
class ArmSpec:
    side: str                 # "left" / "right": prefixes the component names
    joints: list[str]         # arm joints, in the schema's order
    fingers: list[str]        # gripper finger joints (driven), averaged
    grip_open: np.ndarray     # finger qpos when open / closed (closure = 0 / 1)
    grip_closed: np.ndarray
    hand_body: str            # the body whose frame is the hand pose


def arm_8dof_spec(side: str) -> ArmSpec:
    from pioneer_humanoid import arm_8dof_params as p

    joints, fingers, grip_open, grip_closed = {
        "left": (p.LEFT_ARM_JOINTS, p.LEFT_GRIPPER_JOINTS, p.LEFT_GRIPPER_OPEN, p.LEFT_GRIPPER_CLOSED),
        "right": (p.RIGHT_ARM_JOINTS, p.RIGHT_GRIPPER_JOINTS, p.RIGHT_GRIPPER_OPEN, p.RIGHT_GRIPPER_CLOSED),
    }[side]
    return ArmSpec(side, list(joints), list(fingers), np.array([grip_open[j] for j in fingers]),
                   np.array([grip_closed[j] for j in fingers]), f"{side}_link7")


def pose_vector(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """[x y z r11 r21 r31 r12 r22 r32]: position, then the rotation's first two columns."""
    return np.concatenate([position, rotation[:, 0], rotation[:, 1]])


class ArmExtras:
    """Computes the extra fields from a MuJoCo model/data; `features` is SimLeRobotRecorder's `extra_features`."""

    def __init__(self, model, arms: list[ArmSpec], base_body: str = "base_link"):
        import mujoco

        self.model, self.arms = model, arms
        self._mujoco = mujoco
        self._base = model.body(base_body).id
        self._scratch = mujoco.MjData(model)     # FK of the commanded joints, never stepped
        self._ids = []
        for arm in arms:
            self._ids.append(dict(
                qpos=[model.joint(j).qposadr[0] for j in arm.joints],
                dofs=[model.joint(j).dofadr[0] for j in arm.joints],
                finger_qpos=[model.joint(j).qposadr[0] for j in arm.fingers],
                finger_dofs=[model.joint(j).dofadr[0] for j in arm.fingers],
                hand=model.body(arm.hand_body).id,
            ))

    @property
    def features(self) -> dict[str, list[str]]:
        pose = [f"{arm.side}_hand_{c}" for arm in self.arms for c in POSE_COMPONENTS]
        joints = [n for arm in self.arms for n in [*arm.joints, f"{arm.side}_gripper"]]
        return {"observation.ee_pose": pose, "action.ee_pose": list(pose),
                "observation.velocity": joints, "observation.effort": list(joints)}

    def _hand_poses(self, data) -> list[np.ndarray]:
        base_r = data.xmat[self._base].reshape(3, 3)
        base_p = data.xpos[self._base]
        out = []
        for ids in self._ids:
            r = base_r.T @ data.xmat[ids["hand"]].reshape(3, 3)
            p = base_r.T @ (data.xpos[ids["hand"]] - base_p)
            out.append(pose_vector(p, r))
        return out

    def values(self, data, commands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """`data`: the sim state the frame records (kinematics up to date); `commands`: {side: commanded arm joints}."""
        measured = self._hand_poses(data)
        s = self._scratch
        s.qpos[:] = data.qpos
        s.mocap_pos[:], s.mocap_quat[:] = data.mocap_pos, data.mocap_quat
        for arm, ids in zip(self.arms, self._ids):
            s.qpos[ids["qpos"]] = commands[arm.side]
        self._mujoco.mj_kinematics(self.model, s)
        commanded = self._hand_poses(s)

        velocity, effort = [], []
        for arm, ids in zip(self.arms, self._ids):
            span = arm.grip_closed - arm.grip_open
            velocity.extend([*data.qvel[ids["dofs"]], float(np.mean(data.qvel[ids["finger_dofs"]] / span))])
            effort.extend([*data.qfrc_actuator[ids["dofs"]], float(np.mean(np.abs(data.qfrc_actuator[ids["finger_dofs"]])))])
        f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
        return {"observation.ee_pose": f32(np.concatenate(measured)), "action.ee_pose": f32(np.concatenate(commanded)),
                "observation.velocity": f32(velocity), "observation.effort": f32(effort)}
