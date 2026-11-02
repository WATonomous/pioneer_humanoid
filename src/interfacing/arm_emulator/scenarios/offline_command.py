"""A Python stand-in for joint_command, for checks that run with no ROS (physics, grasping, unit tests).

NOT the safety layer and never used near the arm: it only produces MotorCmd-shaped values from the real
YAMLs, so the physics can be exercised on a machine without ROS. Same tick as JointCommandCore::
armPoseToMotorCmds (clamp, low-pass, velocity limit, delta limit, clamp, calibration) and checkMitFaults'
tracking check. The real node is what scenarios/ and the emulator run; test_joint_command.py checks it.
"""
from __future__ import annotations

import math

import yaml

from arm_emulator.config import REPO_ACTUATORS
from arm_emulator.physics import EmulatedArm

MIT_CONTROL, MIT_ENTER = 0, 9
CONTROL_DT = 0.02


class OfflineCommand:
    def __init__(self, arm: EmulatedArm, actuators=REPO_ACTUATORS) -> None:
        self.arm = arm
        safety = yaml.safe_load(actuators.read_text(encoding="utf-8"))["safety"]
        self.blocks = []
        for spec in arm.specs:
            group, joint = spec.name.split(".")
            self.blocks.append({**safety["global"], **safety["left"][group][joint]})
        self.targets = None
        self.last_motor_deg = [0.0] * len(arm.specs)
        self.worst_track = {s.name: 0.0 for s in arm.specs}
        self.feedback = {}
        for spec in arm.specs:
            arm.command(spec.motor_id, MIT_ENTER, 0, 0, 0, 0, 0)

    def seed(self) -> None:
        """Start from where the arm is, as joint_command seeds from feedback."""
        self.targets = []
        for spec, angle in zip(self.arm.specs, [*self.arm.joints_deg(), self.arm.closure() * self.arm.specs[-1].upper_deg]):
            self.targets.append(float(angle))

    def tick(self, pose_deg, grip: float) -> None:
        """One 50 Hz tick: pose_deg (six, command frame) and gripper 0..1 -> one MotorCmd per motor."""
        if self.targets is None:
            self.seed()
        wanted = [*pose_deg, grip * self.arm.specs[-1].upper_deg]
        for i, spec in enumerate(self.arm.specs):
            b = self.blocks[i]
            prev = self.targets[i]
            lo, hi = self._limits(spec)
            target = min(max(wanted[i], lo), hi)
            target = b["low_pass_alpha"] * target + (1.0 - b["low_pass_alpha"]) * prev
            step = min(abs(b["velocity_max"]) * CONTROL_DT, abs(b["delta_max"]))
            target = min(max(target, prev - step), prev + step)
            target = min(max(target, lo), hi)
            self.targets[i] = target
            motor_deg = spec.direction * (target - spec.zero_offset_deg)
            self.last_motor_deg[i] = motor_deg
            fb = self.arm.command(spec.motor_id, MIT_CONTROL, math.radians(motor_deg), 0.0, b["mit_kp"], b["mit_kd"],
                                  self._feed_forward(i, b))
            if fb is not None:
                self.feedback[spec.motor_id] = fb
                err = abs(fb.position - motor_deg)
                self.worst_track[spec.name] = max(self.worst_track[spec.name], err)

    def _limits(self, spec) -> tuple[float, float]:
        from joint_config import load_joint_map
        from arm_emulator.config import REPO_CALIBRATION

        if not hasattr(self, "_map"):
            self._map = load_joint_map(REPO_CALIBRATION, "left")
        info = self._map[spec.motor_id]
        return info["lower"], info["upper"]

    def _feed_forward(self, i: int, block: dict) -> float:
        """Gravity feed-forward. The real node has its own CAD-mass model; here, MuJoCo's, from the same URDF."""
        spec = self.arm.specs[i]
        scale = block.get("gravity_ff_scale") or 0.0
        if spec.slot is None or not scale:
            return 0.0
        hold = spec.direction * float(self.arm.data.qfrc_bias[self.arm._dadr[spec.motor_id]])
        limit = block["gravity_ff_max_torque"]
        return min(max(scale * hold, -limit), limit)

    def run(self, seconds: float, pose_deg, grip: float = 0.0) -> None:
        """Hold a target for this long: 50 Hz commands, 1 kHz physics."""
        substeps = round(CONTROL_DT / self.arm.dt)
        for _ in range(round(seconds / CONTROL_DT)):
            self.tick(pose_deg, grip)
            for _ in range(substeps):
                self.arm.step()

    def ramp(self, seconds: float, start_deg, goal_deg, grip: float = 0.0) -> None:
        n = max(1, round(seconds / CONTROL_DT))
        substeps = round(CONTROL_DT / self.arm.dt)
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3.0 - 2.0 * s)
            self.tick([a + s * (b - a) for a, b in zip(start_deg, goal_deg)], grip)
            for _ in range(substeps):
                self.arm.step()
