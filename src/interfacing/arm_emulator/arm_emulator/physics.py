"""The arm's mechanics: a scene, the left arm on torque motors, one Drive per motor, 1 kHz. No ROS here.

The right arm keeps its stiff position actuators: it stays at home, out of the way, as in the other sims.
Frames (joint_command applyCalibration): motor = direction * (cmd - zero_offset), cmd = URDF. Gripper: the
motor's command angle maps linearly to finger travel (config/motor_models.yaml).

Thread-safety: everything public takes the lock, because ROS callbacks and renderers run on other threads.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from arm_emulator.config import MOTOR_MODELS, REPO_CALIBRATION, REPO_PROFILES, load_motors
from arm_emulator.drive import Drive, Feedback
from arm_emulator.faults import Fault
from humanoid_mujoco_scenes import make_model, scene_reset, scene_success
from pioneer_humanoid.arm_params import (LEFT_ARM_JOINTS, LEFT_GRIPPER_CLOSED, LEFT_GRIPPER_JOINTS, LEFT_GRIPPER_OPEN)
from pioneer_humanoid.mujoco_bimanual_arm import set_home

LOAD_CHECK_EVERY = 20  # physics steps between contact-force checks: mj_contactForce on every step is slow
# How long a hand stays under the arm after the first stiff command: longer than joint_command's gravity
# feed-forward ramp (kGravityFfRampSec, 1 s). Let go sooner and the wrist, at kp 0.244, falls past its tracking limit.
SUPPORT_AFTER_FIRST_COMMAND_S = 1.5


class EmulatedArm:
    def __init__(self, scene: str = "rubiks_cube", cameras: dict[str, tuple[int, int]] | None = None, seed: int = 0,
                 calibration: Path = REPO_CALIBRATION, profiles: Path = REPO_PROFILES, models: Path = MOTOR_MODELS,
                 start_pose_deg: list[float] | None = None, supported: bool = False) -> None:
        """``start_pose_deg``: six angles, command frame (default: hanging). ``supported``: someone holds the
        arm there, as at the real arm, until the first command with stiffness."""
        self.scene = scene
        self.supported = supported
        self._let_go_at: float | None = None
        self.specs, catalogue = load_motors(calibration, profiles, models)
        self.model = model = make_model(scene, cameras=cameras, torque_joints=[*LEFT_ARM_JOINTS, *LEFT_GRIPPER_JOINTS])
        self.data = mujoco.MjData(model)
        self.dt = model.opt.timestep
        self.rng = np.random.default_rng(seed)
        self.lock = threading.RLock()
        self._randomise = scene_reset(scene)
        self._success = scene_success(scene)

        ambient, timeout = catalogue["ambient_temperature"], catalogue["ak_can_timeout_s"]
        self.drives = {s.motor_id: Drive(s, ambient, timeout) for s in self.specs}
        self._arm = [s for s in self.specs if s.slot is not None]
        self._grip = self.specs[-1]
        self._qadr = {s.motor_id: model.joint(LEFT_ARM_JOINTS[s.slot]).qposadr[0] for s in self._arm}
        self._dadr = {s.motor_id: model.joint(LEFT_ARM_JOINTS[s.slot]).dofadr[0] for s in self._arm}
        self._act = {s.motor_id: model.actuator(LEFT_ARM_JOINTS[s.slot]).id for s in self._arm}

        # Mechanics of each motor. Reflected inertia etc. are ASSUMED until fit (motor_models.yaml).
        for spec in self._arm:
            dof = self._dadr[spec.motor_id]
            model.dof_armature[dof] = spec.model["armature"]
            model.dof_damping[dof] = spec.model["damping"]
            model.dof_frictionloss[dof] = spec.model["frictionloss"]
        self._finger_qadr = [model.joint(j).qposadr[0] for j in LEFT_GRIPPER_JOINTS]
        self._finger_dadr = [model.joint(j).dofadr[0] for j in LEFT_GRIPPER_JOINTS]
        self._finger_acts = [model.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
        self._finger_closed = np.array([LEFT_GRIPPER_CLOSED[j] - LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS])
        grip = catalogue["gripper"]
        self.finger_travel = float(grip["finger_travel_m"])
        self.grip_force_scale = float(grip["force_scale"])
        self._grip_range = math.radians(self._grip.upper_deg)  # motor travel for closure 0..1
        mujoco.mj_setConst(model, self.data)

        self.contact_limit = float(catalogue["load_limits"]["contact_force"])
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
        self._arm_bodies = {b for b, n in enumerate(names) if n.startswith("link") and n[-1] in "lL"}
        # Objects to grasp: anything loose. Fixed things (the table) are what the arm must not hit hard.
        self._loose_bodies = {b for b in range(model.nbody) if model.body_jntnum[b] and b not in self._arm_bodies
                              and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE}
        self._locked: dict[int, float] = {}  # motor_id -> the angle a stalled joint is stuck at
        self.steps = 0
        self.breaches: list[str] = []
        self.peak_contact = 0.0
        self.peak_torque = {s.motor_id: 0.0 for s in self.specs}
        self.start_pose_deg = start_pose_deg
        self._reset_all()

    # ---- resets --------------------------------------------------------------------------------------

    def _reset_all(self) -> None:
        """Right arm at home, left arm hanging (or start_pose_deg), as the real arm is when powered on."""
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        set_home(model, data)
        for spec in self._arm:
            pose = 0.0 if self.start_pose_deg is None else math.radians(self.start_pose_deg[spec.slot])
            data.qpos[self._qadr[spec.motor_id]] = pose
            data.ctrl[self._act[spec.motor_id]] = 0.0
        data.qpos[self._finger_qadr] = 0.0
        data.ctrl[self._finger_acts] = 0.0
        if self._randomise is not None:
            self._randomise(model, data, self.rng)
        mujoco.mj_forward(model, data)

    def reset_objects(self, seed: int | None = None) -> None:
        """A new layout for the scene's objects; the arm stays where it is (teleporting it would be a fault)."""
        if self._randomise is None:
            return
        with self.lock:
            self._randomise(self.model, self.data, np.random.default_rng(seed) if seed is not None else self.rng)
            mujoco.mj_forward(self.model, self.data)

    # ---- frames --------------------------------------------------------------------------------------

    def _closure(self) -> tuple[float, float]:
        """Finger closure (0 open .. 1 closed) and its rate, from the two sliding fingers."""
        data = self.data
        closed = self._finger_closed
        c = float(np.mean(data.qpos[self._finger_qadr] / closed))
        rate = float(np.mean(data.qvel[self._finger_dadr] / closed))
        return c, rate

    def _motor_state(self, spec: Any) -> tuple[float, float]:
        """(angle rad, speed rad/s) of one motor in its own frame, without any fault applied."""
        zero = math.radians(spec.zero_offset_deg)
        if spec.slot is None:
            c, rate = self._closure()
            return spec.direction * (c * self._grip_range - zero), spec.direction * rate * self._grip_range
        data = self.data
        return (spec.direction * (data.qpos[self._qadr[spec.motor_id]] - zero),
                spec.direction * data.qvel[self._dadr[spec.motor_id]])

    # ---- stepping ------------------------------------------------------------------------------------

    def step(self) -> None:
        """One physics tick: every drive's law, then MuJoCo."""
        with self.lock:
            data = self.data
            now = float(data.time)
            for spec in self._arm:
                q, qd = self._motor_state(spec)
                data.ctrl[self._act[spec.motor_id]] = spec.direction * self.drives[spec.motor_id].step(q, qd, now, self.dt)
            q, qd = self._motor_state(self._grip)
            tau_cmd = self._grip.direction * self.drives[self._grip.motor_id].step(q, qd, now, self.dt)
            # Virtual work: tau d(angle) = F * 2 d(finger), both fingers moved by one motor.
            force = self.grip_force_scale * tau_cmd * self._grip_range / (2.0 * self.finger_travel)
            data.ctrl[self._finger_acts] = force * np.sign(self._finger_closed)
            if self.supported and self._let_go_at is None and any(self.drives[s.motor_id].kp > 0.0 for s in self._arm):
                self._let_go_at = now + SUPPORT_AFTER_FIRST_COMMAND_S
            if self._let_go_at is not None and now >= self._let_go_at:
                self.supported = False
            held = self.supported and data.qpos[[self._qadr[s.motor_id] for s in self._arm]].copy()
            mujoco.mj_step(self.model, data)
            if self.supported:
                for spec, angle in zip(self._arm, held):
                    data.qpos[self._qadr[spec.motor_id]] = angle
                    data.qvel[self._dadr[spec.motor_id]] = 0.0
            for motor_id, angle in self._locked.items():
                data.qpos[self._qadr[motor_id]] = angle
                data.qvel[self._dadr[motor_id]] = 0.0
            self.steps += 1
            if self.steps % LOAD_CHECK_EVERY == 0:
                self._check_loads()

    def _check_loads(self) -> None:
        """Limits MuJoCo itself doesn't enforce: the sim arm can't break, so a breach has to fail the run."""
        for spec in self.specs:
            tau = abs(self.drives[spec.motor_id].tau)
            self.peak_torque[spec.motor_id] = max(self.peak_torque[spec.motor_id], tau)
            if tau > 0.999 * spec.model["peak_torque"]:
                self._breach(f"{spec.name}: at its motor's peak torque {spec.model['peak_torque']:g} N.m")
        force = np.zeros(6)
        model, data = self.model, self.data
        for i in range(data.ncon):
            con = data.contact[i]
            b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
            if (b1 in self._arm_bodies) == (b2 in self._arm_bodies) or {b1, b2} & self._loose_bodies:
                continue  # arm-on-world only: gripping the cube is the task
            mujoco.mj_contactForce(model, data, i, force)
            normal = float(abs(force[0]))
            self.peak_contact = max(self.peak_contact, normal)
            if normal > self.contact_limit:
                self._breach(f"arm contact {normal:.0f} N over {self.contact_limit:g} N")

    def _breach(self, text: str) -> None:
        if text not in self.breaches:
            self.breaches.append(text)

    # ---- the bus -------------------------------------------------------------------------------------

    def command(self, motor_id: int, control_type: int, position: float, velocity: float, kp: float, kd: float,
                torque: float) -> Feedback | None:
        """A MotorCmd arrives; -> the drive's answer (None: not this arm's motor, refused, or muted)."""
        drive = self.drives.get(motor_id)
        if drive is None:
            return None
        with self.lock:
            if not drive.command(control_type, position, velocity, kp, kd, torque, float(self.data.time)):
                return None
            return drive.feedback(*self._motor_state(drive.spec))

    def inject(self, fault: Fault) -> None:
        with self.lock:
            drive = self.drives[fault.motor_id]
            if fault.kind == "mute":
                drive.mute = True
            elif fault.kind == "code":
                drive.fault_code = int(fault.value) or 2
            elif fault.kind == "dead":
                drive.torque_scale = 0.0
            elif fault.kind == "zero_shift":
                drive.zero_shift_deg = fault.value
            elif fault.kind == "hot":
                drive.heat_offset = fault.value
            elif fault.kind == "stall":
                if drive.spec.slot is None:
                    raise ValueError("stall: arm joints only")
                self._locked[fault.motor_id] = float(self.data.qpos[self._qadr[fault.motor_id]])

    # ---- what others read ----------------------------------------------------------------------------

    def qpos_copy(self) -> np.ndarray:
        with self.lock:
            return self.data.qpos.copy()

    def joints_deg(self) -> np.ndarray:
        """True joint angles, command frame, ArmPose slot order. For tests: the policy sees only feedback."""
        with self.lock:
            return np.degrees([self.data.qpos[self._qadr[s.motor_id]] for s in self._arm])

    def closure(self) -> float:
        with self.lock:
            return self._closure()[0]

    def success(self) -> bool:
        if self._success is None:
            return False
        with self.lock:
            return bool(self._success(self.model, self.data))
