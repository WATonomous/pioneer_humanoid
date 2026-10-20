"""Leader angles -> left-arm joint targets + gripper closure, shared by the Isaac, MuJoCo and real backends.

One-to-one: the leader is calibrated hanging straight down (calibrate_leader.py), which is also the
URDF zero and the real arm's zero, so a joint target is just sign x leader angle. The arm starts at
home (arm_params.DEFAULT_JOINT_POS, elbow bent 90 deg) and only follows the leader once every
leader joint is within ENGAGE_TOL_DEG of home with the gripper open -- at startup and after every
reset -- so it never jumps to wherever the leader happens to be.

Plain Python: no simulator imports. The leader is an input device with torque always off.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from arm_limits import gripper_fraction, limited_target
from servo_leader import ARM_SERVOS, GRIPPER_SERVO, SERVO_IDS, ServoLeader, load_calibration, parse_signs

# Per-servo direction, A..F then G. Flip one with --signs if a joint moves the wrong way.
DEFAULT_SIGNS = "1,-1,-1,1,1,-1,1"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "leader_calibration.json"
# Follow the leader only once every arm joint is this close to home ...
ENGAGE_TOL_DEG = 3.0
# ... and the gripper this close to open (closure fraction, 0 = open).
ENGAGE_GRIP_TOL = 0.1
# Stock wrist damping (18) caps joint6l near 2 deg/s at the GL40's 0.73 Nm, so the sim wrist
# lags the leader. Each backend lowers it for this teleop only; the shared arm config keeps 18.
WRIST_DAMPING = 2.5
# Control rate: one leader read and target update per 10 ms of sim time.
CONTROL_DT = 0.01


def add_leader_args(parser: argparse.ArgumentParser, *, scene_help: str) -> None:
    parser.add_argument(
        "--target",
        choices=("isaac", "mujoco", "real"),
        default="isaac",
        help="what the leader drives (real: dry run, read-only)",
    )
    parser.add_argument("--scene", type=str, default="bare", help=scene_help)
    parser.add_argument("--port", default="/dev/ttyACM0", help="leader serial port")
    parser.add_argument("--baud", type=int, default=1_000_000, help="leader serial baud rate")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="leader zero file from calibrate_leader.py (default: leader_calibration.json here)",
    )
    parser.add_argument("--signs", default=DEFAULT_SIGNS, help=f"direction per servo A..G (default: {DEFAULT_SIGNS})")
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=0.35,
        help="target low-pass coefficient in (0,1]; 1 disables filtering",
    )


def check_leader_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 < args.filter_alpha <= 1.0:
        parser.error("--filter-alpha must be in (0, 1]")
    try:
        parse_signs(args.signs)
    except ValueError as exc:
        parser.error(f"--signs: {exc}")
    if not args.calibration.is_file():
        parser.error(f"no leader calibration at {args.calibration}; run calibrate_leader.py first")


class LeaderMapping:
    """Filtered, URDF-clamped arm targets (rad) and gripper closure (0 open .. 1 closed)."""

    def __init__(self, args: argparse.Namespace, home_rad: list[float], limits_rad: list[tuple[float, float]]):
        if len(home_rad) != len(ARM_SERVOS):
            raise ValueError(f"leader drives {len(ARM_SERVOS)} joints but the arm has {len(home_rad)}")
        self.signs = parse_signs(args.signs)
        self.alpha = args.filter_alpha
        self.home = list(home_rad)
        self.limits_deg = [(math.degrees(lo), math.degrees(hi)) for lo, hi in limits_rad]
        self.gripper_axis = tuple(SERVO_IDS).index(GRIPPER_SERVO)
        self.reset()

    def reset(self) -> None:
        """Back to home and wait for the leader to reach it (call with every arm reset)."""
        self.engaged = False
        self.target = list(self.home)
        self.grip = 0.0
        self.off_home: list[tuple[str, float]] = []

    def update(self, angles: tuple[float, ...]) -> tuple[list[float], float]:
        desired = [
            limited_target(angles[i], self.signs[i], 1.0, self.limits_deg[i])
            for i in range(len(ARM_SERVOS))
        ]
        grip = gripper_fraction(angles[self.gripper_axis], self.signs[self.gripper_axis])
        if not self.engaged:
            self.off_home = [
                (label, math.degrees(d - h))
                for label, d, h in zip(ARM_SERVOS, desired, self.home)
                if abs(math.degrees(d - h)) > ENGAGE_TOL_DEG
            ]
            if grip > ENGAGE_GRIP_TOL:
                self.off_home.append((GRIPPER_SERVO, grip))
            if self.off_home:
                return self.target, self.grip
            self.engaged = True
        self.target = [t + self.alpha * (d - t) for t, d in zip(self.target, desired)]
        self.grip += self.alpha * (grip - self.grip)
        return self.target, self.grip

    def describe(self, joints: list[str]) -> str:
        return (
            "[LEADER] Mapping (1:1 from hanging zero): "
            + " | ".join(
                f"{label}(ID{SERVO_IDS[label]}) {sign:+.0f} -> {joint}"
                for label, joint, sign in zip(ARM_SERVOS, joints, self.signs)
            )
            + f" | {GRIPPER_SERVO}(ID{SERVO_IDS[GRIPPER_SERVO]}) {self.signs[self.gripper_axis]:+.0f} -> gripper"
        )


class LeaderInput:
    """Calibrated ServoLeader that holds the last reading on a dropped serial frame instead of stopping the sim."""

    def __init__(self, args: argparse.Namespace):
        self.leader = ServoLeader(args.port, args.baud, zeros=load_calibration(args.calibration))
        self.angles = (0.0,) * len(SERVO_IDS)
        self._last_warning = 0.0
        self._last_report = time.monotonic()

    def read(self) -> tuple[float, ...]:
        try:
            self.angles = self.leader.read_radians()
        except RuntimeError as exc:
            now = time.monotonic()
            if now - self._last_warning >= 1.0:
                print(f"[WARN] {exc}; holding last leader target", flush=True)
                self._last_warning = now
        return self.angles

    def counts(self) -> tuple[int, ...]:
        """Raw encoder counts behind the last good reading, in SERVO_IDS order."""
        positions = self.leader.last_positions()
        return tuple(positions[servo_id] for servo_id in SERVO_IDS.values())

    def report(self, mapping: LeaderMapping) -> None:
        now = time.monotonic()
        if now - self._last_report < 0.5:
            return
        self._last_report = now
        if mapping.engaged:
            line = (
                " ".join(f"{label}={math.degrees(a):+6.1f}" for label, a in zip(SERVO_IDS, self.angles))
                + f" grip={mapping.grip:.2f}"
            )
        else:
            line = "move leader to home (elbow 90, gripper open): " + " ".join(
                f"{label} {err:+.0f}deg" if label != GRIPPER_SERVO else f"{label} {err:.2f}"
                for label, err in mapping.off_home
            )
        print(f"\r[LEADER] {line:<100}", end="", flush=True)

    def close(self) -> None:
        self.leader.close()


class WallClock:
    """Keep sim time at or behind wall time, so motion (and recordings) have real-world timing."""

    def __init__(self, dt: float):
        self.dt = dt
        self._next = time.monotonic()

    def wait(self) -> None:
        self._next += self.dt
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next = time.monotonic()
