"""Leader angles -> left-arm joint targets + gripper closure, shared by the Isaac, MuJoCo and real backends.

One-to-one: the leader is calibrated hanging straight down (calibrate_leader.py), which is also the
URDF zero and the real arm's zero, so a joint target is just sign x leader angle. There is no home
gate: every valid leader reading immediately updates the simulated arm target.

Plain Python: no simulator imports. The leader is an input device with torque always off.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from arm_limits import GRIPPER_OPEN_DEG, limited_target
from servo_leader import ARM_SERVOS, GRIPPER_SERVO, SERVO_IDS, ServoLeader, load_calibration, parse_signs

# Per-servo direction, A..F then G. Flip one with --signs if a joint moves the wrong way.
DEFAULT_SIGNS = "1,1,-1,-1,1,-1,1"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "leader_calibration.json"
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
        # Runtime zero offsets are deliberately separate from the encoder calibration.  The
        # calibration says what the physical hanging pose is; these offsets say what simulated
        # joint angle that physical pose should command.  The Isaac UI can change them live.
        self.offsets = [0.0] * len(ARM_SERVOS)
        self.grip_offset = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset the filtered target to home; the next reading follows immediately."""
        self.engaged = True
        self.target = list(self.home)
        self.grip = 0.0

    def _desired(self, angles: tuple[float, ...]) -> tuple[list[float], float]:
        desired = [
            limited_target(angles[i], self.signs[i], 1.0, self.limits_deg[i], self.offsets[i])
            for i in range(len(ARM_SERVOS))
        ]
        # At the calibrated gripper-open pose the closure is zero.  Apply the runtime offset
        # before clamping so a live re-zero remains reversible at either end of travel.
        grip = (
            -self.signs[self.gripper_axis]
            * math.degrees(angles[self.gripper_axis])
            / GRIPPER_OPEN_DEG
            + self.grip_offset
        )
        grip = max(0.0, min(1.0, grip))
        return desired, grip

    def set_directions(self, angles: tuple[float, ...], signs: list[float]) -> tuple[list[float], float]:
        """Change directions live without making the simulated arm jump.

        The current physical pose keeps commanding the same desired target.  Motion after this
        point follows the new directions, which makes a mistaken inversion safe to correct while
        Isaac is running.
        """
        if len(signs) != len(SERVO_IDS) or any(sign not in (-1, 1, -1.0, 1.0) for sign in signs):
            raise ValueError(f"directions must contain {len(SERVO_IDS)} values, each +1 or -1")
        previous_arm, previous_grip = self._desired(angles)
        self.signs = [float(sign) for sign in signs]
        self.offsets = [
            target - self.signs[i] * angles[i]
            for i, target in enumerate(previous_arm)
        ]
        grip_angle_deg = math.degrees(angles[self.gripper_axis])
        self.grip_offset = previous_grip + self.signs[self.gripper_axis] * grip_angle_deg / GRIPPER_OPEN_DEG
        return self.snap(angles)

    def set_current_pose_defaults(
        self,
        angles: tuple[float, ...],
        arm_defaults_deg: list[float],
        gripper_default: float,
    ) -> tuple[list[float], float]:
        """Map the physical pose being held now to the requested simulated defaults."""
        if len(arm_defaults_deg) != len(ARM_SERVOS):
            raise ValueError(f"expected {len(ARM_SERVOS)} arm defaults")
        if not all(math.isfinite(value) for value in arm_defaults_deg):
            raise ValueError("arm defaults must be finite")
        if not math.isfinite(gripper_default):
            raise ValueError("gripper default must be finite")
        defaults_rad = [math.radians(value) for value in arm_defaults_deg]
        self.offsets = [
            default - self.signs[i] * angles[i]
            for i, default in enumerate(defaults_rad)
        ]
        gripper_default = max(0.0, min(1.0, gripper_default))
        grip_angle_deg = math.degrees(angles[self.gripper_axis])
        self.grip_offset = gripper_default + self.signs[self.gripper_axis] * grip_angle_deg / GRIPPER_OPEN_DEG
        return self.snap(angles)

    def clear_runtime_zero(self, angles: tuple[float, ...]) -> tuple[list[float], float]:
        """Return to the saved encoder calibration and snap to the resulting absolute target."""
        self.offsets = [0.0] * len(ARM_SERVOS)
        self.grip_offset = 0.0
        return self.snap(angles)

    def snap(self, angles: tuple[float, ...]) -> tuple[list[float], float]:
        """Match the filtered target exactly to the current physical leader pose."""
        desired, grip = self._desired(angles)
        self.target = list(desired)
        self.grip = grip
        return list(self.target), self.grip

    def update(self, angles: tuple[float, ...]) -> tuple[list[float], float]:
        desired, grip = self._desired(angles)
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

    def report(self, mapping: LeaderMapping) -> None:
        now = time.monotonic()
        if now - self._last_report < 0.5:
            return
        self._last_report = now
        line = (
            " ".join(f"{label}={math.degrees(a):+6.1f}" for label, a in zip(SERVO_IDS, self.angles))
            + f" grip={mapping.grip:.2f}"
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
