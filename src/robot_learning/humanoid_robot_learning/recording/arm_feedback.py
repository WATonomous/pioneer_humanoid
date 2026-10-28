"""MotorFeedback -> the dataset's observation.state: six joints in rad + gripper closure 0..1.

Feedback arrives in each drive's motor frame (deg). joint_config.py, the arm_calibration.yaml reader
that joint_command's Python tools share, turns it into the command frame (= URDF frame), the
frame policies were trained in. Slots follow ArmPose (arm_pose_io.JOINT_ORDER_DOC). No ROS imports:
on_feedback takes any object with motor_id and position.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# joint_config lives with the CAN tools, which are scripts rather than a package (as in real_arm.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "interfacing" / "can" / "scripts"))

from joint_config import gripper_position, load_joint_map, motor_to_cmd_deg  # noqa: E402

# Same as joint_command's kSeedFeedbackMaxAgeS: older feedback is a motor that stopped reporting.
FEEDBACK_STALE_S = 0.5
GRIPPER_NAME = "gripper.open_close"


class ArmFeedbackState:
    def __init__(self, calibration: Path, arm_side: str = "left", stale_s: float = FEEDBACK_STALE_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        joint_map = load_joint_map(calibration, arm_side)
        slots = {info["slot"]: dict(info, motor_id=mid) for mid, info in joint_map.items() if info["slot"] is not None}
        missing = [i for i in range(6) if i not in slots]
        gripper = [dict(info, motor_id=mid) for mid, info in joint_map.items() if info["name"] == GRIPPER_NAME]
        if missing or not gripper:
            raise SystemExit(f"{calibration}: {arm_side} arm needs 6 ArmPose joints and {GRIPPER_NAME} "
                             f"(missing slots {missing}, gripper {'ok' if gripper else 'missing'})")
        self.joints = [slots[i] for i in range(6)]
        self.gripper = gripper[0]
        self._by_id = {j["motor_id"]: j for j in self.joints + [self.gripper]}
        self._stale_s, self._clock = stale_s, clock
        self._lock = threading.Lock()
        self._latest: dict[int, tuple[float, float]] = {}  # motor_id -> (command-frame deg, time)

    def on_feedback(self, msg: Any) -> None:
        info = self._by_id.get(int(msg.motor_id))
        if info is not None and math.isfinite(msg.position):
            with self._lock:
                self._latest[info["motor_id"]] = (motor_to_cmd_deg(info, float(msg.position)), self._clock())

    def fresh_deg(self) -> dict[int, float]:
        """Command-frame angle (deg) of every motor that reported within stale_s."""
        now = self._clock()
        with self._lock:
            return {mid: deg for mid, (deg, t) in self._latest.items() if now - t < self._stale_s}

    def missing(self) -> list[str]:
        """Joints without fresh feedback; the state is complete when this is empty."""
        fresh = self.fresh_deg()
        return [j["name"] for j in self.joints + [self.gripper] if j["motor_id"] not in fresh]

    def joints_deg(self) -> np.ndarray | None:
        """(6,) command-frame deg in ArmPose slot order, or None if any joint is stale."""
        fresh = self.fresh_deg()
        if any(j["motor_id"] not in fresh for j in self.joints):
            return None
        return np.array([fresh[j["motor_id"]] for j in self.joints])

    def state(self) -> np.ndarray | None:
        """(7,) float32 observation.state: joints in rad, then gripper 0 open .. 1 closed."""
        q_deg, fresh = self.joints_deg(), self.fresh_deg()
        if q_deg is None or self.gripper["motor_id"] not in fresh:
            return None
        grip = float(np.clip(gripper_position(self.gripper, fresh[self.gripper["motor_id"]]), 0.0, 1.0))
        return np.append(np.radians(q_deg), grip).astype(np.float32)

    def out_of_limits(self) -> list[str]:
        """Joints measured outside their calibrated limits: joint_command excludes those (stale calibration)."""
        fresh = self.fresh_deg()
        return [f"{j['name']}: {fresh[j['motor_id']]:+.1f} deg outside [{j['lower']:.1f}, {j['upper']:.1f}]"
                for j in self.joints
                if j["limit_range"] and j["motor_id"] in fresh and not j["lower"] <= fresh[j["motor_id"]] <= j["upper"]]
