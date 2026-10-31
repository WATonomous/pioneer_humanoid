"""Add the left gripper to real-arm recordings, as the sim recorders do.

The schema's 7th value per arm is `left_gripper`, a closure fraction (0 = open, 1 = closed):

    action[6]  ArmPose.gripper_position on /arm/joint_targets -- the leader's gripper. The leader
               teleop fills it on every live message, whether or not include_gripper is set.
    state[6]   the real gripper's measured closure: GL40 (CAN id 21) feedback on
               /interfacing/motorFeedback, converted with arm_calibration.yaml the same way
               joint_command and real_arm.py do. The sim uses the mean of its two finger joints;
               the real gripper is one motor, so its position is the equivalent.

Read-only: this subscribes, it never publishes, so it cannot move a motor.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from humanoid_robot_learning.recording.ros_buffer import RosRecordBuffer
from humanoid_robot_learning.recording.snapshot import ObservationSnapshot

logger = logging.getLogger(__name__)

GRIPPER_JOINT_NAME = "left_gripper"
GRIPPER_CAN_ID = 21
FEEDBACK_TOPIC = "/interfacing/motorFeedback"
# arm_actuators.yaml gripper mit_feedback_timeout: older than this, the reading is treated as missing.
FEEDBACK_MAX_AGE_S = 0.2
WARN_PERIOD_S = 2.0

_SRC = Path(__file__).resolve().parents[3]
# Same search order as interfacing/can/scripts/joint_config.py DEFAULT_CALIBRATIONS.
DEFAULT_CALIBRATIONS = (
    "/calibration/arm_calibration.yaml",
    "/root/ament_ws/src/interfacing/joint_command/config/arm_calibration.yaml",
    "/root/ament_ws/src/joint_command/config/arm_calibration.yaml",
    str(_SRC / "interfacing" / "joint_command" / "config" / "arm_calibration.yaml"),
)


def find_calibration(explicit: str | None) -> Path:
    for path in [explicit] if explicit else DEFAULT_CALIBRATIONS:
        try:
            if path and Path(path).is_file():
                return Path(path)
        except OSError:  # e.g. the container paths under /root, unreadable on a host
            continue
    raise SystemExit(
        "could not find arm_calibration.yaml; pass --gripper_calibration PATH "
        f"(looked in: {', '.join(DEFAULT_CALIBRATIONS)})"
    )


def load_gripper_calibration(path: Path, arm_side: str = "left") -> dict[str, float]:
    """{direction, zero_offset, upper} of the gripper block (can_id 21) in arm_calibration.yaml."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = ((data.get(arm_side) or {}).get("gripper") or {}).get("open_close")
    if not isinstance(block, dict) or int(block.get("can_id", -1)) != GRIPPER_CAN_ID:
        raise SystemExit(f"{path}: no {arm_side}.gripper.open_close block with can_id {GRIPPER_CAN_ID}")
    upper = float(block["upper_limit"])
    if upper == 0.0:
        raise SystemExit(f"{path}: gripper upper_limit is 0, can't map it to a 0..1 closure")
    return {
        "direction": float(block.get("direction", 1)) or 1.0,
        "zero_offset": float(block.get("zero_offset", 0.0)),
        "upper": upper,
    }


def motor_deg_to_closure(calib: dict[str, float], motor_deg: float) -> float:
    """Motor-frame degrees -> closure, unclamped (joint_config.motor_to_cmd_deg + gripper_position)."""
    cmd_deg = calib["zero_offset"] + motor_deg / calib["direction"]
    return cmd_deg / calib["upper"]


class GripperRecordBuffer(RosRecordBuffer):
    """RosRecordBuffer with the gripper as a 7th value in action and state.

    gripper_state: "feedback" records the measured closure and skips frames without a fresh
    reading; "action" copies the commanded closure (for runs with the gripper unpowered).
    """

    def __init__(
        self,
        node: Any,
        cfg: dict[str, Any],
        *,
        gripper_state: str = "feedback",
        calibration: str | None = None,
    ) -> None:
        if gripper_state not in ("feedback", "action"):
            raise ValueError(f"gripper_state must be 'feedback' or 'action', got {gripper_state!r}")
        self._gripper_state = gripper_state
        self._grip_lock = threading.Lock()
        self._grip_feedback: float | None = None
        self._grip_feedback_t = 0.0
        self._started_t = time.monotonic()
        self._last_warn_t = 0.0
        self._last_range_warn_t = 0.0
        self._calib: dict[str, float] | None = None
        if gripper_state == "feedback":
            path = find_calibration(calibration)
            self._calib = load_gripper_calibration(path)
            logger.info("Gripper calibration: %s", path)
        super().__init__(node, cfg)
        if gripper_state == "feedback":
            self._subscribe_feedback(node)

    def _subscribe_feedback(self, node: Any) -> None:
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        try:
            from common_msgs.msg import MotorFeedback
        except ImportError as exc:
            raise ImportError("common_msgs not found; source the workspace (install/setup.bash)") from exc
        # Same QoS as real_arm.py's feedback subscriber.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
        node.create_subscription(MotorFeedback, FEEDBACK_TOPIC, self._on_feedback, qos)
        logger.info("Subscribed gripper state: %s (motor id %d)", FEEDBACK_TOPIC, GRIPPER_CAN_ID)

    def _on_feedback(self, msg) -> None:
        if int(msg.motor_id) != GRIPPER_CAN_ID:
            return
        closure = motor_deg_to_closure(self._calib, float(msg.position))
        with self._grip_lock:
            self._grip_feedback = closure
            self._grip_feedback_t = time.monotonic()

    def _clamp(self, value: float, what: str) -> float:
        if not np.isfinite(value):
            raise ValueError(f"non-finite gripper {what}")
        if not 0.0 <= value <= 1.0:
            now = time.monotonic()
            if now - self._last_range_warn_t > WARN_PERIOD_S:
                self._last_range_warn_t = now
                logger.warning("gripper %s %.3f outside 0..1, clamped (check the calibration)", what, value)
        return float(min(1.0, max(0.0, value)))

    def _on_action(self, msg) -> None:
        # joint_command ignores a non-finite pose; drop it here too rather than raise in a callback.
        try:
            super()._on_action(msg)
        except ValueError as exc:
            logger.warning("ignoring ArmPose: %s", exc)

    def _vector_from_pose(self, msg) -> np.ndarray:
        joints = super()._vector_from_pose(msg)  # 6 arm joints, converted to the dataset unit
        grip = self._clamp(float(msg.gripper_position), "action")
        return np.append(joints, np.float32(grip)).astype(np.float32)

    def _fresh_feedback(self) -> float | None:
        now = time.monotonic()
        with self._grip_lock:
            value, t = self._grip_feedback, self._grip_feedback_t
        if value is not None and now - t <= FEEDBACK_MAX_AGE_S:
            return value
        if now - self._last_warn_t > WARN_PERIOD_S:
            self._last_warn_t = now
            age = "never received" if value is None else f"{now - t:.2f} s old"
            logger.warning(
                "skipping frames: gripper feedback (id %d on %s) %s. Is the gripper powered? "
                "To record with it unpowered, use --gripper_state action.",
                GRIPPER_CAN_ID, FEEDBACK_TOPIC, age,
            )
        return None

    def snapshot(self) -> ObservationSnapshot:
        snap = super().snapshot()
        if snap.state is None or self._gripper_state == "action":
            return snap  # state[6] already mirrors the commanded closure (state_topic is null)
        grip = self._fresh_feedback()
        if grip is None:
            # No state -> build_lerobot_frame raises ValueError -> the record loop skips the frame.
            return ObservationSnapshot(state=None, action=snap.action, images=snap.images)
        state = snap.state.copy()
        state[-1] = self._clamp(grip, "state")
        return ObservationSnapshot(state=state, action=snap.action, images=snap.images)
