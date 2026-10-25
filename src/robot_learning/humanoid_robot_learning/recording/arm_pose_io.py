"""Convert common_msgs/ArmPose to flat joint vectors for LeRobot datasets."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

# Same order as joint_command_core.cpp armPoseToMotorCmds.
JOINT_ORDER_DOC = (
    "shoulder pitch, roll, yaw; elbow pitch, roll; wrist pitch"
)
# Schema joint names ArmPose can supply, in arm_pose_to_vector order.
ARM_POSE_JOINT_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_roll",
    "left_wrist_pitch",
)


def deg_to_rad(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32) * (math.pi / 180.0)


def rad_to_deg(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float64) * (180.0 / math.pi)


def arm_pose_to_vector_deg(msg: Any) -> np.ndarray:
    """Extract six arm joint angles in degrees from ArmPose."""
    if len(msg.shoulder.position) < 3:
        raise ValueError("ArmPose.shoulder.position needs 3 elements")
    if len(msg.elbow.position) < 2:
        raise ValueError("ArmPose.elbow.position needs 2 elements")
    if len(msg.wrist.position) < 1:
        raise ValueError("ArmPose.wrist.position needs 1 element")

    return np.array(
        [
            float(msg.shoulder.position[0]),
            float(msg.shoulder.position[1]),
            float(msg.shoulder.position[2]),
            float(msg.elbow.position[0]),
            float(msg.elbow.position[1]),
            float(msg.wrist.position[0]),
        ],
        dtype=np.float64,
    )


def arm_pose_to_vector(
    msg: Any,
    *,
    input_unit: str = "deg",
    output_unit: str = "rad",
) -> np.ndarray:
    """Flatten ArmPose to (6,) float32 in the requested output unit."""
    vec = arm_pose_to_vector_deg(msg)
    if input_unit == output_unit:
        return vec.astype(np.float32)
    if input_unit == "deg" and output_unit == "rad":
        return deg_to_rad(vec)
    if input_unit == "rad" and output_unit == "deg":
        return rad_to_deg(vec).astype(np.float32)
    raise ValueError(f"Unsupported unit conversion: {input_unit} -> {output_unit}")
