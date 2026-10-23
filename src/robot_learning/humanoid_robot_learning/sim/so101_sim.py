"""SO101 leader ↔ Isaac Sim follower mapping (NVIDIA workshop convention).

Stores joint values in the same normalized space as the physical SO101 Leader
(-100..100 for arm joints, 0..100 gripper) so sim datasets transfer to real
follower arms trained with LeRobot SO101 pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

SO101_LEADER_KEYS: tuple[str, ...] = tuple(f"{name}.pos" for name in SO101_JOINT_NAMES)

# Sim USD joint limits in degrees (lerobot so101_follower_good.usd).
SO101_USD_DEG_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}


def _joint_mins_maxs() -> tuple[np.ndarray, np.ndarray]:
    mins = np.array([SO101_USD_DEG_LIMITS[n][0] for n in SO101_JOINT_NAMES], dtype=np.float32)
    maxs = np.array([SO101_USD_DEG_LIMITS[n][1] for n in SO101_JOINT_NAMES], dtype=np.float32)
    return mins, maxs


def leader_action_to_array(action: dict[str, Any] | np.ndarray) -> np.ndarray:
    """Convert LeRobot leader ``get_action()`` output to a flat (6,) vector."""
    if isinstance(action, np.ndarray):
        return np.asarray(action, dtype=np.float32).reshape(-1)[:6]
    return np.array([float(action[key]) for key in SO101_LEADER_KEYS], dtype=np.float32)


def leader_raw_to_sim_rad(raw: np.ndarray) -> np.ndarray:
    """Map leader raw values to sim joint targets in radians."""
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)[:6]
    mins, maxs = _joint_mins_maxs()

    normalized = np.zeros_like(raw)
    normalized[:-1] = (raw[:-1] + 100.0) / 200.0
    normalized[-1] = raw[-1] / 100.0

    mapped_deg = mins + normalized * (maxs - mins)
    return mapped_deg * (np.pi / 180.0)


def sim_rad_to_leader_raw(joint_rad: np.ndarray) -> np.ndarray:
    """Map measured sim joint positions (rad) to leader raw values for logging."""
    joint_rad = np.asarray(joint_rad, dtype=np.float32).reshape(-1)[:6]
    mins, maxs = _joint_mins_maxs()

    mapped_deg = joint_rad * (180.0 / np.pi)
    normalized = (mapped_deg - mins) / (maxs - mins)

    raw = np.zeros_like(normalized)
    raw[:-1] = normalized[:-1] * 200.0 - 100.0
    raw[-1] = normalized[-1] * 100.0
    return raw.astype(np.float32)
