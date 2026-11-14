"""The arm's limits (arm_calibration.yaml, arm_actuators.yaml), read the way joint_command reads
them. Used by arm_roundtrip.py, telemetry_record.py and calibrate_arm.py. No ROS imports.

Frames (see joint_command_core.cpp applyCalibration):
  command frame  degrees, what ArmPose carries and what the limits are written in. It IS the
                 URDF frame: zero = hanging (URDF zero), direction = the URDF's positive way
  motor frame    direction * (q_cmd - zero_offset) degrees; MIT drives take it in radians
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# The six ArmPose slots, in message order -- identical to joint_command_core.cpp's jointPaths().
ARM_POSE_JOINTS = [
    ("shoulder", "pitch"), ("shoulder", "roll"), ("shoulder", "yaw"),
    ("elbow", "pitch"), ("elbow", "yaw"), ("wrist", "pitch"),
]
ARM_POSE_NAMES = [f"{group}.{joint}" for group, joint in ARM_POSE_JOINTS]


def arm_pose_topic(arm_side: str) -> str:
    """The ArmPose topic joint_command listens on for this arm (joint_command_node.cpp)."""
    return f"/arm/{arm_side}/joint_targets"


# The repo copy, found relative to this file (src/interfacing/can/scripts -> src/interfacing).
_REPO_CONFIG = Path(__file__).resolve().parents[2] / "joint_command" / "config"

DEFAULT_CALIBRATIONS = [
    "/calibration/arm_calibration.yaml",  # bind-mounted in both containers
    "/root/ament_ws/src/interfacing/joint_command/config/arm_calibration.yaml",
    "/root/ament_ws/src/joint_command/config/arm_calibration.yaml",
    str(_REPO_CONFIG / "arm_calibration.yaml"),
]

ACTUATOR_CONFIGS = [
    "/opt/joint_command_config/arm_actuators.yaml",
    "/root/ament_ws/src/joint_command/config/arm_actuators.yaml",
    "/root/ament_ws/src/interfacing/joint_command/config/arm_actuators.yaml",
    str(_REPO_CONFIG / "arm_actuators.yaml"),
]

# can/config/mit_profiles.yaml: which protocol family each drive speaks (gl2 | ak).
MIT_PROFILES = [
    str(Path(__file__).resolve().parent.parent / "config" / "mit_profiles.yaml"),
    "/opt/watonomous/can/share/can/config/mit_profiles.yaml",
]

# Keys a joint block may override; anything missing falls back to safety.global, exactly as
# JointCommandCore::loadJointSafetyConfig does.
_SAFETY_KEYS = ("active", "velocity_max", "delta_max", "control_type", "mit_kp", "mit_kd",
                "mit_max_torque", "mit_max_track_err", "mit_feedback_timeout",
                "mit_max_temperature", "mit_family",
                "mit_fault_kd", "enable_position_clamp", "enable_velocity_limit",
                "gravity_ff_scale", "gravity_ff_max_torque", "gravity_assume_deg")


def _first_existing(candidates: List[str]) -> Optional[Path]:
    for path in candidates:
        if path and Path(path).exists():
            return Path(path)
    return None


def find_calibration(explicit: Optional[str]) -> Path:
    found = _first_existing([explicit] if explicit else DEFAULT_CALIBRATIONS)
    if found is None:
        raise SystemExit("could not find arm_calibration.yaml; pass --calibration PATH "
                         f"(looked in: {', '.join(DEFAULT_CALIBRATIONS)})")
    return found


def find_actuators(explicit: Optional[str]) -> Path:
    candidates = [explicit] if explicit else ACTUATOR_CONFIGS
    found = _first_existing(candidates)
    if found is None:
        raise SystemExit("could not find arm_actuators.yaml; pass --actuators PATH "
                         f"(looked in: {', '.join(candidates)})")
    return found


def drive_family(motor_id: int) -> Optional[str]:
    """mit_profiles.yaml family for this id (gl2 / ak), or None if the id is not listed there."""
    path = _first_existing(MIT_PROFILES)
    if path is None:
        raise SystemExit(f"could not find mit_profiles.yaml (looked in: {', '.join(MIT_PROFILES)})")
    motors = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("motors", {}) or {}
    return (motors.get(str(motor_id)) or {}).get("family")


def installed_joint_command_config() -> Path:
    """joint_command_node's INSTALLED config dir -- what it enforces until rebuilt (needs ROS)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("joint_command")) / "config"
    except (ImportError, LookupError) as exc:  # no ROS, or joint_command not installed here
        raise SystemExit(f"cannot find joint_command's installed config ({exc!r}), so cannot "
                         "check it matches the repo. Source the workspace joint_command is "
                         "built in, or pass both --calibration and --actuators explicitly.")


def same_yaml(a: Path, b: Path) -> bool:
    """Equal content, ignoring comments and formatting."""
    return (yaml.safe_load(a.read_text(encoding="utf-8")) ==
            yaml.safe_load(b.read_text(encoding="utf-8")))


def load_joint_map(path: Path, arm_side: str) -> Dict[int, dict]:
    """-> {motor_id: {"name", "side", "direction", "zero_offset", "lower", "upper", "slot"}}"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if arm_side not in data:
        raise SystemExit(f"arm side '{arm_side}' not in {path} (have {list(data)})")

    out: Dict[int, dict] = {}

    def walk(node, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        if "can_id" in node:
            # joint_command requires every one of these keys (joint_command_core.cpp).
            missing = [key for key in ("direction", "zero_offset", "lower_limit", "upper_limit",
                                       "limit_range") if key not in node]
            if missing:
                raise SystemExit(f"{arm_side}.{prefix} in {path} is missing {missing}")
            if node["direction"] not in (1, -1):
                raise SystemExit(f"{arm_side}.{prefix} in {path}: direction must be 1 or -1, "
                                 f"got {node['direction']!r}")
            out[int(node["can_id"])] = {
                "name": prefix,
                "side": arm_side,
                "direction": float(node["direction"]),
                "zero_offset": float(node["zero_offset"]),
                "lower": float(node["lower_limit"]),
                "upper": float(node["upper_limit"]),
                "limit_range": bool(node["limit_range"]),
                "slot": None,
            }
            return
        for key, child in node.items():
            walk(child, f"{prefix}.{key}" if prefix else key)

    walk(data[arm_side], "")
    # Tag the six joints an ArmPose carries, so their setpoints can be matched up.
    for slot, (group, joint) in enumerate(ARM_POSE_JOINTS):
        node = data[arm_side].get(group, {}).get(joint)
        if node and int(node["can_id"]) in out:
            out[int(node["can_id"])]["slot"] = slot
    return out


def load_joint_safety(path: Optional[Path],
                      arm_side: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """-> (global block, {"group.joint": arm_side's joint block merged over global})."""
    if path is None:
        return {}, {}
    cfg = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("safety", {}) or {}
    default = cfg.get("global", {}) or {}
    joints: Dict[str, Dict[str, Any]] = {}
    for group, members in (cfg.get(arm_side) or {}).items():
        for joint, node in (members or {}).items():
            node = node or {}
            joints[f"{group}.{joint}"] = {key: node.get(key, default.get(key))
                                          for key in _SAFETY_KEYS}
    return default, joints


def joint_safety(path: Optional[Path], arm_side: str, name: str) -> Dict[str, Any]:
    """One joint's effective block; a joint with no entry (e.g. the gripper) gets global."""
    default, joints = load_joint_safety(path, arm_side)
    if name in joints:
        return joints[name]
    return {key: default.get(key) for key in _SAFETY_KEYS}


def load_actuator_limits(explicit: Optional[str], arm_side: str):
    """({joint: velocity_max_dps}, {joint: mit thresholds}) for the plots."""
    _, joints = load_joint_safety(find_actuators(explicit), arm_side)
    vmax, mit = {}, {}
    for name, block in joints.items():
        vmax[name] = float(block.get("velocity_max") or 0)
        if int(block.get("control_type") if block.get("control_type") is not None else -1) == 0:
            mit[name] = {
                "mit_kp": block.get("mit_kp"),
                "mit_kd": block.get("mit_kd"),
                "max_torque_nm": block.get("mit_max_torque"),
                "max_track_err_deg": block.get("mit_max_track_err"),
                "max_temperature_c": block.get("mit_max_temperature"),
            }
    return vmax, mit


def max_torque_by_joint(explicit: Optional[str], arm_side: str) -> Dict[str, float]:
    """Every joint's mit_max_torque, MIT or not -- the cap a run is judged against."""
    _, joints = load_joint_safety(find_actuators(explicit), arm_side)
    return {name: float(block["mit_max_torque"]) for name, block in joints.items()
            if block.get("mit_max_torque") is not None}


def motor_to_cmd_deg(info: dict, motor_deg: float) -> float:
    """Inverse of joint_command's calibration: cmd = zero_offset + motor / direction."""
    return info["zero_offset"] + motor_deg / (info["direction"] or 1.0)


def gripper_position(info: dict, cmd_deg: float) -> float:
    """Gripper command angle -> position, 0 = open (its calibrated zero) .. 1 = closed
    (upper_limit), as joint_command maps it. Not clamped, so a bad calibration shows."""
    return cmd_deg / info["upper"]
