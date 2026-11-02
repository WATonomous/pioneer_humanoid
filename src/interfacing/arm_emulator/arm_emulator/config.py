"""Reads the arm's real YAMLs, read-only, and joins them with config/motor_models.yaml."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SRC = Path(__file__).resolve().parents[3]
# joint_config (the arm's calibration reader), pioneer_humanoid and the MuJoCo scenes aren't installed here.
for _path in (_SRC / "interfacing" / "can" / "scripts", _SRC / "pioneer_humanoid", _SRC / "simulation" / "mujoco_scenes"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from joint_config import load_joint_map  # noqa: E402

REPO_CONFIG = _SRC / "interfacing" / "joint_command" / "config"
REPO_CALIBRATION = REPO_CONFIG / "arm_calibration.yaml"
REPO_ACTUATORS = REPO_CONFIG / "arm_actuators.yaml"
REPO_PROFILES = _SRC / "interfacing" / "can" / "config" / "mit_profiles.yaml"
MOTOR_MODELS = Path(__file__).resolve().parents[1] / "config" / "motor_models.yaml"
GRIPPER_ID = 21


@dataclass(frozen=True)
class MotorSpec:
    """One motor on the arm: where it sits, how it packs values, and what it can physically do."""

    motor_id: int
    name: str
    slot: int | None           # ArmPose slot = LEFT_ARM_JOINTS index; None for the gripper
    direction: float
    zero_offset_deg: float
    upper_deg: float
    family: str                # ak | gl2
    profile: dict[str, Any]    # mit_profiles.yaml entry
    model: dict[str, Any]      # motor_models.yaml entry


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_motors(calibration: Path = REPO_CALIBRATION, profiles: Path = REPO_PROFILES,
                models: Path = MOTOR_MODELS, side: str = "left") -> tuple[list[MotorSpec], dict[str, Any]]:
    """-> (one MotorSpec per motor, slot order then gripper; the whole of motor_models.yaml)."""
    joint_map = load_joint_map(calibration, side)
    mit = _load(profiles)["motors"]
    catalogue = _load(models)
    motors = []
    for motor_id, info in joint_map.items():
        profile = mit.get(str(motor_id))
        if profile is None:
            raise SystemExit(f"{profiles}: no MIT profile for motor {motor_id} ({info['name']})")
        model = catalogue["models"].get(profile["model"])
        if model is None:
            raise SystemExit(f"{models}: no model {profile['model']!r} for motor {motor_id}")
        motors.append(MotorSpec(motor_id, info["name"], info["slot"], info["direction"], info["zero_offset"],
                                info["upper"], profile["family"], profile, model))
    motors.sort(key=lambda m: 99 if m.slot is None else m.slot)
    if [m.slot for m in motors] != [0, 1, 2, 3, 4, 5, None] or motors[-1].motor_id != GRIPPER_ID:
        raise SystemExit(f"{calibration}: {side} arm needs six ArmPose joints and gripper {GRIPPER_ID}")
    return motors, catalogue


def assumptions(path: Path = MOTOR_MODELS) -> list[str]:
    """The ASSUMED lines of motor_models.yaml, which a run's results depend on."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if "ASSUMED" in line and not line.lstrip().startswith("#")]
