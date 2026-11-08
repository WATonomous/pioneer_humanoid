"""Teleop-only joint limits: ranges narrower than the hardware's that keep the IK out of unnatural poses
(an elbow bent backward, a shoulder swung behind the body). The robot model is unchanged.

The limits live in joint_limits.yaml (jN: [min, max], with the reasons); --joint-limits PATH loads another.
They are given once, in degrees, in the LEFT arm's convention. The right arm is the left mirrored through
the body's mid-plane, which negates every joint angle (FK check, exact for all six joints), so the right
arm gets [-max, -min].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_FILE = Path(__file__).with_name("joint_limits.yaml")
JOINTS = tuple(f"j{i}" for i in range(1, 7))


def load(path) -> dict:
    """{jN: (min, max)} degrees from a YAML of jN: [min, max] (left convention); missing joints are left out."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    limits = {}
    for joint, bounds in raw.items():
        if joint not in JOINTS:
            raise ValueError(f"{path}: unknown joint {joint!r}; expected j1..j6")
        if not (isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(b, (int, float)) for b in bounds)
                and bounds[0] <= bounds[1]):
            raise ValueError(f"{path}: {joint} must be [min, max] in degrees, min <= max")
        limits[joint] = (float(bounds[0]), float(bounds[1]))
    return limits


DEFAULTS = load(DEFAULT_FILE)


def add_args(parser) -> None:
    parser.add_argument("--joint-limits", type=Path, default=DEFAULT_FILE,
                        help="YAML of teleop joint limits, jN: [min, max] degrees, left-arm convention (default: %(default)s)")


def physical_deg() -> np.ndarray:
    """(6, 2) left-arm hardware limits in degrees."""
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS
    from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS

    return np.degrees([JOINT_POS_LIMITS[name] for name in LEFT_ARM_JOINTS])


def side_limits(limits: dict, side: str) -> np.ndarray:
    """(6, 2) radians for one arm from {jN: (min, max)} degrees (left convention); missing joints are hardware."""
    left = physical_deg()
    for i, joint in enumerate(JOINTS):
        left[i] = limits.get(joint, left[i])
    radians = np.radians(left)
    return radians if side == "left" else -radians[:, ::-1]
