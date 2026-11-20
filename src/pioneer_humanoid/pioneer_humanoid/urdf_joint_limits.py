"""Joint position limits for pioneer_bimanual_arm (the old 6-DoF arms), parsed from the URDF.

The URDF is the single source of truth for joint limits; the MuJoCo armv2 model and the real-arm stack read them
from here. Stdlib only -- no isaaclab, no numpy.
"""
import os
import xml.etree.ElementTree as ET

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
URDF_PATH = os.path.join(_REPO_ROOT, "assets", "pioneer_bimanual_arm", "urdf", "pioneer_bimanual_arm.urdf")

# Joint types whose <limit lower/upper> describe a real position range. A
# "continuous" joint has no limits by definition and "fixed" has no DOF.
_LIMITED_JOINT_TYPES = ("revolute", "prismatic")

# Gripper stroke pinned to match GRIPPER_OPEN/CLOSED in arm_params.py.
GRIPPER_LIMIT_OVERRIDES = {
    "joint7": (-0.041, 0.0),
    "joint8": (0.0, 0.041),
    "joint7l": (-0.041, 0.0),
    "joint8l": (0.0, 0.041),
}


def load_joint_limits(
    urdf_path: str | None = None,
    joint_types: tuple[str, ...] = _LIMITED_JOINT_TYPES,
) -> dict[str, tuple[float, float]]:
    """{joint_name: (lower, upper)} from the URDF (rad for revolute, m for prismatic).

    Degenerate limits (lower >= upper) are skipped: the exporter writes lower=upper=0
    for unfilled joints and PhysX reads that as "no limit", so propagating it would
    unlock the joint. Skipping leaves the asset's own limit in place.
    """
    root = ET.parse(urdf_path or URDF_PATH).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") not in joint_types:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if lower >= upper:
            continue
        limits[joint.get("name")] = (lower, upper)
    return limits


# The table every consumer imports: URDF limits, with the gripper stroke pinned.
JOINT_POS_LIMITS = {**load_joint_limits(), **GRIPPER_LIMIT_OVERRIDES}
