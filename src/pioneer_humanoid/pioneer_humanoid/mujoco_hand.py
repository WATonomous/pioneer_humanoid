"""20-DOF pioneer hand for plain MuJoCo: the URDF plus PD position actuators.

Joint limits come from the URDF (they match hand_cfg.JOINT_POS_LIMITS); gains mirror the Isaac
actuator groups in hand_cfg.py. CPU only; needs ``pip install mujoco``. No isaaclab import, so
hand_cfg's names are repeated here.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
HAND_URDF_PATH = _REPO_ROOT / "assets" / "pioneer_hand" / "urdf" / "hand_urdf.urdf"
_MESH_DIR = HAND_URDF_PATH.parents[1] / "meshes"

# Same order as hand_cfg.ALL_JOINT_NAMES (thumb chain first, then fingers 1-4 = index..pinky).
JOINT_NAMES = (
    "circumduction", "MCP_A_thumb", "PIP_thumb", "DIP_thumb",
    "MCP_A_1", "MCP_1", "PIP_1", "DIP_1",
    "MCP_A_2", "MCP_2", "PIP_2", "DIP_2",
    "MCP_A_3", "MCP_3", "PIP_3", "DIP_3",
    "MCP_A_4", "MCP_4", "PIP_4", "DIP_4",
)

# Flexion joints: 0 is the straight finger and the range lies on one side of it, so the sign of
# the far limit is the closing direction (the URDF axes flip sign link to link).
FLEX_JOINTS = tuple(n for n in JOINT_NAMES if n.startswith(("MCP_", "PIP_", "DIP_")) and not n.startswith("MCP_A"))
SPREAD_JOINTS = ("MCP_A_1", "MCP_A_2", "MCP_A_3", "MCP_A_4")

# (stiffness Nm/rad, damping Nms/rad) per joint, from the actuator groups in hand_cfg.HAND_CFG.
_GAINS = {
    "circumduction": (50.0, 2.0),
    "MCP_A_thumb": (30.0, 1.5),
    "PIP_thumb": (10.0, 0.8), "DIP_thumb": (10.0, 0.8),
    **{f"MCP_A_{i}": (50.0, 2.0) for i in range(1, 5)},
    **{f"MCP_{i}": (50.0, 2.0) for i in range(1, 5)},
    **{f"PIP_{i}": (30.0, 1.5) for i in range(1, 5)},
    **{f"DIP_{i}": (10.0, 0.8) for i in range(1, 5)},
}

# Reflected rotor inertia (kg m^2); the URDF has none. The distal links weigh ~2 g, and against
# kp 10-50 they would make modes far faster than the 1 ms step without it.
HAND_ARMATURE = 1e-4

# Collision bits: hand geoms touch the world but not each other (Isaac's in-hand task also runs
# with self-collisions off). World geoms keep MuJoCo's defaults (1/1).
HAND_CONTYPE = 2
HAND_CONAFFINITY = 1

PALM_BODY = "hand_origin"
# Last link of each digit; the fingertip site sits at its far end.
DISTAL_BODIES = ("thumb_distal", "distal_1", "distal_2", "distal_3", "distal_4")


def hand_spec(thumb_pos=None, thumb_yaw_deg: float = 0.0, finger_collisions: bool = False,
              torque_limit: float | None = None) -> mujoco.MjSpec:
    """MjSpec of the hand: hand_origin as the root body, one position actuator per joint (named after it).

    The root has no joint: attach it to a frame (or add a freejoint/mocap weld) to place it.
    Each digit gets a ``tip_<distal body>`` site at its fingertip.

    ``thumb_pos`` / ``thumb_yaw_deg``: a hypothetical thumb mount for design studies -- the thumb base
    (the circumduction joint) moved to ``thumb_pos`` in the palm frame and the whole thumb chain turned
    about the palm normal (Z) by ``thumb_yaw_deg``. Defaults are the hand as built.

    ``finger_collisions``: digits collide with each other (not with their own links or the palm, whose
    convex hull overlaps every finger base). Off by default, like the Isaac in-hand task.

    ``torque_limit``: clamp every actuator to +-this many Nm. None (default) leaves them unlimited, as in
    the Isaac config; then a 0.08 rad squeeze at kp 50 is ~4 Nm per joint, ~100+ N on a small object --
    far more than a hand this size produces. The real motors' torque isn't in the repo (URDF effort=0).
    """
    urdf = HAND_URDF_PATH.read_text()
    urdf = re.sub(r'filename="\.\./meshes/', 'filename="', urdf)
    urdf = urdf.replace(
        "</robot>",
        f'<mujoco><compiler meshdir="{_MESH_DIR}" discardvisual="false" fusestatic="false"/></mujoco></robot>',
    )
    spec = mujoco.MjSpec.from_string(urdf)
    thumb = spec.body("thumb")
    if thumb_pos is not None:
        thumb.pos = list(thumb_pos)
    if thumb_yaw_deg:
        half = np.radians(thumb_yaw_deg) / 2
        thumb.quat = [np.cos(half), 0.0, 0.0, np.sin(half)]

    for joint in spec.joints:
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.armature = HAND_ARMATURE
        joint.ref = 0.0
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:  # collision geoms; the URDF's visual copies stay 0/0
            geom.contype = HAND_CONTYPE
            geom.conaffinity = HAND_CONAFFINITY
            geom.friction = [1.0, 0.01, 0.001]

    for name in JOINT_NAMES:
        kp, kv = _GAINS[name]
        act = spec.add_actuator()
        act.name = name
        act.target = name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=kp, kv=kv)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.inheritrange = 1.0  # ctrlrange = joint range
        if torque_limit is not None:
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-torque_limit, torque_limit]

    if finger_collisions:
        _digit_collision_bits(spec)
    _add_tip_sites(spec)
    return spec


_DIGIT_ROOTS = ("thumb", "abduction_1", "finger_2", "finger_3", "finger_4")  # first body of each digit


def _digit_collision_bits(spec: mujoco.MjSpec) -> None:
    """Give each digit its own contype bit and let it collide with the other digits' bits (and the world)."""
    bits = {root: 1 << (2 + i) for i, root in enumerate(_DIGIT_ROOTS)}
    all_bits = sum(bits.values())

    def walk(body, bit):
        for geom in body.geoms:
            if geom.contype:
                geom.contype = HAND_CONTYPE | bit
                geom.conaffinity = HAND_CONAFFINITY | (all_bits & ~bit)
        for child in body.bodies:
            walk(child, bit)

    for root, bit in bits.items():
        walk(spec.body(root), bit)


def _add_tip_sites(spec: mujoco.MjSpec) -> None:
    """Site at each distal link's farthest collision-mesh vertex from its joint (the fingertip)."""
    model = spec.compile()
    for body in DISTAL_BODIES:
        bid = model.body(body).id
        geom_id = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == bid and model.geom_contype[i])
        mesh = model.geom_dataid[geom_id]
        verts = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
        verts = verts @ rot.reshape(3, 3).T + model.geom_pos[geom_id]  # mesh frame -> body frame
        tip = verts[np.argmax(np.linalg.norm(verts, axis=1))]
        spec.body(body).add_site(name=f"tip_{body}", pos=tip, size=[0.003, 0, 0], rgba=[1, 0.3, 0.3, 1])


def joint_ranges(model: mujoco.MjModel, prefix: str = "") -> np.ndarray:
    """(20, 2) joint limits in JOINT_NAMES order."""
    return np.array([model.jnt_range[model.joint(prefix + n).id] for n in JOINT_NAMES])


def close_direction(ranges: np.ndarray) -> np.ndarray:
    """(20,) +1/-1 per joint: the sign that curls it toward the palm. 0 for spread/thumb-base joints."""
    sign = np.zeros(len(JOINT_NAMES))
    for i, name in enumerate(JOINT_NAMES):
        if name in FLEX_JOINTS:
            lo, hi = ranges[i]
            sign[i] = 1.0 if abs(hi) > abs(lo) else -1.0
    return sign
