"""pioneer_bimanual_arm for plain MuJoCo: the URDF plus joint limits and PD position actuators.

Same numbers as the Isaac config (bimanual_arm.py), from arm_params.py and urdf_joint_limits.py.
CPU only; needs ``pip install mujoco``.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from .arm_params import ACTUATOR_GROUPS, CAMERAS, DEFAULT_JOINT_POS, vertical_fov_deg
from .urdf_joint_limits import JOINT_POS_LIMITS, URDF_PATH

_MESH_DIR = Path(URDF_PATH).resolve().parents[1] / "meshes"

# Collision bits: arm geoms touch the world but not each other (Isaac also runs with
# self-collisions off: the finger hulls overlap the jaw gap). World geoms keep the defaults (1/1).
ARM_CONTYPE = 2
ARM_CONAFFINITY = 1

FINGER_BODIES = ("link7", "link8", "link7l", "link8l")

# Reflected motor inertia (kg m^2) on every revolute arm joint; the URDF has none. Without it the
# light distal links (forearm roll ~7e-4 kg m^2 against kp 1550) form a mode too fast for the
# 1-2 ms step: grasp contacts excite it, the fingers chatter, and a held object spins out of the
# jaws. Any geared motor has far more than this. Isaac's implicit PD drives don't need it.
ARM_ARMATURE = 0.01


def arm_spec(cameras: dict[str, tuple[int, int]] | None = None) -> mujoco.MjSpec:
    """MjSpec of the arm: base_link fixed at the origin, one position actuator per joint (named after it).

    ``cameras``: {name: (height, width)} from arm_params.CAMERAS, added as MuJoCo cameras of that name.
    """
    urdf = Path(URDF_PATH).read_text()
    urdf = re.sub(r'filename="package://[^"]*/meshes/', 'filename="', urdf)
    urdf = urdf.replace(
        "</robot>",
        f'<mujoco><compiler meshdir="{_MESH_DIR}" discardvisual="false" fusestatic="false"/></mujoco></robot>',
    )
    spec = mujoco.MjSpec.from_string(urdf)

    for joint in spec.joints:
        if joint.name in JOINT_POS_LIMITS:
            joint.range = JOINT_POS_LIMITS[joint.name]
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.ref = 0.0
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:  # collision geoms; the URDF's visual copies stay 0/0
            geom.contype = ARM_CONTYPE
            geom.conaffinity = ARM_CONAFFINITY

    for group_name, group in ACTUATOR_GROUPS.items():
        for name in group["joints"]:
            act = spec.add_actuator()
            act.name = name
            act.target = name
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.set_to_position(kp=group["stiffness"], kv=group["damping"])
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-group["effort_limit"], group["effort_limit"]]
            if "gripper" not in group_name:
                spec.joint(name).armature = ARM_ARMATURE

    _box_fingers(spec)
    for name, (height, width) in (cameras or {}).items():
        cam = CAMERAS[name]
        spec.body(cam.body).add_camera(
            name=name,
            pos=list(cam.pos),
            quat=list(cam.rot),  # OpenGL convention (-Z forward, +Y up) == MuJoCo's camera frame
            fovy=vertical_fov_deg(name, height, width),
            resolution=[width, height],
        )
    return spec


def _box_fingers(spec: mujoco.MjSpec) -> None:
    """Replace each finger's collision mesh with its bounding box in the finger link's frame.

    MuJoCo collides meshes as convex hulls; the finger hulls are rounded, so a held object
    rests on one curved point and slips out when the arm moves. A box aligned with the link
    frame (fingers travel along its Y) gives flat, parallel jaw faces.
    """
    model = spec.compile()
    for body in FINGER_BODIES:
        geom_id = next(
            i for i in range(model.ngeom)
            if model.geom_bodyid[i] == model.body(body).id and model.geom_contype[i]
        )
        mesh = model.geom_dataid[geom_id]
        verts = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
        verts = verts @ rot.reshape(3, 3).T + model.geom_pos[geom_id]  # mesh frame -> link frame
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        spec.delete(next(g for g in spec.body(body).geoms if g.contype))
        spec.body(body).add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(hi - lo) / 2,
            pos=(hi + lo) / 2,
            contype=ARM_CONTYPE,
            conaffinity=ARM_CONAFFINITY,
            group=3,
            density=0,
        )


def set_home(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> None:
    """Put every arm joint and its actuator target at DEFAULT_JOINT_POS (elbows bent)."""
    for name, value in DEFAULT_JOINT_POS.items():
        data.qpos[model.joint(prefix + name).qposadr[0]] = value
        data.ctrl[model.actuator(prefix + name).id] = value
