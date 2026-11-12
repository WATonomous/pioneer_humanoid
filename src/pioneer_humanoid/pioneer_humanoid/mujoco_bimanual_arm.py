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
FINGER_PAIRS = (("joint7", "joint8"), ("joint7l", "joint8l"))
FINGER_COUPLING_SOLREF = [0.002, 1.0]  # 2 x the scenes' 1 ms step: MuJoCo's default 20 ms lets the fingers drift mm apart
FINGER_SLABS = 16  # collision boxes per finger, 8.5 mm each along the 136 mm finger
# TPU fingers: estimates, not measured. Rubber-like sliding friction, and contact twice as soft
# as the scenes' stiff default (humanoid_mujoco_scenes CONTACT_SOLREF 0.004 s).
# Sliding 1.5: soft TPU grips better than plastic on plastic. MuJoCo takes the larger friction of a pair, and 1.0 was
# its default for every object, so the old value gave the pads no extra grip. Still an estimate: a tilt test on the
# real pads (mu = tan(slip angle)) would replace it.
FINGER_FRICTION = [1.5, 0.005, 0.0001]
# condim 4: sliding + torsional friction, so a held object can't spin about the axis through both fingers (a TPU pad's
# contact patch resists twisting). MuJoCo takes the larger condim of a pair; at 3 (the default) torsion was off for
# every default-condim object.
FINGER_CONDIM = 4
FINGER_SOLREF = [0.008, 1.0]
FINGER_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]

# Reflected motor inertia (kg m^2) on every revolute arm joint; the URDF has none. Without it the
# light distal links form a mode too fast for the 1-2 ms step: the fingers chatter and a held
# object spins out of the jaws. Isaac's implicit PD drives don't need it.
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
            geom.group = 3  # hidden: drawn over the coloured visual meshes, they z-fight and grey them out

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
    # One motor drives both fingers through the linkage: they always move evenly (joint8 = -joint7).
    for a, b in FINGER_PAIRS:
        eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=b, name2=a, solref=FINGER_COUPLING_SOLREF)
        eq.data[:5] = [0, -1, 0, 0, 0]

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
    """Replace each finger's collision mesh with boxes: its bounding box in each of FINGER_SLABS
    slices along the finger, in the finger link's frame.

    MuJoCo collides meshes as convex hulls; the finger hulls are rounded, so a held object
    rests on one curved point and slips out when the arm moves. Boxes aligned with the link
    frame (fingers travel along its Y) give flat, parallel jaw faces, and slicing follows the
    fin-ray wedge (thin at the tip) instead of one box filling it out to the wrist's width.
    TPU contact: FINGER_FRICTION, FINGER_CONDIM and the softer FINGER_SOLREF.
    """
    model = spec.compile()
    for body in FINGER_BODIES:
        geom_id = next(
            i for i in range(model.ngeom)
            if model.geom_bodyid[i] == model.body(body).id and model.geom_contype[i]
        )
        mesh = model.geom_dataid[geom_id]
        verts = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
        verts = verts @ rot.reshape(3, 3).T + model.geom_pos[geom_id]  # mesh frame -> link frame
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        a, b = verts[edges[:, 0]], verts[edges[:, 1]]
        cuts = np.linspace(verts[:, 2].min(), verts[:, 2].max(), FINGER_SLABS + 1)
        spec.delete(next(g for g in spec.body(body).geoms if g.contype))
        for z0, z1 in zip(cuts[:-1], cuts[1:]):
            # the slice's exact extent: its vertices plus where mesh edges cross its two planes
            pts = [verts[(verts[:, 2] >= z0) & (verts[:, 2] <= z1)]]
            for z in (z0, z1):
                cross = (a[:, 2] - z) * (b[:, 2] - z) < 0
                t = (z - a[cross, 2]) / (b[cross, 2] - a[cross, 2])
                pts.append(a[cross] + t[:, None] * (b[cross] - a[cross]))
            pts = np.concatenate(pts)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            spec.body(body).add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(hi - lo) / 2,
                pos=(hi + lo) / 2,
                contype=ARM_CONTYPE,
                conaffinity=ARM_CONAFFINITY,
                group=3,
                density=0,
                friction=FINGER_FRICTION,
                condim=FINGER_CONDIM,
                solref=FINGER_SOLREF,
                solimp=FINGER_SOLIMP,
            )


def set_home(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> None:
    """Put every arm joint and its actuator target at DEFAULT_JOINT_POS (elbows bent)."""
    for name, value in DEFAULT_JOINT_POS.items():
        data.qpos[model.joint(prefix + name).qposadr[0]] = value
        data.ctrl[model.actuator(prefix + name).id] = value
