"""pioneer_bimanual_arm_8dof for plain MuJoCo: the bimanual rig (pioneer_bimanual_arm's base_link) with the new 7-DoF arms +
1-DoF grippers; the URDF plus its joint limits, PD position actuators and gripper coupling.

Drop-in for mujoco_bimanual_arm (arm_spec / set_home) for the 7-DoF arm. CPU only; needs ``pip install mujoco``.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from .arm_params import vertical_fov_deg
from .mujoco_bimanual_arm import FINGER_CONDIM, FINGER_FRICTION, FINGER_SOLIMP
from .arm_8dof_params import (ACTUATOR_GROUPS, ARM_ARMATURE, CAMERAS, DEFAULT_JOINT_POS, GRIPPER_FRICTIONLOSS, MESH_DIR,
                              URDF_PATH, urdf_limits)
from .urdf_joint_limits import URDF_PATH as ARMV2_URDF_PATH

ARMV2_MESH_DIR = str(Path(ARMV2_URDF_PATH).resolve().parents[1] / "meshes")

ARM_CONTYPE = 2          # as mujoco_bimanual_arm: arm geoms touch the world, not each other
ARM_CONAFFINITY = 1
FINGER_PAD_SLABS = 16
FINGER_PAD_MARGIN = 0.001  # catch a closing handle within one 1 ms step; visually sub-millimetre after compliance
# The shared older-arm value is intentionally soft (8 ms).  These jaws have a
# high-force 41 mm closing stroke around 24 mm task handles; use the scene's
# stiffer 4 ms contact so the actuator cannot visibly bury a pad in the handle.
FINGER_CONTACT_SOLREF = [0.004, 1.0]


def arm_spec(cameras: dict[str, tuple[int, int]] | None = None) -> mujoco.MjSpec:
    """MjSpec of both arms: base_link at the origin, one position actuator per actuated joint (named after it).

    ``cameras``: {name: (height, width)} from arm_8dof_params.CAMERAS, added as MuJoCo cameras of that name.
    """
    urdf = Path(URDF_PATH).read_text()
    # the rig's base_link meshes live in pioneer_bimanual_arm, the arms' in pioneer_bimanual_arm_8dof: absolute paths for both
    urdf = urdf.replace('filename="package://pioneer_bimanual_arm_8dof/meshes/', f'filename="{MESH_DIR}/')
    urdf = urdf.replace('filename="package://pioneer_bimanual_arm/meshes/', f'filename="{ARMV2_MESH_DIR}/')
    urdf = urdf.replace(
        "</robot>", '<mujoco><compiler discardvisual="false" fusestatic="false"/></mujoco></robot>')
    spec = mujoco.MjSpec.from_string(urdf)
    # The URDF importer turns <mimic> into a rigid joint equality.  Delete it:
    # the differential tendon below is the physical coupling, and retaining
    # this hidden equality prevents the unblocked jaw from moving at all.
    mimic_joints = {"left_gripper_mimic", "right_gripper_mimic"}
    for equality in list(spec.equalities):
        if equality.type == mujoco.mjtEq.mjEQ_JOINT and equality.name1 in mimic_joints:
            spec.delete(equality)
    limits = urdf_limits()
    for joint in spec.joints:
        if joint.name in limits:
            joint.range = limits[joint.name]
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.ref = 0.0
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = ARM_CONTYPE, ARM_CONAFFINITY
            geom.group = 3
    for group_name, group in ACTUATOR_GROUPS.items():
        for name in group["joints"]:
            act = spec.add_actuator()
            act.name = name
            if "gripper" in group_name:
                # A single motor drives a differential linkage: command the
                # average finger travel, but let either slider stop on an
                # off-centre object while the opposite slider keeps closing.
                # A rigid mimic equality made both jaws stop as soon as the
                # first one touched, leaving the other visibly open.
                side = name.removesuffix("_gripper")
                tendon = spec.add_tendon(name=f"{side}_gripper_differential")
                tendon.wrap_joint(f"{side}_gripper", 0.5)
                tendon.wrap_joint(f"{side}_gripper_mimic", 0.5)
                act.target = tendon.name
                act.trntype = mujoco.mjtTrn.mjTRN_TENDON
            else:
                act.target = name
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.set_to_position(kp=group["stiffness"], kv=group["damping"])
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            effort = 2 * group["effort_limit"] if "gripper" in group_name else group["effort_limit"]
            act.forcerange = [-effort, effort]
            if "gripper" not in group_name:
                spec.joint(name).armature = ARM_ARMATURE
            else:   # split the linkage's back-drive resistance across its two rails
                for joint_name in (name, f"{side}_gripper_mimic"):
                    joint = spec.joint(joint_name)
                    joint.frictionloss = GRIPPER_FRICTIONLOSS / 2
                    joint.solref_friction = [0.002, 1.0]
                    joint.solimp_friction = [0.999, 0.9999, 0.001, 0.5, 2.0]
    _box_fingers(spec)
    for name, (height, width) in (cameras or {}).items():
        cam = CAMERAS[name]
        spec.body(cam.body).add_camera(name=name, pos=list(cam.pos), quat=list(cam.rot),
                                       fovy=vertical_fov_deg(name, height, width), resolution=[width, height])
    return spec


def _box_fingers(spec: mujoco.MjSpec) -> None:
    """Make contact follow the visible jaws instead of their combined convex hull.

    The generated URDF historically used one convex hull for the orange fin-ray
    jaw and its grey carriage.  MuJoCo convexifies mesh collisions, so that hull
    bridges the concavity between those parts and reaches roughly 9 mm inside
    the visible gripping surface.  It can therefore move an object while the
    rendered claw is still visibly clear of it.

    Build thin boxes from slices of the *visible orange pad* and a separate box
    for the carriage.  The slice AABBs preserve the pad's actual inner face and
    follow its taper; separating the carriage prevents either part from filling
    the open jaw gap.
    """
    model = spec.compile()
    for side in ("left", "right"):
        for finger in ("finger_a", "finger_b"):
            body_name = f"{side}_{finger}"
            body_id = model.body(body_name).id

            visual_geoms = {}
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] != body_id or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_id = model.geom_dataid[geom_id]
                mesh_name = model.mesh(mesh_id).name
                if model.geom_contype[geom_id] == 0 and mesh_name in (f"{body_name}_c0", f"{body_name}_c1"):
                    visual_geoms[mesh_name.rsplit("_", 1)[1]] = geom_id
            if set(visual_geoms) != {"c0", "c1"}:
                raise RuntimeError(f"could not locate both visible contact meshes for {body_name}")

            # Remove every imported finger collision.  With the corrected URDF
            # there are two, while older generated assets have one combined hull.
            for geom in list(spec.body(body_name).geoms):
                if geom.contype or geom.conaffinity:
                    spec.delete(geom)

            def vertices(geom_id: int) -> tuple[np.ndarray, np.ndarray]:
                mesh_id = model.geom_dataid[geom_id]
                start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                verts = model.mesh_vert[start:start + count].copy()
                fstart, fcount = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
                faces = model.mesh_face[fstart:fstart + fcount]
                rot = np.empty(9)
                mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
                return verts @ rot.reshape(3, 3).T + model.geom_pos[geom_id], faces

            def add_box(name: str, lo: np.ndarray, hi: np.ndarray, margin: float = 0.0) -> None:
                spec.body(body_name).add_geom(
                    name=name,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(hi - lo) / 2,
                    pos=(hi + lo) / 2,
                    contype=ARM_CONTYPE,
                    conaffinity=ARM_CONAFFINITY,
                    group=3,
                    density=0,
                    friction=FINGER_FRICTION,
                    condim=FINGER_CONDIM,
                    solref=FINGER_CONTACT_SOLREF,
                    solimp=FINGER_SOLIMP,
                    margin=margin,
                )

            # c0 is the long orange contact jaw.  Use exact mesh-edge
            # intersections at each slice boundary so no box extends inward
            # farther than the visible pad in that section.
            verts, faces = vertices(visual_geoms["c0"])
            edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
            edge_a, edge_b = verts[edges[:, 0]], verts[edges[:, 1]]
            cuts = np.linspace(verts[:, 2].min(), verts[:, 2].max(), FINGER_PAD_SLABS + 1)
            for i, (z0, z1) in enumerate(zip(cuts[:-1], cuts[1:])):
                points = [verts[(verts[:, 2] >= z0) & (verts[:, 2] <= z1)]]
                for z in (z0, z1):
                    crosses = (edge_a[:, 2] - z) * (edge_b[:, 2] - z) < 0
                    t = (z - edge_a[crosses, 2]) / (edge_b[crosses, 2] - edge_a[crosses, 2])
                    points.append(edge_a[crosses] + t[:, None] * (edge_b[crosses] - edge_a[crosses]))
                points = np.concatenate(points)
                add_box(
                    f"{body_name}_contact_pad_{i:02d}",
                    points.min(axis=0),
                    points.max(axis=0),
                    FINGER_PAD_MARGIN,
                )

            # c1 is the small grey carriage at the palm.  It must collide too,
            # but as its own box so it cannot bridge into the orange jaw.
            verts, _ = vertices(visual_geoms["c1"])
            add_box(f"{body_name}_contact_carriage", verts.min(axis=0), verts.max(axis=0))


def set_home(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> None:
    """Every joint and its actuator target at DEFAULT_JOINT_POS (arms hanging, elbows 90 deg, grippers open)."""
    for name, value in DEFAULT_JOINT_POS.items():
        data.qpos[model.joint(prefix + name).qposadr[0]] = value
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
        if aid >= 0:
            data.ctrl[aid] = value
