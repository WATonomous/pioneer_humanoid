"""pioneer_bimanual_arm_8dof for plain MuJoCo: the bimanual rig (pioneer_bimanual_arm's base_link) with the new 7-DoF arms +
1-DoF grippers; the URDF plus its joint limits, PD position actuators and gripper coupling.

Drop-in for mujoco_bimanual_arm (arm_spec / set_home) for the 7-DoF arm. CPU only; needs ``pip install mujoco``.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco

from .arm_params import vertical_fov_deg
from .mujoco_bimanual_arm import FINGER_CONDIM, FINGER_COUPLING_SOLREF, FINGER_FRICTION, FINGER_SOLIMP, FINGER_SOLREF
from .arm_8dof_params import (ACTUATOR_GROUPS, CAMERAS, DEFAULT_JOINT_POS, GRIPPER_FRICTIONLOSS, MESH_DIR, URDF_PATH,
                              urdf_limits)
from .urdf_joint_limits import URDF_PATH as ARMV2_URDF_PATH

ARMV2_MESH_DIR = str(Path(ARMV2_URDF_PATH).resolve().parents[1] / "meshes")

ARM_CONTYPE = 2          # as mujoco_bimanual_arm: arm geoms touch the world, not each other
ARM_CONAFFINITY = 1
ARM_ARMATURE = 0.01      # reflected motor inertia, as mujoco_bimanual_arm


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
    for side in ("left", "right"):   # TPU finger contact, as mujoco_bimanual_arm (else held objects slip out)
        for finger in ("finger_a", "finger_b"):
            for geom in spec.body(f"{side}_{finger}").geoms:
                if geom.contype or geom.conaffinity:
                    geom.friction, geom.solref, geom.solimp = FINGER_FRICTION, FINGER_SOLREF, FINGER_SOLIMP
                    geom.condim = FINGER_CONDIM
    for group_name, group in ACTUATOR_GROUPS.items():
        for name in group["joints"]:
            act = spec.add_actuator()
            act.name, act.target = name, name
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.set_to_position(kp=group["stiffness"], kv=group["damping"])
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-group["effort_limit"], group["effort_limit"]]
            if "gripper" not in group_name:
                spec.joint(name).armature = ARM_ARMATURE
            else:   # near-rigid friction: MuJoCo's default (20 ms, 0.9-0.95) lets a pushed finger creep ~5 cm/s
                joint = spec.joint(name)
                joint.frictionloss = GRIPPER_FRICTIONLOSS
                joint.solref_friction, joint.solimp_friction = [0.002, 1.0], [0.999, 0.9999, 0.001, 0.5, 2.0]
    for side in ("left", "right"):   # the URDF <mimic>: finger B follows finger A (one motor drives both)
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1, eq.name2 = f"{side}_gripper_mimic", f"{side}_gripper"
        eq.data[:5] = [0, 1, 0, 0, 0]
        eq.solref = FINGER_COUPLING_SOLREF
    for name, (height, width) in (cameras or {}).items():
        cam = CAMERAS[name]
        spec.body(cam.body).add_camera(name=name, pos=list(cam.pos), quat=list(cam.rot),
                                       fovy=vertical_fov_deg(name, height, width), resolution=[width, height])
    return spec


def set_home(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> None:
    """Every joint and its actuator target at DEFAULT_JOINT_POS (arms hanging, elbows 90 deg, grippers open)."""
    for name, value in DEFAULT_JOINT_POS.items():
        data.qpos[model.joint(prefix + name).qposadr[0]] = value
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
        if aid >= 0:
            data.ctrl[aid] = value
