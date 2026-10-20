"""The whole-body Pioneer v2 humanoid for plain MuJoCo (assets/pioneer_v2_humanoid).

``robot_spec()`` is the URDF with:

- its CAD colours on the visual meshes (MuJoCo's URDF reader drops them), the collision boxes hidden;
- a movable pelvis: three slides, a yaw and a pitch hinge (at hip height) above ``base_link``, driven by stiff position actuators. The
  robot doesn't balance in this sim; the controller moves the pelvis (step, crouch, turn) and solves the legs so the
  feet stay where they are planted, like a walking controller's output with the balancing taken as solved;
- one position actuator per joint (named after it), gravity compensated (``gravcomp``), as a model-based
  feed-forward on the real robot would;
- collisions: the arms and hands touch the world, not each other; legs and feet are visual (they never touch the
  props, and the pelvis carries the weight).

The robot faces +X once attached with ``FACING_QUAT`` (the URDF's front is -Y).
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "pioneer_v2_humanoid"
URDF_PATH = ASSET_DIR / "pioneer_humanoid_v2.urdf"

ROBOT_CONTYPE = 2       # hands / arms collide with world geoms (contype 1), not with each other
ROBOT_CONAFFINITY = 1
FACING_QUAT = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))   # URDF -Y -> world +X

SIDES = ("l", "r")
PALM_OFFSET = np.array([0.028, -0.012, -0.075])   # palm face centre in hand_H (left hand: x mirrored)
ARM = ("shoulder_pitch", "shoulder_roll", "upper_arm_yaw", "elbow", "forearm_roll", "wrist_dev", "wrist_flex")
FINGERS = ("", "i_", "r_", "l_")            # middle, index, ring, little (the URDF's names)
# the DIP joints follow the PIP (URDF mimic -> joint equality), so they have no actuator of their own
HAND = tuple(f"{f}{j}" for f in FINGERS for j in ("abd", "mcp", "pip")) + ("th_c", "th_f", "th_mcp", "th_ip")
LEG = ("Hip_F", "Hip_A", "Hip_R", "Knee", "Ankle_P", "Ankle_R")
BASE = ("base_x", "base_y", "base_z", "base_yaw", "base_pitch")
HIP_Z = 0.83                 # the pelvis pivots here (hip height): leaning forward bends at the hips


def arm_joints(side: str) -> list[str]:
    return [f"{side}_{j}" for j in ARM]


def hand_joints(side: str) -> list[str]:
    return [f"{side}_{j}" for j in HAND]


def leg_joints(side: str) -> list[str]:
    return [f"{j}_{side.upper()}" for j in LEG]


# position gains (N m / rad or N / m), torque limits; damping from dampratio 1 on the joint's own inertia
GAINS = {
    "base": (2e5, 2e4, 1.0),          # kp, force limit, armature
    "waist": (400.0, 80.0, 0.05),
    "neck": (40.0, 6.0, 0.005),
    "shoulder": (500.0, 60.0, 0.02),
    "elbow": (300.0, 40.0, 0.02),
    "forearm": (120.0, 15.0, 0.01),
    "wrist": (80.0, 10.0, 0.01),
    "finger": (4.0, 1.0, 0.0005),
    "leg": (800.0, 120.0, 0.02),
}


def _group(joint: str) -> str:
    if joint in BASE:
        return "base"
    if joint == "waist_yaw":
        return "waist"
    if joint.startswith("neck"):
        return "neck"
    if joint[:-2] in LEG:
        return "leg"
    j = joint[2:]
    if j in HAND:
        return "finger"
    if j.startswith("shoulder") or j == "upper_arm_yaw":
        return "shoulder"
    if j == "elbow":
        return "elbow"
    if j == "forearm_roll":
        return "forearm"
    return "wrist"


def _link_colors() -> dict[str, list[list[float]]]:
    out = {}
    for link in ET.parse(URDF_PATH).getroot().iter("link"):
        out[link.get("name")] = [[float(c) for c in v.find("material/color").get("rgba").split()]
                                 for v in link.findall("visual")]
    return out


def robot_spec() -> mujoco.MjSpec:
    """MjSpec whose world holds ``pelvis`` (the movable base) -> ``base_link`` -> the robot."""
    urdf = URDF_PATH.read_text()
    urdf = re.sub(r"<mujoco>.*?</mujoco>",
                  f'<mujoco><compiler meshdir="{ASSET_DIR}" balanceinertia="true" discardvisual="false" '
                  f'fusestatic="false"/></mujoco>', urdf)
    spec = mujoco.MjSpec.from_string(urdf)
    colors = _link_colors()

    materials = {}
    for body in spec.bodies:
        if body.name == "world":
            continue
        meshes = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]
        for g, rgba in zip(meshes, colors.get(body.name, [])):
            g.contype = g.conaffinity = 0
            g.group = 2
            key = tuple(round(c, 3) for c in rgba)
            if key not in materials:      # satin, not the default glossy plastic: white shells burn out otherwise
                materials[key] = spec.add_material(name=f"cad{len(materials)}", rgba=rgba, specular=0.25,
                                                   shininess=0.35).name
            g.material = materials[key]
            g.density = 0
        leg = any(body.name.startswith(p) for p in ("l_hip", "r_hip", "l_thigh", "r_thigh", "l_shank", "r_shank",
                                                      "l_ankle", "r_ankle", "l_foot", "r_foot", "base_link"))
        hand = "_hand_" in body.name
        for g in body.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                continue
            g.group = 3
            g.rgba = [0.9, 0.3, 0.3, 0.3]
            g.density = 0
            g.contype, g.conaffinity = (0, 0) if leg else (ROBOT_CONTYPE, ROBOT_CONAFFINITY)
            if hand:            # rubber finger pads
                g.friction = [1.5, 0.02, 0.002]
                g.condim = 4
        body.gravcomp = 1.0

    # palm frames: on the palm's face, x along the fingers, z out of the palm (the palms face inward at rest)
    for side, sx in (("l", -1.0), ("r", 1.0)):
        rot = np.column_stack([[0, 0, -1], np.cross([sx, 0, 0], [0, 0, -1]), [sx, 0, 0]])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        spec.body(f"{side}_hand_H").add_site(name=f"{side}_palm", pos=PALM_OFFSET * np.array([sx, 1, 1]), quat=quat,
                                             size=[0.006, 0, 0], rgba=[1, 0.2, 0.2, 1], group=4)
    # fingertip pads (the far end of each distal link's collision box)
    model = spec.compile()
    for side in SIDES:
        for tip, link in (("thumb_tip", "TD"), ("index_tip", "D_I"), ("middle_tip", "D")):
            b = model.body(f"{side}_hand_{link}").id
            g = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == b and model.geom_contype[i])
            pos = model.geom_pos[g] + np.array([0, 0, -model.geom_size[g][2]])
            spec.body(f"{side}_hand_{link}").add_site(name=f"{side}_{tip}", pos=pos, size=[0.004, 0, 0], group=4)
    spec.body("head").add_site(name="eyes", pos=[0, -0.09, 0.05], size=[0.006, 0, 0], group=4)
    # the head camera, behind the visor, looking where the face looks (URDF -Y), tilted 15 degrees down;
    # MuJoCo cameras look along their -Z with +Y up
    tilt = math.radians(20)
    fwd = np.array([0, -math.cos(tilt), -math.sin(tilt)])
    up = np.array([0, -math.sin(tilt), math.cos(tilt)])
    rot = np.column_stack([np.cross(up, -fwd), up, -fwd])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    spec.body("head").add_camera(name="head_cam", pos=[0, -0.135, 0.05], quat=q, fovy=70)   # just outside the visor

    # movable pelvis: x, y, z slides and a yaw hinge between the world and base_link
    base_link = spec.body("base_link")
    root = mujoco.MjSpec()
    pelvis = root.worldbody.add_body(name="pelvis", pos=[0, 0, HIP_Z])
    pelvis.mass = 5.0
    pelvis.inertia = [0.1, 0.1, 0.1]
    for name, axis, kind in (("base_x", [1, 0, 0], mujoco.mjtJoint.mjJNT_SLIDE),
                             ("base_y", [0, 1, 0], mujoco.mjtJoint.mjJNT_SLIDE),
                             ("base_z", [0, 0, 1], mujoco.mjtJoint.mjJNT_SLIDE),
                             ("base_yaw", [0, 0, 1], mujoco.mjtJoint.mjJNT_HINGE),
                             ("base_pitch", [0, 1, 0], mujoco.mjtJoint.mjJNT_HINGE)):
        pelvis.add_joint(name=name, type=kind, axis=axis)
    pelvis.gravcomp = 1.0
    frame = pelvis.add_frame(pos=[0, 0, -HIP_Z], quat=list(FACING_QUAT))
    frame.attach_body(base_link, "", "")

    for joint in root.joints:
        kp, limit, armature = GAINS[_group(joint.name)]
        joint.armature = armature
        if joint.name.endswith("dip"):
            continue
        act = root.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, dampratio=1.0)
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-limit, limit]
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE and joint.name not in BASE:
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = joint.range
    for body in root.bodies:
        body.gravcomp = 1.0
    return root


if __name__ == "__main__":
    m = robot_spec().compile()
    print(m.nbody, m.njnt, m.nu, m.nmeshface)
