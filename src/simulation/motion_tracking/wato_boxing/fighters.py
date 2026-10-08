"""Wato fighter entity: the tracking robot plus boxing gloves and hitboxes.

Collision layout (MuJoCo contype/conaffinity bits):
  bit 0 (1)  floor        - feet (box under each sole, as in the tracking task)
                            and every hitbox, so a fallen fighter lands on it
  bit 1 (2)  red fighter  - gloves + hitboxes, collide with blue and the ropes
  bit 2 (4)  blue fighter - gloves + hitboxes, collide with red and the ropes
  bit 3 (8)  ropes / posts

A fighter's hitboxes never touch its own body; they meet the opponent
(gloves block gloves, gloves land on head/torso, bodies clinch), the ropes
and the floor. Hitboxes are simple shapes fitted to the CAD meshes; the
meshes stay visual only.

Hitbox geoms (per fighter): glove_l, glove_r (spheres on the wrists, shown),
head, torso, pelvis, and capsules along the arms and legs (hidden, group 3).
"""

import dataclasses
from typing import Literal

import mujoco
import numpy as np

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from wato_tracking.robot import ACTUATORS, FEET_COLLISION, get_spec

Team = Literal["red", "blue"]

TEAM_BIT = {"red": 2, "blue": 4}
ROPE_BIT = 8
TEAM_RGBA = {"red": (0.80, 0.30, 0.28, 1.0), "blue": (0.30, 0.40, 0.80, 1.0)}
GLOVE_RGBA = {"red": (0.65, 0.05, 0.05, 1.0), "blue": (0.05, 0.12, 0.60, 1.0)}

GLOVE_RADIUS = 0.06
HEAD_RADIUS = 0.10
ARM_RADIUS = 0.045
LEG_RADIUS = 0.06

WRISTS = {"l": "Mirrorlink6__1__1", "r": "link6__1__1"}
CLAWS = {"l": ("Mirrorlink7__1__1", "Mirrorlink8__1__1"), "r": ("link7__1__1", "link8__1__1")}
HEAD_BODY = "baseLink__1__1"  # shoulder frame; the robot has no head, the target sits on top
TORSO_BODY = "Torso_1"
PELVIS_BODY = "base_link"
# capsule from the body's origin to its child's origin (fixed offsets, exact)
LIMB_CHAINS = {
  "arm": [
    ("Mirrorlink2__1__1", "Mirrorlink3__1__1"),
    ("Mirrorlink3__1__1", "Mirrorlink4__1__1"),
    ("Mirrorlink4__1__1", "Mirrorlink5__1__1"),
    ("Mirrorlink5__1__1", "Mirrorlink6__1__1"),
    ("link2__1__1", "link3__1__1"),
    ("link3__1__1", "link4__1__1"),
    ("link4__1__1", "link5__1__1"),
    ("link5__1__1", "link6__1__1"),
  ],
  "leg": [
    ("left_hip_r_1", "left_thigh_1"),
    ("left_thigh_1", "left_calf_1"),
    ("left_calf_1", "left_foot_joint_1"),
    ("right_hip_r_1", "right_thigh_1"),
    ("right_thigh_1", "right_calf_1"),
    ("right_calf_1", "right_foot_joint_1"),
  ],
}

# Orthodox boxing stance: frame 0 of scripts/generate_moves.py (boxing take at
# 8.2 s, stance scaled to 0.75, hips 6 cm lower, both soles flat, weight 50/50).
# Root pose is relative to the centre between the feet, facing +x (the
# opponent). Regenerate from data/motions/moves.csv row 0 if the stance changes.
STANCE_ROOT_POS = (-0.0524, -0.0069, 0.7325)
STANCE_ROOT_QUAT = (0.5629, -0.0713, 0.082, -0.8194)  # wxyz
STANCE_JOINTS = {
  "left_hip_a_akh70": 0.8392,
  "left_hip_r_rs04": -0.2128,
  "left_thigh_rs03": 0.0229,
  "left_knee_akh70": 1.1640,
  "left_foot_joint_simMotor1": -0.5223,
  "left_foot_joint_simMotor2": 0.0262,
  "right_hip_a_akh70": 0.4150,
  "right_hip_r_rs04": -0.1368,
  "right_thigh_rs03": -0.4047,
  "right_knee_akh70": -0.8805,
  "right_foot_joint_simMotor1": -0.6171,
  "right_foot_joint_simMotor2": -0.0062,
  "left_shoulder_ak10_1_pitch": 0.4223,
  "left_shoulder_ak10_2_roll": 0.4125,
  "left_elbow_ak80_1_yaw": 0.4813,
  "left_elbow_ak80_2_bend": -0.3951,
  "left_elbow_ak80_3_forearm": -1.9173,
  "left_wrist_gl40": 0.3040,
  "left_claw_1": 0.0000,
  "left_claw_2": 0.0000,
  "right_shoulder_ak10_1_pitch": 1.6594,
  "right_shoulder_ak10_2_roll": -0.2783,
  "right_elbow_ak80_1_yaw": -1.1606,
  "right_elbow_ak80_2_bend": -0.5358,
  "right_elbow_ak80_3_forearm": 0.3020,
  "right_wrist_gl40": 0.4943,
  "right_claw_1": 0.0000,
  "right_claw_2": 0.0000,
}


def _quat_mul(a, b) -> np.ndarray:
  out = np.zeros(4)
  mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
  return out


def _rotate(q, v) -> np.ndarray:
  out = np.zeros(3)
  mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
  return out


def _mesh_points_world(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> np.ndarray:
  b = model.body(body).id
  pts = []
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != b or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    mid = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    pts.append(data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)
  return np.vstack(pts)


def _to_body(data: mujoco.MjData, model: mujoco.MjModel, body: str, p_world) -> np.ndarray:
  b = model.body(body).id
  return data.xmat[b].reshape(3, 3).T @ (np.asarray(p_world) - data.xpos[b])


def _body_quat_of_world_axes(data: mujoco.MjData, model: mujoco.MjModel, body: str) -> np.ndarray:
  """Quaternion (in the body frame) of a box aligned with the world axes at qpos0."""
  q = data.xquat[model.body(body).id].copy()
  q[1:] *= -1
  return q


def add_hitboxes(spec: mujoco.MjSpec, team: Team) -> None:
  """Recolour the robot and add gloves + hitboxes for `team`."""
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
      geom.rgba = TEAM_RGBA[team]

  # measure at qpos0 (upright, all joints 0): offsets in body frames are fixed
  model = spec.copy().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)

  def add(body: str, name: str, group: int = 3, rgba=(0.9, 0.9, 0.2, 0.4), **kw):
    spec.body(body).add_geom(
      name=name,
      group=group,
      rgba=rgba,
      mass=0.0,
      density=0.0,
      **kw,
    )

  # gloves: between the claws, on the wrist body
  for side, wrist in WRISTS.items():
    a, b = (data.xpos[model.body(n).id] for n in CLAWS[side])
    add(
      wrist,
      f"glove_{side}",
      group=1,
      rgba=GLOVE_RGBA[team],
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(GLOVE_RADIUS, 0, 0),
      pos=_to_body(data, model, wrist, (a + b) / 2),
    )

  # head: sphere at the top of the shoulder frame
  pts = _mesh_points_world(model, data, HEAD_BODY)
  top = pts[:, 2].max()
  cap = pts[pts[:, 2] > top - 0.15]
  centre = np.array([*cap[:, :2].mean(0), top - HEAD_RADIUS])
  add(HEAD_BODY, "head", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(HEAD_RADIUS, 0, 0), pos=_to_body(data, model, HEAD_BODY, centre))

  # torso / pelvis: world-axis boxes around the meshes
  for body, name in ((TORSO_BODY, "torso"), (PELVIS_BODY, "pelvis")):
    pts = _mesh_points_world(model, data, body)
    lo, hi = pts.min(0), pts.max(0)
    add(
      body,
      name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(hi - lo) / 2,
      pos=_to_body(data, model, body, (lo + hi) / 2),
      quat=_body_quat_of_world_axes(data, model, body),
    )

  # limbs
  for kind, chain in LIMB_CHAINS.items():
    r = ARM_RADIUS if kind == "arm" else LEG_RADIUS
    for parent, child in chain:
      end = np.array(spec.body(child).pos)
      add(parent, f"{kind}_{parent}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(r, 0, 0), fromto=(0, 0, 0, *end))


def stance_pose(position_xy=(0.0, 0.0), yaw: float = 0.0) -> EntityCfg.InitialStateCfg:
  """Stance with its foot-midpoint at position_xy, facing `yaw` (rad, 0 = +x)."""
  qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
  pos = _rotate(qz, STANCE_ROOT_POS) + np.array([*position_xy, 0.0])
  return EntityCfg.InitialStateCfg(
    pos=tuple(float(x) for x in pos),
    rot=tuple(float(x) for x in _quat_mul(qz, STANCE_ROOT_QUAT)),
    joint_pos=dict(STANCE_JOINTS),
    joint_vel={".*": 0.0},
  )


HITBOX_NAMES = (r"^glove_[lr]$", r"^head$", r"^torso$", r"^pelvis$", r"^arm_.*$", r"^leg_.*$")


def hitbox_collision(team: Team) -> CollisionCfg:
  """Hitbox bits. Applied after FEET_COLLISION, which zeroes every geom it does
  not match; this one leaves the feet alone (disable_other_geoms=False)."""
  other = TEAM_BIT["blue" if team == "red" else "red"]
  return CollisionCfg(
    geom_names_expr=HITBOX_NAMES,
    contype=TEAM_BIT[team],
    conaffinity=other | ROPE_BIT | 1,  # 1 = floor
    condim=3,
    priority=0,
    friction=(0.6,),
    disable_other_geoms=False,
  )


def scaled_actuators(gain_scale: float) -> tuple:
  """Motors with stiffness x gain_scale and damping x sqrt(gain_scale) (same
  damping ratio); torque and speed limits unchanged."""
  if gain_scale == 1.0:
    return ACTUATORS
  return tuple(
    dataclasses.replace(a, stiffness=a.stiffness * gain_scale, damping=a.damping * gain_scale**0.5) for a in ACTUATORS
  )


# Sensors. The bar on top of the shoulder frame matches a RealSense D455
# (132 x 30 mm vs 124 x 29 mm); the camera sits at its front face looking
# forward. The ultrasonic sensor sits at the front of the chest. Each has a
# site (used by wato_boxing/perception.py; frame: -z forward, +y up, +x right,
# like a MuJoCo camera) and the camera also a MuJoCo <camera> for rendering.
CAMERA_SITE = "d455"
ULTRASONIC_SITE = "ultrasonic"
D455_FOVY = 60.5  # deg, vertical: 86 deg horizontal on a 16:10 image (perception.CameraCfg)
ULTRASONIC_BODY = TORSO_BODY


def _forward_frame_quat(data: mujoco.MjData, model: mujoco.MjModel, body: str) -> np.ndarray:
  """Body-frame quat of a frame looking along the robot's forward axis (world
  +y at qpos0), up = world +z: camera convention (-z forward, +y up)."""
  rot_world = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # columns: x, y, z of the frame
  rot_body = data.xmat[model.body(body).id].reshape(3, 3).T @ rot_world
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, rot_body.flatten())
  return q


def add_sensors(spec: mujoco.MjSpec) -> None:
  model = spec.copy().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)

  # camera: front face of the bar on top of the shoulder frame
  pts = _mesh_points_world(model, data, HEAD_BODY)
  bar = pts[pts[:, 2] > pts[:, 2].max() - 0.03]
  cam = np.array([bar[:, 0].mean(), bar[:, 1].max(), bar[:, 2].mean()])
  quat = _forward_frame_quat(data, model, HEAD_BODY)
  pos = _to_body(data, model, HEAD_BODY, cam)
  spec.body(HEAD_BODY).add_site(name=CAMERA_SITE, pos=pos, quat=quat, size=(0.01, 0.01, 0.01), group=4)
  spec.body(HEAD_BODY).add_camera(name=CAMERA_SITE, pos=pos, quat=quat, fovy=D455_FOVY)

  # ultrasonic: front of the chest, upper third of the torso
  pts = _mesh_points_world(model, data, ULTRASONIC_BODY)
  lo, hi = pts.min(0), pts.max(0)
  us = np.array([(lo[0] + hi[0]) / 2, hi[1], lo[2] + 0.66 * (hi[2] - lo[2])])
  spec.body(ULTRASONIC_BODY).add_site(
    name=ULTRASONIC_SITE,
    pos=_to_body(data, model, ULTRASONIC_BODY, us),
    quat=_forward_frame_quat(data, model, ULTRASONIC_BODY),
    size=(0.01, 0.01, 0.01),
    group=4,
  )


def get_fighter_cfg(team: Team, position_xy=(0.0, 0.0), yaw: float = 0.0, gain_scale: float = 1.0) -> EntityCfg:
  def spec_fn() -> mujoco.MjSpec:
    spec = get_spec()
    add_hitboxes(spec, team)
    add_sensors(spec)
    return spec

  return EntityCfg(
    init_state=stance_pose(position_xy, yaw),
    collisions=(FEET_COLLISION, hitbox_collision(team)),
    spec_fn=spec_fn,
    articulation=EntityArticulationInfoCfg(actuators=scaled_actuators(gain_scale), soft_joint_pos_limit_factor=0.9),
  )

