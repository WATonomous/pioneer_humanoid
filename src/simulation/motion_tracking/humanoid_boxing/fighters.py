"""Pioneer fighter entity: the tracking robot plus boxing gloves and hitboxes.

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
from humanoid_tracking.robot import ACTUATORS, FEET_COLLISION, get_spec

Team = Literal["red", "blue"]

TEAM_BIT = {"red": 2, "blue": 4}
ROPE_BIT = 8
TEAM_RGBA = {"red": (0.80, 0.30, 0.28, 1.0), "blue": (0.30, 0.40, 0.80, 1.0)}
GLOVE_RGBA = {"red": (0.65, 0.05, 0.05, 1.0), "blue": (0.05, 0.12, 0.60, 1.0)}

GLOVE_RADIUS = 0.07  # round the fist (12.6 x 13.4 x 16.9 cm)
HEAD_RADIUS = 0.11
ARM_RADIUS = 0.045
LEG_RADIUS = 0.06

WRISTS = {"l": "l_hand_H", "r": "r_hand_H"}  # the hands, closed into fists (humanoid_tracking/robot.py)
HEAD_BODY = "head"
TORSO_BODY = "torso"
PELVIS_BODY = "base_link"
# capsule from the body's origin to its child's origin (fixed offsets, exact)
LIMB_CHAINS = {
  "arm": [
    ("l_arm_link2", "l_arm_link3"),  # shoulder
    ("l_arm_link3", "l_arm_link4"),  # upper arm, to the elbow
    ("l_arm_link4", "l_forearm"),  # forearm
    ("l_forearm", "l_hand_W"),  # to the wrist
    ("r_arm_link2", "r_arm_link3"),
    ("r_arm_link3", "r_arm_link4"),
    ("r_arm_link4", "r_forearm"),
    ("r_forearm", "r_hand_W"),
  ],
  "leg": [
    ("l_hip_roll_link", "l_thigh"),
    ("l_thigh", "l_shank_foot"),  # thigh, to the knee
    ("l_shank_foot", "l_ankle_cross"),  # shank, to the ankle
    ("r_hip_roll_link", "r_thigh"),
    ("r_thigh", "r_shank_foot"),
    ("r_shank_foot", "r_ankle_cross"),
  ],
}

# Orthodox boxing stance: frame 0 of scripts/generate_moves.py (Wato's stance made on this robot:
# full size with the DESIGN joint ranges (robot.py), half its forward lean, hips 12 cm lower, both
# soles flat, weight 50/50; see that script). Root pose is relative to the centre between the feet,
# facing +x (the opponent). Regenerate from data/motions/moves.csv row 0 if the stance changes.
STANCE_ROOT_POS = (-0.0960, 0.0166, 0.7279)
STANCE_ROOT_QUAT = (0.5659, -0.0357, 0.0411, -0.8226)  # wxyz
STANCE_JOINTS = {
  "waist_yaw": 0.0000,
  "neck_yaw": 0.0000,
  "neck_pitch": 0.0000,
  "r_shoulder_pitch": -1.2274,
  "r_shoulder_roll": 0.1689,
  "r_upper_arm_yaw": 0.5897,
  "r_elbow": -1.5708,
  "r_forearm_roll": -0.9657,
  "r_wrist_dev": 0.5236,
  "r_wrist_flex": 0.6603,
  "l_shoulder_pitch": -1.1789,
  "l_shoulder_roll": 0.3712,
  "l_upper_arm_yaw": 0.3175,
  "l_elbow": -1.5708,
  "l_forearm_roll": -0.1493,
  "l_wrist_dev": 0.4846,
  "l_wrist_flex": 0.4879,
  "Hip_F_L": 0.8927,
  "Hip_A_L": 0.1140,
  "Hip_R_L": 0.0046,
  "Knee_L": -0.9681,
  "Ankle_P_L": -0.1812,
  "Ankle_R_L": 0.0651,
  "Hip_F_R": 0.5598,
  "Hip_A_R": 0.2740,
  "Hip_R_R": 0.3617,
  "Knee_R": -1.1168,
  "Ankle_P_R": -0.5929,
  "Ankle_R_R": 0.0844,
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

  # gloves: around the fist, on the hand body
  for side, wrist in WRISTS.items():
    pts = _mesh_points_world(model, data, wrist)
    add(
      wrist,
      f"glove_{side}",
      group=1,
      rgba=GLOVE_RGBA[team],
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(GLOVE_RADIUS, 0, 0),
      pos=_to_body(data, model, wrist, (pts.min(0) + pts.max(0)) / 2),
    )

  # head: sphere at the centre of the head
  pts = _mesh_points_world(model, data, HEAD_BODY)
  centre = (pts.min(0) + pts.max(0)) / 2
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


# Sensors. The camera (a RealSense D455) sits on the front of the head's visor at
# eye height (the centre of the head), looking forward. The ultrasonic sensor sits at the front of the chest. Each has a
# site (used by humanoid_boxing/perception.py; frame: -z forward, +y up, +x right,
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

  # camera: front of the head at eye height (the head's centre)
  pts = _mesh_points_world(model, data, HEAD_BODY)
  zc = (pts[:, 2].min() + pts[:, 2].max()) / 2
  band = pts[np.abs(pts[:, 2] - zc) < 0.02]
  cam = np.array([(band[:, 0].min() + band[:, 0].max()) / 2, band[:, 1].max(), zc])
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

