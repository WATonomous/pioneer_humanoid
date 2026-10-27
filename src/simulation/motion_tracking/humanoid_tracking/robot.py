"""mjlab entity config for the Pioneer humanoid (assets/pioneer_whole_body_humanoid in this repo).

The model is the whole-body URDF, loaded with a free-floating base. Changes made when it is loaded:
  * re-rooted to the same conventions the rest of the package uses: the base frame sits at the hip
    joints (pelvis height, not on the floor as in the URDF) and faces +y with +x the robot's right
    (the URDF's robot faces -y). Only frames are relabelled; the robot's shape and joints are unchanged
  * the hands are closed into fists and their 40 finger joints removed: boxing gloves go over them
    (fighters.py). The CSV joint order (CSV_JOINT_NAMES) is the remaining 29 joints' qpos order
  * the URDF's per-joint effort limit becomes a hidden actuator force clamp in MuJoCo: removed, the
    actuators below carry the limits
  * the CAD meshes are visual only and the URDF's bounding-box collisions are dropped; each foot gets
    a box collider fitted to the sole of its mesh (only the feet touch the ground here)
  * an IMU site + gyro/velocimeter on base_link (the tracking observations read robot/imu_ang_vel and
    robot/imu_lin_vel)

Motor values are the URDF's (its README: exact for the legs and arms, estimates for the waist, neck and
wrists); armatures are the matching motors' (the legs and arms use the same motor types as Wato).
"""

import re
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "pioneer_whole_body_humanoid"
PIONEER_URDF = ASSET_DIR / "pioneer_whole_body_humanoid.urdf"

FEET = ("l_foot", "r_foot")
WRISTS = ("l_hand_H", "r_hand_H")  # left, right: the hands (fists)
SOLE_THICKNESS = 0.02
# Boxing-glove props in place of the hand meshes (visual only: no mass, no contact), centred on the
# forearm's line, with a site "<hand>_glove" at the centre: where the glove hitbox goes and what the move
# generator aims. The hand meshes stay in the model, hidden in HIDDEN_GROUP.
GLOVE_PROP_RGBA = (0.70, 0.06, 0.06, 1.0)
GLOVE_PAD = 1.1  # the glove's half-axes over the fist's
HIDDEN_GROUP = 5

# The fist the hands are closed into [deg]; positive curls both hands the same way (URDF README).
FIST = {
  "abd": 0.0, "mcp": 85.0, "pip": 90.0, "dip": 70.0,
  "th_c": 45.0, "th_f": 50.0, "th_mcp": 35.0, "th_ip": 40.0,
}  # fmt: skip
FINGER_JOINT = re.compile(r"^[lr]_(?:[irl]_)?(abd|mcp|pip|dip)$|^[lr]_(th_c|th_f|th_mcp|th_ip)$")

##
# Motor specs.
#   ARMATURE [kg m^2]   EFFORT [N m]   VELOCITY [rad/s]
##

ARMATURE_AKH70 = 0.030  # hip pitch, knee
ARMATURE_RS04 = 0.025  # hip roll
ARMATURE_RS03 = 0.012  # hip yaw, waist
ARMATURE_ANKLE = 2.0 * ARMATURE_RS03  # parallel linkage of two RS03s, modelled as two serial joints
ARMATURE_AK10 = 0.0081  # shoulders
ARMATURE_AK80 = 0.0049  # upper arm yaw, elbow, forearm roll
ARMATURE_SMALL = 0.0005  # wrists, neck

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


# joint-name regex -> motor speed cap [rad/s], filled by _motor
SPEED_CAPS: dict[str, float] = {}


def _motor(names: tuple[str, ...], armature: float, effort: float, velocity: float, implicit: bool = False):
  """PD joint with Isaac's effort_limit_sim / velocity_limit_sim behaviour.

  MuJoCo has no joint speed cap. A DC motor whose stall torque is far above the effort limit keeps the
  full effort limit up to `velocity` and then can only brake, which matches a hard velocity limit
  closely. Joints with very little inertia about their axis (forearm roll, wrists, neck) shake under
  the explicit DC-motor PD and use MuJoCo's implicit position actuator instead (no speed cap).
  """
  for n in names:
    SPEED_CAPS[n] = velocity
  if implicit:
    return BuiltinPositionActuatorCfg(
      target_names_expr=names, stiffness=_stiffness(armature), damping=_damping(armature),
      effort_limit=effort, armature=armature,
    )  # fmt: skip
  return DcMotorActuatorCfg(
    target_names_expr=names, stiffness=_stiffness(armature), damping=_damping(armature),
    effort_limit=effort, saturation_effort=1e3 * effort, velocity_limit=velocity, armature=armature,
  )  # fmt: skip


ACTUATORS = (
  # Legs.
  _motor(("Hip_F_.*", "Knee_.*"), ARMATURE_AKH70, 222.0, 3.6652),
  _motor(("Hip_A_.*",), ARMATURE_RS04, 120.0, 20.944),
  _motor(("Hip_R_.*",), ARMATURE_RS03, 60.0, 20.42),
  _motor(("Ankle_P_.*", "Ankle_R_.*"), ARMATURE_ANKLE, 60.0, 20.42),
  # Torso.
  _motor(("waist_yaw",), ARMATURE_RS03, 60.0, 6.0),
  _motor(("neck_yaw", "neck_pitch"), ARMATURE_SMALL, 5.0, 6.0, implicit=True),
  # Arms.
  _motor((".*_shoulder_pitch", ".*_shoulder_roll"), ARMATURE_AK10, 53.0, 6.0),
  _motor((".*_upper_arm_yaw", ".*_elbow"), ARMATURE_AK80, 22.0, 6.0),
  _motor((".*_forearm_roll",), ARMATURE_AK80, 22.0, 6.0, implicit=True),
  _motor((".*_wrist_dev", ".*_wrist_flex"), ARMATURE_SMALL, 2.0, 6.0, implicit=True),
)

ACTION_SCALE: dict[str, float] = {}
for _a in ACTUATORS:
  for _n in _a.target_names_expr:
    ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


def _quat_axis_angle(axis, angle: float) -> np.ndarray:
  q = np.zeros(4)
  mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
  return q


def _quat_mul(a, b) -> np.ndarray:
  out = np.zeros(4)
  mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
  return out


def _make_fists(spec: mujoco.MjSpec) -> None:
  """Turn each finger joint's child link by its FIST angle about the joint axis (URDF link frames sit
  on their joint's pivot), then remove the finger joints: rigid fists."""
  for j in list(spec.joints):
    m = FINGER_JOINT.match(j.name)
    if not m:
      continue
    angle = np.radians(FIST[m.group(1) or m.group(2)])
    body = j.parent
    body.quat = _quat_mul(body.quat, _quat_axis_angle(j.axis, angle))
    spec.delete(j)
  for eq in list(spec.equalities):  # mimic couplings between the removed joints, if any
    spec.delete(eq)


def _reroot(spec: mujoco.MjSpec, base: mujoco.MjsBody) -> None:
  """Move the base frame up to the midpoint of the hip pitch joints and turn it 180 deg about z, by
  re-expressing everything attached to it in the new frame."""
  model = spec.copy().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  hips = [data.xpos[model.body(b).id] - data.xpos[model.body(base.name).id] for b in ("l_hip_pitch_link", "r_hip_pitch_link")]
  off = (hips[0] + hips[1]) / 2
  qz = _quat_axis_angle([0, 0, 1], np.pi)
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, qz)
  R = R.reshape(3, 3)

  def move(obj, pos_attr="pos", quat_attr="quat"):
    setattr(obj, pos_attr, R @ (np.asarray(getattr(obj, pos_attr)) - off))
    setattr(obj, quat_attr, _quat_mul(qz, getattr(obj, quat_attr)))

  for child in base.bodies:
    move(child)
  for g in base.geoms:
    move(g)
  for s in base.sites:
    move(s)
  move(base, "ipos", "iquat")


# DESIGN TARGETS, not the robot as built: joint ranges [deg] wide enough for a Wato-size boxing stance
# and moves (knee, ankles, hips past the URDF's) and a tight guard (elbows). Mechanical blockers (part
# collisions, the ankle linkage, cables) are ignored for now. USE_DESIGN_RANGES = False puts back the
# URDF's ranges.
USE_DESIGN_RANGES = True
DESIGN_RANGES = {
  "Knee_.": (-110.0, 0.0),  # URDF -65..0
  "Hip_F_.": (-20.0, 115.0),  # URDF -20..90
  "Hip_R_.": (-50.0, 90.0),  # URDF -30..90
  "Ankle_P_.": (-50.0, 50.0),  # URDF -30..50
  "Ankle_R_.": (-30.0, 30.0),  # URDF -15..15 (it was -30..30 before #403)
  "Hip_A_.": (-25.0, 60.0),  # URDF -10..60 (the pivots swing the lead leg in)
  "[lr]_elbow": (-135.0, 0.0),  # URDF -90..0 (a tight guard folds the arm past 90 deg)
}


def _glove_props(spec: mujoco.MjSpec) -> None:
  """Hide each hand's meshes and put a boxing glove on the fist: a padded ellipsoid the fist's size,
  centred on the forearm's line (the hand frame's z axis; the fist's box sits off it towards the thumb),
  its long axis along the forearm, and a cuff round the wrist."""
  model = spec.copy().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  for hand in WRISTS:
    b = model.body(hand).id
    pts = []
    for g in range(model.ngeom):
      if model.geom_bodyid[g] == b and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pts.append(data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)
    local = (np.vstack(pts) - data.xpos[b]) @ data.xmat[b].reshape(3, 3)
    lo, hi = local.min(0), local.max(0)
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    centre[0] = 0.0  # on the forearm's line
    body = spec.body(hand)
    for g in body.find_all(mujoco.mjtObj.mjOBJ_GEOM):  # the palm and the finger links under it
      if g.type == mujoco.mjtGeom.mjGEOM_MESH:
        g.group, g.material, g.rgba = HIDDEN_GROUP, "", (0, 0, 0, 0)
    common = dict(group=2, rgba=GLOVE_PROP_RGBA, contype=0, conaffinity=0, mass=0.0, density=0.0)
    body.add_geom(name=f"{hand}_glove", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=GLOVE_PAD * half, pos=centre, **common)
    body.add_site(name=f"{hand}_glove", pos=centre, size=(0.01, 0.01, 0.01))
    # the cuff: a short cylinder round the wrist end of the fist (+z in the hand frame, along the forearm)
    body.add_geom(
      name=f"{hand}_glove_cuff", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.8 * min(half[:2]), 0.035),
      pos=(centre[0], centre[1], hi[2] + 0.01), **common,
    )  # fmt: skip


def _urdf_text() -> str:
  text = PIONEER_URDF.read_text()
  free = ('<link name="world"/><joint name="floating_base" type="floating">'
          '<parent link="world"/><child link="base_link"/></joint>')  # fmt: skip
  return re.sub(r"(<robot[^>]*>)", lambda m: m.group(1) + free, text, count=1)


def _sole_box(spec: mujoco.MjSpec, body_name: str) -> tuple[np.ndarray, np.ndarray]:
  """Half-size and centre (body frame) of a thin box under a foot's meshes."""
  model = spec.copy().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  b = model.body(body_name).id
  pts = []
  for g in range(model.ngeom):
    if model.geom_bodyid[g] == b and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      mid = model.geom_dataid[g]
      v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
      pts.append(data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)
  local = (np.vstack(pts) - data.xpos[b]) @ data.xmat[b].reshape(3, 3)
  lo, hi = local.min(0), local.max(0)
  half = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, SOLE_THICKNESS / 2])
  centre = np.array([(hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, lo[2] + SOLE_THICKNESS / 2])
  return half, centre


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_string(_urdf_text())
  spec.meshdir = str(ASSET_DIR)
  base = spec.body("base_link")

  _make_fists(spec)
  for g in list(spec.geoms):  # the URDF's bounding-box collisions
    if g.type != mujoco.mjtGeom.mjGEOM_MESH:
      spec.delete(g)
  _reroot(spec, base)

  # URDF effort -> MuJoCo actuatorfrcrange: a hidden clamp at the URDF value; the actuators carry the limits
  for joint in spec.joints:
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
    if USE_DESIGN_RANGES:
      for pattern, deg in DESIGN_RANGES.items():
        if re.fullmatch(pattern, joint.name):
          joint.range = np.radians(deg)
  for geom in spec.geoms:
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 2
  _glove_props(spec)

  for foot in FEET:
    half, centre = _sole_box(spec, foot)
    spec.body(foot).add_geom(
      name=f"{foot}_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=centre, group=3, rgba=(0.8, 0.3, 0.3, 0.5)
    )

  # the default pose stands on the floor (the root starts at hip height, as in the scene's qpos0)
  model = spec.copy().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  lows = [data.geom_xpos[model.geom(f"{f}_collision").id][2] - model.geom_size[model.geom(f"{f}_collision").id][2] for f in FEET]
  base.pos = [0.0, 0.0, -min(lows)]

  base.add_site(name="imu", size=(0.01, 0.01, 0.01))
  spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  return spec


def _joint_order() -> tuple[str, ...]:
  spec = get_spec()
  return tuple(j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE)


# Joint order of the motion CSVs (= qpos order after the free joint).
CSV_JOINT_NAMES = _joint_order()


def _standing_height() -> float:
  """Base height with the soles on the floor at the zero pose."""
  return float(get_spec().body("base_link").pos[2])


STANDING_HEIGHT = _standing_height()

FEET_COLLISION = CollisionCfg(
  geom_names_expr=(r"^[lr]_foot_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, STANDING_HEIGHT + 0.005),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)


def get_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=ACTUATORS, soft_joint_pos_limit_factor=0.9),
  )
