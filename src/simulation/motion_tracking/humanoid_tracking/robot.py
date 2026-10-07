"""mjlab entity config for the Wato humanoid, ported from the Isaac Lab config
in WATonomous/Humanoid_motion_tracking (whole_body_tracking/.../robots/wato.py).

The model is GMR's watonomous.xml (fetched into data/robot/ by
scripts/fetch_assets.sh): the same robot the motions were retargeted onto, so
the CSV joint order is this model's qpos order. base_link frame: x = robot
right, y = robot forward, z = up.

Changes made to the MJCF when it is loaded:
  * its <motor>s, floor and light are dropped; mjlab adds the actuators and
    the terrain
  * its +-10 Nm per-joint actuator force cap is removed (the motors are
    18-222 Nm; the actuators carry the real limits)
  * the CAD meshes are visual only; each foot gets a box collider fitted to
    the sole of its mesh (Isaac uses convex hulls of every mesh with
    self-collisions off; only the feet are meant to touch the ground here)
  * an IMU site + gyro/velocimeter on base_link (the tracking observations
    read robot/imu_ang_vel and robot/imu_lin_vel)

Motor values are copied from wato.py; see the sources noted there.
"""

from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

WATO_XML = Path(__file__).resolve().parents[1] / "data" / "robot" / "watonomous.xml"

# Joint order of the retargeted CSVs (= qpos order of watonomous.xml).
CSV_JOINT_NAMES = (
  "left_hip_a_akh70",
  "left_hip_r_rs04",
  "left_thigh_rs03",
  "left_knee_akh70",
  "left_foot_joint_simMotor1",
  "left_foot_joint_simMotor2",
  "right_hip_a_akh70",
  "right_hip_r_rs04",
  "right_thigh_rs03",
  "right_knee_akh70",
  "right_foot_joint_simMotor1",
  "right_foot_joint_simMotor2",
  "left_shoulder_ak10_1_pitch",
  "left_shoulder_ak10_2_roll",
  "left_elbow_ak80_1_yaw",
  "left_elbow_ak80_2_bend",
  "left_elbow_ak80_3_forearm",
  "left_wrist_gl40",
  "left_claw_1",
  "left_claw_2",
  "right_shoulder_ak10_1_pitch",
  "right_shoulder_ak10_2_roll",
  "right_elbow_ak80_1_yaw",
  "right_elbow_ak80_2_bend",
  "right_elbow_ak80_3_forearm",
  "right_wrist_gl40",
  "right_claw_1",
  "right_claw_2",
)

FEET = ("left_foot_1", "right_foot_1")
WRISTS = ("Mirrorlink6__1__1", "link6__1__1")  # left, right
SOLE_THICKNESS = 0.02

##
# Motor specs (wato.py).
#   ARMATURE [kg m^2]   EFFORT [N m] (N for claws)   VELOCITY [rad/s] (m/s for claws)
##

ARMATURE_AK10 = 0.0081
ARMATURE_AK80 = 0.0049
ARMATURE_RS03 = 0.012
ARMATURE_RS04 = 0.025
ARMATURE_AKH70 = 0.030
ARMATURE_GL40 = 0.0005

EFFORT_AKH70 = 222.0
EFFORT_RS04 = 120.0
EFFORT_RS03 = 60.0
EFFORT_AK10 = 53.0
EFFORT_AK80 = 18.0
EFFORT_GL40 = 0.73
EFFORT_CLAW = 30.0

VELOCITY_AKH70 = 3.6652
VELOCITY_RS04 = 20.944
VELOCITY_RS03 = 20.42
VELOCITY_AK10 = 6.0
VELOCITY_AK80 = 6.0
VELOCITY_GL40 = 6.0
VELOCITY_CLAW = 0.2

STIFFNESS_CLAW = 400.0
DAMPING_CLAW = 40.0
ARMATURE_CLAW = 0.001

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def _motor(
  names: tuple[str, ...],
  armature: float,
  effort: float,
  velocity: float,
  stiffness: float | None = None,
  damping: float | None = None,
  implicit: bool = False,
) -> DcMotorActuatorCfg | BuiltinPositionActuatorCfg:
  """PD joint with Isaac's effort_limit_sim / velocity_limit_sim behaviour.

  MuJoCo has no joint speed cap. A DC motor whose stall torque is far above
  the effort limit keeps the full effort limit up to `velocity` and then can
  only brake, which matches a hard velocity limit closely.

  The DC motor's PD is explicit, so on joints with very little inertia about
  their axis (forearm roll, wrist, claws) its damping overshoots every 5 ms
  step and the joint shakes. Those use MuJoCo's implicit position actuator
  (`implicit=True`, like Isaac's implicit actuators and mjlab's G1) and give
  up the speed cap.
  """
  if implicit:
    return BuiltinPositionActuatorCfg(
      target_names_expr=names,
      stiffness=_stiffness(armature) if stiffness is None else stiffness,
      damping=_damping(armature) if damping is None else damping,
      effort_limit=effort,
      armature=armature,
    )
  return DcMotorActuatorCfg(
    target_names_expr=names,
    stiffness=_stiffness(armature) if stiffness is None else stiffness,
    damping=_damping(armature) if damping is None else damping,
    effort_limit=effort,
    saturation_effort=1e3 * effort,
    velocity_limit=velocity,
    armature=armature,
  )


ACTUATORS = (
  # Legs.
  _motor((".*_hip_a_akh70", ".*_knee_akh70"), ARMATURE_AKH70, EFFORT_AKH70, VELOCITY_AKH70),
  _motor((".*_hip_r_rs04",), ARMATURE_RS04, EFFORT_RS04, VELOCITY_RS04),
  _motor((".*_thigh_rs03",), ARMATURE_RS03, EFFORT_RS03, VELOCITY_RS03),
  # Ankle: parallel linkage of two RS03s modelled as two serial joints, each
  # with 2x one motor's gains/armature and one motor's 60 Nm.
  _motor(
    (".*_foot_joint_simMotor1", ".*_foot_joint_simMotor2"),
    2.0 * ARMATURE_RS03,
    EFFORT_RS03,
    VELOCITY_RS03,
  ),
  # Arms.
  _motor((".*_shoulder_ak10_1_pitch", ".*_shoulder_ak10_2_roll"), ARMATURE_AK10, EFFORT_AK10, VELOCITY_AK10),
  _motor((".*_elbow_ak80_1_yaw", ".*_elbow_ak80_2_bend"), ARMATURE_AK80, EFFORT_AK80, VELOCITY_AK80),
  _motor((".*_elbow_ak80_3_forearm",), ARMATURE_AK80, EFFORT_AK80, VELOCITY_AK80, implicit=True),
  _motor((".*_wrist_gl40",), ARMATURE_GL40, EFFORT_GL40, VELOCITY_GL40, implicit=True),
  # Claws: position-held grippers with the team's hand-tuned PD.
  _motor(
    (".*_claw_1", ".*_claw_2"),
    ARMATURE_CLAW,
    EFFORT_CLAW,
    VELOCITY_CLAW,
    stiffness=STIFFNESS_CLAW,
    damping=DAMPING_CLAW,
    implicit=True,
  ),
)

WATO_ACTION_SCALE: dict[str, float] = {}
for _a in ACTUATORS:
  for _n in _a.target_names_expr:
    WATO_ACTION_SCALE[_n] = 0.25 * _a.effort_limit / _a.stiffness


def _sole_box(spec: mujoco.MjSpec, body_name: str) -> tuple[np.ndarray, np.ndarray]:
  """Half-size and centre (body frame) of a thin box under a foot mesh."""
  model = spec.copy().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  b = model.body(body_name).id
  g = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == b)
  mid = model.geom_dataid[g]
  verts = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
  world = data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T
  local = (world - data.xpos[b]) @ data.xmat[b].reshape(3, 3)
  lo, hi = local.min(0), local.max(0)
  half = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, SOLE_THICKNESS / 2])
  centre = np.array([(hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, lo[2] + SOLE_THICKNESS / 2])
  return half, centre


def get_spec() -> mujoco.MjSpec:
  if not WATO_XML.exists():
    raise FileNotFoundError(f"{WATO_XML} is missing; run scripts/fetch_assets.sh first")
  spec = mujoco.MjSpec.from_file(str(WATO_XML))

  for act in list(spec.actuators):
    spec.delete(act)
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.worldbody.lights):
    spec.delete(light)
  # the MJCF caps every joint's actuator force at +-10 Nm (actuatorfrcrange),
  # far below the motors; the actuators below carry the real effort limits
  for joint in spec.joints:
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE

  for geom in spec.geoms:
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 2

  for foot in FEET:
    half, centre = _sole_box(spec, foot)
    spec.body(foot).add_geom(
      name=f"{foot}_collision",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=half,
      pos=centre,
      group=3,
      rgba=(0.8, 0.3, 0.3, 0.5),
    )

  spec.body("base_link").add_site(name="imu", size=(0.01, 0.01, 0.01))
  spec.add_sensor(
    name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu"
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  return spec


FEET_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_1_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.86),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)


def get_wato_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=ACTUATORS, soft_joint_pos_limit_factor=0.9),
  )
