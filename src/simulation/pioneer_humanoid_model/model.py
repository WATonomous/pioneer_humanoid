"""The Pioneer humanoid as a MuJoCo physics model, built from its URDF.

`build_model(base)` loads `assets/pioneer_humanoid/pioneer_humanoid.urdf` and adds what the
URDF can't say:

  * a base: "hang" holds the pelvis fixed HANG_HEIGHT above the floor (every limb moves
    freely, nothing touches the floor); "stand" gives the pelvis a free joint on the floor
  * per-joint specs from `joint_specs.yaml`: joint limits, motor torque cap, max speed and
    rotor inertia (armature), and real link masses where known (legs, from the team model);
    every value there is labelled real / assumed / placeholder
  * one torque motor per joint (`motor_<joint>`), capped at the joint's spec; motor_test.py's
    Controller drives it with a PD (stiff enough to reach full torque at FULL_TORQUE_ERR of error) and a
    speed- and acceleration-limited reference, so it never overshoots
  * the 8 finger DIP joints that mimic their PIP (URDF <mimic>), as equality constraints:
    they follow, they have no motor
  * a floor and lights; the robot's collision boxes touch the floor only (they are each
    link's bounding box, which overlap their neighbours, so self-collision is off)
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
ASSET = REPO / "assets" / "pioneer_humanoid"
URDF = ASSET / "pioneer_humanoid.urdf"
# what MuJoCo needs to read the URDF: keep the visual meshes, fix the placeholder inertias up
MUJOCO_BLOCK = '<mujoco><compiler meshdir="." balanceinertia="true" discardvisual="false"/></mujoco>'
HANG_HEIGHT = 0.35  # m the pelvis is lifted in "hang" mode, so the feet clear the floor
SPECS = Path(__file__).resolve().parent / "joint_specs.yaml"
FULL_TORQUE_ERR = math.radians(5.0)  # a motor reaches its torque cap at this much error

# joint groups for the GUI and the report, in display order
GROUPS = {
  "Torso & head": r"^(waist_yaw|neck_yaw|neck_pitch)$",
  "Left arm": r"^l_(shoulder_pitch|shoulder_roll|upper_arm_yaw|elbow|forearm_roll|wrist_dev|wrist_flex)$",
  "Right arm": r"^r_(shoulder_pitch|shoulder_roll|upper_arm_yaw|elbow|forearm_roll|wrist_dev|wrist_flex)$",
  "Left hand": r"^l_",
  "Right hand": r"^r_",
  "Left leg": r"_L$",
  "Right leg": r"_R$",
}


@dataclass
class Robot:
  model: mujoco.MjModel
  motors: list[str]  # joint names with a motor, in actuator order
  mimics: dict[str, tuple[str, float]]  # follower joint -> (leader joint, multiplier)
  base: str
  specs: dict  # joint_specs.yaml, as loaded
  speed: np.ndarray  # per motor: max joint speed [rad/s]
  kp: np.ndarray  # per motor: position gain [Nm/rad]
  lower: np.ndarray  # per motor: joint limits [rad]
  upper: np.ndarray

  def group_of(self, joint: str) -> str:
    for g, pat in GROUPS.items():
      if re.search(pat, joint):
        return g
    return "Other"

  def qadr(self, joint: str) -> int:
    return int(self.model.jnt_qposadr[self.model.joint(joint).id])

  def dofadr(self, joint: str) -> int:
    return int(self.model.jnt_dofadr[self.model.joint(joint).id])


def _urdf_text(base: str) -> str:
  """The URDF with a root added: floating for "stand", fixed in mid-air for "hang"."""
  text = URDF.read_text()
  root = ET.fromstring(text)
  children = {j.find("child").get("link") for j in root.findall("joint")}
  base_link = next(l.get("name") for l in root.findall("link") if l.get("name") not in children)
  if base == "stand":
    joint = (f'<joint name="floating_base" type="floating"><parent link="world"/>'
             f'<child link="{base_link}"/></joint>')  # fmt: skip
  else:
    joint = (f'<joint name="hang" type="fixed"><origin xyz="0 0 {HANG_HEIGHT}" rpy="0 0 0"/>'
             f'<parent link="world"/><child link="{base_link}"/></joint>')  # fmt: skip
  return re.sub(r"(<robot[^>]*>)", lambda m: m.group(1) + MUJOCO_BLOCK + '<link name="world"/>' + joint, text, count=1)


def _mimics() -> dict[str, tuple[str, float]]:
  out = {}
  for j in ET.parse(URDF).getroot().findall("joint"):
    m = j.find("mimic")
    if m is not None:
      out[j.get("name")] = (m.get("joint"), float(m.get("multiplier", "1")))
  return out


def build_model(base: str = "hang") -> Robot:
  if base not in ("hang", "stand"):
    raise ValueError(f"base must be 'hang' or 'stand', not {base!r}")
  spec = mujoco.MjSpec.from_string(_urdf_text(base))
  spec.meshdir = str(ASSET)
  spec.option.timestep = 0.001  # 1 ms: lets the motors be ~4x stiffer than at 2 ms and stay stable (motor_test.Controller)
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960

  for g in spec.geoms:  # the robot's boxes touch only the floor
    g.contype, g.conaffinity = 2, 0
  spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05],
                          rgba=[0.55, 0.6, 0.65, 1], contype=0, conaffinity=2)  # fmt: skip
  spec.worldbody.add_light(pos=[1.5, -1.5, 3], dir=[-0.5, 0.5, -1], diffuse=[0.7, 0.7, 0.7])
  spec.worldbody.add_light(pos=[-1.5, 1.5, 3], dir=[0.5, -0.5, -1], diffuse=[0.4, 0.4, 0.4])

  specs = yaml.safe_load(SPECS.read_text())
  for b in spec.bodies:  # real link masses where known; inertia scaled with the mass
    s = specs.get("links", {}).get(b.name)
    if s and b.mass > 0:
      ratio = float(s["mass"]) / b.mass
      b.mass, b.inertia = float(s["mass"]), [x * ratio for x in b.inertia]

  mimics = _mimics()
  motors, speed, kp, lower, upper = [], [], [], [], []
  for j in spec.joints:
    if j.type != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    j.damping = [0.05, 0.0, 0.0]  # a little joint friction (the other two are nonlinear coefficients)
    if j.name in mimics:
      leader, mult = mimics[j.name]
      j.armature = 0.01
      spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=j.name, name2=leader, data=[0, mult, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      continue
    s = specs["joints"][j.name]
    j.range = [math.radians(s["lower"]), math.radians(s["upper"])]
    j.armature = float(s["armature"])
    torque = float(s["torque"])
    j.limited = True
    # the URDF importer turns each joint's effort="10" placeholder into a joint-level force clamp
    # (actfrcrange) that silently caps every motor at 10 Nm: use the spec's torque there too
    j.actfrclimited, j.actfrcrange = True, [-torque, torque]
    a = spec.add_actuator(name=f"motor_{j.name}", target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.set_to_motor()  # torque in, torque out: motor_test.Controller computes it
    a.forcelimited, a.forcerange = True, [-torque, torque]
    a.ctrllimited, a.ctrlrange = True, [-torque, torque]
    motors.append(j.name)
    speed.append(float(s["speed"]))
    kp.append(torque / FULL_TORQUE_ERR)
    lower.append(j.range[0])
    upper.append(j.range[1])
  model = spec.compile()
  return Robot(model=model, motors=motors, mimics=mimics, base=base, specs=specs, speed=np.array(speed),
               kp=np.array(kp), lower=np.array(lower), upper=np.array(upper))  # fmt: skip


def visual_meshes(robot: Robot, cell: float = 0.003):
  """Per body: its visual meshes merged in the body's frame and simplified by snapping
  vertices to a `cell`-sized grid (the URDF's meshes are 1 mm, too heavy for a browser).
  Returns {body_id: (vertices [V,3], faces [F,3], rgb)}."""
  m = robot.model
  per_body: dict[int, list] = {}
  for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 1:
      continue
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(np.int64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.geom_quat[g])
    v = v @ rot.reshape(3, 3).T + m.geom_pos[g]  # geom frame -> body frame
    rgb = m.mat_rgba[m.geom_matid[g]][:3] if m.geom_matid[g] >= 0 else m.geom_rgba[g][:3]
    per_body.setdefault(int(m.geom_bodyid[g]), []).append((v, f, rgb))
  out = {}
  for b, parts in per_body.items():
    verts, faces, colors, n = [], [], [], 0
    for v, f, rgb in parts:
      verts.append(v)
      faces.append(f + n)
      colors.append(np.repeat(rgb[None], len(v), 0))
      n += len(v)
    v, f, c = np.vstack(verts), np.vstack(faces), np.vstack(colors)
    key = np.round(v / cell).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    f = inverse.reshape(-1)[f]  # each vertex -> its grid cell's representative
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]  # drop collapsed triangles
    out[b] = (v[first].astype(np.float32), f.astype(np.int32), tuple(float(x) for x in c[first].mean(0)))
  return out


def deg(x: float) -> float:
  return math.degrees(x)
