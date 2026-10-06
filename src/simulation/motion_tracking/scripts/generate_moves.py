"""Generate boxing movement (footwork, ducks, slips, bob and weave, pivots)
for the Wato robot as a retargeted CSV, chained freely.

The stance and guard are copied from one frame of a retargeted clip (default:
the boxing take at 8.2 s, orthodox: left foot forward, rear heel up, torso
bladed), shortened to fit Wato's legs and held a few cm lower. Moves:

  F B L R   step-drag forward / back / left / right: the foot nearest the
            direction moves first while the other pushes, then follows the
            same distance; weight onto the pushing foot before the first foot
            lifts and onto the landed foot before the second lifts; feet glide
            low and never cross
  D  DD     duck, shallow and fast / deep: bend the knees with a small forward
            lean (back straight), feet planted, guard up, straight back up
  SL SR     slip left / right: weight onto the foot on that side, dip and tilt
            the head off the centre line, return
  WL WR     bob and weave ending on the left / right: the head travels a U
            under a hook, dipping on the near side, lowest in the middle,
            rising on the far side
  PL PR     pivot left / right on the lead foot: weight on the lead foot, the
            rear foot swings around the arc in short steps while the body
            turns, then the lead foot turns to match; later moves face the new
            direction. The flat robot foot cannot spin under load, so the lead
            foot stays planted and the lead hip turns
  H         hold the stance

Moves chain without stopping: the body passes through the stance between
moves on one smooth curve (centre of mass, hip height, heading and lean),
except at H and the ends. Feet move on minimum-jerk paths. Leg joint angles
come from whole-body IK on the Wato MJCF; the arms keep the guard.

After the IK the script checks every leg joint against its motor speed cap;
a move that goes over --max-cap-use of a cap is slowed down and the clip is
solved again, until every move fits. Output: CSV at --fps (root pos, root
quat xyzw, 28 joints), opponent along world +x at the start.

  uv run scripts/generate_moves.py --sequence F F D PL SL SR WL WR DD PR B B \
      --output-file data/motions/moves.csv
  uv run scripts/csv_to_npz.py --input-file data/motions/moves.csv \
      --input-fps 50 --output-file data/motions/moves.npz --video True
"""

import os
import re
import sys
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import tyro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wato_tracking.robot import ACTUATORS, CSV_JOINT_NAMES, WATO_XML, _sole_box  # noqa: E402

LEAD, REAR = "left_foot_1", "right_foot_1"  # orthodox
LEG_JOINTS = CSV_JOINT_NAMES[:12]
UP = np.array([0.0, 0.0, 1.0])

Move = Literal["F", "B", "L", "R", "D", "DD", "SL", "SR", "WL", "WR", "PL", "PR", "H"]

# step direction in the stance frame (x towards the opponent, y left) and the
# foot that moves first
STEPS = {
  "F": (np.array([1.0, 0.0]), LEAD),
  "B": (np.array([-1.0, 0.0]), REAR),
  "L": (np.array([0.0, 1.0]), LEAD),
  "R": (np.array([0.0, -1.0]), REAR),
}


@dataclass
class StepCfg:
  forward: float = 0.10
  """forward/back step length [m]"""
  side: float = 0.08
  """sideways step length [m]"""
  weight_shift: float = 0.6
  """CoM fraction from neutral towards the support foot before a foot lifts"""
  push: float = 0.5
  """fraction of the step the CoM travels while the first foot is in the air"""
  lift: float = 0.03
  """swing foot clearance [m]"""
  shift: float = 0.35
  swing: float = 0.35
  transfer: float = 0.45
  drag: float = 0.35
  reset: float = 0.3


@dataclass
class DuckCfg:
  depth: float
  """how far the hips drop [m]"""
  lean: float
  """forward lean at the bottom [deg]; hips hinge, back straight"""
  down: float
  bottom: float
  up: float


@dataclass
class SlipCfg:
  depth: float = 0.06
  lean: float = 12.0
  """sideways head tilt [deg]"""
  weight: float = 0.4
  """CoM fraction towards the foot on the slip side"""
  out: float = 0.3
  back: float = 0.35


@dataclass
class WeaveCfg:
  depth: float = 0.14
  """hip drop at the bottom of the U [m]"""
  side_depth: float = 0.06
  lean: float = 8.0
  """sideways tilt at the sides of the U [deg]"""
  forward_lean: float = 10.0
  """forward lean at the bottom [deg]"""
  weight: float = 0.3
  """CoM fraction towards the foot on each side"""
  phase: float = 0.3
  """time between the U's points [s]"""


@dataclass
class PivotCfg:
  angle: float = 45.0
  """turn per pivot [deg]"""
  max_arc: float = 0.25
  """longest rear-foot swing per step [m]; longer arcs take several steps"""
  weight: float = 0.6
  """CoM fraction towards the lead foot while the rear foot swings"""
  lift: float = 0.03
  load: float = 0.35
  swing: float = 0.4
  touch: float = 0.1
  """rear foot down between swing steps [s]"""
  unload: float = 0.4
  """weight onto the rear foot before the lead foot turns [s]"""
  turn: float = 0.3
  """lead foot turns to the new direction [s]"""
  reset: float = 0.35


def min_jerk(s):
  s = np.clip(s, 0.0, 1.0)
  return s**3 * (10 - 15 * s + 6 * s**2)


def rot_z(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s], [s, c]])


def quat_z(a: float) -> np.ndarray:
  return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  out = np.zeros(4)
  mujoco.mju_mulQuat(out, a, b)
  return out


def quat_tilt(t: np.ndarray) -> np.ndarray:
  """Rotation that tips the vertical towards horizontal vector t by |t| rad."""
  ang = float(np.linalg.norm(t))
  if ang < 1e-9:
    return np.array([1.0, 0, 0, 0])
  axis = np.cross(UP, np.array([t[0], t[1], 0.0]) / ang)
  out = np.zeros(4)
  mujoco.mju_axisAngle2Quat(out, axis, ang)
  return out


def smooth_through(waypoints: list[tuple[int, np.ndarray, bool]], n_frames: int) -> np.ndarray:
  """C1 cubic Hermite curve through (frame, value, stop) waypoints: zero
  velocity at stop waypoints, finite-difference (Catmull-Rom) velocity
  elsewhere. Waypoint 0 is the value at frame 0."""
  t = np.array([w[0] for w in waypoints], dtype=float)
  p = np.array([w[1] for w in waypoints])
  v = np.zeros_like(p)
  for i in range(1, len(p) - 1):
    if not waypoints[i][2] and t[i + 1] > t[i - 1]:
      v[i] = (p[i + 1] - p[i - 1]) / (t[i + 1] - t[i - 1])
  out = np.empty((n_frames, p.shape[1]))
  for k in range(n_frames):
    i = min(max(np.searchsorted(t, k + 1, side="left") - 1, 0), len(t) - 2)
    h = t[i + 1] - t[i]
    if h <= 0:
      out[k] = p[i + 1]
      continue
    s = np.clip((k + 1 - t[i]) / h, 0.0, 1.0)
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    out[k] = h00 * p[i] + h10 * h * v[i] + h01 * p[i + 1] + h11 * h * v[i + 1]
  return out


class Wato:
  def __init__(self):
    spec = mujoco.MjSpec.from_file(str(WATO_XML))
    self.m = spec.compile()
    self.d = mujoco.MjData(self.m)
    self.feet = {f: self.m.body(f).id for f in (LEAD, REAR)}
    self.base = self.m.body("base_link").id
    self.torso = self.m.body("Torso_1").id
    self.sole = {f: _sole_box(spec, f)[1] for f in (LEAD, REAR)}
    leg_dofs = [self.m.jnt_dofadr[self.m.joint(j).id] for j in LEG_JOINTS]
    self.dofs = np.array(list(range(6)) + leg_dofs)  # free joint + legs
    self.leg_qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in LEG_JOINTS])
    self.leg_range = np.array([self.m.jnt_range[self.m.joint(j).id] for j in LEG_JOINTS])

  def set_csv_row(self, row: np.ndarray) -> None:
    self.d.qpos[:3] = row[:3]
    self.d.qpos[3:7] = row[[6, 3, 4, 5]]
    self.d.qpos[7:] = row[7:]
    self.forward()

  def forward(self) -> None:
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)

  def csv_row(self) -> np.ndarray:
    q = self.d.qpos
    return np.concatenate([q[:3], q[[4, 5, 6, 3]], q[7:]])

  def com(self) -> np.ndarray:
    return self.d.subtree_com[self.base].copy()

  def sole_xy(self, foot: str) -> np.ndarray:
    b = self.feet[foot]
    return (self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.sole[foot])[:2]

  def foot_pose(self, foot: str) -> tuple[np.ndarray, np.ndarray]:
    b = self.feet[foot]
    return self.d.xpos[b].copy(), self.d.xquat[b].copy()

  def solve(self, foot_pos, foot_quat, com_xy, base_z, base_quat, iters=50, tol=1e-6) -> float:
    m, d, nv = self.m, self.d, self.m.nv
    jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
    err = np.inf
    for _ in range(iters):
      rows, errs, w = [], [], []
      for f, b in self.feet.items():
        mujoco.mj_jacBody(m, d, jp, jr, b)
        rows += [jp.copy(), jr.copy()]
        e_r = np.zeros(3)
        mujoco.mju_subQuat(e_r, foot_quat[f], d.xquat[b])
        errs += [foot_pos[f] - d.xpos[b], d.xmat[b].reshape(3, 3) @ e_r]  # local -> world
        w += [10.0] * 6
      mujoco.mj_jacBody(m, d, jp, jr, self.base)
      e_r = np.zeros(3)
      mujoco.mju_subQuat(e_r, base_quat, d.xquat[self.base])
      rows += [jp[2:3].copy(), jr.copy()]
      errs += [np.array([base_z - d.xpos[self.base][2]]), d.xmat[self.base].reshape(3, 3) @ e_r]
      w += [10.0] * 4
      jc = np.zeros((3, nv))
      mujoco.mj_jacSubtreeCom(m, d, jc, self.base)
      rows.append(jc[:2])
      errs.append(com_xy - self.com()[:2])
      w += [5.0] * 2

      J = np.vstack(rows)[:, self.dofs]
      e = np.concatenate(errs)
      err = float(np.abs(e).max())
      if err < tol:
        break
      W = np.array(w)
      dq_sub = np.linalg.solve(J.T @ (W[:, None] * J) + 1e-6 * np.eye(len(self.dofs)), J.T @ (W * e))
      dq_sub *= min(1.0, 0.1 / max(np.abs(dq_sub).max(), 1e-12))  # no big jumps
      dq = np.zeros(nv)
      dq[self.dofs] = dq_sub
      mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
      d.qpos[self.leg_qadr] = np.clip(d.qpos[self.leg_qadr], self.leg_range[:, 0], self.leg_range[:, 1])
      self.forward()
    return err


@dataclass
class Stance:
  """Stance geometry (world frame at heading 0) the moves are built from."""

  feet_pos: dict  # foot origin positions
  feet_quat: dict
  sole_off: dict  # foot origin -> sole centre, xy
  com: np.ndarray  # neutral CoM xy
  base_z: float
  base_quat: np.ndarray


@dataclass
class Feet:
  """Current foot placement: origin position and yaw offset from the stance."""

  pos: dict
  yaw: dict

  def copy(self) -> "Feet":
    return Feet({f: p.copy() for f, p in self.pos.items()}, dict(self.yaw))


class Timeline:
  """Per-frame foot targets plus waypoints for the body channels
  [com x, com y, hip drop, heading, tilt x, tilt y]."""

  def __init__(self, stance: Stance, fps: float):
    self.st, self.fps = stance, fps
    self.feet_frames: list[Feet] = []
    self.waypoints: list[tuple[int, np.ndarray, bool]] = []
    self.spans: list[tuple[int, int]] = []  # frames of each move
    self.feet = Feet({f: p.copy() for f, p in stance.feet_pos.items()}, {LEAD: 0.0, REAR: 0.0})
    self.heading = 0.0

  # -- stance geometry --------------------------------------------------------
  def sole(self, foot: str, feet: Feet | None = None) -> np.ndarray:
    feet = feet or self.feet
    return feet.pos[foot][:2] + rot_z(feet.yaw[foot]) @ self.st.sole_off[foot]

  def neutral(self) -> np.ndarray:
    """Neutral CoM: the stance's CoM carried along with the feet."""
    st, feet = self.st, self.feet
    mid0 = (st.feet_pos[LEAD][:2] + st.feet_pos[REAR][:2]) / 2
    mid = (feet.pos[LEAD][:2] + feet.pos[REAR][:2]) / 2
    return mid + rot_z(self.heading) @ (st.com - mid0)

  def towards(self, foot: str, frac: float) -> np.ndarray:
    n = self.neutral()
    return n + frac * (self.sole(foot) - n)

  def world(self, v: np.ndarray) -> np.ndarray:
    """Stance-frame xy vector -> world."""
    return rot_z(self.heading) @ v

  # -- building blocks ----------------------------------------------------------
  def body(self, com, drop=0.0, tilt=(0.0, 0.0), heading=None, stop=False) -> None:
    h = self.heading if heading is None else heading
    self.waypoints.append((len(self.feet_frames), np.array([*com, drop, h, *tilt]), stop))

  def stance_body(self, stop=False) -> None:
    self.body(self.neutral(), stop=stop)

  def frames(self, duration: float, path=None) -> None:
    """Append frames; path(s) -> Feet for s in (0, 1], else feet stay put."""
    n = max(1, int(round(duration * self.fps)))
    for k in range(1, n + 1):
      self.feet_frames.append(path(k / n) if path else self.feet.copy())

  def glide(self, foot: str, target: np.ndarray, lift: float, duration: float) -> None:
    start = self.feet.copy()

    def path(u):
      f = start.copy()
      f.pos[foot] = start.pos[foot] + min_jerk(u) * (target - start.pos[foot])
      f.pos[foot][2] += lift * 16 * u**2 * (1 - u) ** 2
      return f

    self.frames(duration, path)
    self.feet.pos[foot] = target.copy()

  def swing_about(self, foot: str, centre: np.ndarray, angle: float, lift: float, duration: float) -> None:
    """Foot travels an arc about centre (xy), turning with it."""
    start = self.feet.copy()

    def path(u):
      a = min_jerk(u) * angle
      f = start.copy()
      f.pos[foot] = start.pos[foot].copy()
      f.pos[foot][:2] = centre + rot_z(a) @ (start.pos[foot][:2] - centre)
      f.pos[foot][2] += lift * 16 * u**2 * (1 - u) ** 2
      f.yaw[foot] = start.yaw[foot] + a
      return f

    self.frames(duration, path)
    self.feet = path(1.0)
    self.feet.pos[foot][2] = start.pos[foot][2]

  # -- moves --------------------------------------------------------------------
  def step(self, move: str, c: StepCfg, T: float) -> None:
    direction, first = STEPS[move]
    second = REAR if first == LEAD else LEAD
    length = c.forward if direction[0] else c.side
    step = np.array([*self.world(direction * length), 0.0])

    self.frames(c.shift * T)
    self.body(self.towards(second, c.weight_shift))
    pushed = self.towards(second, c.weight_shift) + c.push * step[:2]
    self.glide(first, self.feet.pos[first] + step, c.lift, c.swing * T)
    self.body(pushed)
    self.frames(c.transfer * T)
    self.body(self.towards(first, c.weight_shift))
    loaded = self.towards(first, c.weight_shift)
    self.glide(second, self.feet.pos[second] + step, c.lift, c.drag * T)
    self.body(loaded)
    self.frames(c.reset * T)
    self.stance_body()

  def duck(self, c: DuckCfg, T: float) -> None:
    lean = np.radians(c.lean) * self.world(np.array([1.0, 0.0]))
    self.frames(c.down * T)
    self.body(self.neutral(), c.depth, lean)
    self.frames(c.bottom * T)
    self.body(self.neutral(), c.depth, lean)
    self.frames(c.up * T)
    self.stance_body()

  def slip(self, side: int, c: SlipCfg, T: float) -> None:
    # side +1 = left (lead foot side), -1 = right (rear foot side)
    foot = LEAD if side > 0 else REAR
    tilt = np.radians(c.lean) * self.world(np.array([0.0, float(side)]))
    self.frames(c.out * T)
    self.body(self.towards(foot, c.weight), c.depth, tilt)
    self.frames(c.back * T)
    self.stance_body()

  def weave(self, side: int, c: WeaveCfg, T: float) -> None:
    # side = where the head ends: +1 left, -1 right; the U starts on the other side
    near, far = (REAR, LEAD) if side > 0 else (LEAD, REAR)
    lateral = self.world(np.array([0.0, 1.0]))
    fwd = np.radians(c.forward_lean) * self.world(np.array([1.0, 0.0]))
    t = np.radians(c.lean)
    self.frames(c.phase * T)
    self.body(self.towards(near, c.weight), c.side_depth, -side * t * lateral)
    self.frames(c.phase * T)
    self.body(self.neutral(), c.depth, fwd)
    self.frames(c.phase * T)
    self.body(self.towards(far, c.weight), c.side_depth, side * t * lateral)
    self.frames(c.phase * T)
    self.stance_body()

  def pivot(self, direction: int, c: PivotCfg, T: float) -> None:
    # direction +1 = left (counter-clockwise), -1 = right
    centre = self.sole(LEAD)
    total = direction * np.radians(c.angle)
    radius = np.linalg.norm(self.feet.pos[REAR][:2] - centre)
    n_steps = max(1, int(np.ceil(radius * abs(total) / c.max_arc)))

    self.frames(c.load * T)
    self.body(self.towards(LEAD, c.weight))
    for _ in range(n_steps):
      a = total / n_steps
      self.swing_about(REAR, centre, a, c.lift, c.swing * T)
      self.heading += a
      self.body(self.towards(LEAD, c.weight))
      self.frames(c.touch * T)
    # the lead foot turns to match, weight on the rear foot
    self.frames(c.unload * T)
    self.body(self.towards(REAR, c.weight))
    start = self.feet.copy()

    def path(u):
      f = start.copy()
      a = min_jerk(u) * total
      f.pos[LEAD] = start.pos[LEAD].copy()
      f.pos[LEAD][:2] = centre + rot_z(a) @ (start.pos[LEAD][:2] - centre)
      f.pos[LEAD][2] += 0.015 * 16 * u**2 * (1 - u) ** 2
      f.yaw[LEAD] = start.yaw[LEAD] + a
      return f

    self.frames(c.turn * T, path)
    self.feet = path(1.0)
    self.feet.pos[LEAD][2] = start.pos[LEAD][2]
    self.body(self.towards(REAR, c.weight))
    self.frames(c.reset * T)
    self.stance_body()


def build(
  stance: Stance,
  sequence: list[str],
  tempo: list[float],
  fps: float,
  hold: float,
  step: StepCfg,
  ducks: dict,
  slip: SlipCfg,
  weave: WeaveCfg,
  pivot: PivotCfg,
) -> Timeline:
  tl = Timeline(stance, fps)
  tl.stance_body(stop=True)
  tl.frames(hold)
  tl.stance_body(stop=True)
  for move, T in zip(sequence, tempo):
    start = len(tl.feet_frames)
    if move in STEPS:
      tl.step(move, step, T)
    elif move in ducks:
      tl.duck(ducks[move], T)
    elif move in ("SL", "SR"):
      tl.slip(1 if move == "SL" else -1, slip, T)
    elif move in ("WL", "WR"):
      tl.weave(1 if move == "WL" else -1, weave, T)
    elif move in ("PL", "PR"):
      tl.pivot(1 if move == "PL" else -1, pivot, T)
    elif move == "H":
      tl.stance_body(stop=True)
      tl.frames(hold * T)
      tl.stance_body(stop=True)
    tl.spans.append((start, len(tl.feet_frames)))
  tl.stance_body(stop=True)
  tl.frames(hold)
  tl.stance_body(stop=True)
  return tl


def solve_timeline(robot: Wato, stance: Stance, tl: Timeline) -> tuple[np.ndarray, float]:
  body = smooth_through(tl.waypoints, len(tl.feet_frames))
  out, worst = [], 0.0
  for feet, (cx, cy, drop, heading, tx, ty) in zip(tl.feet_frames, body):
    feet_quat = {f: quat_mul(quat_z(feet.yaw[f]), stance.feet_quat[f]) for f in feet.pos}
    base_quat = quat_mul(quat_tilt(np.array([tx, ty])), quat_mul(quat_z(heading), stance.base_quat))
    err = robot.solve(feet.pos, feet_quat, np.array([cx, cy]), stance.base_z - drop, base_quat)
    worst = max(worst, err)
    out.append(robot.csv_row())
  return np.array(out), worst


def make_stance(robot: Wato, stance_file: str, stance_time: float, stance_fps: float, scale: float, crouch: float) -> Stance:
  rows = np.loadtxt(stance_file, delimiter=",")
  robot.set_csv_row(rows[int(round(stance_time * stance_fps))])
  feet0 = {f: robot.foot_pose(f) for f in (LEAD, REAR)}
  origin = np.array([*(feet0[LEAD][0][:2] + feet0[REAR][0][:2]) / 2, 0.0])
  robot.d.qpos[:3] -= origin
  robot.forward()
  base_z, base_quat = robot.d.xpos[robot.base][2], robot.d.xquat[robot.base].copy()
  feet_quat = {f: q for f, (p, q) in feet0.items()}
  clip_feet = {f: p - origin for f, (p, q) in feet0.items()}
  clip_com = robot.com()[:2]

  # shorten the stance about its centre and sink into it, in small IK steps
  feet_pos = {f: np.array([*(scale * p[:2]), p[2]]) for f, p in clip_feet.items()}
  com = scale * clip_com
  for k in range(1, 21):
    s = k / 20
    robot.solve(
      {f: clip_feet[f] + s * (feet_pos[f] - clip_feet[f]) for f in feet_pos},
      feet_quat,
      clip_com + s * (com - clip_com),
      base_z - s * crouch,
      base_quat,
    )
  sole_off = {f: robot.sole_xy(f) - feet_pos[f][:2] for f in (LEAD, REAR)}
  return Stance(feet_pos, feet_quat, sole_off, com, base_z - crouch, base_quat)


def speed_use(out: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
  """Per-frame |joint velocity| / cap for the legs, and the caps."""
  caps = np.array(
    [
      next(a.velocity_limit for a in ACTUATORS if any(re.fullmatch(p, n) for p in a.target_names_expr))
      for n in LEG_JOINTS
    ]
  )
  vel = np.abs(np.gradient(out[:, 7:19], 1.0 / fps, axis=0))
  return vel / caps, caps


def main(
  sequence: list[Move] = ["F", "F", "D", "PL", "SL", "SR", "WL", "WR", "DD", "PR", "B", "B"],  # noqa: B006
  output_file: str = "data/motions/moves.csv",
  stance_file: str = "data/motions/boxing.csv",
  stance_time: float = 8.2,
  stance_fps: float = 120.0,
  stance_scale: float = 0.75,
  crouch: float = 0.06,
  hold: float = 1.0,
  fps: float = 50.0,
  max_cap_use: float = 0.8,
  step: StepCfg = StepCfg(),  # noqa: B008
  duck: DuckCfg = DuckCfg(depth=0.10, lean=10.0, down=0.25, bottom=0.1, up=0.3),  # noqa: B008
  deep_duck: DuckCfg = DuckCfg(depth=0.20, lean=25.0, down=0.5, bottom=0.2, up=0.55),  # noqa: B008
  slip: SlipCfg = SlipCfg(),  # noqa: B008
  weave: WeaveCfg = WeaveCfg(),  # noqa: B008
  pivot: PivotCfg = PivotCfg(),  # noqa: B008
):
  """Args:
    sequence: moves in order (see the module docstring).
    output_file: CSV to write.
    stance_file: retargeted CSV to copy the stance and guard from.
    stance_time: time of the stance frame in stance_file [s].
    stance_fps: frame rate of stance_file.
    stance_scale: scales the distance between the feet of the copied stance
      (the boxing take's 60 cm stance is long for Wato's legs; 0.75 = 45 cm).
    crouch: how much lower than the copied stance the hips stay [m].
    hold: stance pause at the start, end and for each H [s].
    fps: output frame rate.
    max_cap_use: slow a move down until no leg joint exceeds this fraction of
      its motor speed cap (headroom for the policy to balance).
    step / duck / deep_duck / slip / weave / pivot: per-move settings.
  """
  robot = Wato()
  stance = make_stance(robot, stance_file, stance_time, stance_fps, stance_scale, crouch)
  ducks = {"D": duck, "DD": deep_duck}
  base_qpos = robot.d.qpos.copy()

  tempo = [1.0] * len(sequence)
  for attempt in range(8):
    robot.d.qpos[:] = base_qpos
    robot.forward()
    tl = build(stance, sequence, tempo, fps, hold, step, ducks, slip, weave, pivot)
    out, worst = solve_timeline(robot, stance, tl)
    use, caps = speed_use(out, fps)
    per_move = [use[a:b].max() if b > a else 0.0 for a, b in tl.spans]
    over = [i for i, u in enumerate(per_move) if u > max_cap_use]
    if not over:
      break
    for i in over:
      tempo[i] *= per_move[i] / max_cap_use * 1.05
    print(f"pass {attempt + 1}: slowing {', '.join(f'{sequence[i]}#{i + 1}' for i in over)}")

  os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
  np.savetxt(output_file, out, delimiter=",")
  print(f"\nWrote {len(out)} frames ({len(out) / fps:.1f} s at {fps:g} fps) to {output_file}")
  print(f"max IK residual: {worst:.2e}")
  if worst > 1e-3:
    print("WARNING: some targets were out of reach; reduce the move sizes")

  # per-move report: duration, slow-down, peak speed use, hip drop, turn
  torso_z, head_yaw = [], []
  for row in out:
    robot.set_csv_row(row)
    torso_z.append(robot.d.xpos[robot.torso][2])
  torso_z = np.array(torso_z)
  print(f"\n{'#':>3s} {'move':4s} {'time':>6s} {'slowed':>7s} {'peak cap use':>13s}  {'torso drop':>10s}")
  for i, ((a, b), mv) in enumerate(zip(tl.spans, sequence)):
    j = int(np.argmax(use[a:b].max(0))) if b > a else 0
    print(
      f"{i + 1:3d} {mv:4s} {(b - a) / fps:5.2f}s {tempo[i]:6.2f}x {per_move[i]:6.0%} {LEG_JOINTS[j][:13]:>13s}"
      f"  {torso_z[0] - torso_z[a:b].min():7.2f} m"
    )
  if max(per_move) > max_cap_use:
    print(f"WARNING: still above {max_cap_use:.0%} of a speed cap after 8 passes")


if __name__ == "__main__":
  tyro.cli(main)
