"""Generate boxing footwork (step-drag) for the Wato robot as a retargeted CSV.

The stance and guard are copied from one frame of a retargeted clip (default:
the boxing take at 8.2 s, orthodox: left foot forward, rear heel up, torso
bladed). Each step follows the standard step-drag rules:

  * the foot nearest the direction of travel moves first, the other pushes
    (forward: lead foot first; back: rear foot first; left: left foot first;
    right: right foot first)
  * the second foot follows the same distance, so stance length and width are
    restored after every step; feet never cross or come together
  * weight goes onto the pushing foot before the first foot lifts, the push
    carries the body into the step, the weight goes onto the landed foot
    before the second foot lifts, then back to a neutral stance
  * feet glide low; hips stay low at one height (no bobbing); guard and torso
    angle stay fixed; no stance switch

Foot and centre-of-mass paths are minimum-jerk curves; leg joint angles come
from whole-body IK on the Wato MJCF (feet pose, CoM over the support foot,
pelvis height and orientation). The CoM follows one smooth curve through
the weight-shift points, so the body flows through each step. The defaults keep every leg joint under the
motor speed caps (the AKH70 hip/knee cap of 3.67 rad/s is the tight one); the
script prints the check. Output: CSV at --fps with the usual
layout (root pos, root quat xyzw, 28 joints), opponent along world +x.

  uv run scripts/generate_footwork.py --sequence F F F B B B L L L R R R \
      --output-file data/motions/footwork.csv
  uv run scripts/csv_to_npz.py --input-file data/motions/footwork.csv \
      --input-fps 50 --output-file data/motions/footwork.npz --video True
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

# direction -> (unit step in the stance frame, foot that moves first)
# stance frame: x = towards the opponent, y = left
MOVES = {
  "F": (np.array([1.0, 0.0]), LEAD),
  "B": (np.array([-1.0, 0.0]), REAR),
  "L": (np.array([0.0, 1.0]), LEAD),
  "R": (np.array([0.0, -1.0]), REAR),
}


@dataclass
class Timing:
  shift: float = 0.35
  """weight onto the pushing foot [s]"""
  swing: float = 0.35
  """first foot steps [s]"""
  transfer: float = 0.45
  """weight onto the landed foot [s]"""
  drag: float = 0.35
  """second foot follows [s]"""
  reset: float = 0.3
  """weight back to neutral stance [s]"""


def min_jerk(s: np.ndarray | float) -> np.ndarray | float:
  s = np.clip(s, 0.0, 1.0)
  return s**3 * (10 - 15 * s + 6 * s**2)


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
    self.sole = {f: _sole_box(spec, f)[1] for f in (LEAD, REAR)}
    # velocity dofs the IK may move: free joint + leg joints
    leg_dofs = [self.m.jnt_dofadr[self.m.joint(j).id] for j in LEG_JOINTS]
    self.dofs = np.array(list(range(6)) + leg_dofs)
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

  def solve(self, foot_pos: dict, foot_quat: dict, com_xy, base_z, base_quat, iters=50, tol=1e-6) -> float:
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


def main(
  sequence: list[Literal["F", "B", "L", "R"]] = ["F", "F", "F", "B", "B", "B", "L", "L", "L", "R", "R", "R"],  # noqa: B006
  output_file: str = "data/motions/footwork.csv",
  stance_file: str = "data/motions/boxing.csv",
  stance_time: float = 8.2,
  stance_fps: float = 120.0,
  stance_scale: float = 0.75,
  step_forward: float = 0.10,
  step_side: float = 0.08,
  weight_shift: float = 0.6,
  push: float = 0.5,
  crouch: float = 0.06,
  lift: float = 0.03,
  hold: float = 1.0,
  fps: float = 50.0,
  timing: Timing = Timing(),  # noqa: B008
):
  """Args:
    sequence: steps in order: F(orward) B(ack) L(eft) R(ight).
    output_file: CSV to write.
    stance_file: retargeted CSV to copy the stance and guard from.
    stance_time: time of the stance frame in stance_file [s].
    stance_fps: frame rate of stance_file.
    stance_scale: scales the distance between the feet of the copied stance
      (the boxing take's 60 cm stance is long for Wato's legs; 0.75 = 45 cm).
    step_forward: forward/back step length [m].
    step_side: sideways step length [m].
    weight_shift: how far the CoM moves from neutral towards the support
      foot's sole centre before a foot lifts (0 = none, 1 = fully over it).
    push: fraction of the step the CoM travels while the first foot is in the
      air (the pushing foot drives the body into the step).
    crouch: how much lower than the copied stance the hips stay throughout [m];
      bent knees keep the straightening leg in reach when the weight shifts.
    lift: swing foot clearance [m] (feet glide low).
    hold: pause in stance at the start, end and on every change of direction [s].
    fps: output frame rate.
    timing: phase durations of one step.
  """
  robot = Wato()
  rows = np.loadtxt(stance_file, delimiter=",")
  robot.set_csv_row(rows[int(round(stance_time * stance_fps))])

  # stance frame: origin between the feet on the ground, x towards the
  # opponent (world +x in the retargeted clips), y left
  feet0 = {f: robot.foot_pose(f) for f in (LEAD, REAR)}
  origin = np.array([*(feet0[LEAD][0][:2] + feet0[REAR][0][:2]) / 2, 0.0])
  robot.d.qpos[:3] -= origin
  robot.forward()
  base_z, base_quat = robot.d.xpos[robot.base][2], robot.d.xquat[robot.base].copy()
  feet_quat = {f: q for f, (p, q) in feet0.items()}
  clip_feet = {f: p - origin for f, (p, q) in feet0.items()}
  clip_com = robot.com()[:2]

  # shorten the stance about its centre and sink into it, in small IK steps
  # from the clip pose
  feet_pos = {f: np.array([*(stance_scale * p[:2]), p[2]]) for f, p in clip_feet.items()}
  com_neutral = stance_scale * clip_com
  stance_z = base_z - crouch
  for k in range(1, 21):
    s = k / 20
    robot.solve(
      {f: clip_feet[f] + s * (feet_pos[f] - clip_feet[f]) for f in feet_pos},
      feet_quat,
      clip_com + s * (com_neutral - clip_com),
      base_z - s * crouch,
      base_quat,
    )
  sole_off = {f: robot.sole_xy(f) - feet_pos[f][:2] for f in (LEAD, REAR)}

  # Feet: still, or a min-jerk glide while stepping. CoM: one smooth curve
  # through the weight-shift waypoints, stopping only in the holds, so the
  # body flows through a step instead of halting between its phases.
  dt = 1.0 / fps
  feet_frames: list[dict] = []
  waypoints: list[tuple[int, np.ndarray, bool]] = []  # (frame, CoM xy, stop there)

  def feet_phase(duration, feet_a, feet_b, swing_foot=None):
    n = max(1, int(round(duration * fps)))
    for k in range(1, n + 1):
      s = min_jerk(k / n)
      feet = {f: feet_a[f] + s * (feet_b[f] - feet_a[f]) for f in feet_a}
      if swing_foot is not None:
        u = k / n
        feet[swing_foot] = feet[swing_foot] + np.array([0, 0, lift * 16 * u**2 * (1 - u) ** 2])
      feet_frames.append(feet)

  def waypoint(com, stop=False):
    waypoints.append((len(feet_frames), com, stop))

  def neutral_at(feet):
    # neutral CoM moves with the stance (both feet moved by the same offset)
    shift = (feet[LEAD][:2] + feet[REAR][:2] - feet_pos[LEAD][:2] - feet_pos[REAR][:2]) / 2
    return com_neutral + shift

  def over(foot, feet):
    n = neutral_at(feet)
    return n + weight_shift * (feet[foot][:2] + sole_off[foot] - n)

  def hold_stance(feet):
    waypoint(neutral_at(feet), stop=True)
    feet_phase(hold, feet, feet)
    waypoint(neutral_at(feet), stop=True)

  feet = {f: p.copy() for f, p in feet_pos.items()}
  hold_stance(feet)
  prev = None
  for move in sequence:
    if prev is not None and move != prev:
      hold_stance(feet)
    prev = move
    direction, first = MOVES[move]
    second = REAR if first == LEAD else LEAD
    step = np.array([*(direction * (step_forward if direction[0] else step_side)), 0.0])

    # 1. load the pushing foot
    feet_phase(timing.shift, feet, feet)
    waypoint(over(second, feet))
    # 2. first foot glides out, the push carrying the body with it
    landed = dict(feet, **{first: feet[first] + step})
    feet_phase(timing.swing, feet, landed, swing_foot=first)
    waypoint(over(second, feet) + push * step[:2])
    feet = landed
    # 3. weight onto the landed foot
    feet_phase(timing.transfer, feet, feet)
    waypoint(over(first, feet))
    # 4. second foot follows the same distance: stance restored
    closed = dict(feet, **{second: feet[second] + step})
    feet_phase(timing.drag, feet, closed, swing_foot=second)
    waypoint(over(first, feet))
    feet = closed
    # 5. back to a balanced stance
    feet_phase(timing.reset, feet, feet)
    waypoint(neutral_at(feet))
  hold_stance(feet)

  com_frames = smooth_through(waypoints, len(feet_frames))

  out, worst = [], 0.0
  for feet_f, com_f in zip(feet_frames, com_frames):
    err = robot.solve(feet_f, feet_quat, com_f, stance_z, base_quat)
    worst = max(worst, err)
    out.append(robot.csv_row())
  out = np.array(out)

  os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
  np.savetxt(output_file, out, delimiter=",")
  print(f"Wrote {len(out)} frames ({len(out) * dt:.1f} s at {fps:g} fps) to {output_file}")
  print(f"max IK residual: {worst:.2e}")
  if worst > 1e-3:
    print("WARNING: some targets were out of reach; lower --weight-shift / step lengths or raise --crouch")

  # feasibility against the motor speed caps and joint limits
  vel = np.abs(np.gradient(out[:, 7:], dt, axis=0))
  print(f"{'joint':28s} {'max |vel|':>9s} {'cap':>6s}")
  for i, name in enumerate(LEG_JOINTS):
    cap = next(a.velocity_limit for a in ACTUATORS if any(re.fullmatch(p, name) for p in a.target_names_expr))
    lo, hi = robot.leg_range[i]
    q = out[:, 7 + i]
    flags = (" OVER CAP" if vel[:, i].max() > cap else "") + (
      " AT LIMIT" if (q.min() <= lo + 1e-3 or q.max() >= hi - 1e-3) else ""
    )
    print(f"{name:28s} {vel[:, i].max():9.2f} {cap:6.2f}{flags}")


if __name__ == "__main__":
  tyro.cli(main)
