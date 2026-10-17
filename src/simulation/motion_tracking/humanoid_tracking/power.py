"""Stage 2, punch power: tracking plus a real target and power / balance terms
around each punch.

The training clip's .yaml (scripts/build_move_library.py) gives every punch
its impact frame, glove and drive direction. `PowerMotionCommand` turns that
into per-frame tables and, each step, puts every env's target ball where the
current punch is aimed (relative to where the robot actually is, like the
tracking targets), holds it to the punch's line (a rail: straight forward,
hook across, uppercut up) and parks it between punches. The ball's mass is
sampled per env at startup, log-uniform 2-40 kg (env_cfg.py): light balls
reward speed, heavy ones the body behind the punch, and push back.

Windows, in frames from the impact frame k (50 Hz):
  pre       k-7 .. k-2   hips lead, weight transfer
  impact    k-3 .. k+3   glove on target (square on the ball's line)
  momentum  k   .. k+10  ball momentum along the line (mass x speed); the
                         punch is scored for the streak when this closes
  retract   k+4 .. k+25  punching hand moving back towards its guard
  post      k+1 .. k+25  steady after impact
  loose     k-15 .. k+25 pelvis may over-rotate the punch's way, feet tracked
                         at half weight, feet may be 35 cm off in height,
                         no penalty for the punching arm / pelvis turning
                         faster than the clip in the punch's direction
  free      k-15 .. k+15 tracking of the pelvis, legs and punching arm
                         tolerates FREE_LOOSEN x the error (the policy picks
                         its own mechanics; power and balance decide), easing
                         back to normal by k+FREE_BACK so it recovers to the
                         clip's guard; the guard arm stays strict; the
                         punching hand may be 50 cm off in height
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import yaml
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

from humanoid_tracking.target_ball import GLOVE_RADIUS, TARGET_RADIUS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

PRE = (-7, -2)
IMPACT = (-3, 3)
MOMENTUM = (0, 10)
POST = (1, 25)
LOOSE = (-15, 25)

# Footwork: during step forward / back / left / right the policy is rewarded for
# getting its pelvis ahead of the clip's along the step's direction
# (step_progress: 1.0 at STEP_EXTRA_MAX ahead), the legs and pelvis are tracked with STEP_LOOSEN x the
# tolerance (upper body and guard stay strict), velocity tracking does not
# penalise moving faster along the step, and feet may be 35 cm off in height.
# When a step ends, the extra distance it covered along its direction (up to
# STEP_EXTRA_MAX) is banked: the reference is shifted by it, so tracking carries
# on from where the robot is instead of pulling it back. Balance: the same
# always-on terms, plus steady_after_step for STEP_POST frames after landing.
STEP_MOVES = ("step_forward", "step_back", "step_left", "step_right")
STEP_LOOSEN = 2.5
STEP_EXTRA_MAX = 0.30  # m banked per step
STEP_POST = 25  # frames
FREE = (-15, 15)
FREE_BACK = 30  # frames after impact where tracking is strict again
FREE_LOOSEN = 3.0  # tracking error scale (std x this) in the free window

TARGET_JITTER = 0.0  # m: none; the policy cannot see the target, so jitter would only add noise

# Momentum is scored against each punch's own baseline: the momentum the
# tracking policy (before power training) gives a target of that mass with that
# punch, from <motion>_baseline.yaml (scripts/measure_power_baseline.py). 1.0 =
# as good as before, so a jab improving by half counts as much as a rear hook
# improving by half. Without the file, every punch uses MOMENTUM_REF kg m/s.
#
# Streak: a punch is "good" when it lands square (glove-on-target >= GOOD_ACCURACY
# at its best moment) and gives the ball at least GOOD_RATIO x its baseline
# momentum along its line. Each good punch raises the env's streak by 1, any other punch resets it,
# and so does an episode reset. Every good punch pays the streak_bonus term (the
# streak, so n in a row pay 1+2+..+n) and the momentum term is scaled by
# 1 + STREAK_GAIN x streak, capped at STREAK_CAP. Glove speed is not rewarded.
GOOD_ACCURACY = 0.5
GOOD_RATIO = 1.0  # x the punch's baseline momentum
MOMENTUM_REF = 3.0  # kg m/s: baseline when there is no baseline file
MOMENTUM_CAP = 3.0  # momentum reward cap, in baselines
RETRACT = (4, 25)
RETRACT_SPEED = 1.5  # m/s towards the guard that scores 1.0
STREAK_GAIN = 0.5
STREAK_CAP = 3.0
ACCURACY_STD = 0.05  # m: accuracy halves ~4 cm off the ball's line (or short of contact)
BALL_PARK = (0.0, 0.0, -3.0)  # relative to the env origin; touches nothing
FOOT_WEIGHT_LOOSE = 0.5
OVER_ROTATION = 0.52  # rad (30 deg) of extra pelvis yaw forgiven the punch's way
FOOT_DOWN_Z = 0.09  # m, foot body origin below this = on the ground (stance ~0.05)

WRISTS = ("Mirrorlink6__1__1", "link6__1__1")  # left, right
ELBOWS = ("Mirrorlink4__1__1", "link4__1__1")
SHOULDERS = ("Mirrorlink2__1__1", "link2__1__1")
FEET = ("left_foot_1", "right_foot_1")
LEGS = ("left_hip_r_1", "left_calf_1", "left_foot_1", "right_hip_r_1", "right_calf_1", "right_foot_1")


def _yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_z(a: torch.Tensor) -> torch.Tensor:
  return torch.stack([torch.cos(a / 2), torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2)], -1)


def _in(o: torch.Tensor, window: tuple[int, int], active: torch.Tensor) -> torch.Tensor:
  return active & (o >= window[0]) & (o <= window[1])


class PowerMotionCommand(MotionCommand):
  cfg: PowerMotionCommandCfg

  def __init__(self, cfg: PowerMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    names = list(cfg.body_names)
    self.wrist_idx = torch.tensor([names.index(b) for b in WRISTS], device=self.device)
    self.elbow_idx = torch.tensor([names.index(b) for b in ELBOWS], device=self.device)
    self.foot_idx = torch.tensor([names.index(b) for b in FEET], device=self.device)
    self.leg_idx = torch.tensor([names.index(b) for b in LEGS], device=self.device)
    # glove sphere geoms (left, right), global ids into sim.data.geom_xpos
    local = self.robot.find_geoms(("glove_l", "glove_r"), preserve_order=True)[0]
    self.glove_geom = self.robot.indexing.geom_ids[local].long()
    # arm bodies per side (shoulder, elbow, wrist): the guard arm stays strict
    self.arm_idx = torch.tensor(
      [[names.index(b[s]) for b in (SHOULDERS, ELBOWS, WRISTS)] for s in range(2)], device=self.device
    )
    self._build_tables()
    n = self.num_envs
    self.ball_seg = torch.full((n,), -2, dtype=torch.long, device=self.device)
    self.ball_centre = torch.zeros(n, 3, device=self.device)
    self.ball_dir = torch.zeros(n, 3, device=self.device)
    self.ball_dir[:, 2] = 1.0
    self.ball_peak = torch.zeros(n, device=self.device)
    self.ref_offset = torch.zeros(n, 3, device=self.device)
    self.since_step = torch.full((n,), 10_000, dtype=torch.long, device=self.device)
    # per-punch bests (over the impact window) and the streak
    self.best_accuracy = torch.zeros(n, device=self.device)
    self.best_momentum = torch.zeros(n, device=self.device)
    self.streak = torch.zeros(n, device=self.device)
    self.just_good = torch.zeros(n, dtype=torch.bool, device=self.device)
    for k in ("power_ball_speed", "power_glove_speed", "power_hip_lead", "power_streak", "power_accuracy"):
      self.metrics[k] = torch.zeros(n, device=self.device)

  # -- per-frame tables from the clip's impact labels ---------------------------
  def _build_tables(self) -> None:
    meta = yaml.safe_load(Path(self.cfg.motion_file).with_suffix(".yaml").read_text())
    T, dev = self.motion.time_step_total, self.device
    self.seg = torch.full((T,), -1, dtype=torch.long, device=dev)
    self.seg_end = torch.zeros(T, dtype=torch.long, device=dev)
    self.impact = torch.zeros(T, dtype=torch.long, device=dev)
    self.hand = torch.zeros(T, dtype=torch.long, device=dev)
    self.direction = torch.zeros(T, 3, device=dev)
    self.aim = torch.zeros(T, 3, device=dev)
    self.v_ref = torch.ones(T, device=dev)
    self.turn = torch.zeros(T, device=dev)
    self.turn_ref = torch.ones(T, device=dev)
    yaw_rate = self.motion.body_ang_vel_w[:, self.motion_anchor_body_index, 2]
    # per-punch momentum baselines (kg m/s in log-mass bins)
    base_path = Path(self.cfg.motion_file).with_name(Path(self.cfg.motion_file).stem + "_baseline.yaml")
    base = yaml.safe_load(base_path.read_text()) if base_path.exists() else None
    moves = sorted({s["move"] for s in meta["segments"] if s.get("impact")})
    self.move_id = torch.zeros(T, dtype=torch.long, device=dev)
    if base is not None:
      self.base_log_mass = torch.tensor(base["mass_log_bin_centres"], device=dev)
      self.base_momentum = torch.tensor(
        [base["punches"].get(m, [MOMENTUM_REF] * len(self.base_log_mass)) for m in moves], device=dev
      )
      print(f"[power] momentum baselines from {base_path}")
    else:
      self.base_log_mass = torch.zeros(1, device=dev)
      self.base_momentum = torch.full((len(moves), 1), MOMENTUM_REF, device=dev)
      print(f"[power] no {base_path}: every punch's momentum baseline is {MOMENTUM_REF} kg m/s")
    if len(moves) == 0:  # a clip without punches (footwork): one placeholder row
      self.base_momentum = torch.full((1, len(self.base_log_mass)), MOMENTUM_REF, device=dev)
    # footwork steps: travel direction (clip frame, unit xy)
    self.stepping = torch.zeros(T, dtype=torch.bool, device=dev)
    self.step_dir = torch.zeros(T, 3, device=dev)
    self.step_end = torch.zeros(T, dtype=torch.long, device=dev)
    root = self.motion.body_pos_w[:, self.motion_anchor_body_index]
    for s in meta["segments"]:
      if s["move"] not in STEP_MOVES:
        continue
      a, b = s["start"], min(s["end"], T - 1)
      d = root[b] - root[a]
      d[2] = 0.0
      dist = float(torch.norm(d))
      if dist < 1e-3:
        continue
      self.stepping[a : b + 1] = True
      self.step_dir[a : b + 1] = d / dist
      self.step_end[a : b + 1] = b
    for i, s in enumerate(meta["segments"]):
      imp = s.get("impact")
      if imp is None:
        continue
      a, b = s["start"], min(s["end"], T - 1)
      k = min(imp["frame"], T - 1)
      hand = 0 if imp["hand"] == "left" else 1
      d = torch.tensor(imp["direction"], dtype=torch.float32, device=dev)
      self.seg[a : b + 1] = i
      self.seg_end[a : b + 1] = b
      self.move_id[a : b + 1] = moves.index(s["move"])
      self.impact[a : b + 1] = k
      self.hand[a : b + 1] = hand
      self.direction[a : b + 1] = d
      self.aim[a : b + 1] = torch.tensor(imp["aim_pos"], dtype=torch.float32, device=dev)
      v = float(self.motion.body_lin_vel_w[k, self.wrist_idx[hand]] @ d)
      self.v_ref[a : b + 1] = max(v, 0.5)
      w = yaw_rate[max(k + PRE[0], 0) : k + PRE[1] + 1]
      self.turn[a : b + 1] = float(torch.sign(w.mean())) if len(w) else 0.0
      self.turn_ref[a : b + 1] = max(float(w.abs().max()) if len(w) else 0.0, 1.0)

  # -- per-env view of the current punch --------------------------------------
  @property
  def punching(self) -> torch.Tensor:
    return self.seg[self.time_steps] >= 0

  @property
  def offset(self) -> torch.Tensor:
    """Frames from the current punch's impact (meaningless outside punches)."""
    return self.time_steps - self.impact[self.time_steps]

  def window(self, w: tuple[int, int]) -> torch.Tensor:
    return _in(self.offset, w, self.punching)

  def loosen(self) -> torch.Tensor:
    """Per-env tracking std scale: FREE_LOOSEN in the free window, easing
    linearly back to 1 at k+FREE_BACK, 1 elsewhere."""
    o = self.offset.float()
    back = 1.0 + (FREE_LOOSEN - 1.0) * ((FREE_BACK - o) / (FREE_BACK - FREE[1])).clamp(0.0, 1.0)
    f = torch.where(o <= FREE[1], torch.full_like(o, FREE_LOOSEN), back)
    return torch.where(self.punching & (o >= FREE[0]), f, torch.ones_like(o))

  @property
  def in_step(self) -> torch.Tensor:
    return self.stepping[self.time_steps]

  def root_loosen(self) -> torch.Tensor:
    """Pelvis tracking std scale: loosen(), or STEP_LOOSEN during a step."""
    return torch.where(self.in_step, torch.full_like(self.loosen(), STEP_LOOSEN), self.loosen())

  def body_loosen(self) -> torch.Tensor:
    """Per-env, per-tracked-body std scale: loosen() for everything except the
    guard (non-punching) arm; during a step, STEP_LOOSEN for the legs only."""
    m = self.loosen()[:, None].repeat(1, len(self.cfg.body_names))
    guard = self.arm_idx[1 - self.hand[self.time_steps]]
    m.scatter_(1, guard, 1.0)
    legs = torch.zeros_like(m, dtype=torch.bool)
    legs[:, self.leg_idx] = True
    legs[:, self.motion_anchor_body_index] = True
    return torch.where(legs & self.in_step[:, None], torch.maximum(m, torch.full_like(m, STEP_LOOSEN)), m)

  @property
  def step_dir_w(self) -> torch.Tensor:
    """The current step's travel direction in the world, turned to the robot's heading."""
    return quat_apply(self._heading_delta(), self.step_dir[self.time_steps])

  # -- banked extra step distance: the reference shifted by it ------------------
  @property
  def body_pos_w(self) -> torch.Tensor:
    pos = MotionCommand.body_pos_w.fget(self)
    off = getattr(self, "ref_offset", None)
    return pos if off is None else pos + off[:, None, :]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    pos = MotionCommand.anchor_pos_w.fget(self)
    off = getattr(self, "ref_offset", None)
    return pos if off is None else pos + off

  def _resample_command(self, env_ids: torch.Tensor):
    if hasattr(self, "ref_offset"):
      self.ref_offset[env_ids] = 0.0
      self.since_step[env_ids] = 10_000
    super()._resample_command(env_ids)

  def _update_steps(self) -> None:
    """When a step ends, bank the extra distance it covered along its direction."""
    t = self.time_steps
    ending = self.in_step & (t == self.step_end[t])
    if torch.any(ending):
      d = self.step_dir_w
      ahead = ((self.robot_anchor_pos_w - self.anchor_pos_w) * d).sum(-1).clamp(0.0, STEP_EXTRA_MAX)
      self.ref_offset += torch.where(ending, ahead, torch.zeros_like(ahead))[:, None] * d
      self.ref_offset[:, 2] = 0.0
    self.since_step = torch.where(ending, torch.zeros_like(self.since_step), self.since_step + 1)

  @property
  def punch_dir(self) -> torch.Tensor:
    """Drive direction in the world, turned to the robot's heading."""
    return quat_apply(self._heading_delta(), self.direction[self.time_steps])

  def _heading_delta(self) -> torch.Tensor:
    return yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))

  # -- the target ball --------------------------------------------------------
  def reset(self, env_ids):
    out = super().reset(env_ids)
    if hasattr(self, "streak"):
      ids = slice(None) if env_ids is None else env_ids
      self.ref_offset[ids] = 0.0
      self.since_step[ids] = 10_000
      self.streak[ids] = 0.0
      self.best_accuracy[ids] = 0.0
      self.best_momentum[ids] = 0.0
    return out

  def _update_command(self, env_ids: torch.Tensor | None = None):
    super()._update_command(env_ids)
    self._update_steps()
    if "target" in self._env.scene.entities:
      self._update_streak()
      self._update_ball()

  def glove_pos(self) -> torch.Tensor:
    """Centre of the punching glove (current punch's hand)."""
    xpos = self._env.sim.data.geom_xpos[:, self.glove_geom]
    return xpos[torch.arange(self.num_envs, device=self.device), self.hand[self.time_steps]]

  def accuracy(self) -> torch.Tensor:
    """1 for a square hit: the glove on the line through the ball's centre along
    the punch direction, at (or through) contact; off the line or short of
    contact falls off with ACCURACY_STD."""
    rel = self.glove_pos() - self.ball_centre
    along = (rel * self.ball_dir).sum(-1)
    off_line = torch.norm(rel - along[:, None] * self.ball_dir, dim=-1)
    short = (-along - GLOVE_RADIUS - TARGET_RADIUS).clamp(min=0.0)
    return torch.exp(-(off_line**2 + short**2) / ACCURACY_STD**2)

  def streak_scale(self) -> torch.Tensor:
    return (1.0 + STREAK_GAIN * self.streak).clamp(max=STREAK_CAP)

  @property
  def target_mass(self) -> torch.Tensor:
    ball = self._env.scene["target"]
    return self._env.sim.model.body_mass[:, ball.indexing.body_ids[0]]

  def baseline_momentum(self) -> torch.Tensor:
    """This punch's baseline momentum for this env's target mass (kg m/s),
    interpolated in log mass between the baseline bins."""
    table = self.base_momentum[self.move_id[self.time_steps]]  # (n, bins)
    if table.shape[1] == 1:
      return table[:, 0]
    x = torch.log(self.target_mass).clamp(self.base_log_mass[0], self.base_log_mass[-1])
    i = torch.searchsorted(self.base_log_mass, x[:, None]).squeeze(-1).clamp(1, len(self.base_log_mass) - 1)
    x0, x1 = self.base_log_mass[i - 1], self.base_log_mass[i]
    rows = torch.arange(self.num_envs, device=self.device)
    w = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return table[rows, i - 1] * (1 - w) + table[rows, i] * w

  def momentum_ratio(self) -> torch.Tensor:
    """Ball momentum along the line / this punch's baseline (1.0 = as before)."""
    return self.ball_momentum_now() / self.baseline_momentum()

  def ball_momentum_now(self) -> torch.Tensor:
    """Ball momentum along its line from the last step (0 outside a held punch)."""
    speed = getattr(self, "ball_speed", torch.zeros(self.num_envs, device=self.device))
    return self.target_mass * speed

  def _update_streak(self) -> None:
    """Track each punch's best accuracy (impact window) and peak ball momentum
    (momentum window); when the momentum window closes, score it:
    good -> streak + 1, else streak = 0."""
    o = self.offset
    acc = self.accuracy()
    self.best_accuracy = torch.where(self.window(IMPACT), torch.maximum(self.best_accuracy, acc), self.best_accuracy)
    p = self.momentum_ratio()
    self.best_momentum = torch.where(self.window(MOMENTUM), torch.maximum(self.best_momentum, p), self.best_momentum)
    done = self.punching & (o == MOMENTUM[1] + 1)
    good = (self.best_accuracy >= GOOD_ACCURACY) & (self.best_momentum >= GOOD_RATIO)
    self.just_good = done & good
    self.streak = torch.where(done, torch.where(good, self.streak + 1.0, torch.zeros_like(self.streak)), self.streak)
    reset = done | ~self.punching
    self.best_accuracy = torch.where(reset, torch.zeros_like(self.best_accuracy), self.best_accuracy)
    self.best_momentum = torch.where(reset, torch.zeros_like(self.best_momentum), self.best_momentum)

  def guard_pos(self) -> torch.Tensor:
    """Where the punching wrist should end up: the clip's wrist at the end of
    this punch's segment (back in guard), relative to where the robot is."""
    end = self.seg_end[self.time_steps]
    wrist = self.wrist_idx[self.hand[self.time_steps]]
    ref = self.motion.body_pos_w[end, wrist] + self._env.scene.env_origins
    anchor = self.anchor_pos_w
    delta_pos = torch.cat([self.robot_anchor_pos_w[:, :2], anchor[:, 2:]], -1)
    return delta_pos + quat_apply(self._heading_delta(), ref - anchor)

  def _update_ball(self) -> None:
    ball = self._env.scene["target"]
    seg = self.seg[self.time_steps]
    origins = self._env.scene.env_origins

    # momentum along the line from the step that just ran (before moving the ball)
    vel = ball.data.root_link_lin_vel_w
    speed = (vel * self.ball_dir).sum(-1)
    held = (self.ball_seg >= 0) & (seg == self.ball_seg)
    self.ball_speed = torch.where(held, speed, torch.zeros_like(speed))
    self.ball_peak = torch.where(held, torch.maximum(self.ball_peak, speed), torch.zeros_like(speed))

    # new punch (or reset into one): place the ball at its aim point, at rest,
    # relative to where the robot is (as body_pos_relative_w is)
    new = seg != self.ball_seg
    if torch.any(new):
      ids = new.nonzero(as_tuple=False).squeeze(-1)
      delta = self._heading_delta()[ids]
      anchor = self.anchor_pos_w[ids]
      robot = self.robot_anchor_pos_w[ids]
      pos_ref = self.aim[self.time_steps[ids]] + origins[ids] + self.ref_offset[ids]
      d = quat_apply(delta, self.direction[self.time_steps[ids]])
      centre = torch.cat([robot[:, :2], anchor[:, 2:]], -1) + quat_apply(delta, pos_ref - anchor)
      centre = centre + d * (GLOVE_RADIUS + TARGET_RADIUS)
      centre = centre + (torch.rand_like(centre) * 2 - 1) * TARGET_JITTER
      park = origins[ids] + torch.tensor(BALL_PARK, device=self.device)
      in_punch = (seg[ids] >= 0)[:, None]
      self.ball_centre[ids] = torch.where(in_punch, centre, park)
      self.ball_dir[ids] = torch.where(in_punch, d, self.ball_dir[ids])
      self.ball_seg[ids] = seg[ids]

    # hold every ball on its line (rail), at rest when parked
    along = ((ball.data.root_link_pos_w - self.ball_centre) * self.ball_dir).sum(-1)
    along = torch.where(new | (seg < 0), torch.zeros_like(along), along)
    v = torch.where(new | (seg < 0), torch.zeros_like(speed), speed)
    state = torch.zeros(self.num_envs, 13, device=self.device)
    state[:, :3] = self.ball_centre + along[:, None] * self.ball_dir
    state[:, 3] = 1.0
    state[:, 7:10] = v[:, None] * self.ball_dir
    ball.write_root_state_to_sim(state)
    self._env.sim.forward()

  def _update_metrics(self):
    super()._update_metrics()
    if not hasattr(self, "seg"):
      return
    hit = self.window(MOMENTUM)
    self.metrics["power_ball_speed"] = torch.where(hit, self.momentum_ratio(), torch.zeros_like(hit, dtype=torch.float))
    self.metrics["power_glove_speed"] = glove_speed(self._env, self.cfg.command_name) * self.v_ref[self.time_steps]
    self.metrics["power_hip_lead"] = hips_lead(self._env, self.cfg.command_name)
    self.metrics["power_streak"] = self.streak.clone()
    self.metrics["power_accuracy"] = glove_on_target(self._env, self.cfg.command_name)


@dataclass(kw_only=True)
class PowerMotionCommandCfg(MotionCommandCfg):
  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> PowerMotionCommand:
    return PowerMotionCommand(self, env)


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> PowerMotionCommand:
  return cast(PowerMotionCommand, env.command_manager.get_term(command_name))


def _punch_body(cmd: PowerMotionCommand, idx: torch.Tensor) -> torch.Tensor:
  return idx[cmd.hand[cmd.time_steps]]


# -- power rewards (positive, only around punches) ------------------------------
def ball_momentum(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ball momentum (mass x speed) along the punch's line / this punch's
  baseline at this target mass (1.0 = as good as before power training),
  capped, from impact to k+10, times the streak scale."""
  cmd = _cmd(env, command_name)
  r = cmd.momentum_ratio().clamp(0.0, MOMENTUM_CAP) * cmd.streak_scale()
  return torch.where(cmd.window(MOMENTUM), r, torch.zeros_like(r))


def retract(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """After the hit, the punching hand moving back towards its guard
  (RETRACT_SPEED scores 1, capped at 1.5); nothing once within 3 cm of it."""
  cmd = _cmd(env, command_name)
  rows = torch.arange(cmd.num_envs, device=cmd.device)
  wrist = cmd.wrist_idx[cmd.hand[cmd.time_steps]]
  to_guard = cmd.guard_pos() - cmd.robot_body_pos_w[rows, wrist]
  dist = torch.norm(to_guard, dim=-1)
  u = to_guard / dist.clamp(min=1e-6)[:, None]
  r = ((cmd.robot_body_lin_vel_w[rows, wrist] * u).sum(-1) / RETRACT_SPEED).clamp(0.0, 1.5)
  return torch.where(cmd.window(RETRACT) & (dist > 0.03), r, torch.zeros_like(r))


def glove_speed(env: ManagerBasedRlEnv, command_name: str, cap: float = 1.5) -> torch.Tensor:
  """Punching wrist speed along the punch's line / the clip's, around impact.
  Logged as a metric only (not a reward)."""
  cmd = _cmd(env, command_name)
  wrist = _punch_body(cmd, cmd.wrist_idx)
  vel = cmd.robot_body_lin_vel_w[torch.arange(cmd.num_envs, device=cmd.device), wrist]
  r = ((vel * cmd.punch_dir).sum(-1) / cmd.v_ref[cmd.time_steps]).clamp(0.0, cap)
  return torch.where(cmd.window(IMPACT), r, torch.zeros_like(r))


def glove_on_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Square hit around impact (PowerMotionCommand.accuracy): the glove driving
  along the line through the ball's centre. A glancing touch scores little, so
  the policy may change its mechanics but not where, or how squarely, the
  punch lands."""
  cmd = _cmd(env, command_name)
  r = cmd.accuracy()
  return torch.where(cmd.window(IMPACT), r, torch.zeros_like(r))


def streak_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Paid on the step a good punch is scored: the streak after it (1, 2, 3, ..),
  so n good punches in a row pay n(n+1)/2."""
  cmd = _cmd(env, command_name)
  return torch.where(cmd.just_good, cmd.streak, torch.zeros_like(cmd.streak))


def step_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """During a step: how far the pelvis is ahead of the clip's (shifted by the
  distance already banked) along the step's direction, / STEP_EXTRA_MAX:
  0 = keeping pace with the clip, 1 = STEP_EXTRA_MAX further. The distance it
  ends the step with is banked (PowerMotionCommand._update_steps)."""
  cmd = _cmd(env, command_name)
  ahead = ((cmd.robot_anchor_pos_w - cmd.anchor_pos_w) * cmd.step_dir_w).sum(-1)
  r = (ahead / STEP_EXTRA_MAX).clamp(0.0, 1.0)
  return torch.where(cmd.in_step, r, torch.zeros_like(r))


def steady_after_step(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Pelvis roll/pitch rate differing from the clip's for STEP_POST frames after a step lands."""
  cmd = _cmd(env, command_name)
  err = torch.sum(torch.square(cmd.robot_anchor_ang_vel_w[:, :2] - cmd.anchor_ang_vel_w[:, :2]), dim=-1)
  return torch.where(cmd.since_step <= STEP_POST, err, torch.zeros_like(err))


def hips_lead(env: ManagerBasedRlEnv, command_name: str, cap: float = 1.5) -> torch.Tensor:
  """Pelvis turning the punch's way before impact / the clip's peak turn rate."""
  cmd = _cmd(env, command_name)
  rate = cmd.robot_anchor_ang_vel_w[:, 2] * cmd.turn[cmd.time_steps]
  r = (rate / cmd.turn_ref[cmd.time_steps]).clamp(0.0, cap)
  return torch.where(cmd.window(PRE) & (cmd.turn[cmd.time_steps] != 0), r, torch.zeros_like(r))


def weight_transfer(env: ManagerBasedRlEnv, command_name: str, ref_speed: float = 0.3, cap: float = 1.5) -> torch.Tensor:
  """Centre of mass moving along the punch's line just before impact."""
  cmd = _cmd(env, command_name)
  com_vel = cmd.robot.data.root_com_vel_w[:, :3]
  r = ((com_vel * cmd.punch_dir).sum(-1) / ref_speed).clamp(0.0, cap)
  return torch.where(_in(cmd.offset, (PRE[0], 0), cmd.punching), r, torch.zeros_like(r))


# -- balance (penalties are returned positive; weights are negative) ------------
def _feet_down(z: torch.Tensor) -> torch.Tensor:
  return z < FOOT_DOWN_Z


def feet_off_ground(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Feet in the air during a punch where the clip has them down."""
  cmd = _cmd(env, command_name)
  ref = _feet_down(cmd.body_pos_relative_w[:, cmd.foot_idx, 2])
  up = ~_feet_down(cmd.robot_body_pos_w[:, cmd.foot_idx, 2])
  n = (ref & up).float().sum(-1)
  return torch.where(cmd.window(LOOSE), n, torch.zeros_like(n))


def feet_slide(env: ManagerBasedRlEnv, command_name: str, free_speed: float = 0.1) -> torch.Tensor:
  """Planted feet sliding faster than `free_speed` (pivoting in place is fine)."""
  cmd = _cmd(env, command_name)
  down = _feet_down(cmd.robot_body_pos_w[:, cmd.foot_idx, 2])
  speed = torch.norm(cmd.robot_body_lin_vel_w[:, cmd.foot_idx, :2], dim=-1)
  s = ((speed - free_speed).clamp(min=0.0) * down).sum(-1)
  return torch.where(cmd.window(LOOSE), s, torch.zeros_like(s))


def tilt_excess(env: ManagerBasedRlEnv, command_name: str, allow: float = 0.17) -> torch.Tensor:
  """Pelvis roll/pitch beyond the clip's by more than `allow` rad (10 deg). Always on."""
  cmd = _cmd(env, command_name)
  up = torch.tensor([0.0, 0.0, 1.0], device=cmd.device).expand(cmd.num_envs, 3)

  def tilt(q):
    return torch.acos(quat_apply(q, up)[:, 2].clamp(-1.0, 1.0))

  return (tilt(cmd.robot_anchor_quat_w) - tilt(cmd.anchor_quat_w) - allow).clamp(min=0.0)


def com_over_feet(env: ManagerBasedRlEnv, command_name: str, std: float = 0.05) -> torch.Tensor:
  """Centre of mass (horizontal) on the segment between the feet. Always on."""
  cmd = _cmd(env, command_name)
  com = cmd.robot.data.data.subtree_com[:, cmd.robot.indexing.root_body_id, :2]
  a, b = cmd.robot_body_pos_w[:, cmd.foot_idx[0], :2], cmd.robot_body_pos_w[:, cmd.foot_idx[1], :2]
  ab = b - a
  t = (((com - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-6)).clamp(0.0, 1.0)
  dist = torch.norm(com - (a + t[:, None] * ab), dim=-1)
  return torch.exp(-((dist / std) ** 2))


def steady_after_impact(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Pelvis roll/pitch rate differing from the clip's after impact."""
  cmd = _cmd(env, command_name)
  err = torch.sum(torch.square(cmd.robot_anchor_ang_vel_w[:, :2] - cmd.anchor_ang_vel_w[:, :2]), dim=-1)
  return torch.where(cmd.window(POST), err, torch.zeros_like(err))


# -- tracking terms with the punch-window allowances ---------------------------
def root_ori_loose(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """motion_global_anchor_orientation_error_exp, but in a punch's loose window
  up to OVER_ROTATION of extra yaw the punch's way is not an error."""
  cmd = _cmd(env, command_name)
  yaw_err = torch.atan2(
    torch.sin(_yaw(cmd.robot_anchor_quat_w) - _yaw(cmd.anchor_quat_w)),
    torch.cos(_yaw(cmd.robot_anchor_quat_w) - _yaw(cmd.anchor_quat_w)),
  )
  s = cmd.turn[cmd.time_steps]
  forgive = (yaw_err * s).clamp(0.0, OVER_ROTATION) * s
  forgive = torch.where(cmd.window(LOOSE), forgive, torch.zeros_like(forgive))
  robot_q = quat_mul(_quat_z(-forgive), cmd.robot_anchor_quat_w)
  error = quat_error_magnitude(cmd.anchor_quat_w, robot_q) ** 2
  return torch.exp(-error / (std * cmd.loosen()) ** 2)


def root_pos_free(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """motion_global_anchor_position_error_exp with the free window's looser std."""
  cmd = _cmd(env, command_name)
  error = torch.sum(torch.square(cmd.anchor_pos_w - cmd.robot_anchor_pos_w), dim=-1)
  return torch.exp(-error / (std * cmd.root_loosen()) ** 2)


def _body_weights(cmd: PowerMotionCommand) -> torch.Tensor:
  w = torch.ones(cmd.num_envs, len(cmd.cfg.body_names), device=cmd.device)
  loose = cmd.window(LOOSE)
  w[:, cmd.foot_idx] = torch.where(loose, FOOT_WEIGHT_LOOSE, 1.0)[:, None]
  return w / w.sum(-1, keepdim=True)


def body_pos_feet_loose(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  cmd = _cmd(env, command_name)
  err = torch.sum(torch.square(cmd.body_pos_relative_w - cmd.robot_body_pos_w), dim=-1)
  err = err / cmd.body_loosen() ** 2
  return torch.exp(-(err * _body_weights(cmd)).sum(-1) / std**2)


def body_ori_feet_loose(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  cmd = _cmd(env, command_name)
  err = quat_error_magnitude(cmd.body_quat_relative_w, cmd.robot_body_quat_w) ** 2
  err = err / cmd.body_loosen() ** 2
  return torch.exp(-(err * _body_weights(cmd)).sum(-1) / std**2)


def body_lin_vel_no_speed_cap(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Body linear velocity tracking, except the punching wrist and elbow are not
  penalised for going faster than the clip along the punch's line (loose window)."""
  cmd = _cmd(env, command_name)
  robot = cmd.robot_body_lin_vel_w.clone()
  ref = cmd.body_lin_vel_w
  rows = torch.arange(cmd.num_envs, device=cmd.device)
  d = cmd.punch_dir
  loose = cmd.window(LOOSE)
  for idx in (cmd.wrist_idx, cmd.elbow_idx):
    b = _punch_body(cmd, idx)
    extra = (((robot[rows, b] - ref[rows, b]) * d).sum(-1)).clamp(min=0.0) * loose
    robot[rows, b] -= extra[:, None] * d
  # stepping: moving faster than the clip along the step's direction is fine
  sd = cmd.step_dir_w[:, None, :]
  extra = (((robot - ref) * sd).sum(-1)).clamp(min=0.0) * cmd.in_step[:, None]
  robot = robot - extra[..., None] * sd
  err = torch.sum(torch.square(ref - robot), dim=-1) / cmd.body_loosen() ** 2
  return torch.exp(-err.mean(-1) / std**2)


def body_ang_vel_no_turn_cap(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Body angular velocity tracking, except turning (yaw rate) faster than the
  clip the punch's way is not penalised (loose window)."""
  cmd = _cmd(env, command_name)
  robot = cmd.robot_body_ang_vel_w.clone()
  ref = cmd.body_ang_vel_w
  s = cmd.turn[cmd.time_steps][:, None]
  extra = ((robot[..., 2] - ref[..., 2]) * s).clamp(min=0.0) * cmd.window(LOOSE)[:, None]
  robot[..., 2] -= extra * s
  err = torch.sum(torch.square(ref - robot), dim=-1) / cmd.body_loosen() ** 2
  return torch.exp(-err.mean(-1) / std**2)


# -- termination ----------------------------------------------------------------
def bad_body_z_feet_loose(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  feet_threshold_loose: float,
  body_names: tuple[str, ...],
  punch_hand_threshold_free: float = 0.5,
) -> torch.Tensor:
  """bad_motion_body_pos_z_only on `body_names`, with the feet allowed
  `feet_threshold_loose` in a punch's loose window and the punching wrist
  `punch_hand_threshold_free` in its free window."""
  cmd = _cmd(env, command_name)
  names = list(cmd.cfg.body_names)
  idx = torch.tensor([names.index(b) for b in body_names], device=cmd.device)
  err = torch.abs(cmd.body_pos_relative_w[:, idx, 2] - cmd.robot_body_pos_w[:, idx, 2])
  limit = torch.full_like(err, threshold)
  is_foot = torch.tensor([b in FEET for b in body_names], device=cmd.device)
  loose = (cmd.window(LOOSE) | cmd.in_step)[:, None] & is_foot[None]
  limit = torch.where(loose, torch.full_like(err, feet_threshold_loose), limit)
  wrist_pos = [i for i, b in enumerate(body_names) if b in WRISTS]
  if wrist_pos:
    side = torch.tensor([WRISTS.index(body_names[i]) for i in wrist_pos], device=cmd.device)
    is_punch = side[None] == cmd.hand[cmd.time_steps][:, None]
    free = cmd.window(FREE)[:, None] & is_punch
    cols = torch.tensor(wrist_pos, device=cmd.device)
    limit[:, cols] = torch.where(free, torch.full_like(limit[:, cols], punch_hand_threshold_free), limit[:, cols])
  return torch.any(err > limit, dim=-1)


class ramp_from_run_start:
  """mjlab's reward_curriculum, with stage steps counted from this run's first
  step. A resumed run restores common_step_counter (16000 iterations = 384k
  steps), which would put every stage in the past and skip the ramp."""

  JUMP = 1000  # a counter change bigger than this between calls is a restore, not training

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    from mjlab.envs.mdp.curriculums import reward_curriculum

    self._inner = reward_curriculum(cfg, env)
    self._start = int(env.common_step_counter)
    self._last = self._start

  def __call__(self, env: ManagerBasedRlEnv, env_ids, reward_name: str, stages: list) -> dict:
    from mjlab.envs.mdp.curriculums import _apply_stages

    now = int(env.common_step_counter)
    if abs(now - self._last) > self.JUMP:  # checkpoint restore: keep the run-relative count
      self._start += now - self._last
    self._last = now
    return _apply_stages(self._inner._term_cfg, now - self._start, stages)
