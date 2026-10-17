"""Move-by-move teleop for the tracking policy.

A `MoveScheduler` plays clips from the move library (scripts/build_move_library.py)
one after another: queued moves in order, the guard clip in a loop when the
queue is empty. Every clip is recorded from the same stance at the origin, so
each new clip is re-anchored (yaw + xy) onto where the previous one ended;
otherwise a step forward followed by a jab would pull the robot back to where
the jab was recorded.

`TeleopMotionCommand` is mjlab's MotionCommand with a one-frame motion that the
scheduler rewrites every step. The policy, the observations and the ghost see
the same reference terms as in training (current frame only), so a policy
trained on a single clip can be driven without changes.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

from humanoid_tracking.target_ball import TARGET_MASS, TARGET_RADIUS, TARGET_RGBA

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

GUARD = "guard"
MAX_QUEUE = 4


def _yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_z(angle: torch.Tensor) -> torch.Tensor:
  half = angle / 2
  zero = torch.zeros_like(half)
  return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], -1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    -1,
  )


def _rot_z(v: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
  x, y = v[..., 0], v[..., 1]
  return torch.stack([c * x - s * y, s * x + c * y, v[..., 2]], -1)


@dataclass
class Clip:
  joint_pos: torch.Tensor  # (T, J)
  joint_vel: torch.Tensor  # (T, J)
  body_pos_w: torch.Tensor  # (T, B, 3), tracked bodies only
  body_quat_w: torch.Tensor  # (T, B, 4) wxyz
  body_lin_vel_w: torch.Tensor  # (T, B, 3)
  body_ang_vel_w: torch.Tensor  # (T, B, 3)

  def __len__(self) -> int:
    return self.joint_pos.shape[0]

  def placed(self, yaw: torch.Tensor, xy: torch.Tensor) -> Clip:
    """This clip turned and moved so its first root pose has `yaw` and `xy`."""
    dyaw = yaw - _yaw(self.body_quat_w[0, 0])
    c, s = torch.cos(dyaw), torch.sin(dyaw)
    pos = _rot_z(self.body_pos_w, c, s)
    pos[..., :2] += xy - pos[0, 0, :2]
    return Clip(
      self.joint_pos,
      self.joint_vel,
      pos,
      _quat_mul(_quat_z(dyaw).expand_as(self.body_quat_w), self.body_quat_w),
      _rot_z(self.body_lin_vel_w, c, s),
      _rot_z(self.body_ang_vel_w, c, s),
    )


def load_library(library_dir: str | Path, body_indexes: torch.Tensor, device: str) -> dict[str, Clip]:
  root = Path(library_dir)
  spec = yaml.safe_load((root / "library.yaml").read_text())
  clips = {}
  for move, info in spec["moves"].items():
    d = np.load(root / info["file"])

    def t(k: str, d=d) -> torch.Tensor:
      return torch.tensor(d[k], dtype=torch.float32, device=device)

    clips[move] = Clip(
      t("joint_pos"),
      t("joint_vel"),
      t("body_pos_w")[:, body_indexes],
      t("body_quat_w")[:, body_indexes],
      t("body_lin_vel_w")[:, body_indexes],
      t("body_ang_vel_w")[:, body_indexes],
    )
  if GUARD not in clips:
    raise ValueError(f"{root / 'library.yaml'} has no '{GUARD}' clip")
  return clips


class MoveScheduler:
  """Plays queued moves back to back; loops the guard when idle.

  `request()` is thread-safe (keyboard and GUI threads call it); everything
  else runs on the sim thread.
  """

  def __init__(self, clips: dict[str, Clip], on_start=None):
    self.clips = clips
    self.requests: queue.Queue[str] = queue.Queue()
    self.pending: list[str] = []
    self.on_start = on_start or (lambda move, pending: None)
    self.reset()

  def reset(self) -> None:
    self.pending.clear()
    guard = self.clips[GUARD]
    self.move = GUARD
    self.clip = guard.placed(_yaw(guard.body_quat_w[0, 0]), torch.zeros(2, device=guard.joint_pos.device))
    self.frame = 0

  def request(self, move: str) -> None:
    self.requests.put(move)

  def _start(self, move: str) -> None:
    yaw = _yaw(self.clip.body_quat_w[self.frame, 0])
    xy = self.clip.body_pos_w[self.frame, 0, :2]
    self.move, self.clip, self.frame = move, self.clips[move].placed(yaw, xy), 0
    self.on_start(move, list(self.pending))

  def step(self) -> None:
    while not self.requests.empty():
      move = self.requests.get_nowait()
      if move == "__clear__":
        self.pending.clear()
      elif move in self.clips and len(self.pending) < MAX_QUEUE:
        self.pending.append(move)

    # the guard is a still pose, so a move may start from any guard frame
    if self.move == GUARD and self.pending:
      self._start(self.pending.pop(0))
      return
    if self.frame + 1 < len(self.clip):
      self.frame += 1
      return
    self._start(self.pending.pop(0) if self.pending else GUARD)


BALL_PARK = (0.0, 0.0, -3.0)  # where the ball waits between punches (touches nothing)


class TeleopMotionCommand(MotionCommand):
  cfg: TeleopMotionCommandCfg

  def __init__(self, cfg: TeleopMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)  # loads cfg.motion_file (the guard) only to set up
    clips = load_library(cfg.library_dir, self.body_indexes, self.device)
    self._status = None
    # punch -> impact label (scripts/build_move_library.py), for the target ball
    spec = yaml.safe_load((Path(cfg.library_dir) / "library.yaml").read_text())
    self._impacts = {m: i["impact"] for m, i in spec["moves"].items() if "impact" in i}
    # the punch's target: (centre, direction) while a punch plays; the ball is
    # a real body when the scene has a "target" entity (scripts/teleop.py),
    # else only drawn
    self._ball: tuple[np.ndarray, np.ndarray] | None = None
    self._ball_move: str | None = None
    self._ball_peak = 0.0
    self.scheduler = MoveScheduler(clips, on_start=self._on_start)
    # one-frame motion that _write_frame refills; time_steps stays 0
    self.motion.time_step_total = 1
    self._write_frame()

  def _write_frame(self) -> None:
    c, f = self.scheduler.clip, self.scheduler.frame
    m = self.motion
    m.joint_pos = c.joint_pos[f : f + 1]
    m.joint_vel = c.joint_vel[f : f + 1]
    m.body_pos_w = c.body_pos_w[f : f + 1]
    m.body_quat_w = c.body_quat_w[f : f + 1]
    m.body_lin_vel_w = c.body_lin_vel_w[f : f + 1]
    m.body_ang_vel_w = c.body_ang_vel_w[f : f + 1]
    self.time_steps.zero_()

  def _on_start(self, move: str, pending: list[str]) -> None:
    self._report_ball()
    self._ball = self._ball_position(move)
    self._ball_move, self._ball_peak = (move if self._ball is not None else None), 0.0
    self._place_ball()
    if self._status is not None:
      self._status.content = f"**{move}**" + (f"  · next: {', '.join(pending)}" if pending else "")
    if self.cfg.on_start is not None:
      self.cfg.on_start(move, pending)

  @property
  def _ball_entity(self):
    return self._env.scene["target"] if "target" in self._env.scene.entities else None

  def _place_ball(self) -> None:
    """Ball to this punch's target, at rest; parked out of the way otherwise."""
    ball = self._ball_entity
    if ball is None:
      return
    state = ball.data.default_root_state.clone()
    pos = self._ball[0] if self._ball is not None else np.array(BALL_PARK)
    state[:, :3] = torch.tensor(pos, dtype=torch.float32, device=self.device) + self._env.scene.env_origins[:1]
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
    state[:, 7:] = 0.0
    ball.write_root_state_to_sim(state)
    self._env.sim.forward()  # so the line-hold below reads the new position, not the old one

  def _hold_ball_on_line(self) -> None:
    """Keep the ball on the punch's line (a rail): only motion along the
    punch's direction counts. Records the peak speed along it."""
    ball = self._ball_entity
    if ball is None or self._ball is None:
      return
    centre = torch.tensor(self._ball[0], dtype=torch.float32, device=self.device) + self._env.scene.env_origins[0]
    d = torch.tensor(self._ball[1], dtype=torch.float32, device=self.device)
    pos = ball.data.root_link_pos_w[:1]
    vel = ball.data.root_link_lin_vel_w[:1]
    along, speed = ((pos - centre) @ d), (vel @ d)
    self._ball_peak = max(self._ball_peak, float(speed[0]))
    state = ball.data.default_root_state[:1].clone()
    state[:, :3] = centre + along[:, None] * d
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
    state[:, 7:10] = speed[:, None] * d
    state[:, 10:] = 0.0
    ball.write_root_state_to_sim(state)
    self._pending_forward = True

  def _report_ball(self) -> None:
    if self._ball_move is None or self._ball_entity is None:
      return
    v = self._ball_peak
    msg = f"{self._ball_move}: ball {TARGET_MASS * v:.2f} kg m/s, would travel {v:.2f} m in 1 s" if v > 0.01 else f"{self._ball_move}: missed the ball"
    print(f"[teleop] {msg}", flush=True)
    if self._status is not None:
      self._status.content += f"  \n{msg}"

  def _ball_position(self, move: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Where this punch is aimed at impact (build_move_library.py: mid-strike,
    straights towards the centreline), plus the radii along the punch
    direction, carried to where the clip was placed (after earlier steps and
    pivots). None for a non-punch."""
    imp = self._impacts.get(move)
    if imp is None:
      return None
    src, placed = self.scheduler.clips[move], self.scheduler.clip
    dyaw = float(_yaw(placed.body_quat_w[0, 0]) - _yaw(src.body_quat_w[0, 0]))
    c, s = np.cos(dyaw), np.sin(dyaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    src0 = src.body_pos_w[0, 0].cpu().numpy()
    at0 = placed.body_pos_w[0, 0].cpu().numpy()
    centre = np.array(imp["aim_pos"]) + np.array(imp["direction"]) * (imp["glove_radius"] + TARGET_RADIUS)
    out = rot @ (centre - [src0[0], src0[1], 0.0])
    out[:2] += at0[:2]
    return out, rot @ np.array(imp["direction"])

  def _debug_vis_impl(self, visualizer) -> None:
    super()._debug_vis_impl(visualizer)
    if self._ball_entity is None and self._ball is not None and self.scheduler.move in self._impacts:
      visualizer.add_sphere(self._ball[0], TARGET_RADIUS, TARGET_RGBA, label="target")

  def reset(self, env_ids):
    self._ball, self._ball_move = None, None
    self._place_ball()
    self.scheduler.reset()
    self._write_frame()
    return super().reset(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None):
    self.scheduler.step()
    self._write_frame()
    if self.cfg.kinematic:
      # replay the reference exactly: teleport the robot onto it every step
      ids = torch.arange(self.num_envs, device=self.device)
      self._write_reference_state_to_sim(
        ids,
        self.body_pos_w[:, 0],
        self.body_quat_w[:, 0],
        self.body_lin_vel_w[:, 0],
        self.body_ang_vel_w[:, 0],
        self.joint_pos,
        self.joint_vel,
      )
      self._pending_forward = True
    self._hold_ball_on_line()
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()

  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
    """Viser: one button per move instead of the frame scrubber."""
    with server.gui.add_folder("Moves"):
      self._status = server.gui.add_markdown("**guard**")
      for label, move in self.cfg.buttons:
        if move not in self.scheduler.clips:
          continue
        btn = server.gui.add_button(label)
        btn.on_click(lambda _, m=move: self.scheduler.request(m))
      server.gui.add_button("Clear queue").on_click(lambda _: self.scheduler.request("__clear__"))


@dataclass(kw_only=True)
class TeleopMotionCommandCfg(MotionCommandCfg):
  library_dir: str = "data/moves"
  kinematic: bool = False
  """Replay the reference without physics (checks the clips, not the policy)."""
  buttons: tuple[tuple[str, str], ...] = ()
  """(label, move) pairs for the viser GUI."""
  on_start: object = None
  """Called as on_start(move, pending) when a clip starts."""

  def build(self, env: ManagerBasedRlEnv) -> TeleopMotionCommand:
    return TeleopMotionCommand(self, env)
