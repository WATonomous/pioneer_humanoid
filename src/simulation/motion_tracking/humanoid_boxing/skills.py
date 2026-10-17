"""Fighters that box with moves: a fight policy picks moves, frozen move skills
(trained motion-tracking policies) carry them out.

Each fighter has a FighterSkills action term. It holds a move player that
streams the chosen move's reference (from the move library, data/moves) for
every env at once, placing each move where the fighter actually stands when
it starts, and two frozen skill policies: the punch policy (guard and
punches) and the footwork policy (steps, pivots, defence). Every env step
(50 Hz) the term builds the tracking observation each skill was trained on,
runs the skill that owns the current move and sets the motor targets, exactly
as in the tracking task (joint targets = action x scale, offset 0).

Moves are chosen only in guard: a move plays to its end, then the fighter is
back in guard and the next choice takes effect (guard = keep waiting).

  learner   the action is the fight policy's move index (one number)
  opponent  no action; the move comes from the opponent mode, per env:
            hold (stays in guard), scripted (random moves) or self-play
            (a frozen snapshot of the learner's fight policy)
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import yaml
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, quat_mul, subtract_frame_transforms
from tensordict import TensorDict

from humanoid_tracking.robot import WATO_ACTION_SCALE

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

MOVES = (
  "guard",
  "jab", "cross", "lead_hook", "rear_hook", "lead_uppercut",
  "step_forward", "step_back", "step_left", "step_right", "pivot_left", "pivot_right",
  "slip_left", "slip_right", "weave_left", "weave_right", "duck", "deep_duck",
)  # fmt: skip
GUARD = 0
PUNCHES = ("jab", "cross", "lead_hook", "rear_hook", "lead_uppercut")
FOOTWORK = tuple(m for m in MOVES if m != "guard" and m not in PUNCHES)

# opponent modes
HOLD, SCRIPTED, SELFPLAY = 0, 1, 2
SCRIPT_RATE = 1.5  # scripted opponent: moves started per second while in guard
SCRIPT_WEIGHTS = {"punch": 0.35, "footwork": 0.65}
SNAPSHOT_EVERY = 4800  # env steps (~200 PPO iterations) between self-play snapshots
SNAPSHOT_POOL = 8  # the learner's last N snapshots; each self-play env faces one (5 drifted to a passive stand-off: fight_v7d)

# Technique: with each punch the fight policy also sets offsets on the arm
# joints and the punch's speed. They change the reference the frozen punch
# skill tracks (not the motors), so balance and physics stay the skill's; the
# offsets fade in and out over the punch (sin envelope), so it still starts
# and ends in the guard.
TECH_JOINTS = tuple(
  f"{side}_{j}"
  for side in ("left", "right")
  for j in ("shoulder_ak10_1_pitch", "shoulder_ak10_2_roll", "elbow_ak80_1_yaw", "elbow_ak80_2_bend", "elbow_ak80_3_forearm")
)
MAX_OFFSET = 0.1  # rad per joint at technique value +-1 (0.3, then 0.2, toppled red in 5-10% of rounds: fight_v7b/v7c; the policy saturates it)
RATE_SPAN = 0.25  # punch speed 0.75x .. 1.25x
TECH_DIM = len(TECH_JOINTS) + 1  # joint offsets, then speed


def _yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_z(a: torch.Tensor) -> torch.Tensor:
  z = torch.zeros_like(a)
  return torch.stack([torch.cos(a / 2), z, z, torch.sin(a / 2)], -1)


def _rot_xy(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
  c, s = torch.cos(a), torch.sin(a)
  return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


class FrozenSkill(nn.Module):
  """A trained tracking policy's actor (observation normaliser + MLP), deterministic."""

  def __init__(self, checkpoint: str, device: str):
    super().__init__()
    sd = torch.load(checkpoint, map_location=device, weights_only=False)["actor_state_dict"]
    self.register_buffer("mean", sd["obs_normalizer._mean"])
    self.register_buffer("std", sd["obs_normalizer._std"])
    sizes = [sd[f"mlp.{i}.weight"].shape for i in (0, 2, 4, 6)]
    layers: list[nn.Module] = []
    for k, (out_f, in_f) in enumerate(sizes):
      lin = nn.Linear(in_f, out_f)
      lin.weight.data.copy_(sd[f"mlp.{2 * k}.weight"])
      lin.bias.data.copy_(sd[f"mlp.{2 * k}.bias"])
      layers += [lin, nn.ELU()] if k < 3 else [lin]
    self.mlp = nn.Sequential(*layers).to(device)
    self.requires_grad_(False)

  @torch.no_grad()
  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp((obs - self.mean) / (self.std + 1e-2))


class MoveLibrary:
  """All MOVES' references as padded tensors [move, frame, ...]."""

  def __init__(self, library_dir: str, anchor_body: int, device: str):
    spec = yaml.safe_load((Path(library_dir) / "library.yaml").read_text())["moves"]
    clips = []
    for m in MOVES:
      if m not in spec:
        raise ValueError(f"{m} is not in {library_dir}/library.yaml")
      d = np.load(Path(library_dir) / spec[m]["file"])
      clips.append(d)
    n = [len(c["joint_pos"]) for c in clips]
    L = max(n)

    def pad(key, sl=None):
      out = []
      for c in clips:
        a = c[key] if sl is None else c[key][:, sl]
        a = np.concatenate([a, np.repeat(a[-1:], L - len(a), 0)])  # hold the last frame
        out.append(a)
      return torch.tensor(np.stack(out), dtype=torch.float32, device=device)

    self.length = torch.tensor(n, device=device)
    self.joint_pos = pad("joint_pos")
    self.joint_vel = pad("joint_vel")
    self.anchor_pos = pad("body_pos_w", anchor_body)
    self.anchor_quat = pad("body_quat_w", anchor_body)
    self.anchor_yaw0 = _yaw(self.anchor_quat[:, 0])


class MovePlayer:
  """Per-env current move and frame; each move is placed where the fighter
  stood (pelvis position and heading) when it started."""

  def __init__(self, lib: MoveLibrary, num_envs: int, num_joints: int, dt: float, device: str):
    self.lib = lib
    self.dt = dt
    self.move = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.phase = torch.zeros(num_envs, device=device)  # clip frames played (fractional)
    self.rate = torch.ones(num_envs, device=device)
    self.offset = torch.zeros(num_envs, num_joints, device=device)
    self.yaw_off = torch.zeros(num_envs, device=device)
    self.xy0 = torch.zeros(num_envs, 2, device=device)
    self.needs_place = torch.ones(num_envs, dtype=torch.bool, device=device)

  def start(self, ids, moves, root_pos, root_quat, rate=None, offset=None) -> None:
    self.move[ids] = moves
    self.phase[ids] = 0.0
    self.rate[ids] = 1.0 if rate is None else rate
    self.offset[ids] = 0.0 if offset is None else offset
    self.yaw_off[ids] = _yaw(root_quat[ids]) - self.lib.anchor_yaw0[moves]
    self.xy0[ids] = root_pos[ids, :2]
    self.needs_place[ids] = False

  @property
  def ready(self) -> torch.Tensor:
    return self.move == GUARD

  @property
  def frame(self) -> torch.Tensor:
    return self.phase.long().clamp(max=self.lib.length[self.move] - 1)

  @property
  def progress(self) -> torch.Tensor:
    return (self.phase / (self.lib.length[self.move] - 1).clamp(min=1).float()).clamp(max=1.0)

  def advance(self) -> torch.Tensor:
    """Next frame(s) at the move's speed; returns the envs whose move just finished."""
    self.phase += self.rate
    return self.phase >= self.lib.length[self.move] - 1

  def reference(self):
    m, f = self.move, self.frame
    lib = self.lib
    p = self.progress[:, None]
    dur = ((lib.length[m] - 1).clamp(min=1).float() / self.rate * self.dt)[:, None]  # s
    jp = lib.joint_pos[m, f] + self.offset * torch.sin(math.pi * p)
    jv = lib.joint_vel[m, f] * self.rate[:, None] + self.offset * math.pi * torch.cos(math.pi * p) / dur
    a0 = lib.anchor_pos[m, 0]
    a = lib.anchor_pos[m, f]
    xy = _rot_xy(a[:, :2] - a0[:, :2], self.yaw_off) + self.xy0
    pos = torch.cat([xy, a[:, 2:]], -1)
    quat = quat_mul(_quat_z(self.yaw_off), lib.anchor_quat[m, f])
    return jp, jv, pos, quat


@dataclass(kw_only=True)
class FighterSkillsCfg(ActionTermCfg):
  role: str = "learner"  # "learner" | "opponent"
  punch_checkpoint: str = ""
  footwork_checkpoint: str = ""
  library_dir: str = "data/moves"
  technique: bool = True  # the action also carries TECH_DIM technique values

  def build(self, env: ManagerBasedRlEnv) -> FighterSkills:
    return FighterSkills(self, env)


class FighterSkills(ActionTerm):
  cfg: FighterSkillsCfg

  def __init__(self, cfg: FighterSkillsCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    ent = self._entity
    dev = self.device
    self.anchor_body = ent.find_bodies("base_link")[0][0]
    self.lib = MoveLibrary(cfg.library_dir, self.anchor_body, dev)
    self.player = MovePlayer(self.lib, self.num_envs, len(ent.joint_names), env.step_dt, dev)
    self.tech_ids = torch.tensor([ent.joint_names.index(j) for j in TECH_JOINTS], device=dev)
    self.is_punch = torch.tensor([m in PUNCHES for m in MOVES], device=dev)
    self.tech = torch.zeros(self.num_envs, TECH_DIM, device=dev)
    self.punch_skill = FrozenSkill(cfg.punch_checkpoint, dev)
    self.footwork_skill = FrozenSkill(cfg.footwork_checkpoint or cfg.punch_checkpoint, dev)
    self.is_footwork = torch.tensor([m in FOOTWORK for m in MOVES], device=dev)
    self.scale = torch.tensor(
      [next(v for p, v in WATO_ACTION_SCALE.items() if re.fullmatch(p, n)) for n in ent.joint_names], device=dev
    )
    self.skill_action = torch.zeros(self.num_envs, len(ent.joint_names), device=dev)
    self.targets = torch.zeros_like(self.skill_action)
    self._raw = torch.zeros(self.num_envs, self.action_dim, device=dev)
    self.choice = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
    # opponent
    self.mode = torch.full((self.num_envs,), HOLD, dtype=torch.long, device=dev)
    self.mode_probs = torch.tensor([1.0, 0.0, 0.0], device=dev)
    self.learner_actor: nn.Module | None = None  # set by FightRunner
    self.snapshots: list[nn.Module] = []
    self.snapshot_of = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
    self._steps = 0
    self.team = cfg.entity_name

  # -- ActionTerm ---------------------------------------------------------------
  @property
  def action_dim(self) -> int:
    if self.cfg.role != "learner":
      return 0
    return 1 + (TECH_DIM if self.cfg.technique else 0)

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.player.move[ids] = GUARD
    self.player.phase[ids] = 0.0
    self.player.rate[ids] = 1.0
    self.player.offset[ids] = 0.0
    self.player.needs_place[ids] = True
    self.skill_action[ids] = 0.0
    self.choice[ids] = GUARD
    if self.cfg.role == "opponent":
      n = self.mode[ids].numel()
      self.mode[ids] = torch.multinomial(self.mode_probs, n, replacement=True)
      if self.snapshots:
        self.snapshot_of[ids] = torch.randint(0, len(self.snapshots), (n,), device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    ent = self._entity
    root_pos, root_quat = ent.data.root_link_pos_w, ent.data.root_link_quat_w
    if self.cfg.role == "learner":
      self._raw[:] = actions
      self.choice = actions[:, 0].round().long().clamp(0, len(MOVES) - 1)
      self.tech = actions[:, 1:] if actions.shape[1] > 1 else self.tech.zero_()
    else:
      self.choice, self.tech = self._opponent_choice()

    # (re)place guard after a reset, start chosen moves in guard, advance
    place = self.player.needs_place
    if torch.any(place):
      ids = place.nonzero().flatten()
      self.player.start(ids, torch.full_like(ids, GUARD), root_pos, root_quat)
    start = self.player.ready & (self.choice != GUARD)
    if torch.any(start):
      ids = start.nonzero().flatten()
      rate, offset = self._technique(ids)
      self.player.start(ids, self.choice[ids], root_pos, root_quat, rate, offset)
    done = self.player.advance()
    if torch.any(done):  # back to guard (or the next guard loop), placed where it stands
      ids = done.nonzero().flatten()
      self.player.start(ids, torch.full_like(ids, GUARD), root_pos, root_quat)
    self._steps += 1

    # the skill's tracking observation (same terms and order as the tracking task)
    jp_ref, jv_ref, ref_pos, ref_quat = self.player.reference()
    pos_b, ori_b = subtract_frame_transforms(root_pos, root_quat, ref_pos, ref_quat)
    ori6 = matrix_from_quat(ori_b)[..., :2].reshape(self.num_envs, -1)
    lin_b = quat_apply_inverse(root_quat, ent.data.root_link_lin_vel_w)
    ang_b = quat_apply_inverse(root_quat, ent.data.root_link_ang_vel_w)
    obs = torch.cat([jp_ref, jv_ref, pos_b, ori6, lin_b, ang_b, ent.data.joint_pos, ent.data.joint_vel, self.skill_action], -1)
    fw = self.is_footwork[self.player.move][:, None]
    self.skill_action = torch.where(fw, self.footwork_skill(obs), self.punch_skill(obs))
    self.targets = self.skill_action * self.scale

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self.targets)

  def _technique(self, ids):
    """Punch speed and joint offsets for the moves starting in `ids` (punches only)."""
    t = self.tech[ids].clamp(-1, 1) if self.tech.shape[1] == TECH_DIM else torch.zeros(len(ids), TECH_DIM, device=self.device)
    punch = self.is_punch[self.choice[ids]][:, None]
    t = torch.where(punch, t, torch.zeros_like(t))
    offset = torch.zeros(len(ids), self.player.offset.shape[1], device=self.device)
    offset[:, self.tech_ids] = t[:, :-1] * MAX_OFFSET
    return 1.0 + t[:, -1] * RATE_SPAN, offset

  # -- opponent ------------------------------------------------------------------
  def _opponent_choice(self) -> tuple[torch.Tensor, torch.Tensor]:
    n, dev = self.num_envs, self.device
    choice = torch.full((n,), GUARD, dtype=torch.long, device=dev)
    tech = torch.zeros(n, TECH_DIM, device=dev)
    # scripted: in guard, start a random move at SCRIPT_RATE per second
    scripted = self.mode == SCRIPTED
    go = scripted & (torch.rand(n, device=dev) < SCRIPT_RATE * self._env.step_dt)
    if torch.any(go):
      is_punch = torch.rand(n, device=dev) < SCRIPT_WEIGHTS["punch"]
      punch = torch.tensor([MOVES.index(m) for m in PUNCHES], device=dev)
      foot = torch.tensor([MOVES.index(m) for m in FOOTWORK], device=dev)
      pick = torch.where(
        is_punch,
        punch[torch.randint(0, len(punch), (n,), device=dev)],
        foot[torch.randint(0, len(foot), (n,), device=dev)],
      )
      choice = torch.where(go, pick, choice)
    # self-play: a frozen snapshot of the learner's fight policy, from this side
    self._maybe_snapshot()
    selfplay = self.mode == SELFPLAY
    if torch.any(selfplay) and self.snapshots:
      obs = TensorDict({"actor": fight_actor_obs(self._env, self.team)}, batch_size=[n])
      for k, snap in enumerate(self.snapshots):
        sel = selfplay & (self.snapshot_of == k)
        if torch.any(sel):
          with torch.no_grad():
            out = snap(obs, stochastic_output=True)
          a = out[:, 0].round().long().clamp(0, len(MOVES) - 1)
          choice = torch.where(sel, a, choice)
          if out.shape[1] == 1 + TECH_DIM:
            tech = torch.where(sel[:, None], out[:, 1:], tech)
    return choice, tech

  def _maybe_snapshot(self) -> None:
    if self.learner_actor is None:
      return
    if not self.snapshots or self._steps % SNAPSHOT_EVERY == 0:
      self.snapshots.append(copy.deepcopy(self.learner_actor).eval())
      self.snapshots = self.snapshots[-SNAPSHOT_POOL:]


# --------------------------------------------------------------------------- #
# Observations for the fight policy
# --------------------------------------------------------------------------- #


def _skills(env: ManagerBasedRlEnv, team: str) -> FighterSkills:
  return env.action_manager.get_term(team)  # type: ignore[return-value]


def own_state(env: ManagerBasedRlEnv, team: str) -> torch.Tensor:
  """Own pelvis up-vector, velocities (pelvis frame), current move (one-hot),
  its progress and whether a new move can start (in guard)."""
  ent = env.scene[team]
  q = ent.data.root_link_quat_w
  down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
  sk = _skills(env, team)
  onehot = torch.nn.functional.one_hot(sk.player.move, len(MOVES)).float()
  return torch.cat(
    [
      quat_apply_inverse(q, down),
      quat_apply_inverse(q, ent.data.root_link_lin_vel_w),
      quat_apply_inverse(q, ent.data.root_link_ang_vel_w),
      onehot,
      sk.player.progress[:, None],
      sk.player.ready.float()[:, None],
    ],
    -1,
  )


def opponent_move(env: ManagerBasedRlEnv, team: str) -> torch.Tensor:
  """Privileged (critic): the opponent's current move and its progress."""
  from humanoid_boxing.mdp import other

  sk = _skills(env, other(team))
  return torch.cat([torch.nn.functional.one_hot(sk.player.move, len(MOVES)).float(), sk.player.progress[:, None]], -1)


def fight_actor_obs(env: ManagerBasedRlEnv, team: str) -> torch.Tensor:
  """The fight policy's actor observation from `team`'s side (also used to run
  self-play snapshots on the other fighter)."""
  from humanoid_boxing import perception as vision

  return torch.cat([own_state(env, team), vision.opponent_perceived(env, team)], -1)


# --------------------------------------------------------------------------- #
# Curriculum: the opponent's mode
# --------------------------------------------------------------------------- #


class opponent_curriculum:
  """Opponent mode mix by training progress (run-relative, like the power
  ramp): stages = [(fraction of the run, (p_hold, p_scripted, p_selfplay))].
  Each env draws its opponent's mode at reset."""

  JUMP = 1000

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._start = int(env.common_step_counter)
    self._last = self._start

  def __call__(self, env: ManagerBasedRlEnv, env_ids, team: str, total_steps: int, stages: list) -> dict:
    now = int(env.common_step_counter)
    if abs(now - self._last) > self.JUMP:  # checkpoint restore
      self._start += now - self._last
    self._last = now
    frac = (now - self._start) / max(total_steps, 1)
    probs = stages[-1][1]
    for until, p in stages:
      if frac < until:
        probs = p
        break
    term = _skills(env, team)
    term.mode_probs = torch.tensor(probs, dtype=torch.float32, device=env.device)
    return {"hold": torch.tensor(probs[0]), "scripted": torch.tensor(probs[1]), "selfplay": torch.tensor(probs[2])}
