"""Fight rewards and round endings for the two-fighter arena.

One FightState per env is updated once per step (the termination manager runs
first, the reward terms read the same numbers). Rewards are written from one
fighter's side (`team`); the opponent is the other one. With
scale_rewards_by_dt=False the weights are points.

A fighter is DOWN when any non-foot hitbox touches the canvas, its pelvis is
below DOWN_PELVIS_Z or its torso tilts past DOWN_TILT. The robot cannot get
up, so the first time anyone is down the round ends. It counts as a
KNOCKDOWN when the fighter took a glove hit in the last HIT_WINDOW seconds,
otherwise as a slip.

Unrealistic-movement checks: joint speed over SPEED_SHARE of the motor cap,
motors at their torque limit, joint limits, joint acceleration, feet sliding
while planted, both feet off the floor, touching the ropes, and
self-collision (legs crossing through each other, an arm through the torso),
which ends the round as a loss.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactMatch, ContactSensorCfg
from wato_boxing.fighters import HITBOX_NAMES, LEG_RADIUS, STANCE_ROOT_QUAT
from wato_tracking.robot import ACTUATORS, SPEED_CAPS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

TEAMS = ("red", "blue")
DOWN_PELVIS_Z = 0.40  # stance pelvis is ~0.73 m
DOWN_TILT = math.radians(60.0)
HIT_WINDOW = 1.0  # s
STANCE_PELVIS_Z = 0.72
# the boxing stance leans the pelvis ~12 deg; stability measures lean beyond it
_w, _x, _y, _z = STANCE_ROOT_QUAT
STANCE_TILT = math.acos(1 - 2 * (_x * _x + _y * _y))
HIT_FORCE_REF = 100.0  # N, a glove contact this hard scores 1
SPEED_SHARE = 0.8
TORQUE_SHARE = 0.95
HEAD_WEIGHT, BODY_WEIGHT = 1.0, 0.5

CORE_HITBOXES = (r"^head$", r"^torso$", r"^pelvis$", r"^leg_.*$")
FEET = ("left_foot_1", "right_foot_1")
LEGS = {  # capsule segments: (from body, to body)
  "left": [("left_thigh_1", "left_calf_1"), ("left_calf_1", "left_foot_joint_1")],
  "right": [("right_thigh_1", "right_calf_1"), ("right_calf_1", "right_foot_joint_1")],
}
ARMS = [
  ("Mirrorlink3__1__1", "Mirrorlink4__1__1"),
  ("Mirrorlink5__1__1", "Mirrorlink6__1__1"),
  ("link3__1__1", "link4__1__1"),
  ("link5__1__1", "link6__1__1"),
]
LEG_CROSS_DIST = 2 * LEG_RADIUS - 0.02  # capsules overlapping by more than 2 cm
TORSO_SHRINK = 0.02  # arm centreline this far inside the torso box


def other(team: str) -> str:
  return "blue" if team == "red" else "red"


# --------------------------------------------------------------------------- #
# Sensors the state needs (added to the scene by the env cfg)
# --------------------------------------------------------------------------- #


def fight_sensors() -> tuple[ContactSensorCfg, ...]:
  sensors = []
  for team in TEAMS:
    opp = other(team)
    for target, geom in (("head", "head"), ("body", "torso")):
      sensors.append(
        ContactSensorCfg(
          name=f"{team}_hits_{opp}_{target}",
          primary=ContactMatch(mode="geom", pattern=r"glove_[lr]", entity=team),
          secondary=ContactMatch(mode="geom", pattern=geom, entity=opp),
          fields=("found", "force"),
          reduce="netforce",
        )
      )
    sensors += [
      ContactSensorCfg(  # anything but the feet on the canvas = down
        name=f"{team}_floor",
        primary=ContactMatch(mode="geom", pattern=HITBOX_NAMES, entity=team),
        secondary=ContactMatch(mode="geom", pattern="terrain"),
        fields=("found",),
      ),
      ContactSensorCfg(
        name=f"{team}_feet",
        primary=ContactMatch(mode="geom", pattern=r"^(left|right)_foot_1_collision$", entity=team),
        secondary=ContactMatch(mode="geom", pattern="terrain"),
        fields=("found",),
      ),
      ContactSensorCfg(
        name=f"{team}_rope",
        primary=ContactMatch(mode="geom", pattern=HITBOX_NAMES, entity=team),
        secondary=ContactMatch(mode="subtree", pattern="ring", entity="ring"),
        fields=("found",),
      ),
    ]
    for target in ("torso", "pelvis"):  # body-to-body contact = shoving / clinching
      sensors.append(
        ContactSensorCfg(
          name=f"{team}_shove_{target}",
          primary=ContactMatch(mode="geom", pattern=CORE_HITBOXES, entity=team),
          secondary=ContactMatch(mode="geom", pattern=target, entity=opp),
          fields=("found", "force"),
          reduce="netforce",
        )
      )
  return tuple(sensors)


# --------------------------------------------------------------------------- #
# Per-step state
# --------------------------------------------------------------------------- #


@dataclass
class TeamState:
  down: torch.Tensor
  pelvis_z: torch.Tensor
  tilt: torch.Tensor
  hit_taken: torch.Tensor  # glove on this fighter this step
  hit_force_dealt: torch.Tensor  # weighted glove force on the opponent this step / HIT_FORCE_REF
  since_hit: torch.Tensor  # s since this fighter last took a glove hit
  landed: torch.Tensor  # glove-hit onsets dealt this round
  dealing: torch.Tensor  # glove on the opponent this step
  rope: torch.Tensor
  airborne: torch.Tensor
  foot_slip: torch.Tensor
  speed_over: torch.Tensor
  torque_sat: torch.Tensor
  joint_acc: torch.Tensor
  shove: torch.Tensor
  self_collision: torch.Tensor


@dataclass
class FightState:
  step: int = -1
  teams: dict[str, TeamState] = field(default_factory=dict)


def _sensor(env: ManagerBasedRlEnv, name: str):
  return env.scene[name].data


def _seg_dist(p1, q1, p2, q2) -> torch.Tensor:
  """Distance between segments p1-q1 and p2-q2, batched over [N, 3]."""
  d1, d2, r = q1 - p1, q2 - p2, p1 - p2
  a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
  b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
  denom = (a * e - b * b).clamp_min(1e-9)
  s = ((b * f - c * e) / denom).clamp(0, 1)
  t = ((b * s + f) / e.clamp_min(1e-9)).clamp(0, 1)
  s = ((b * t - c) / a.clamp_min(1e-9)).clamp(0, 1)
  return ((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None])).norm(dim=-1)


def _self_collision(env: ManagerBasedRlEnv, team: str) -> torch.Tensor:
  ent = env.scene[team]
  pos = ent.data.body_link_pos_w

  def body(name):
    return pos[:, ent.find_bodies(name)[0][0]]

  hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for a0, a1 in LEGS["left"]:
    for b0, b1 in LEGS["right"]:
      hit |= _seg_dist(body(a0), body(a1), body(b0), body(b1)) < LEG_CROSS_DIST

  m, d = env.sim.mj_model, env.sim.data
  gid = m.geom(f"{team}/torso").id
  centre, rot = d.geom_xpos[:, gid], d.geom_xmat[:, gid].reshape(-1, 3, 3)
  half = torch.as_tensor(m.geom_size[gid], device=env.device, dtype=torch.float32) - TORSO_SHRINK
  for a0, a1 in ARMS:
    p0, p1 = body(a0), body(a1)
    for s in (0.25, 0.5, 0.75):
      local = torch.einsum("nji,nj->ni", rot, p0 + s * (p1 - p0) - centre)
      hit |= (local.abs() < half).all(-1)
  return hit


def _caps(names: list[str], table: dict[str, float], device) -> torch.Tensor:
  return torch.tensor(
    [next(v for k, v in table.items() if re.fullmatch(k, n)) for n in names], device=device, dtype=torch.float32
  )


def fight_state(env: ManagerBasedRlEnv) -> FightState:
  st: FightState = getattr(env, "_fight_state", None) or FightState()
  env._fight_state = st
  if st.step == env.common_step_counter:
    return st
  dt, n, dev = env.step_dt, env.num_envs, env.device
  zeros = torch.zeros(n, device=dev)

  for team in TEAMS:
    opp = other(team)
    ent = env.scene[team]
    prev = st.teams.get(team)

    pelvis_z = ent.data.root_link_pos_w[:, 2]
    tilt = torch.acos((-ent.data.projected_gravity_b[:, 2]).clamp(-1, 1))
    floor = (_sensor(env, f"{team}_floor").found > 0).any(-1)
    down = floor | (pelvis_z < DOWN_PELVIS_Z) | (tilt > DOWN_TILT)

    def hit(attacker, defender):
      head, body = _sensor(env, f"{attacker}_hits_{defender}_head"), _sensor(env, f"{attacker}_hits_{defender}_body")
      found = (head.found > 0).any(-1) | (body.found > 0).any(-1)
      force = HEAD_WEIGHT * head.force.norm(dim=-1).sum(-1) + BODY_WEIGHT * body.force.norm(dim=-1).sum(-1)
      return found, (force / HIT_FORCE_REF).clamp(max=2.0)

    taken, _ = hit(opp, team)
    dealing, force_dealt = hit(team, opp)
    since_hit = torch.where(taken, zeros, (prev.since_hit if prev else zeros + 1e3) + dt)
    prev_dealing = prev.dealing if prev else torch.zeros_like(dealing)
    landed = (prev.landed if prev else zeros) + (dealing & ~prev_dealing).float()

    feet = _sensor(env, f"{team}_feet").found > 0  # [N, 2]
    foot_ids = [ent.find_bodies(f)[0][0] for f in FEET]
    foot_v = ent.data.body_link_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
    if not hasattr(ent, "_speed_caps"):
      ent._speed_caps = _caps(ent.joint_names, SPEED_CAPS, dev)
      ent._effort_caps = _caps(
        ent.actuator_names, {p: a.effort_limit for a in ACTUATORS for p in a.target_names_expr}, dev
      )
    shove = sum(_sensor(env, f"{team}_shove_{t}").force.norm(dim=-1).sum(-1) for t in ("torso", "pelvis"))

    ts = TeamState(
      down=down,
      pelvis_z=pelvis_z,
      tilt=tilt,
      hit_taken=taken,
      hit_force_dealt=force_dealt,
      since_hit=since_hit,
      landed=landed,
      dealing=dealing,
      rope=(_sensor(env, f"{team}_rope").found > 0).any(-1),
      airborne=~feet.any(-1) & ~down,
      foot_slip=(foot_v * feet.float()).sum(-1),
      speed_over=(ent.data.joint_vel.abs() / ent._speed_caps - SPEED_SHARE).clamp_min(0).sum(-1),
      torque_sat=(ent.data.actuator_force.abs() > TORQUE_SHARE * ent._effort_caps).float().mean(-1),
      joint_acc=(ent.data.joint_acc**2).mean(-1),
      shove=shove / HIT_FORCE_REF,
      self_collision=_self_collision(env, team),
    )
    st.teams[team] = ts
  st.step = env.common_step_counter
  return st


def reset_fight_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  st: FightState | None = getattr(env, "_fight_state", None)
  if st is None or not st.teams:
    return
  ids = slice(None) if env_ids is None else env_ids
  for ts in st.teams.values():
    ts.since_hit[ids] = 1e3
    ts.landed[ids] = 0.0
    ts.dealing[ids] = False


# --------------------------------------------------------------------------- #
# Round endings
# --------------------------------------------------------------------------- #


def round_over(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Someone is down (cannot get up) or a fighter's limbs passed through itself."""
  st = fight_state(env)
  out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for ts in st.teams.values():
    out |= ts.down | ts.self_collision
  return out


# --------------------------------------------------------------------------- #
# Rewards, from `team`'s side
# --------------------------------------------------------------------------- #


def _me_opp(env, team):
  st = fight_state(env)
  return st.teams[team], st.teams[other(team)]


def knockdown(env, team: str) -> torch.Tensor:
  me, opp = _me_opp(env, team)
  return (opp.down & (opp.since_hit <= HIT_WINDOW) & ~me.down).float()


def opponent_slip(env, team: str) -> torch.Tensor:
  me, opp = _me_opp(env, team)
  return (opp.down & (opp.since_hit > HIT_WINDOW) & ~me.down).float()


def knocked_down(env, team: str) -> torch.Tensor:
  """Down while the opponent stays up, from a hit or a fall."""
  me, opp = _me_opp(env, team)
  return (me.down & ~opp.down).float()


def double_down(env, team: str) -> torch.Tensor:
  me, opp = _me_opp(env, team)
  return (me.down & opp.down).float()


def self_collision(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.self_collision.float()


def clean_hits(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.hit_force_dealt


def hits_taken(env, team: str) -> torch.Tensor:
  _, opp = _me_opp(env, team)
  return opp.hit_force_dealt


def stability(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  lean = (me.tilt - STANCE_TILT).clamp_min(0)
  return torch.exp(-((lean / 0.35) ** 2)) * torch.exp(-(((me.pelvis_z - STANCE_PELVIS_Z) / 0.08) ** 2))


def shoving(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.shove


def overspeed(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.speed_over


def torque_saturation(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.torque_sat


def jerkiness(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.joint_acc


def foot_slip(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.foot_slip


def airborne(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.airborne.float()


def rope_contact(env, team: str) -> torch.Tensor:
  me, _ = _me_opp(env, team)
  return me.rope.float()


def time_up_points(env, team: str) -> torch.Tensor:
  """At the bell with nobody down: +-1 by the difference in landed hits."""
  me, opp = _me_opp(env, team)
  bell = env.termination_manager.time_outs & ~env.termination_manager.terminated
  return bell.float() * torch.tanh((me.landed - opp.landed) / 3.0)
