"""Clean punches for the fight task: what a landed punch is worth, and the
balance damage it does.

A CLEAN PUNCH is a glove arriving on the opponent's head or torso (gloves and
arms block) while the attacker is throwing a punch move, with the glove
moving at MIN_SPEED or more relative to the opponent's torso (measured: the
robot's punches land at ~0.5-1.2 m/s; a glove resting in guard ~0.2-0.35).
Leaning a glove on the opponent, or bumping it during a step, does not count.
Each one scores once, as it lands:

  value = target zone (ZONE_VALUE: head; on the torso, by where it lands:
          liver / ribs = lower half, off the centreline; solar plexus = lower
          half, centre; chest = upper half)
        x speed (glove speed / SPEED_REF, capped at SPEED_CAP)
        x COUNTER_BONUS if the opponent was throwing a punch itself
        x force (peak glove contact force over the impact's first
          FORCE_WINDOW / FORCE_REF, capped at FORCE_CAP): weight behind it
The force part is paid as its peak grows over FORCE_WINDOW, so the total is
the value above at the impact's peak force.

Balance damage is paid per clean punch, in the second after it (WINDOW):
  imbalance    degrees the opponent's pelvis tilts past its stance + DEADBAND,
               beyond any tilt it already had when the punch landed
  driven_back  metres the opponent is pushed away
  stagger      once, if it tilts past STAGGER_TILT
Each pays as its peak grows, so a punch is paid once for its biggest effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from humanoid_boxing import mdp as fight
from humanoid_boxing import skills
from humanoid_boxing.fighters import TORSO_BODY, WRISTS

ZONE_VALUE = {"head": 1.0, "liver_ribs": 0.8, "solar_plexus": 0.7, "chest": 0.4}
ZONES = tuple(ZONE_VALUE)
SIDE_OFFSET = 0.08  # m off the torso's centreline counts as the side (half-width ~0.20)
MIN_SPEED = 0.4  # m/s glove speed (relative to the target) for a punch to count
SPEED_REF = 1.0  # m/s for value 1
SPEED_CAP = 1.5
COUNTER_BONUS = 1.5
FORCE_WINDOW = 0.1  # s after landing over which the peak contact force is taken
FORCE_REF = 50.0  # N for force 1 (measured: the robot's clean punches peak ~50 N on average)
FORCE_CAP = 2.0
WINDOW = 1.0  # s of balance damage credited to a punch
DEADBAND = math.radians(5.0)
STAGGER_TILT = math.radians(35.0)


@dataclass
class _Side:
  dealing: torch.Tensor  # glove on the opponent's head/torso last step
  scored: torch.Tensor  # the current punch move has already landed clean (one clean punch per punch)
  impact: torch.Tensor  # points-value paid this step for clean punches (0 if none)
  base: torch.Tensor  # latest clean punch's zone x speed x counter
  force_peak: torch.Tensor  # its peak force factor so far
  force_sum: torch.Tensor  # sum of clean punches' peak forces this round (N)
  zone: torch.Tensor  # its zone index into ZONES (-1 if none)
  count: torch.Tensor  # clean punches landed this round
  zone_count: torch.Tensor  # [N, len(ZONES)] of them by zone
  since: torch.Tensor  # s since this fighter's last clean punch
  tilt0: torch.Tensor  # opponent's excess tilt when it landed
  tilt_peak: torch.Tensor
  push0: torch.Tensor  # opponent's distance away when it landed
  push_peak: torch.Tensor
  staggered: torch.Tensor
  d_tilt: torch.Tensor  # growth of the peaks this step (the rewards)
  d_push: torch.Tensor
  stagger: torch.Tensor


class _Impacts:
  def __init__(self, env):
    n, dev = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    self.step = -1
    self.sides = {
      t: _Side(
        dealing=torch.zeros(n, dtype=torch.bool, device=dev), impact=z(), since=z() + 1e3,
        scored=torch.zeros(n, dtype=torch.bool, device=dev),
        zone=torch.full((n,), -1, dtype=torch.long, device=dev), count=z(),
        zone_count=torch.zeros(n, len(ZONES), device=dev), base=z(), force_peak=z(), force_sum=z(),
        tilt0=z(), tilt_peak=z(), push0=z(), push_peak=z(),
        staggered=torch.zeros(n, dtype=torch.bool, device=dev), d_tilt=z(), d_push=z(),
        stagger=torch.zeros(n, dtype=torch.bool, device=dev),
      )  # fmt: skip
      for t in fight.TEAMS
    }
    self.punch_ids = torch.tensor([skills.MOVES.index(m) for m in skills.PUNCHES], device=dev)
    self.zone_value = torch.tensor([ZONE_VALUE[z] for z in ZONES], device=dev)


def landing_sensors():
  """Where a glove lands on the torso (added to the scene by the fight cfg)."""
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  return tuple(
    ContactSensorCfg(
      name=f"{team}_lands_{fight.other(team)}_body",
      primary=ContactMatch(mode="geom", pattern=r"glove_[lr]", entity=team),
      secondary=ContactMatch(mode="geom", pattern="torso", entity=fight.other(team)),
      fields=("found", "pos"),
      reduce="maxforce",
    )
    for team in fight.TEAMS
  )


def _hit(env, attacker, defender):
  head = fight._sensor(env, f"{attacker}_hits_{defender}_head")
  body = fight._sensor(env, f"{attacker}_hits_{defender}_body")
  return (head.found > 0).any(-1), (body.found > 0).any(-1)


def _force(env, attacker, defender):
  """[N] glove contact force on the defender's head + torso this step (N)."""
  head = fight._sensor(env, f"{attacker}_hits_{defender}_head")
  body = fight._sensor(env, f"{attacker}_hits_{defender}_body")
  return head.force.norm(dim=-1).sum(-1) + body.force.norm(dim=-1).sum(-1)


def _zone(env, attacker, defender, head):
  """[N] zone index (into ZONES) of this step's landing."""
  land = fight._sensor(env, f"{attacker}_lands_{defender}_body")
  found = land.found > 0  # [N, P]
  first = found.float().argmax(-1)
  pos = land.pos[torch.arange(env.num_envs, device=env.device), first]  # [N, 3]
  them = env.scene[defender]
  geom = them.find_geoms(("torso",))[0][0]
  centre = them.data.geom_pose_w[:, geom, :3]
  rel = pos - centre
  # lateral offset in the defender's heading frame (robot forward = base +y)
  w, x, y, z = them.data.root_link_quat_w.unbind(-1)
  fx, fy = 2 * (x * y - w * z), 1 - 2 * (x * x + z * z)
  yaw = torch.atan2(fy, fx)
  lateral = -torch.sin(yaw) * rel[:, 0] + torch.cos(yaw) * rel[:, 1]
  low = rel[:, 2] < 0
  side = lateral.abs() > SIDE_OFFSET
  zone = torch.where(low & side, 1, torch.where(low, 2, 3))
  return torch.where(head, 0, zone)


def impacts(env) -> _Impacts:
  im: _Impacts | None = getattr(env, "_impacts", None)
  if im is None:
    im = env._impacts = _Impacts(env)
  if im.step == env.common_step_counter:
    return im
  st = fight.fight_state(env)
  dt = env.step_dt
  for team in fight.TEAMS:
    opp = fight.other(team)
    s, me, them = im.sides[team], env.scene[team], env.scene[opp]
    head, body = _hit(env, team, opp)
    dealing = head | body
    onset = dealing & ~s.dealing
    s.dealing = dealing

    # speed of the faster glove (wrist) relative to the opponent's torso; hooks
    # and uppercuts arrive across or up, so the full speed counts, not just
    # the component towards the torso centre
    wrist_ids = [me.find_bodies(b)[0][0] for b in WRISTS.values()]
    torso = them.find_bodies(TORSO_BODY)[0][0]
    tgt_v = them.data.body_link_lin_vel_w[:, torso]
    closing = (me.data.body_link_lin_vel_w[:, wrist_ids] - tgt_v[:, None]).norm(dim=-1).max(-1).values

    player = env.action_manager.get_term(team).player
    my_move = player.move
    their_move = env.action_manager.get_term(opp).player.move
    punching = torch.isin(my_move, im.punch_ids)
    # one clean punch per punch move: the glove's contact flickers on and off
    # as it lands and rubs, and each new contact used to count again (up to
    # ~17 per punch: fight_v7 learned to rub rather than punch)
    s.scored = s.scored & ~(player.phase <= player.rate + 1e-6)  # a new move started this step
    clean = onset & punching & (closing >= MIN_SPEED) & ~s.scored
    s.scored = s.scored | clean
    zone = _zone(env, team, opp, head)
    value = im.zone_value[zone] * (closing / SPEED_REF).clamp(0, SPEED_CAP)
    s.zone = torch.where(clean, zone, torch.full_like(zone, -1))
    value = value * torch.where(torch.isin(their_move, im.punch_ids), COUNTER_BONUS, 1.0)
    s.count = s.count + clean.float()
    s.zone_count[clean, zone[clean]] += 1

    # force: paid as the impact's peak contact force grows over FORCE_WINDOW
    s.since = torch.where(clean, torch.zeros_like(s.since), s.since + dt)
    f = (_force(env, team, opp) / FORCE_REF).clamp(0, FORCE_CAP)
    s.base = torch.where(clean, value, s.base)
    prev_peak = torch.where(clean, torch.zeros_like(f), s.force_peak)
    in_window = (s.since <= FORCE_WINDOW + 1e-6) & dealing
    peak = torch.where(in_window, torch.maximum(prev_peak, f), prev_peak)
    s.impact = s.base * (peak - prev_peak)
    s.force_sum = s.force_sum + (peak - prev_peak) * FORCE_REF
    s.force_peak = peak

    # balance damage credited to the latest clean punch
    excess = (st.teams[opp].tilt - fight.STANCE_TILT - DEADBAND).clamp_min(0)
    away = (them.data.root_link_pos_w[:, :2] - me.data.root_link_pos_w[:, :2]).norm(dim=-1)
    s.tilt0 = torch.where(clean, excess, s.tilt0)
    s.tilt_peak = torch.where(clean, excess, s.tilt_peak)
    s.push0 = torch.where(clean, away, s.push0)
    s.push_peak = torch.where(clean, away, s.push_peak)
    s.staggered = torch.where(clean, st.teams[opp].tilt > STAGGER_TILT, s.staggered)
    live = s.since <= WINDOW
    new_tilt = torch.where(live, torch.maximum(s.tilt_peak, excess), s.tilt_peak)
    new_push = torch.where(live, torch.maximum(s.push_peak, away), s.push_peak)
    s.d_tilt, s.d_push = new_tilt - s.tilt_peak, new_push - s.push_peak
    s.tilt_peak, s.push_peak = new_tilt, new_push
    s.stagger = live & ~s.staggered & (st.teams[opp].tilt > STAGGER_TILT) & ~st.teams[opp].down
    s.staggered = s.staggered | s.stagger
  im.step = env.common_step_counter
  return im


def reset_impacts(env, env_ids) -> None:
  im: _Impacts | None = getattr(env, "_impacts", None)
  if im is None:
    return
  ids = slice(None) if env_ids is None else env_ids
  for s in im.sides.values():
    s.dealing[ids] = False
    s.scored[ids] = False
    s.since[ids] = 1e3
    s.staggered[ids] = False
    s.count[ids] = 0.0
    s.zone_count[ids] = 0.0
    s.force_peak[ids] = 0.0
    s.force_sum[ids] = 0.0


# --------------------------------------------------------------------------- #
# Reward terms (from `team`'s side)
# --------------------------------------------------------------------------- #


def clean_punches(env, team: str) -> torch.Tensor:
  return impacts(env).sides[team].impact


def clean_punches_taken(env, team: str) -> torch.Tensor:
  return impacts(env).sides[fight.other(team)].impact


def punch_imbalance(env, team: str) -> torch.Tensor:
  """Degrees of extra tilt a clean punch caused (paid as its peak grows)."""
  return torch.rad2deg(impacts(env).sides[team].d_tilt)


def punch_driven_back(env, team: str) -> torch.Tensor:
  """Metres a clean punch pushed the opponent away (paid as its peak grows)."""
  return impacts(env).sides[team].d_push


def punch_stagger(env, team: str) -> torch.Tensor:
  return impacts(env).sides[team].stagger.float()


RANGE = (0.8, 1.05)  # m pelvis-to-pelvis: clean punches land from here (guard gloves touch at ~0.7)
CROWDING = 0.75  # m: inside this a punch arrives at ~0 m/s


def _gap(env, team: str) -> torch.Tensor:
  me, them = env.scene[team], env.scene[fight.other(team)]
  return (me.data.root_link_pos_w[:, :2] - them.data.root_link_pos_w[:, :2]).norm(dim=-1)


def in_range_vs_still(env, team: str) -> torch.Tensor:
  """Per step in punching range of the opponent, only while it holds its guard
  (the still stage's shaping: learn where punches land from)."""
  hold = skills._skills(env, fight.other(team)).mode == skills.HOLD
  g = _gap(env, team)
  return (hold & (g >= RANGE[0]) & (g <= RANGE[1])).float()


def in_range_vs_moving(env, team: str) -> torch.Tensor:
  """Per step in punching range of a scripted or self-play opponent. Without it
  self-play drifted to a stand-off out of range (~1.3 m, guard ~70%, ~1.5
  clean punches a round each: fight_v7d/v7e), where neither side lands."""
  hold = skills._skills(env, fight.other(team)).mode == skills.HOLD
  g = _gap(env, team)
  return (~hold & (g >= RANGE[0]) & (g <= RANGE[1])).float()


def punch_from_range(env, team: str) -> torch.Tensor:
  """Once per punch move started from punching range (a correct setup; paid at
  the start, whether it lands or not, so it is small)."""
  p = skills._skills(env, team).player
  # a move restarts in the step its predecessor ends, so detect the start by
  # its phase (one advance played), not by a change of move
  started = torch.isin(p.move, impacts(env).punch_ids) & (p.phase <= p.rate + 1e-6)
  g = _gap(env, team)
  return (started & (g >= RANGE[0]) & (g <= RANGE[1])).float()


def crowding(env, team: str) -> torch.Tensor:
  """Per step pressed inside CROWDING (smothering in the guard)."""
  return (_gap(env, team) < CROWDING).float()


def time_up_clean(env, team: str) -> torch.Tensor:
  """At the bell with nobody down: +-1 by the difference in clean punches
  (the arena's time_up_points counts any glove contact, bumps included)."""
  im = impacts(env)
  bell = env.termination_manager.time_outs & ~env.termination_manager.terminated
  return bell.float() * torch.tanh((im.sides[team].count - im.sides[fight.other(team)].count) / 3.0)


def punch_stats(env, env_ids, team: str) -> dict:
  """Curriculum-slot logger: per finished round, clean punches and by zone
  (runs before the reset clears the counts)."""
  if env_ids is None or len(env_ids) == 0:
    return {}
  im = impacts(env)
  s = im.sides[team]
  out = {"clean_per_round": s.count[env_ids].mean()}
  for k, z in enumerate(ZONES):
    out[z] = s.zone_count[env_ids, k].mean()
  out["force_per_punch_N"] = (s.force_sum[env_ids].sum() / s.count[env_ids].sum().clamp_min(1))
  out["taken_per_round"] = im.sides[fight.other(team)].count[env_ids].mean()
  return out
