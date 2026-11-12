"""Clean punches for the fight task: what a landed punch is worth, and the
balance damage it does.

A CLEAN PUNCH is a glove arriving on the opponent's head or torso (gloves and
arms block) while the attacker is throwing a punch move, with the glove
moving at MIN_SPEED or more relative to the opponent's torso (measured: the
robot's punches land at ~0.5-1.2 m/s; a glove resting in guard ~0.2-0.35).
Leaning a glove on the opponent, or bumping it during a step, does not count.
Each one scores once, as it lands:

  value = where it lands (HEAD_VALUE on the head; on the torso BODY_VALUE,
          rising the closer it lands to the head, over GRADIENT_REACH)
        x mix-up (x1 to x3, from the last MIX_MEMORY clean punches only:
          +TYPE_BONUS x the share of them that were another punch type
          (PUNCHES) and +LEVEL_BONUS x the share at another level (LEVELS);
          a round's first clean punch has nothing to mix with (x1), and the
          same punch to the same level every time stays x1)
        x speed (glove speed / SPEED_REF, capped at SPEED_CAP)
        x COUNTER_BONUS if the opponent was throwing a punch itself
        x force (peak glove contact force over the impact's first
          FORCE_WINDOW / FORCE_REF, capped at FORCE_CAP): weight behind it
The force part is paid as its peak grows over FORCE_WINDOW, so the total is
the value above at the impact's peak force.

A BLOCKED PUNCH is a punch move that touched the opponent's arms or gloves and
ended without landing clean (counted once, when the next move starts). A
MISSED one touched nothing. From the defender's side these are its blocks and
evasions: its DEFENCE RATE is the share of the opponent's punches that did not
land clean (logged, not rewarded: a block reward invites a guard-only stalemate).

ZONES (head, liver / ribs, solar plexus, chest) are kept for the logs only.

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
from humanoid_boxing.fighters import HEAD_RADIUS, TORSO_BODY, WRISTS

HEAD_VALUE = 3.5
BODY_VALUE = (1.5, 2.5)  # torso: at GRADIENT_REACH from the head's surface, and next to it
GRADIENT_REACH = 0.52  # m from the head's surface to the torso's bottom corners
MIX_MEMORY = 4  # clean punches remembered per round
TYPE_BONUS = 2.0  # + on a punch type not among the last MIX_MEMORY clean punches
LEVEL_BONUS = 1.5  # + on a level not among them
ZONES = ("head", "liver_ribs", "solar_plexus", "chest")
LEVELS = ("head", "chest", "body")  # for the mix-up; ZONES -> level below (left / right does not count:
# fight_still_v4 farmed a side-by-side target bonus with one lead hook to the chest's centreline)
ZONE_LEVEL = (0, 2, 1, 1)
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
  recent_type: torch.Tensor  # [N, MIX_MEMORY] PUNCHES index of the last clean punches (-1 = none yet)
  recent_level: torch.Tensor  # [N, MIX_MEMORY] LEVELS index of them
  type_count: torch.Tensor  # [N, len(PUNCHES)] clean punches this round by type
  level_count: torch.Tensor  # [N, len(LEVELS)] and by level
  mix: torch.Tensor  # latest clean punch's mix-up factor
  mix_sum: torch.Tensor  # sum of the mix-up factors this round
  points: torch.Tensor  # clean-punch points dealt this round (the clean_punches term before its weight)
  guard_hit: torch.Tensor  # the current punch move has touched the opponent's arms / gloves
  was_punch: torch.Tensor  # last step's move was a punch
  blocked: torch.Tensor  # a blocked punch ended this step
  blocked_count: torch.Tensor  # blocked punches this round
  thrown_count: torch.Tensor  # punch moves finished this round
  missed_count: torch.Tensor  # of them, touched neither the target nor the guard
  guard_steps: torch.Tensor  # steps spent in guard this round
  steps: torch.Tensor  # steps this round
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
        recent_type=torch.full((n, MIX_MEMORY), -1, dtype=torch.long, device=dev),
        recent_level=torch.full((n, MIX_MEMORY), -1, dtype=torch.long, device=dev), mix=z(), mix_sum=z(), points=z(),
        type_count=torch.zeros(n, len(skills.PUNCHES), device=dev), level_count=torch.zeros(n, len(LEVELS), device=dev),
        guard_hit=torch.zeros(n, dtype=torch.bool, device=dev), was_punch=torch.zeros(n, dtype=torch.bool, device=dev),
        blocked=torch.zeros(n, dtype=torch.bool, device=dev), blocked_count=z(),
        thrown_count=z(), missed_count=z(), guard_steps=z(), steps=z(),
        zone_count=torch.zeros(n, len(ZONES), device=dev), base=z(), force_peak=z(), force_sum=z(),
        tilt0=z(), tilt_peak=z(), push0=z(), push_peak=z(),
        staggered=torch.zeros(n, dtype=torch.bool, device=dev), d_tilt=z(), d_push=z(),
        stagger=torch.zeros(n, dtype=torch.bool, device=dev),
      )  # fmt: skip
      for t in fight.TEAMS
    }
    self.punch_ids = torch.tensor([skills.MOVES.index(m) for m in skills.PUNCHES], device=dev)
    self.punch_type = torch.full((len(skills.MOVES),), -1, dtype=torch.long, device=dev)  # MOVES -> PUNCHES index
    self.punch_type[self.punch_ids] = torch.arange(len(skills.PUNCHES), device=dev)
    self.zone_level = torch.tensor(ZONE_LEVEL, device=dev)


def landing_sensors():
  """Where a glove lands on the torso, and gloves on the opponent's guard (added
  to the scene by the fight cfg)."""
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  sensors = []
  for team in fight.TEAMS:
    opp = fight.other(team)
    sensors += [
      ContactSensorCfg(
        name=f"{team}_lands_{opp}_body",
        primary=ContactMatch(mode="geom", pattern=r"glove_[lr]", entity=team),
        secondary=ContactMatch(mode="geom", pattern="torso", entity=opp),
        fields=("found", "pos"),
        reduce="maxforce",
      ),
    ]
    sensors += [  # one per glove: a sensor's secondary must be a single geom
      ContactSensorCfg(
        name=f"{team}_glove_{g}_on_{opp}_guard",
        primary=ContactMatch(mode="geom", pattern=(r"^arm_.*$", r"^glove_[lr]$"), entity=opp),
        secondary=ContactMatch(mode="geom", pattern=f"glove_{g}", entity=team),
        fields=("found",),
      )
      for g in ("l", "r")
    ]
  return tuple(sensors)


def _hit(env, attacker, defender):
  head = fight._sensor(env, f"{attacker}_hits_{defender}_head")
  body = fight._sensor(env, f"{attacker}_hits_{defender}_body")
  return (head.found > 0).any(-1), (body.found > 0).any(-1)


def _force(env, attacker, defender):
  """[N] glove contact force on the defender's head + torso this step (N)."""
  head = fight._sensor(env, f"{attacker}_hits_{defender}_head")
  body = fight._sensor(env, f"{attacker}_hits_{defender}_body")
  return head.force.norm(dim=-1).sum(-1) + body.force.norm(dim=-1).sum(-1)


def _landing(env, attacker, defender, head):
  """This step's landing: [N] zone index (into ZONES) and [N] location value
  (HEAD_VALUE / BODY_VALUE)."""
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
  head_geom = them.find_geoms(("head",))[0][0]
  head_pos = them.data.geom_pose_w[:, head_geom, :3]
  near = (1 - ((pos - head_pos).norm(dim=-1) - HEAD_RADIUS) / GRADIENT_REACH).clamp(0, 1)
  value = BODY_VALUE[0] + (BODY_VALUE[1] - BODY_VALUE[0]) * near
  return torch.where(head, 0, zone), torch.where(head, HEAD_VALUE, value)


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
    new_move = player.phase <= player.rate + 1e-6  # a new move started this step
    # blocked: the punch that just ended touched the guard and never landed clean
    s.blocked = new_move & s.was_punch & s.guard_hit & ~s.scored
    s.blocked_count = s.blocked_count + s.blocked.float()
    ended = new_move & s.was_punch
    s.thrown_count = s.thrown_count + ended.float()
    s.missed_count = s.missed_count + (ended & ~s.guard_hit & ~s.scored).float()
    s.guard_steps = s.guard_steps + (my_move == skills.GUARD).float()
    s.steps = s.steps + 1
    guard = torch.stack([(fight._sensor(env, f"{team}_glove_{g}_on_{opp}_guard").found > 0).any(-1) for g in ("l", "r")]).any(0)
    s.guard_hit = (s.guard_hit & ~new_move) | (guard & punching)
    s.was_punch = punching
    # one clean punch per punch move: the glove's contact flickers on and off
    # as it lands and rubs, and each new contact used to count again (up to
    # ~17 per punch: fight_v7 learned to rub rather than punch)
    s.scored = s.scored & ~new_move
    clean = onset & punching & (closing >= MIN_SPEED) & ~s.scored
    s.scored = s.scored | clean
    zone, where = _landing(env, team, opp, head)
    # mix-up over landed punches only: misses and blocked punches leave the memory alone
    kind = im.punch_type[my_move].clamp_min(0)
    level = im.zone_level[zone]
    # shares over the filled memory only: counting empty slots as new gave every
    # round's first punches a free x3 (fight_still_v5 threw only lead hooks at ~x2)
    filled = (s.recent_type >= 0).sum(-1).float()
    other = lambda mem, x: ((mem >= 0) & (mem != x[:, None])).sum(-1).float() / filled.clamp_min(1)  # noqa: E731
    mix = 1 + TYPE_BONUS * other(s.recent_type, kind) + LEVEL_BONUS * other(s.recent_level, level)
    push = lambda mem, x: torch.where(clean[:, None], torch.cat([mem[:, 1:], x[:, None]], -1), mem)  # noqa: E731
    s.recent_type, s.recent_level = push(s.recent_type, kind), push(s.recent_level, level)
    s.type_count[clean, kind[clean]] += 1
    s.level_count[clean, level[clean]] += 1
    s.mix = torch.where(clean, mix, s.mix)
    s.mix_sum = s.mix_sum + torch.where(clean, mix, torch.zeros_like(mix))
    value = where * mix * (closing / SPEED_REF).clamp(0, SPEED_CAP)
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
    s.points = s.points + s.impact
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
    s.recent_type[ids] = -1
    s.recent_level[ids] = -1
    s.type_count[ids] = 0.0
    s.level_count[ids] = 0.0
    s.mix_sum[ids] = 0.0
    s.points[ids] = 0.0
    s.guard_hit[ids] = False
    s.was_punch[ids] = False
    s.blocked_count[ids] = 0.0
    s.thrown_count[ids] = 0.0
    s.missed_count[ids] = 0.0
    s.guard_steps[ids] = 0.0
    s.steps[ids] = 0.0


# --------------------------------------------------------------------------- #
# Reward terms (from `team`'s side)
# --------------------------------------------------------------------------- #


def clean_punches(env, team: str) -> torch.Tensor:
  return impacts(env).sides[team].impact


def blocked_punches(env, team: str) -> torch.Tensor:
  """1 when a punch that only hit the opponent's guard ends."""
  return impacts(env).sides[team].blocked.float()


def blocks_made(env, team: str) -> torch.Tensor:
  """1 when an opponent's punch that only hit this fighter's guard ends. Weight 0
  by default: a fallback for self-play that trades without defending (paying for
  blocks invites a guard-only stalemate, so keep it well under a clean punch)."""
  return impacts(env).sides[fight.other(team)].blocked.float()


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


def _exchange(im, ids, team: str) -> dict:
  """Both sides' punching and defence over the rounds `ids`."""
  s, o = im.sides[team], im.sides[fight.other(team)]
  thrown = o.thrown_count[ids].sum()
  return {
    "clean_per_round": s.count[ids].mean(),
    "taken_per_round": o.count[ids].mean(),
    "points_per_round": s.points[ids].mean(),
    "points_taken_per_round": o.points[ids].mean(),
    "opp_thrown_per_round": o.thrown_count[ids].mean(),
    "blocks_made_per_round": o.blocked_count[ids].mean(),  # the opponent's punches stopped by this guard
    "evaded_per_round": o.missed_count[ids].mean(),  # ... that touched nothing (slipped, out of reach)
    # share of the opponent's punches that did not land clean (0 if it threw none)
    "defence_rate": torch.where(thrown > 0, 1 - o.count[ids].sum() / thrown.clamp_min(1), torch.zeros_like(thrown)),
    "guard_share": s.guard_steps[ids].sum() / s.steps[ids].sum().clamp_min(1),
    "mix_per_punch": s.mix_sum[ids].sum() / s.count[ids].sum().clamp_min(1),
  }


def punch_stats(env, env_ids, team: str) -> dict:
  """Curriculum-slot logger: per finished round, clean punches and by zone, and
  the same against each opponent mode (vs_still/, vs_scripted/, vs_selfplay/;
  runs before the reset clears the counts and redraws the mode)."""
  if env_ids is None or len(env_ids) == 0:
    return {}
  im = impacts(env)
  s = im.sides[team]
  mode = skills._skills(env, fight.other(team)).mode[env_ids]
  extra = {}
  for k, name in ((skills.HOLD, "vs_still"), (skills.SCRIPTED, "vs_scripted"), (skills.SELFPLAY, "vs_selfplay")):
    sel = env_ids[mode == k]
    if len(sel):
      extra.update({f"{name}/{key}": v for key, v in _exchange(im, sel, team).items()})
  out = {"clean_per_round": s.count[env_ids].mean()}
  for k, z in enumerate(ZONES):
    out[z] = s.zone_count[env_ids, k].mean()
  out["force_per_punch_N"] = (s.force_sum[env_ids].sum() / s.count[env_ids].sum().clamp_min(1))
  out["taken_per_round"] = im.sides[fight.other(team)].count[env_ids].mean()
  out["points_per_round"] = s.points[env_ids].mean()
  out["points_taken_per_round"] = im.sides[fight.other(team)].points[env_ids].mean()
  out["blocked_per_round"] = s.blocked_count[env_ids].mean()
  out["mix_per_punch"] = s.mix_sum[env_ids].sum() / s.count[env_ids].sum().clamp_min(1)
  for k, m in enumerate(skills.PUNCHES):
    out[f"type/{m}"] = s.type_count[env_ids, k].mean()
  for k, lv in enumerate(LEVELS):
    out[f"level/{lv}"] = s.level_count[env_ids, k].mean()
  out["guard_share"] = s.guard_steps[env_ids].sum() / s.steps[env_ids].sum().clamp_min(1)
  return {**out, **extra}
