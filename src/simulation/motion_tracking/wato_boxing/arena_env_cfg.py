"""Two Wato fighters in a ring: the scene for fighting RL.

Red starts on the -x side facing +x, blue on the +x side facing -x, both in
the orthodox stance, `distance` apart (centre between the feet to centre
between the feet). At the default 1.3 m the lead gloves start about 0.2 m
apart and a jab reaches the opponent's head only after stepping in about
0.3 m (three 10 cm steps), so neither fighter starts in range.

Rewards and round endings (wato_boxing/mdp.py, weights in REWARD_WEIGHTS) are
scored from the `learner`'s side: knockdown +100 / knocked down -100 (the
robot cannot get up, so the first fighter down ends the round), clean hits,
stability, and penalties for shoving and unrealistic movement; self-collision
ends the round as a loss; at the bell, points by landed-hit difference.

Observations: the actor sees the opponent only through its sensors, an EKF
fusing a simulated RGB camera (D455 on the shoulder-frame bar) and a chest
ultrasonic sensor (wato_boxing/perception.py); the critic also gets the exact
opponent state. Actions are still a placeholder: each fighter's action is a
joint position target around the stance (the tracking task's action); the
fight task replaces it with skill commands.

Standing without a policy: the tracking gains (10 Hz, `gain_scale=1`) are
soft on purpose, the policy does the balancing, and a crouched fighter folds
under them in about a second. The arena holds the stance with 4x stiffness
(same damping ratio, same torque limits) so the fighters stand on their own:
the hips settle ~2 cm and stay. A fight task with a trained policy should go
back to `gain_scale=1`, the gains the skills are trained with.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from wato_boxing import mdp as fight
from wato_boxing import perception as vision
from wato_boxing.perception import PerceptionCfg
from wato_boxing.fighters import get_fighter_cfg
from wato_boxing.ring import get_ring_cfg
from wato_tracking.robot import WATO_ACTION_SCALE

TEAMS = ("red", "blue")


# Points per event (scale_rewards_by_dt=False): terminal events are worth
# their weight once; per-step terms are summed over the 50 Hz steps.
REWARD_WEIGHTS = {
  # outcome (the round ends)
  "knockdown": 100.0,  # opponent down within 1 s of taking a glove hit
  "opponent_slip": 10.0,  # opponent down without being hit
  "knocked_down": -100.0,  # down while the opponent stays up (hit or own fall)
  "double_down": -50.0,
  "self_collision": -100.0,  # limbs through own body: round ends
  "time_up_points": 10.0,  # at the bell: +-10 by landed-hit difference
  # boxing
  "clean_hits": 2.0,  # per step of glove contact, x force / 100 N, head x1 body x0.5
  "hits_taken": -1.0,
  "stability": 0.02,  # upright at stance height, per step
  "shoving": -0.5,  # head/torso/pelvis/legs pressing on the opponent's torso or pelvis
  # unrealistic movement
  "overspeed": -1.0,  # sum over joints of (|speed| / cap - 0.8)+
  "torque_saturation": -0.5,  # share of motors at 95% of their torque limit
  "joint_limits": -1.0,  # past 90% of a joint's range (claws excluded)
  "jerkiness": -1e-6,  # mean joint acceleration^2
  "foot_slip": -0.5,  # planted-foot speed, m/s
  "airborne": -2.0,  # both feet off the floor
  "rope_contact": -2.0,
}


def fight_rewards(team: str = "red", weights: dict[str, float] | None = None) -> dict[str, RewardTermCfg]:
  """Reward terms from `team`'s side (the learner)."""
  weights = {**REWARD_WEIGHTS, **(weights or {})}
  terms = {}
  for name, w in weights.items():
    if name == "joint_limits":
      terms[name] = RewardTermCfg(
        # claws are held closed at 0, just outside their soft band; not boxing joints
        func=mdp.joint_pos_limits,
        weight=w,
        params={"asset_cfg": SceneEntityCfg(team, joint_names=(r"^(?!.*claw).*$",))},
      )
    else:
      terms[name] = RewardTermCfg(func=getattr(fight, name), weight=w, params={"team": team})
  return terms


def boxing_arena_env_cfg(
  distance: float = 1.3,
  ring_size: float = 4.0,
  num_envs: int = 1,
  gain_scale: float = 4.0,
  learner: str = "red",
  round_s: float = 20.0,
  perception: PerceptionCfg | None = None,
) -> ManagerBasedRlEnvCfg:
  half = distance / 2
  scene = SceneCfg(
    num_envs=num_envs,
    env_spacing=ring_size + 1.5,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={
      "red": get_fighter_cfg("red", position_xy=(-half, 0.0), yaw=0.0, gain_scale=gain_scale),
      "blue": get_fighter_cfg("blue", position_xy=(half, 0.0), yaw=3.141592653589793, gain_scale=gain_scale),
      "ring": get_ring_cfg(ring_size),
    },
    sensors=fight.fight_sensors(),
  )

  actions = {
    team: JointPositionActionCfg(
      entity_name=team,
      actuator_names=(".*",),
      scale=WATO_ACTION_SCALE,
      use_default_offset=True,
    )
    for team in TEAMS
  }
  joint_terms = {
    f"{team}_joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(team)})
    for team in TEAMS
  }
  pcfg = perception or PerceptionCfg()
  observations = {
    # the policy sees the opponent only through its sensors (EKF estimate)
    "actor": ObservationGroupCfg(
      terms={
        **joint_terms,
        "opponent": ObservationTermCfg(func=vision.opponent_perceived, params={"team": learner, "cfg": pcfg}),
      },
      concatenate_terms=True,
    ),
    # the critic (and a teacher policy) gets the exact opponent state
    "critic": ObservationGroupCfg(
      terms={
        **joint_terms,
        "opponent_perceived": ObservationTermCfg(func=vision.opponent_perceived, params={"team": learner, "cfg": pcfg}),
        "opponent_true": ObservationTermCfg(func=vision.opponent_true, params={"team": learner, "cfg": pcfg}),
      },
      concatenate_terms=True,
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    rewards=fight_rewards(learner),
    scale_rewards_by_dt=False,
    terminations={
      "round_over": TerminationTermCfg(func=fight.round_over),
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    },
    events={
      "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
      "reset_fight_state": EventTermCfg(func=fight.reset_fight_state, mode="reset"),
      "reset_trackers": EventTermCfg(func=vision.reset_trackers, mode="reset"),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.WORLD,
      lookat=(0.0, 0.0, 0.8),
      distance=4.0,
      elevation=-15.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=80,
      njmax=600,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=round_s,
  )
