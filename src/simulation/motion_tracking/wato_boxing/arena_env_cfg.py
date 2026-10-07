"""Two Wato fighters in a ring: the scene for fighting RL.

Red starts on the -x side facing +x, blue on the +x side facing -x, both in
the orthodox stance, `distance` apart (centre between the feet to centre
between the feet). At the default 1.3 m the lead gloves start about 0.2 m
apart and a jab reaches the opponent's head only after stepping in about
0.3 m (three 10 cm steps), so neither fighter starts in range.

Contact sensors for later rewards: <attacker>_hits_<defender>_head / _body
(a glove touching the opponent's head / torso), with found + net force.

This env has no task yet: no rewards, and each fighter's action is a joint
position target around the stance (the tracking task's action). It exists to
build and view the scene; the fight task replaces the actions (skill
commands), observations and rewards.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from wato_boxing.fighters import get_fighter_cfg
from wato_boxing.ring import get_ring_cfg
from wato_tracking.robot import WATO_ACTION_SCALE

TEAMS = ("red", "blue")


def hit_sensors() -> tuple[ContactSensorCfg, ...]:
  sensors = []
  for attacker, defender in (("red", "blue"), ("blue", "red")):
    for target, geom in (("head", "head"), ("body", "torso")):
      sensors.append(
        ContactSensorCfg(
          name=f"{attacker}_hits_{defender}_{target}",
          primary=ContactMatch(mode="geom", pattern=r"glove_[lr]", entity=attacker),
          secondary=ContactMatch(mode="geom", pattern=geom, entity=defender),
          fields=("found", "force"),
          reduce="netforce",
        )
      )
  return tuple(sensors)


def boxing_arena_env_cfg(distance: float = 1.3, ring_size: float = 4.0, num_envs: int = 1) -> ManagerBasedRlEnvCfg:
  half = distance / 2
  scene = SceneCfg(
    num_envs=num_envs,
    env_spacing=ring_size + 1.5,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={
      "red": get_fighter_cfg("red", position_xy=(-half, 0.0), yaw=0.0),
      "blue": get_fighter_cfg("blue", position_xy=(half, 0.0), yaw=3.141592653589793),
      "ring": get_ring_cfg(ring_size),
    },
    sensors=hit_sensors(),
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
  observations = {
    "actor": ObservationGroupCfg(
      terms={
        f"{team}_joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(team)})
        for team in TEAMS
      },
      concatenate_terms=True,
    ),
  }
  observations["critic"] = observations["actor"]

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    rewards={},
    terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
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
    episode_length_s=10.0,
  )
