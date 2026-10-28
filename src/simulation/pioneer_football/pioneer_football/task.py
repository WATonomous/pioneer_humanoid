"""Flat-ground astronaut locomotion. One 12-action policy, full body mass."""

from copy import deepcopy

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as builtin
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg as Event
from mjlab.managers.metrics_manager import MetricsTermCfg as Metric
from mjlab.managers.observation_manager import (
    ObservationGroupCfg as Group,
)
from mjlab.managers.observation_manager import (
    ObservationTermCfg as Obs,
)
from mjlab.managers.reward_manager import RewardTermCfg as Reward
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg as Done
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Noise
from mjlab.viewer import ViewerConfig

from . import mdp
from .model import JOINTS, REST, SCALES, robot_spec, spawn_height

TASK = "Pioneer-Football-Locomotion-v0"


def make_env_cfg(play=False):
    height = spawn_height()
    robot = EntityCfg(
        spec_fn=robot_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, height), joint_pos={n + "$": q for n, q in REST.items()}
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=JOINTS),),
            soft_joint_pos_limit_factor=0.95,
        ),
    )
    feet = SceneEntityCfg("robot", body_names=("l_foot", "r_foot"), preserve_order=True)
    sensors = (
        ContactSensorCfg(
            name="feet",
            primary=ContactMatch(
                mode="body", pattern=("l_foot", "r_foot"), entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="body_ground",
            primary=ContactMatch(
                mode="body", pattern=".*", entity="robot", exclude=("l_foot", "r_foot")
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="netforce",
        ),
    )
    terms = {
        "linear_velocity": Obs(
            func=builtin.base_lin_vel, noise=Noise(n_min=-0.05, n_max=0.05)
        ),
        "angular_velocity": Obs(
            func=builtin.base_ang_vel, noise=Noise(n_min=-0.1, n_max=0.1)
        ),
        "gravity": Obs(
            func=builtin.projected_gravity, noise=Noise(n_min=-0.02, n_max=0.02)
        ),
        "commands": Obs(
            func=builtin.generated_commands, params={"command_name": "base_velocity"}
        ),
        "joint_positions": Obs(
            func=builtin.joint_pos_rel, noise=Noise(n_min=-0.01, n_max=0.01)
        ),
        "joint_velocities": Obs(
            func=builtin.joint_vel_rel, noise=Noise(n_min=-0.3, n_max=0.3), scale=0.05
        ),
        "actions": Obs(func=builtin.last_action),
    }
    commands = {
        "base_velocity": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 6.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            rel_forward_envs=0.0,
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.6), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.5, 0.5)
            ),
        )
    }
    events = {
        "reset_base": Event(
            func=builtin.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"yaw": (-3.14, 3.14), "z": (0.005, 0.015)},
                "velocity_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "yaw": (-0.05, 0.05),
                },
            },
        ),
        "reset_joints": Event(
            func=builtin.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.01, 0.01), "velocity_range": (-0.05, 0.05)},
        ),
        "foot_friction": Event(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=("l_foot_collision_0", "r_foot_collision_0")
                ),
                "operation": "abs",
                "ranges": (0.6, 1.0),
                "shared_random": True,
            },
        ),
    }
    rewards = {
        "track_linear": Reward(func=mdp.track_linear, weight=1.0),
        "track_yaw": Reward(func=mdp.track_yaw, weight=1.0),
        "air_time": Reward(func=mdp.air_time, weight=2.0),
        "feet_slide": Reward(
            func=mdp.feet_slide, weight=-0.1, params={"asset_cfg": feet}
        ),
        "feet_crossing": Reward(
            func=mdp.feet_crossing, weight=-40.0, params={"asset_cfg": feet}
        ),
        "upright": Reward(func=builtin.flat_orientation_l2, weight=-1.0),
        "vertical_velocity": Reward(func=mdp.vertical_velocity, weight=-0.2),
        "action_rate": Reward(func=builtin.action_rate_l2, weight=-0.005),
        "joint_acceleration": Reward(func=builtin.joint_acc_l2, weight=-1e-7),
        "torques": Reward(func=builtin.joint_torques_l2, weight=-2e-6),
        "joint_limits": Reward(func=builtin.joint_pos_limits, weight=-1.0),
        "hip_deviation": Reward(
            func=mdp.hip_deviation,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("Hip_A_.*", "Hip_R_.*")
                )
            },
        ),
        "overspeed": Reward(func=mdp.overspeed, weight=-1.0),
        "fall": Reward(func=builtin.is_terminated, weight=-100.0),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=512,
            env_spacing=3.0,
            entities={"robot": robot},
            sensors=sensors,
            terrain=TerrainEntityCfg(terrain_type="plane"),
        ),
        observations={
            "actor": Group(terms=terms, enable_corruption=not play),
            "critic": Group(terms=deepcopy(terms), enable_corruption=False),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=JOINTS,
                scale=SCALES,
                use_default_offset=True,
            )
        },
        commands=commands,
        events=events,
        rewards=rewards,
        terminations={
            "timeout": Done(func=builtin.time_out, time_out=True),
            "tilt": Done(func=builtin.bad_orientation, params={"limit_angle": 0.8}),
            "low_pelvis": Done(
                func=builtin.root_height_below_minimum, params={"minimum_height": 0.55}
            ),
            "body_ground": Done(func=mdp.body_ground),
        },
        metrics={
            name: Metric(func=func, reduce=reduce)
            for name, func, reduce in (
                ("speed_error_mps", mdp.tracking_error, "mean"),
                ("yaw_error_rps", mdp.yaw_error, "mean"),
                ("forward_speed_mps", mdp.forward_speed, "mean"),
                ("peak_effort_fraction", mdp.effort_fraction, "max"),
                ("peak_speed_fraction", mdp.speed_fraction, "max"),
            )
        },
        sim=SimulationCfg(
            nconmax=128,
            njmax=512,
            contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(timestep=0.002),
        ),
        decimation=10,
        episode_length_s=20.0,
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="base",
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            distance=3.0,
        ),
    )
    if play:
        cfg.scene.num_envs = 1
    return cfg


def make_rl_cfg():
    return RslRlOnPolicyRunnerCfg(
        experiment_name="pioneer_football_locomotion",
        logger="tensorboard",
        upload_model=False,
        num_steps_per_env=24,
        max_iterations=10000,
        save_interval=100,
        clip_actions=1.0,
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 128),
            obs_normalization=True,
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 128), obs_normalization=True, activation="elu"
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1e-3,
            entropy_coef=0.008,
            num_learning_epochs=5,
            num_mini_batches=4,
        ),
    )


def register():
    from mjlab.tasks.registry import register_mjlab_task

    register_mjlab_task(
        TASK,
        env_cfg=make_env_cfg(),
        play_env_cfg=make_env_cfg(play=True),
        rl_cfg=make_rl_cfg(),
    )
