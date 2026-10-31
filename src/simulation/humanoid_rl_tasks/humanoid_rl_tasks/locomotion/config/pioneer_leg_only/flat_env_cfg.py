"""Pioneer leg-only and whole-body humanoids, flat-ground velocity tracking with a gait clock."""
import os

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from pioneer_humanoid.leg_only import LEG_ONLY_HUMANOID_CFG, WHOLE_BODY_ARM_SWING_CFG

from ... import mdp as loco_mdp
from . import gait
from .arm_swing import ArmSwingActionCfg

FEET = ".*_foot"
LEGS = ["Hip_.*", "Knee_.*"]


@configclass
class PioneerLegOnlyRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")})
    # self-collision is off, so nothing else stops the feet stepping in line or through each other (ankles stand 0.31 m apart)
    feet_too_close = RewTerm(
        func=loco_mdp.feet_crossing_l2,
        weight=-40.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True), "margin": 0.2},
    )


@configclass
class PioneerLegOnlyFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: PioneerLegOnlyRewards = PioneerLegOnlyRewards()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = LEG_ONLY_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # randomised so the policy survives another simulator (MuJoCo) and the real robot
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEGS)
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEGS)

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        self.observations.policy.gait_clock = ObsTerm(func=gait.gait_clock)
        # a single robot hitting a physics glitch once sent the critic's loss to inf: clip what it can see, end such episodes
        for term in vars(self.observations.policy).values():
            if isinstance(term, ObsTerm):
                term.modifiers = [ModifierCfg(func=gait.finite)]
                term.clip = (-100.0, 100.0)
        self.terminations.physics_blowup = DoneTerm(func=gait.physics_blowup)
        self.actions.joint_pos.scale = 0.25

        feet_contact = SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"])
        feet = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"])
        r = self.rewards
        r.feet_air_time = None
        r.contact = RewTerm(func=gait.contact_matches_phase, weight=0.18, params={"sensor_cfg": feet_contact})
        r.feet_swing_height = RewTerm(func=gait.feet_swing_height, weight=-20.0, params={"sensor_cfg": feet_contact, "asset_cfg": feet})
        r.contact_no_vel = RewTerm(func=gait.stance_foot_velocity, weight=-0.2, params={"sensor_cfg": feet_contact, "asset_cfg": feet})
        r.base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.835})  # near-straight stance legs (0.848)
        r.alive = RewTerm(func=mdp.is_alive, weight=0.15)
        # loosening this on hip roll (for side steps) made every run diverge and did not speed up side steps
        r.hip_pos = RewTerm(func=gait.joint_pos_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Hip_A_.*", "Hip_R_.*"])})
        r.dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS + ["Ankle_.*"])})
        r.track_ang_vel_z_exp.weight = 0.5
        r.lin_vel_z_l2.weight = -2.0
        r.ang_vel_xy_l2.weight = -0.05
        r.action_rate_l2.weight = -0.02
        r.dof_acc_l2.weight = -2.5e-7
        r.stance_knee = RewTerm(
            func=gait.stance_knee, weight=-1.0,
            params={"target": -0.15, "asset_cfg": SceneEntityCfg("robot", joint_names=["Knee_L", "Knee_R"], preserve_order=True)},
        )
        # land flat: a sole tipped onto its heel edge is where Isaac and MuJoCo contacts differ most (backward falls in MuJoCo)
        r.feet_flat = RewTerm(func=gait.stance_foot_tilt, weight=-2.0, params={"sensor_cfg": feet_contact, "asset_cfg": feet})

        # motors, joints and ground that differ from the model, so the gait survives another simulator and the real robot
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "stiffness_distribution_params": (0.7, 1.3),
                    "damping_distribution_params": (0.7, 1.3), "operation": "scale"},
        )
        # start every episode a little off the default pose and moving, so the first steps are robust too
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset, mode="reset", params={"position_range": (-0.1, 0.1), "velocity_range": (-0.2, 0.2)}
        )
        self.events.reset_base.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.3, 0.3)}
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}},
        )
        self.events.push_robot.interval_range_s = (5.0, 10.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}
        self.events.joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "armature_distribution_params": (0.5, 2.0), "operation": "scale"},
        )


def _play(cfg) -> None:
    """One robot's worth of play settings: no noise or pushes, walking straight ahead past a side camera."""
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    # plain grey sky (the default sky photo washes out the horizon); camera offsets are from the base, at hip height
    cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.78))
    # straight ahead at 0.5 m/s, facing +X, so the side camera sees the gait
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    # PLAY_COMMAND="vx,vy,wz" picks another command, e.g. "0.3,0,0.6" to turn or "0,0.3,0" to step sideways;
    # "random" keeps the training ranges and draws a new command every 3 s, so the robot wanders around
    command = os.environ.get("PLAY_COMMAND", "0.5,0,0")
    if command == "random":
        cfg.commands.base_velocity.resampling_time_range = (3.0, 3.0)
        cfg.commands.base_velocity.rel_standing_envs = 0.15
        cfg.episode_length_s = 60.0
    else:
        vx, vy, wz = (float(v) for v in command.split(","))
        # exactly this command for the whole video: never a "stand still" draw, no heading target, no re-draw
        cfg.commands.base_velocity.rel_standing_envs = 0.0
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        cfg.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
        cfg.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
        cfg.commands.base_velocity.ranges.ang_vel_z = (wz, wz)
    cfg.viewer = ViewerCfg(eye=(1.4, -3.0, 0.1), lookat=(0.3, 0.0, -0.15), origin_type="asset_root", asset_name="robot")


def _demo_look(cfg) -> None:
    """Demo look: three-quarter front view from slightly above, a key light for shadows, full HD, no command arrows."""
    cfg.viewer = ViewerCfg(eye=(2.6, -2.2, 0.55), lookat=(0.2, 0.0, -0.1), resolution=(1920, 1080), origin_type="asset_root", asset_name="robot")
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=700.0, color=(0.8, 0.82, 0.86))
    cfg.scene.key_light = AssetBaseCfg(
        prim_path="/World/keyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=1.5, color=(1.0, 0.97, 0.92)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.82, 0.25, 0.25, 0.45)),
    )


@configclass
class PioneerLegOnlyFlatEnvCfg_PLAY(PioneerLegOnlyFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)


# the leg-only robot's joint order, which its policies' observations and actions follow
LEG_ORDER = ["Hip_F_L", "Hip_F_R", "Hip_A_L", "Hip_A_R", "Hip_R_L", "Hip_R_R", "Knee_L", "Knee_R", "Ankle_P_L", "Ankle_P_R", "Ankle_R_L", "Ankle_R_R"]


@configclass
class PioneerWholeBodyFlatEnvCfg(PioneerLegOnlyFlatEnvCfg):
    """The same task on the whole-body humanoid (arms and hands, 44.7 kg): fists closed, shoulders and elbows swung on the
    gait clock (arm_swing.py), the rest of the upper body held rigid. The policy sees and drives only the 12 leg joints,
    in the leg-only robot's order, so a leg-only policy can start it."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHOLE_BODY_ARM_SWING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        legs = SceneEntityCfg("robot", joint_names=LEG_ORDER, preserve_order=True)
        self.actions.joint_pos.joint_names = LEG_ORDER
        self.actions.joint_pos.preserve_order = True
        self.actions.arm_swing = ArmSwingActionCfg()
        self.observations.policy.joint_pos.params["asset_cfg"] = legs
        self.observations.policy.joint_vel.params["asset_cfg"] = legs


@configclass
class PioneerWholeBodyFlatEnvCfg_PLAY(PioneerWholeBodyFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)
        _demo_look(self)
