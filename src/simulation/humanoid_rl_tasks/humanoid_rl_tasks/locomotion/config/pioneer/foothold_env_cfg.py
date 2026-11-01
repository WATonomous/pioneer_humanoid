"""The height-scan task with foothold-conditioned perception (foothold.py): the policy sees 7 x 7 height patches around
the two footholds a small predictor expects next, instead of the whole height map."""
import copy
import os

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import foothold, gait
from .flat_env_cfg import _play
from .rough_env_cfg import PioneerLegOnlyPerceptiveEnvCfg

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))
_FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True)
_FEET = SceneEntityCfg("robot", body_names=["l_foot", "r_foot"], preserve_order=True)
_SCAN = {"sensor_cfg": SceneEntityCfg("height_scanner"), "scan_resolution": 0.05, "scan_size": (1.6, 1.0), "patch_size": 7,
         "bootstrap_footholds": ((0.0, 0.15), (0.0, -0.15))}  # feet under the hips (0.15 m to each side) until it has a model


@configclass
class _SystemStateCfg(ObsGroup):
    """The foot predictor's input: base motion, gravity, command, joints."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)


@configclass
class _SystemContactCfg(ObsGroup):
    feet_contact = ObsTerm(func=foothold.body_contact, params={"sensor_cfg": _FEET_CONTACT, "threshold": 1.0})


@configclass
class _SystemPoseCfg(ObsGroup):
    base_pose_w = ObsTerm(func=foothold.base_pose_w)
    feet_pose_w = ObsTerm(func=foothold.body_pose_w, params={"asset_cfg": _FEET})
    feet_pos_b = ObsTerm(func=foothold.body_pos_b, params={"asset_cfg": _FEET})


@configclass
class PioneerLegOnlyFootholdEnvCfg(PioneerLegOnlyPerceptiveEnvCfg):
    """The policy sees its predicted next footholds and a 7 x 7 patch (5 cm cells) around each; the critic (training only)
    also sees the whole height map. The foot predictor is refit during training (foot models go to log_dir). Gravity
    assistance carries up to half the robot's weight on the easiest terrain, fading out by level 4."""

    log_dir: str = os.path.join(_REPO, "outputs", "rl", "pioneer_leg_only_rough_foothold", "foot_models")

    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.log_dir, exist_ok=True)
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0])
        policy = self.observations.policy
        policy.height_scan = None
        policy.foothold_scan = ObsTerm(
            func=foothold.foothold_scan,
            params={**_SCAN, "xy_drift_range": (-0.05, 0.05), "z_drift_range": (-0.05, 0.05)},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            modifiers=[ModifierCfg(func=gait.finite)],
            clip=(-1.0, 1.0),
        )
        critic = copy.deepcopy(policy)
        critic.enable_corruption = False
        critic.height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                                     modifiers=[ModifierCfg(func=gait.finite)], clip=(-1.0, 1.0))
        critic.foothold_scan = ObsTerm(func=foothold.foothold_scan, params=dict(_SCAN), modifiers=[ModifierCfg(func=gait.finite)],
                                       clip=(-1.0, 1.0))
        self.observations.critic = critic
        self.observations.system_state = _SystemStateCfg()
        self.observations.system_contact = _SystemContactCfg()
        self.observations.system_pose = _SystemPoseCfg()

        self.events.gravity_assist = EventTerm(
            func=foothold.apply_gravity_compensation,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "init_offset": 0.5, "cutoff_level": 4.0},
        )
        self.curriculum.gravity_offset_level = CurrTerm(func=foothold.gravity_offset_level, params={"init_offset": 0.5, "cutoff_level": 4.0})


@configclass
class PioneerLegOnlyFootholdEnvCfg_PLAY(PioneerLegOnlyFootholdEnvCfg):
    """Uses the trained foot predictor in models/ and never refits it."""

    log_dir: str = os.path.join(_REPO, "models", "pioneer_leg_only_rough_foothold")

    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)
        self.events.gravity_assist = None
        self.curriculum.gravity_offset_level = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        for group in (self.observations.policy, self.observations.critic):
            group.foothold_scan.params.update(train_fpm=False, force_model=True)
