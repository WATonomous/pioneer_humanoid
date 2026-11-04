from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from pioneer_humanoid.whole_body import WHOLE_BODY_HUMANOID_GAINS_B_CFG

from ... import mdp
from ...locomotion_env_cfg import TerminationsCfg
from .rough_env_cfg import LEG_JOINTS, PioneerHumanoidRewards, PioneerHumanoidRoughEnvCfg, use_robot


@configclass
class PioneerHumanoidFlatRewards(PioneerHumanoidRewards):
    """Kept close to official Isaac Lab G1 flat: no gait-phase clock, no swing-direction
    shaping, no body-spacing constraints. Those custom terms let the policy satisfy reward
    with a stepping *pattern* that produced no net forward displacement (verified: episodes
    with strong phase_* rewards but forward_velocity stuck at exactly 0.0 for 400+ iterations).
    G1's reward set relies on track_lin_vel_xy_exp as the dominant, ungated signal instead.

    A custom double_support_penalty term (curriculum-gated) was carried here for a while
    to break a repeatedly-observed frozen double-support local optimum. Once domain
    randomization was scaled down closer to G1's own (much gentler) reset distribution,
    real stepping emerged naturally without it (reduced_randomization_fresh_002:
    feet_air_time rose 0.0131 -> 0.1364 well before the penalty's curriculum was even
    scheduled to activate) -- removed as unnecessary once that was confirmed live.

    feet_crossing_penalty (anti leg-crossing) now lives in the shared
    PioneerHumanoidRewards base class in rough_env_cfg.py, inherited here -- it applies
    to both terrains equally, not just flat."""


@configclass
class PioneerHumanoidFlatTerminations(TerminationsCfg):
    base_tilt = None
    low_base = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.30})


@configclass
class PioneerHumanoidFlatEnvCfg(PioneerHumanoidRoughEnvCfg):
    rewards: PioneerHumanoidFlatRewards = PioneerHumanoidFlatRewards()
    terminations: PioneerHumanoidFlatTerminations = PioneerHumanoidFlatTerminations()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # base_height_l2 (rough-only) depends on the height_scanner sensor removed
        # above -- flat solves the same sitting/frozen-stance risk via the low_base
        # termination in PioneerHumanoidFlatTerminations instead.
        self.rewards.base_height_l2 = None

        self.actions.joint_pos.scale = {
            "Hip_F_.*": 0.25,
            "Knee_.*": 0.30,
            "Hip_A_.*": 0.06,
            "Hip_R_.*": 0.15,
            "Ankle_R_.*": 0.08,
            "Ankle_P_.*": 0.12,
        }

        # Reverted: tightening reset randomization (base velocity +-0.1, joint
        # position_range 0.9-1.1) converged to the identical ~40-42 step ceiling as
        # full-strength randomization, and the rollout video showed the same rigid
        # forward-topple failure mode -- ruled out reset perturbation as the bottleneck.

        # joint_vel observation noise is inherited from the base ObservationsCfg at a
        # flat +-1.5 rad/s (copied from Isaac Lab's G1-scale default). Hip_F/Knee here
        # have velocity_limit_sim=3.6652 rad/s (see pioneer_humanoid_v1.py) -- that noise
        # is ~41% of their full range, versus a small fraction for G1's much
        # faster-rated joints. diagnose_spawn.py proved zero-action survives 60+ steps
        # under full physical randomization while every trained policy (even near-zero
        # noise/entropy) still falls by ~35-40 steps -- the fall is coming from the
        # policy's closed-loop reaction to something, and corrupted joint velocity
        # proprioception on the slowest, most balance-critical joints is the leading
        # remaining candidate. Scaled down to a value proportionate to Wato's actual
        # actuator speeds instead of G1's.
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.3, n_max=0.3)

        # Matched to official Isaac Lab G1 flat: track_lin_vel_xy_exp (inherited from rough,
        # weight=1.0, std=0.5) stays the dominant, ungated reward -- only retune a handful of
        # weights + command range, same pattern as G1FlatEnvCfg.
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # Raised from 0.75, threshold lowered from 0.4: ankle_stiffness_extended_2000's
        # healthy checkpoints (iter 1000-1500) reached full 1000-step episodes but with
        # feet_air_time reward stuck at exactly 0.0 -- the robot found a static brace
        # pose that already scores ~0.92/1.0 on track_lin_vel_xy_exp (std=0.5 is
        # forgiving enough that standing-still error looks "close enough"), so there
        # was no marginal incentive to risk stepping. Lower threshold makes even a
        # brief foot-lift count; higher weight makes it worth the risk.
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.2
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
        self.rewards.termination_penalty.weight = -100.0
        # Reverted to the original std=0.5: both 0.25 and 0.4 overcorrected and
        # produced too weak a gradient (stepping_fix_fresh_002/003 both plateaued
        # around episode length 45-47, nowhere near the 990/1000 full-episode
        # stability the original std=0.5 config actually reached by iteration 1000).
        # The real fix for "stuck standing" is entropy_coef/empirical_normalization
        # (prevents the divergence that corrupted that run) plus the higher
        # feet_air_time weight below -- not a tighter tracking tolerance.

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = None
        # yaw matched to G1 (full random heading at reset, not locked to 0) -- found
        # while diffing event randomization against G1's actual source; this had never
        # been revisited from Wato's flat-specific override.
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}

        # push_robot and add_base_mass are already disabled entirely in Wato's rough
        # config (rough_env_cfg.py) -- already matches G1's approach there, nothing to
        # change. G1 also locks reset_robot_joints/reset_base randomization down much
        # further than the base defaults Wato has used all session (position_range
        # (0.5,1.5), velocity ±0.5 in every axis) -- not going as far as G1's full zero
        # (the zero-action test proved the robot can handle full randomization
        # passively, so it's not strictly necessary to remove) -- just scaling down
        # toward G1's gentler spirit, to see if it makes training converge more easily.
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
            "roll": (-0.2, 0.2),
            "pitch": (-0.2, 0.2),
            "yaw": (-0.2, 0.2),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)


@configclass
class PioneerHumanoidFlatEnvCfg_PLAY(PioneerHumanoidFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# #335 gain set B (D5): see rough_env_cfg.py.
@configclass
class PioneerHumanoidFlatGainsBEnvCfg(PioneerHumanoidFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        use_robot(self, WHOLE_BODY_HUMANOID_GAINS_B_CFG)


@configclass
class PioneerHumanoidFlatGainsBEnvCfg_PLAY(PioneerHumanoidFlatEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        use_robot(self, WHOLE_BODY_HUMANOID_GAINS_B_CFG)
