from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PioneerHumanoidRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "pioneer_humanoid_rough"
    # Observation normalization is controlled by the per-network policy fields in
    # the current Isaac Lab/RSL-RL bridge. Keep it explicitly disabled for all
    # existing tasks so their observation path and checkpoint layout remain
    # unchanged; the isolated scratch runner below enables it from iteration zero.
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Root cause of the recurring "RuntimeError: normal expects all elements of
        # std >= 0.0" crash (confirmed via torch.autograd.set_detect_anomaly + cross-
        # referenced against rsl_rl's own upstream GitHub issue #33): with the default
        # noise_std_type="scalar", the policy's std is a raw nn.Parameter with no
        # lower bound -- the optimizer can push it negative or to NaN directly, or an
        # out-of-distribution observation (e.g. mid-fall on rough terrain) can drive
        # the PPO ratio's exp() to overflow, both of which crash torch.normal().
        # rsl_rl's "log" mode stores log(std) and exponentiates it. This prevents a
        # finite parameter from directly becoming a negative std, but does not make
        # NaN/Inf parameters safe; the training launcher's non-finite gradient guard
        # remains useful. We were already on rsl-rl-lib 3.1.2 (which has this), just
        # never set it.
        noise_std_type="log",
    )
    # entropy_coef 0.008 matches G1's own default. Tried lowering to 0.004 once noise
    # std got stuck around 2.17 on rough (see git history) -- it genuinely helped:
    # noise annealed 2.17->1.65 over ~80 min and tracking/stepping metrics hit new
    # bests. But two resumes after introducing that change both eventually crashed
    # with "RuntimeError: normal expects all elements of std >= 0.0" (the second one
    # almost immediately after resuming, suggesting the checkpoint was already
    # compromised) -- whereas 0.008 ran the *entire* original
    # rough_terrain_tilt_term_fresh_001 run to full completion (10248 iterations)
    # with zero crashes. Reverted to 0.008 there to prioritize stability, but noise
    # started the same slow creep upward again in the very next fresh run
    # (rough_terrain_lower_term_penalty_fresh_001: 1.75->1.81->1.97->2.00->2.03->2.06
    # within the first ~90 min) -- 0.008 doesn't stop the growth, it just runs longer
    # before whatever eventually destabilizes it. Rather than reactively lowering
    # entropy again only after noise has already ballooned (which is what led to both
    # crashes above -- correcting an already-large, already-dominant noise
    # distribution mid-training), baking a moderate reduction in from iteration 0 of
    # a fresh run should let the policy settle into a lower-noise regime before it
    # ever reaches the range that's previously preceded a crash. 0.006 (not another
    # jump straight to 0.004) as a first, more cautious step in that direction.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PioneerHumanoidRoughNoStairsPPORunnerCfg(PioneerHumanoidRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "pioneer_humanoid_rough_no_stairs"


@configclass
class PioneerHumanoidRoughNoStairsFootTuneFineNormalizedScratchPPORunnerCfg(
    PioneerHumanoidRoughNoStairsPPORunnerCfg
):
    """Fresh FootTuneFine lineage with actor and critic observation normalization."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 6000
        self.experiment_name = "pioneer_humanoid_rough_no_stairs_foot_tune_fine_normalized_scratch"
        # The runner-level option is deprecated. Set the two active policy options
        # explicitly so there is no ambiguity about which normalizers are created.
        self.empirical_normalization = None
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True


@configclass
class PioneerHumanoidRoughNoStairsFootTuneFineKneeAxisFixedNormalizedScratchPPORunnerCfg(
    PioneerHumanoidRoughNoStairsFootTuneFineNormalizedScratchPPORunnerCfg
):
    """Fresh normalized lineage for the corrected knee geometry."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = (
            "pioneer_humanoid_rough_no_stairs_foot_tune_fine_knee_axis_fixed_normalized_scratch"
        )


@configclass
class PioneerHumanoidRoughNoStairsSelectiveSelfCollisionNormalizedScratchPPORunnerCfg(
    PioneerHumanoidRoughNoStairsFootTuneFineKneeAxisFixedNormalizedScratchPPORunnerCfg
):
    """Fresh normalized collision lineage with the existing PPO settings."""

    def __post_init__(self):
        super().__post_init__()

        self.resume = False
        self.experiment_name = "pioneer_humanoid_rough_no_stairs_selective_self_collision_normalized_scratch"


@configclass
class PioneerHumanoidRoughNoStairsSelectiveKneeShapePPORunnerCfg(
    PioneerHumanoidRoughNoStairsSelectiveSelfCollisionNormalizedScratchPPORunnerCfg
):
    """Keep normalized PPO defaults, with separate logs and denser pilot saves."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 25
        self.experiment_name = "pioneer_humanoid_rough_no_stairs_selective_knee_shape"


@configclass
class PioneerHumanoidFlatPPORunnerCfg(PioneerHumanoidRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "pioneer_humanoid_flat"
        # Raised from 0.5 to match G1's flat config: G1FlatPPORunnerCfg never overrides
        # init_noise_std, so it inherits G1RoughPPORunnerCfg's 1.0 -- Wato's flat config
        # had been sitting at half that (0.5) since before this session, unexamined.
        # (Separately: cutting it to 0.1, with entropy_coef also cut, made no difference
        # either -- episode length stayed ~35-36 through iteration 150 in that test,
        # ruling out low noise as a destabilizer. This is a different test: raising
        # toward G1's actual value, not lowering further.)
        self.policy.init_noise_std = 1.0
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.desired_kl = 0.01
        # 0.004 was too conservative: stepping_fix_fresh_004 plateaued hard at episode
        # length ~52 for 250+ iterations (noise std already annealed down to 0.22 by
        # iteration 750, likely too little exploration left to ever discover the
        # "don't fall" behavior the original entropy_coef=0.01 run stumbled onto).
        # Keep the historical entropy choice; this legacy task intentionally retains
        # its original identity-normalized observation path for checkpoint loading.
        self.algorithm.entropy_coef = 0.008
        self.empirical_normalization = False
