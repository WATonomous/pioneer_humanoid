from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PioneerHumanoidRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "pioneer_humanoid_rough"
    # Was False (matching G1's rough default) until rough_terrain_tilt_term_fresh_001
    # crashed once episode length reached 600-750/1000 steps: Mean action noise std
    # froze around 2.17 instead of adapting, then diverged into
    # "RuntimeError: normal expects all elements of std >= 0.0". This is the same root
    # cause already diagnosed and fixed on flat earlier this session -- without
    # observation/return normalization, the value function's regression targets grow
    # unboundedly once episodes get long, and can destabilize training. Flat's fix
    # (empirical_normalization=True) was applied via resume on an already-far-along
    # checkpoint there too and resolved it cleanly (990/1000, no further divergence).
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # Root cause of the recurring "RuntimeError: normal expects all elements of
        # std >= 0.0" crash (confirmed via torch.autograd.set_detect_anomaly + cross-
        # referenced against rsl_rl's own upstream GitHub issue #33): with the default
        # noise_std_type="scalar", the policy's std is a raw nn.Parameter with no
        # lower bound -- the optimizer can push it negative or to NaN directly, or an
        # out-of-distribution observation (e.g. mid-fall on rough terrain) can drive
        # the PPO ratio's exp() to overflow, both of which crash torch.normal().
        # rsl_rl's own maintainers added "log" mode as the structural fix: std is
        # stored as log(std) and exponentiated, so it's mathematically guaranteed
        # positive no matter what the optimizer does to the underlying parameter.
        # We were already on rsl-rl-lib 3.1.2 (which has this), just never set it.
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
        # The actual divergence cause was unnormalized value targets over long
        # episodes, not entropy per se -- empirical_normalization=True below should
        # fix that directly, so raising entropy back partway is safe to retest.
        self.algorithm.entropy_coef = 0.008
        # Episode length jumped from ~60 to ~1000 steps once ankle stiffness fixed the
        # fall-within-1s failure mode. Without observation/return normalization, value
        # function regression targets over a ~40x longer horizon likely blew up --
        # this was previously False, which is unusual and risky for long-horizon tasks.
        self.empirical_normalization = True
