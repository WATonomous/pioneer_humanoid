from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg


@configclass
class PioneerLegOnlyFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Every batch is also trained on its left-right mirror image, for an even, symmetric gait."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "pioneer_leg_only_flat"
    clip_actions = 10.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",  # a plain std parameter can step below zero and crash training
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func="humanoid_rl_tasks.locomotion.config.pioneer.symmetry:mirror"
        ),
    )


@configclass
class PioneerWholeBodyFlatPPORunnerCfg(PioneerLegOnlyFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_whole_body_flat"


@configclass
class PioneerLegOnlyRoughPPORunnerCfg(PioneerLegOnlyFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_leg_only_rough_blind"


@configclass
class PioneerLegOnlyPerceptivePPORunnerCfg(PioneerLegOnlyFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_leg_only_rough_perceptive"
        self.max_iterations = 5000
        self.policy.actor_hidden_dims = [512, 256, 128]  # the height scan adds 187 inputs
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class PioneerLegOnlyFootholdPPORunnerCfg(PioneerLegOnlyFlatPPORunnerCfg):
    """An asymmetric critic: it also sees the whole height map."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_leg_only_rough_foothold"
        self.max_iterations = 4000
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}


@configclass
class PioneerWholeBodyFootholdPPORunnerCfg(PioneerLegOnlyFootholdPPORunnerCfg):
    """Fine-tuned from the leg-only foothold policy."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_whole_body_rough_foothold"
        self.max_iterations = 800
