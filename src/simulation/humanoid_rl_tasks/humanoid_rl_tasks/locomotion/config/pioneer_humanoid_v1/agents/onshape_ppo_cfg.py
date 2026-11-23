"""PPO defaults for the Pioneer Onshape stair-course tasks."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PioneerHumanoidRoughNoStairsSelectiveKneeShapePPORunnerCfg


@configclass
class PioneerHumanoidOnshapePPORunnerCfg(PioneerHumanoidRoughNoStairsSelectiveKneeShapePPORunnerCfg):
    """Fresh normalized pilot; longer runs require an explicit CLI override."""

    def __post_init__(self):
        super().__post_init__()
        self.resume = False
        self.max_iterations = 1000
        self.save_interval = 25
        self.experiment_name = "pioneer_humanoid_onshape"
