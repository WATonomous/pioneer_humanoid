"""Runner config for the foothold-conditioned straight-stairs task.

Inherits the whole-body foothold recipe unchanged -- 12 leg actions, asymmetric
actor/critic observation groups, and left/right mirror data augmentation, which
is the same lever the flat/rough foothold walks were trained with. Only the
experiment name and the iteration budget differ.
"""

from isaaclab.utils import configclass

from ...pioneer.agents.rsl_rl_ppo_cfg import PioneerWholeBodyFootholdPPORunnerCfg


@configclass
class PioneerHumanoidStairsFootholdPPORunnerCfg(PioneerWholeBodyFootholdPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_humanoid_stairs_foothold"
        self.max_iterations = 3000
        self.save_interval = 50
