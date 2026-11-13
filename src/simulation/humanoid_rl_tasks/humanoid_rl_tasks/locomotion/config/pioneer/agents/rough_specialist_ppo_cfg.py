"""Separate rough-specialist experiment with the reference PPO/interface unchanged."""
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PioneerWholeBodyFootholdPPORunnerCfg


@configclass
class PioneerWholeBodyRoughSpecialistPPORunnerCfg(PioneerWholeBodyFootholdPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_whole_body_rough_specialist"
        self.max_iterations = 1500
