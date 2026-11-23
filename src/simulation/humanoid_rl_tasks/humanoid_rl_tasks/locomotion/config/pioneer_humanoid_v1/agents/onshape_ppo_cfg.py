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

        # The course reward is small per step (a 3000-iteration straight-stairs
        # run settled at ~0.06 per step) while the entropy bonus at the inherited
        # 0.006 was ~0.24 per step, so entropy dominated and the exploration std
        # grew monotonically instead of settling: mean action noise 0.97 -> 3.33
        # rad, i.e. 4+ rad on the arm channels whose action scale is 1.0. Past
        # the action clip that exploration is uniform over the whole clip range,
        # which leaves the action means with no usable gradient -- the arms
        # random-walked to their limits (shoulders pinned behind the torso) and
        # the legs ended up mostly saturated too. Bake in a lower coefficient
        # from iteration 0 rather than cutting entropy after it has ballooned
        # (see the note on the base config for why late corrections crashed).
        self.algorithm.entropy_coef = 0.002
