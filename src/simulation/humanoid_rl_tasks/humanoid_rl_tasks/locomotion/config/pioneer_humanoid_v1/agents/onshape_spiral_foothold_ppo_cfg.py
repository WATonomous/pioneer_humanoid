"""Runner config for the foothold-conditioned spiral task.

Same whole-body foothold recipe as the straight stairs -- 12 leg actions,
asymmetric actor/critic observation groups, left/right mirror augmentation.
Only the experiment name and the iteration budget differ.

The learning rate is the shared 1e-3 unless ``SPIRAL_LR`` overrides it. A
reward change at 1e-3 rewrites the policy within a few dozen iterations, which
is how the warm-started gait was being lost on every fine-tune; retraining from
the rough walker uses 2e-4 so the leg action survives the adaptation.
"""

import os

from isaaclab.utils import configclass

from ...pioneer.agents.rsl_rl_ppo_cfg import PioneerWholeBodyFootholdPPORunnerCfg


@configclass
class PioneerHumanoidSpiralFootholdPPORunnerCfg(PioneerWholeBodyFootholdPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pioneer_humanoid_spiral_foothold"
        self.max_iterations = 2000
        self.save_interval = 50
        learning_rate = os.environ.get("SPIRAL_LR")
        if learning_rate:
            self.algorithm.learning_rate = float(learning_rate)
        # Rollout length per env. The PPO buffer is num_envs x this, so at 3000
        # envs the inherited 24 does not fit in the 8 GB card alongside another
        # process (measured OOM at 99 MiB free). Shortening it trades a noisier
        # gradient for memory; 16 keeps the total below the 2560 x 24 that
        # previously ran comfortably.
        # Hard bound on the raw policy output (the inherited 10 is +-2.5 rad at
        # the hip scale, far beyond any joint range, which is what let the mean
        # park on the rail). Training and play must agree on this value.
        # The inherited 10.0 was never applied (no script passed it to the
        # wrapper), so every existing spiral checkpoint was trained unbounded;
        # default to None to reproduce them exactly.
        clip = os.environ.get("SPIRAL_CLIP_ACTIONS")
        self.clip_actions = float(clip) if clip else None
        steps = os.environ.get("SPIRAL_STEPS")
        if steps:
            self.num_steps_per_env = int(steps)
