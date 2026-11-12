"""PPO config: identical to G1's (as in the Isaac WatoFlatPPORunnerCfg, which this task's settings come from) except
the experiment name."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "pioneer_tracking"
  return cfg
