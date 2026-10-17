"""PPO config: identical to G1's (as in the Isaac WatoFlatPPORunnerCfg, which this task's settings come from) except
the experiment name."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "pioneer_tracking"
  return cfg


def power_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Stage 2 (punch power): twice the entropy bonus, so a policy resumed from a
  converged tracker tries new punching mechanics instead of only refining."""
  cfg = tracking_ppo_runner_cfg()
  cfg.algorithm.entropy_coef = 0.01
  return cfg
