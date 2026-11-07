"""PPO config: identical to G1's (as in the Isaac WatoFlatPPORunnerCfg, which this task's settings come from) except
the experiment name."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "pioneer_tracking"
  return cfg


def power_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Stage 2 (punch power): the tracker's entropy bonus. Wato used twice it, so a policy resumed from a
  converged tracker tries new punching mechanics; on Pioneer 2x kept widening the action noise (std
  0.26 -> 0.39 over 2000 iterations of power_v1) until tracking degraded."""
  return tracking_ppo_runner_cfg()
