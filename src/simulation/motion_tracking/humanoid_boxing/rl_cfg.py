"""PPO for the fight policy: a move (categorical) plus, with each punch, its
technique (Gaussian: arm-joint offsets and speed, humanoid_boxing/skills.py).

FightRunner hands the learner's actor to the opponent's action term, which
snapshots it for self-play.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.runner import MjlabOnPolicyRunner

from humanoid_boxing.skills import MOVES


def wato_fight_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="humanoid_boxing.distribution:FightMLPModel",  # masks log-probs/entropy to decision steps
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "humanoid_boxing.distribution:HybridDistribution",
        "num_classes": len(MOVES),
        "init_std": 0.2,
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # entropy is the mean over decision steps (~3% of samples) while the
      # surrogate only has gradient there, so per decision this is ~0.1 (5-10x
      # the old unmasked 0.01, under which the choice in guard collapsed)
      entropy_coef=0.003,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.995,  # knockdowns come seconds after the punch that set them up
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="wato_fight",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6000,
  )


class FightRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    opponent = env.unwrapped.action_manager.get_term("blue")
    opponent.learner_actor = self.alg.actor
