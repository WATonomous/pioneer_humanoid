"""What does the reward manager actually charge for action_l2 per step?

``Episode_Reward/action_l2`` in the training log implies sum(a^2) ~ 37
(RMS |a| ~ 1.7), while every direct measurement of the policy output says
sum(a^2) ~ 1400 (|a| ~ 11). Both cannot be true of the same tensor, and the
weight needed to make the penalty bite depends on which one the reward manager
sees. This prints, per step: the term's weight, the value computed straight
from the sampled action, and the observed increment of the term's own episodic
sum.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src/simulation/humanoid_rl/humanoid_rl/scripts"))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_rl_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0")
CKPT = os.environ.get("CKPT_OVERRIDE",
                      "/workspace/humanoid/outputs/rl/pioneer_humanoid_spiral_foothold/2026-11-16_21-30-53/model_299.pt")
N, STEPS = 4, 60

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.observations.policy.enable_corruption = False


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=N, checkpoint=None, device="cuda:0", distributed=False,
                 seed=5, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)

term = next(t for t in u.reward_manager._term_cfgs if t.func.__name__ == "action_l2")
print(f"[L2] weight {term.weight}")
print(f"[L2] max_episode_length_s {u.max_episode_length_s}  max_episode_length {u.max_episode_length}")

obs = env.get_observations()
prev = float(u.reward_manager._episode_sums["action_l2"].mean())
with torch.inference_mode():
    for step in range(STEPS):
        act = runner.alg.policy.act(obs)
        obs, *_ = env.step(act)
        cur = float(u.reward_manager._episode_sums["action_l2"].mean())
        raw = float(torch.sum(act.square(), dim=1).mean())
        stored = float(torch.sum(u.action_manager.action.square(), dim=1).mean())
        if step % 6 == 0:
            print(f"[L2] {step:3d} sum(a^2) sampled {raw:8.1f}  stored {stored:8.1f}  "
                  f"term/step {abs(term.weight) * stored:7.4f}  episodic delta {cur - prev:+8.4f}")
        prev = cur
print("[L2] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
