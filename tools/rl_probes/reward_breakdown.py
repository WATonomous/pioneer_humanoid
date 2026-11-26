"""Per-step reward breakdown of the spiral foothold policy (PLAY)."""

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

TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
CKPT = "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_900.pt"

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=1, checkpoint=None, device="cuda:0", distributed=False,
                 seed=3, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)

STEPS = 400
obs = env.get_observations()
with torch.inference_mode():
    for _ in range(STEPS):
        obs, *_ = env.step(policy(obs))

sums = {k: float(v[0]) for k, v in u.reward_manager._episode_sums.items()}
total = sum(sums.values())
print(f"[RB] episode so far: {STEPS} steps, total return {total:+.1f} ({total / STEPS:+.3f}/step)")
for name, value in sorted(sums.items(), key=lambda kv: -abs(kv[1])):
    if abs(value) > 0.05:
        print(f"[RB] {name:32s} {value:+8.2f}  {value / STEPS:+.4f}/step")
print("[RB] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
