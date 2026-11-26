"""Completion rate of the final spiral foothold model, training disturbance, many envs."""

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

TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-v0"
CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1200

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.seed = 7
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=N, checkpoint=None, device="cuda:0", distributed=False,
                 seed=7, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
env.clip_actions = agent_cfg.clip_actions   # the runner cfg's bound; the wrapper defaults to none
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)
robot = u.scene["robot"]

terms = list(u.termination_manager.active_terms)
counts = {t: 0 for t in terms}
episodes = 0
gains = []
spawn = u._spiral_spawn_height.clone()
best = robot.data.root_pos_w[:, 2].clone()
obs = env.get_observations()
with torch.inference_mode():
    for _ in range(STEPS):
        obs, *_ = env.step(policy(obs))
        done = u.reset_terminated | u.reset_time_outs
        for t in terms:
            counts[t] += int(u.termination_manager.get_term(t).sum())
        z_now = robot.data.root_pos_w[:, 2]
        best = torch.where(done, best, torch.maximum(best, z_now))
        if done.any():
            gains += [float(v) for v in (best - spawn)[done]]
            episodes += int(done.sum())
            best[done] = z_now[done]
            spawn[done] = u._spiral_spawn_height[done]
g = torch.tensor(gains)
print("[EVAL] n", N, "steps", STEPS, "episodes", episodes)
print("[EVAL] terminations", {k: round(v / max(1, episodes), 3) for k, v in counts.items()})
print("[EVAL] height gain per episode: mean %.2f m (%.1f risers), median %.2f m, p90 %.2f m"
      % (float(g.mean()), float(g.mean()) / 0.12, float(g.median()), float(g.quantile(0.9))))
print("[EVAL] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
