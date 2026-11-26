"""Marker-spawn evaluation: same start point for every checkpoint.

Runs the PLAY config (no pushes, no corruption) with 16 envs sharing the course
origin, and classifies each episode end:
  * complete  -- reached the goal (root y >= goal_distance - 0.1)
  * fall      -- anything else (base contact, tilt, out of bounds)
also reports episode length and mean forward distance. This is the number that
stays comparable across spawn-distribution changes, unlike training logs.
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

TASK = "Isaac-Locomotion-StairsStraightFoothold-PioneerHumanoid-Play-v0"
CKPT = sys.argv[1]
N = 16
STEPS = 4000
GOAL_Y = 20.4

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.log_dir = "/workspace/humanoid/models/pioneer_humanoid_stairs_foothold"
cfg.scene.env_spacing = 0.0


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=N, checkpoint=CKPT, device="cuda:0", distributed=False,
                 seed=None, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)

env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=env.unwrapped.device)

robot = env.unwrapped.scene["robot"]
obs = env.get_observations()

completed = falls = 0
lengths = []
distances = []
ep_len = torch.zeros(N)
prev_y = robot.data.root_pos_w[:, 1].clone()
for step in range(STEPS):
    with torch.inference_mode():
        obs, rew, dones, _ = env.step(policy(obs))
    ep_len += 1
    y = robot.data.root_pos_w[:, 1]
    if bool(dones.any()):
        ids = dones.nonzero(as_tuple=False).squeeze(-1)
        # root state is post-reset, so classify on the y reached just before
        y_peak = prev_y[ids] if step else y[ids]
        for i in ids.tolist():
            lengths.append(float(ep_len[i]))
            distances.append(float(prev_y[i] - (-1.549)))
            if float(prev_y[i]) >= GOAL_Y:
                completed += 1
            else:
                falls += 1
        ep_len[ids] = 0.0
    prev_y = y.clone()

n = completed + falls
print(f"[EVAL] {CKPT.split('/')[-2]}/{CKPT.split('/')[-1]}")
print(f"[EVAL] episodes {n}: complete {completed} ({completed/max(1,n):.3f}), fall {falls} ({falls/max(1,n):.3f})")
if n:
    import statistics
    print(f"[EVAL] episode length mean {statistics.mean(lengths):.0f} steps; "
          f"mean distance {statistics.mean(distances):.2f} m; max {max(distances):.2f} m")
print("[EVAL] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
