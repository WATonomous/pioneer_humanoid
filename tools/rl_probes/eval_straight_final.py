"""Completion of a straight-stairs foothold policy on the (deterministic) Play course: terminations per term."""

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
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3500

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
env.clip_actions = agent_cfg.clip_actions
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)
robot = u.scene["robot"]
origins = u.scene.env_origins

terms = list(u.termination_manager.active_terms)
counts = {t: 0 for t in terms}
episodes = 0
ep_len = torch.zeros(N, device=u.device)
complete_len = []
max_y = torch.full((N,), -1e6, device=u.device)
obs = env.get_observations()
with torch.inference_mode():
    for i in range(STEPS):
        obs, *_ = env.step(policy(obs))
        ep_len += 1
        y = (robot.data.root_pos_w - origins)[:, 1]
        max_y = torch.maximum(max_y, y)
        done = u.reset_terminated | u.reset_time_outs
        if done.any():
            for t in terms:
                m = u.termination_manager.get_term(t)
                counts[t] += int(m.sum())
                if t == "course_complete":
                    complete_len += [float(v) * u.step_dt for v in ep_len[m]]
            episodes += int(done.sum())
            ep_len[done] = 0
        if (i + 1) % 500 == 0:
            print(f"[EVAL] step {i + 1}/{STEPS} episodes {episodes}", flush=True)
print("[EVAL] n", N, "steps", STEPS, "episodes", episodes)
print("[EVAL] terminations", {k: round(v / max(1, episodes), 3) for k, v in counts.items()})
if complete_len:
    t = torch.tensor(complete_len)
    print("[EVAL] time to complete: mean %.1f s, min %.1f s, max %.1f s" % (float(t.mean()), float(t.min()), float(t.max())))
print("[EVAL] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
