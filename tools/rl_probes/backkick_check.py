"""Smoke test + calibration for the swing-leg backward-kick penalty.

Prints the term's value every step alongside the reward income it competes with,
so the weight can be set from a measurement instead of a guess.
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
from humanoid_rl_tasks.locomotion.config.pioneer import gait  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
CKPT = os.environ.get("CKPT_OVERRIDE",
                      "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_299_gain05.pt")
N, STEPS = 4, 200

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=N, checkpoint=None, device="cuda:0", distributed=False,
                 seed=5, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)

TERMS = [t for t in ("feet_swing_behind", "knee_flexion_airborne", "shank_backward_rotation", "swing_start_forward", "knee_swing_rate", "knee_saturation", "stance_tilt_hinge", "contact_mismatch")
         if t in u.reward_manager.active_terms]
for name in TERMS:
    t = u.reward_manager.get_term_cfg(name)
    print(f"[BK] {name}: weight {t.weight}  func {t.func.__name__}", flush=True)

vals = {n: [] for n in TERMS}
rews = []
obs = env.get_observations()
with torch.inference_mode():
    for _ in range(STEPS):
        obs, rew, *_ = env.step(policy(obs))
        rews.append(float(rew.mean()))
        for n in TERMS:
            t = u.reward_manager.get_term_cfg(n)
            vals[n].append(float(t.func(u, **t.params).mean()))

for n in TERMS:
    w = u.reward_manager.get_term_cfg(n).weight
    v = torch.tensor(vals[n])
    print(f"[BK] {n}: raw mean {float(v.mean()):.5f} p90 {float(v.quantile(0.9)):.5f} | "
          f"charge mean {w * float(v.mean()):.5f} p90 {w * float(v.quantile(0.9)):.5f} /step", flush=True)

inc = float(torch.tensor(rews).mean())
print(f"[BK] mean total reward per step (net, dt-scaled): {inc:.4f}", flush=True)
for n in TERMS:
    w = u.reward_manager.get_term_cfg(n).weight
    v = torch.tensor(vals[n])
    print(f"[BK]   {n}: dt-scaled mean {w * float(v.mean()) * u.step_dt:+.5f}/step", flush=True)
print("[BK] OK", flush=True)
sys.stdout.flush()
env.close()
simulation_app.close()
