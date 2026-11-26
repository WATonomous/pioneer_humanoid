"""Does the policy behave differently under training-time observation noise?

The fine-tune's logged ``action_l2`` implies RMS |a| ~ 0.8-1.8 during training,
but every play-time probe on the same checkpoint measures mean |a| ~ 10 with
the knee target on a clip rail 80% of the steps. Same weights, so the input
must differ: the only observation difference left between the two paths is
``enable_corruption`` (on in training, off in play).

This runs the *training* environment (corruption on, no forced slots) and prints
both the deterministic mean action and the sampled action from rsl_rl's own
distribution, so "small in training / large in play" can be pinned to the noise
or dismissed.
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
N, STEPS = 8, 120
CORRUPT = os.environ.get("SAT_CORRUPT", "1") == "1"

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.observations.policy.enable_corruption = CORRUPT


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
at = u.action_manager.get_term("joint_pos")
names = list(at._joint_names)
knee = names.index("Knee_L")

print(f"[C] corruption {CORRUPT}, std {float(runner.alg.policy.log_std.exp().mean()):.2f}")
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS):
        mean = runner.alg.policy.act_inference(obs)
        sampled = runner.alg.policy.act(obs)
        obs, *_ = env.step(sampled)
        if step % 20 == 0 or step == STEPS - 1:
            tgt = at.processed_actions[:, knee]
            on_clip = float(((tgt + 1.10).abs() < 1e-3).float().mean() + ((tgt + 0.05).abs() < 1e-3).float().mean())
            print(f"[C] {step:3d} |mean| {float(mean.abs().mean()):6.2f}  |sampled| {float(sampled.abs().mean()):6.2f}  "
                  f"knee_mean {float(mean[:, knee].abs().mean()):6.2f}  knee_on_clip {on_clip:4.2f}")
print("[C] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
