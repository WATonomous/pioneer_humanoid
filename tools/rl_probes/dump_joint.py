"""One joint, in detail: raw action, scale, target, position, velocity, torque."""

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
CKPT = os.environ.get("CKPT_OVERRIDE", "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_299_gait.pt")
N = 1
SLOTS = ((1.551, -0.777, 0.816, 1.5708),)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
term = u.event_manager.get_term_cfg("reset_base")
term.params["slots"] = SLOTS
term.params["xy_jitter"] = 0.0
term.params["yaw_jitter"] = 0.0


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
robot = u.scene["robot"]
at = u.action_manager.get_term("joint_pos")
print("[K] names", at._joint_names)
print("[K] scale", at._scale.tolist())
print("[K] offset", at._offset.tolist())
print("[K] clip", getattr(at.cfg, "clip", None), "manager clip", getattr(u.action_manager, "_clip", None))
print("[K] limits", robot.data.joint_pos_limits[0].tolist())
clip = getattr(at.cfg, "clip", {}) or {}
import fnmatch
lo = torch.full((len(at._joint_names),), -1e9)
hi = torch.full((len(at._joint_names),), 1e9)
for pat, (a_, b_) in clip.items():
    for k, n in enumerate(at._joint_names):
        if fnmatch.fnmatch(n, pat):
            lo[k], hi[k] = a_, b_
sat_lo = torch.zeros(len(at._joint_names))
sat_hi = torch.zeros(len(at._joint_names))
raw_abs = torch.zeros(len(at._joint_names))
count = 0
obs = env.get_observations()
with torch.inference_mode():
    for step in range(120):
        action = policy(obs)
        obs, *_ = env.step(action)
        if step >= 40:
            t = at.processed_actions[0].cpu()
            raw_abs += action[0].cpu().abs()
            sat_lo += (t <= lo + 1e-3).float()
            sat_hi += (t >= hi - 1e-3).float()
            count += 1
        if step >= 90:
            for name in ("Knee_L", "Hip_F_L"):
                j = robot.joint_names.index(name)
                a_i = at._joint_names.index(name)
                print(f"[K] {step:3d} {name:8s} raw {action[0, a_i]:+7.2f} tgt {at.processed_actions[0, a_i]:+6.2f} "
                      f"q {robot.data.joint_pos[0, j]:+6.2f} v {robot.data.joint_vel[0, j]:+6.2f} "
                      f"tau {robot.data.applied_torque[0, j]:+7.1f}")
print("[K] sat_lo", [f"{v:.2f}" for v in (sat_lo / max(count, 1)).tolist()])
print("[K] sat_hi", [f"{v:.2f}" for v in (sat_hi / max(count, 1)).tolist()])
print("[K] |raw|  ", [f"{v:.1f}" for v in (raw_abs / max(count, 1)).tolist()])
print("[K] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
