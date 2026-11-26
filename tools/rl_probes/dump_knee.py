"""Dump the left knee's velocity and commanded target for two seconds."""

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
sensor = u.scene.sensors["contact_forces"]
feet = [sensor.body_names.index(n) for n in ("l_foot", "r_foot")]
knee = robot.joint_names.index("Knee_L")
print("[D] clip_actions", env.clip_actions if hasattr(env, "clip_actions") else None)
print("[D] action joints", u.action_manager.get_term("joint_pos")._joint_names)
print("[D] action scale", u.action_manager.get_term("joint_pos").cfg.scale,
      "offset", u.action_manager.get_term("joint_pos").cfg.offset)
obs = env.get_observations()
with torch.inference_mode():
    for step in range(140):
        action = policy(obs)
        obs, *_ = env.step(action)
        if step >= 60:
            force = sensor.data.net_forces_w[0, feet, 2].tolist()
            jv = robot.data.joint_vel[0, knee].item()
            jp = robot.data.joint_pos[0, knee].item()
            raw = action[0].tolist()
            stored = u.action_manager.action[0].tolist()
            term_ = u.action_manager.get_term("joint_pos")
            proc = term_.processed_actions[0].tolist()
            print(f"[D] {step:3d} knee_v {jv:+.3f} knee_q {jp:+.3f} "
                  f"F_l {force[0]:6.0f} F_r {force[1]:6.0f}")
            print("[D]   raw   ", " ".join(f"{v:+.2f}" for v in raw))
            print("[D]   stored", " ".join(f"{v:+.2f}" for v in stored))
            print("[D]   target", " ".join(f"{v:+.2f}" for v in proc))
print("[D] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
