"""Quantify the gait wobble: yaw-rate command vs tracking, lateral sway, speed steadiness."""

import math
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

RUN = sys.argv[1]  # spiral | straight
if RUN == "spiral":
    TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
    CKPT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_900.pt"
    STEPS = 2400
else:
    TASK = "Isaac-Locomotion-StairsStraightFoothold-PioneerHumanoid-Play-v0"
    CKPT = "/workspace/humanoid/models/pioneer_humanoid_stairs_foothold/model_599.pt"
    STEPS = 1200

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
robot = u.scene["robot"]
cmd_term = u.command_manager.get_term("base_velocity")

obs = env.get_observations()
fwd, lat, wz, wz_cmd, hdg_err, rad_err = [], [], [], [], [], []
cent = (0.0, 0.0)
with torch.inference_mode():
    for _ in range(STEPS):
        obs, *_ = env.step(policy(obs))
        vb = robot.data.root_lin_vel_b[0]
        fwd.append(float(vb[0]))
        lat.append(float(vb[1]))
        wz.append(float(robot.data.root_ang_vel_b[0, 2]))
        wz_cmd.append(float(cmd_term.vel_command_b[0, 2]))
        heading = float(robot.data.heading_w[0])
        hdg_err.append(abs(float(torch.remainder(cmd_term.heading_target[0] - heading + math.pi, 2 * math.pi) - math.pi)))

        if RUN == "spiral":
            p = robot.data.root_pos_w[0]
            ang = math.atan2(float(p[1]) - cent[1], float(p[0]) - cent[0])
            r = math.hypot(float(p[0]) - cent[0], float(p[1]) - cent[1])
            rad_err.append(abs(r - 1.515))


def stats(name, xs, scale=1.0):
    t = torch.tensor(xs)
    print(f"[WOB] {name:22s} mean {float(t.mean()):.3f}  std {float(t.std()):.3f}  "
          f"p90|.| {float(t.abs().quantile(0.9)):.3f}  max|.| {float(t.abs().max()):.3f}")


print(f"[WOB] {RUN}: {STEPS} steps")
stats("forward speed m/s", fwd)
stats("lateral speed m/s", lat)
stats("yaw rate rad/s", wz)
stats("yaw rate cmd rad/s", wz_cmd)
stats("heading error rad", hdg_err)
if rad_err:
    stats("radial error m", rad_err)
fp = torch.tensor(fwd)
print(f"[WOB] forward-speed variation: std {float(fp.std()):.3f} m/s "
      f"({100 * float(fp.std()) / max(1e-6, float(fp.mean())):.0f}% of mean)")
w = torch.tensor(wz)
wc = torch.tensor(wz_cmd)
print(f"[WOB] yaw-rate tracking: rms(cmd-actual) {float((wc - w).pow(2).mean().sqrt()):.3f} rad/s, "
      f"corr {float(torch.corrcoef(torch.stack([wc, w]))[0, 1]):.2f}")
sign = torch.sign(torch.tensor(wz))
flips = int((sign[1:] * sign[:-1] < 0).sum())
print(f"[WOB] yaw-rate sign flips: {flips} in {STEPS} steps ({STEPS / max(1, flips):.1f} steps/flip)")
print("[WOB] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
