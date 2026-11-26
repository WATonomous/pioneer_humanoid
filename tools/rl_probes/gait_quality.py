"""Gait quality of a spiral checkpoint: speed vs command, sway, yaw flail, foot slip.

Runs the PLAY scene (no pushes) on the six measured spawn slots, with the
deterministic policy, and reports per-env medians so a single unlucky episode
cannot dominate.
"""

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

TASK = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
CKPT = sys.argv[1]
N = 12
STEPS = 3000

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708),
    (0.213, 1.500, 1.417, 3.0005),
    (-1.504, -0.182, 2.017, 4.8331),
    (1.361, -0.666, 2.857, 7.3987),
    (-1.192, 0.935, 3.817, 10.3309),
    (1.500, -0.213, 5.017, 13.9960),
)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
term = u.event_manager.get_term_cfg("reset_base")
term.params["slots"] = SLOTS
term.params["xy_jitter"] = 0.03
term.params["yaw_jitter"] = 0.03


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
cmd = u.command_manager.get_term("base_velocity")
feet = [robot.body_names.index(n) for n in ("l_foot", "r_foot")]
contact = u.scene.sensors["contact_forces"]

obs = env.get_observations()
# Per-env accumulators.
rec = {k: [[] for _ in range(N)] for k in ("vx", "vy", "wz", "wz_err", "roll_pitch_rate", "slip", "heading")}
alive = torch.ones(N, dtype=torch.bool)
with torch.inference_mode():
    for step in range(STEPS):
        obs, *_ = env.step(policy(obs))
        done = u.reset_terminated | u.reset_time_outs
        if step > 100:  # skip the spawn turn-in
            vb = robot.data.root_lin_vel_b
            wz = robot.data.root_ang_vel_b[:, 2]
            wz_cmd = cmd.vel_command_b[:, 2]
            body_rates = robot.data.body_ang_vel_w[:, feet, :]
            in_contact = contact.data.net_forces_w[:, feet, 2] > 1.0
            foot_vel = robot.data.body_lin_vel_w[:, feet, :].norm(dim=-1)
            slip = (foot_vel * in_contact).sum(dim=1)
            for i in range(N):
                if done[i]:
                    continue
                rec["vx"][i].append(float(vb[i, 0]))
                rec["vy"][i].append(float(vb[i, 1]))
                rec["wz"][i].append(float(wz[i]))
                rec["wz_err"][i].append(float(wz[i] - wz_cmd[i]))
                rec["roll_pitch_rate"][i].append(float(torch.linalg.vector_norm(body_rates[i, :, :2], dim=-1).max()))
                rec["slip"][i].append(float(slip[i]))


def per_env(name, key, reduce="std"):
    vals = []
    for seq in rec[key]:
        if len(seq) < 50:
            continue
        t = torch.tensor(seq)
        vals.append(float(t.std()) if reduce == "std" else float(t.abs().mean()))
    v = torch.tensor(vals)
    print(f"[GAIT] {name:26s} median {float(v.median()):.3f}  p90 {float(v.quantile(0.9)):.3f}  (n={v.numel()})")


print(f"[GAIT] {os.path.basename(CKPT)}  {STEPS} steps x {N} envs")
per_env("forward speed std m/s", "vx", "std")
per_env("|forward speed| mean m/s", "vx", "mean")
per_env("lateral speed std m/s", "vy", "std")
per_env("yaw rate std rad/s", "wz", "std")
per_env("yaw err rate std rad/s", "wz_err", "std")
per_env("foot roll/pitch rate max", "roll_pitch_rate", "std")
per_env("foot slip m/s", "slip", "mean")
vx_mean = torch.tensor([float(torch.tensor(s).mean()) for s in rec["vx"] if len(s) > 50])
print(f"[GAIT] forward speed mean median {float(vx_mean.median()):.3f} m/s")
print("[GAIT] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
