"""Is the shipped policy's action saturation real, or an artefact of a fall?

``dump_joint.py`` printed raw actions of -18 and knee targets pinned at the
clip (-1.10 / -0.05), with Hip_F at -222 Nm (its effort limit) and the joint
sitting on its position limit. That is either a policy that has learned to
live in saturation, or a robot that has already fallen and is being driven
into its stops. This separates the two: base height and tilt over time, the
per-joint raw action magnitude, how often each joint's target is outside its
position limit, how often the motor is at its effort limit, and how often the
knee target sits exactly on one of its clip values (the two-state relay).
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

TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ.get("CKPT_OVERRIDE",
                      "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_299_plant.pt")
N, STEPS = 1, 400
SLOTS = ((1.551, -0.777, 0.816, 1.5708),)
if os.environ.get("SAT_SLOTS", "marker") == "spiral":
    SLOTS = (
        (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005),
        (-1.504, -0.182, 2.017, 4.8331), (1.361, -0.666, 2.857, 7.3987),
        (-1.192, 0.935, 3.817, 10.3309), (1.500, -0.213, 5.017, 13.9960),
    )
    N = len(SLOTS)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
if os.environ.get("SAT_CORRUPT", "0") != "1":
    cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
if "Spiral" in TASK:
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
env.clip_actions = agent_cfg.clip_actions   # the runner cfg's bound; the wrapper defaults to none
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)
robot = u.scene["robot"]
at = u.action_manager.get_term("joint_pos")
names = list(at._joint_names)


def _vec(v, n):
    t = torch.as_tensor(v, dtype=torch.float32)
    if t.dim() == 0:
        return t.expand(n).clone()
    if t.dim() == 2:
        t = t[0]
    return t.reshape(-1)[:n].clone()


scale = _vec(at._scale, len(names))
offset = _vec(at._offset, len(names))
lim = robot.data.joint_pos_limits[0].cpu()
jid = [robot.joint_names.index(n) for n in names]
eff = robot.data.joint_effort_limits[0].cpu() if hasattr(robot.data, "joint_effort_limits") else None

raw_abs = torch.zeros(len(names))
raw_max = torch.zeros(len(names))
outside = torch.zeros(len(names))
at_eff = torch.zeros(len(names))
count = 0
knee_hits = []
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS):
        action = policy(obs)
        obs, *_ = env.step(action)
        tgt = at.processed_actions.cpu()
        tau = robot.data.applied_torque[:, jid].cpu()
        raw_abs += action.cpu().abs().mean(0)
        raw_max = torch.maximum(raw_max, action.cpu().abs().amax(0))
        outside += ((tgt < lim[jid, 0] - 1e-3) | (tgt > lim[jid, 1] + 1e-3)).float().mean(0)
        if eff is not None:
            at_eff += (tau.abs() >= 0.98 * eff[jid]).float().mean(0)
        for n in ("Knee_L", "Knee_R"):
            i = names.index(n)
            knee_hits.append(float(((tgt[:, i] + 1.10).abs() < 1e-3).float().mean()
                                   + ((tgt[:, i] + 0.05).abs() < 1e-3).float().mean()))
        count += 1
        if step % 20 == 0:
            z = float(robot.data.root_pos_w[0, 2])
            g = robot.data.projected_gravity_b[0].cpu()
            tilt = float(torch.rad2deg(torch.acos((-g[2]).clamp(-1.0, 1.0))))
            print(f"[A] {step:3d} base_z {z:5.2f} tilt {tilt:5.1f}deg  "
                  f"knee_raw {action[0, names.index('Knee_L')]:+6.2f}  "
                  f"hipF_raw {action[0, names.index('Hip_F_L')]:+6.2f}  "
                  f"knee_tgt {at.processed_actions[0, names.index('Knee_L')]:+5.2f}  "
                  f"knee_q {robot.data.joint_pos[0, robot.joint_names.index('Knee_L')]:+5.2f}")

print("[A] joint        mean|raw|  max|raw|  tgt_outside  at_effort")
for k, n in enumerate(names):
    print(f"[A] {n:10s} {float(raw_abs[k]/count):9.2f} {float(raw_max[k]):9.2f} "
          f"{float(outside[k]/count):11.2f} {float(at_eff[k]/count):10.2f}")
print(f"[A] knee target on a clip value: {100.0*sum(knee_hits)/max(len(knee_hits),1):.1f} %")
print(f"[A] terminated {int(u.reset_terminated.sum())} times, timeouts {int(u.reset_time_outs.sum())}")
print("[A] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
