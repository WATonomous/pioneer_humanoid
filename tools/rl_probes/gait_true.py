"""Debounced gait measurement: footfalls, step length, treads per step, path speed.

The earlier cadence number counted every force flicker across a 1 N threshold as a
new footfall. A loaded foot carries ~350 N, so use 50 N and require the foot to be
unloaded for 0.1 s before a new onset counts.
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
CKPT = os.environ.get("CKPT_OVERRIDE", "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_399_grounded.pt")
N, STEPS = 6, 2500
LOAD_N, DEBOUNCE = 50.0, 5  # 5 steps = 0.1 s unloaded before a new onset counts

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005), (-1.504, -0.182, 2.017, 4.8331),
    (1.361, -0.666, 2.857, 7.3987), (-1.192, 0.935, 3.817, 10.3309), (1.500, -0.213, 5.017, 13.9960),
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
sensor = u.scene.sensors["contact_forces"]
feet = [sensor.body_names.index(n) for n in ("l_foot", "r_foot")]

# Per foot, per env: armed flag and the unloaded-step counter.
armed = [[True, True] for _ in range(N)]
unloaded = [[0, 0] for _ in range(N)]
onsets = [[] for _ in range(N)]   # (time, foot, x, y, z)
loaded_steps = [[0, 0] for _ in range(N)]
speed_b, speed_path, vz = [[] for _ in range(N)], [[] for _ in range(N)], [[] for _ in range(N)]

obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS):
        obs, *_ = env.step(policy(obs))
        done = (u.reset_terminated | u.reset_time_outs).tolist()
        force = sensor.data.net_forces_w[:, feet, 2].cpu()
        pos = robot.data.root_pos_w.cpu()
        vb = robot.data.root_lin_vel_b.cpu()
        vw = robot.data.root_lin_vel_w.cpu()
        for i in range(N):
            if done[i]:
                armed[i] = [True, True]
                unloaded[i] = [0, 0]
                onsets[i] = []
                continue
            if step > 100:
                speed_b[i].append(float(vb[i, 0]))
                ang = math.atan2(float(pos[i, 1]), float(pos[i, 0]))
                tangent = torch.tensor([-math.sin(ang), math.cos(ang)])
                speed_path[i].append(float((vw[i, :2] * tangent).sum()))
                vz[i].append(float(vw[i, 2]))
            for f in range(2):
                if force[i, f] > LOAD_N:
                    loaded_steps[i][f] += 1
                    if armed[i][f]:
                        onsets[i].append((step, f, float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
                        armed[i][f] = False
                    unloaded[i][f] = 0
                else:
                    unloaded[i][f] += 1
                    if unloaded[i][f] >= DEBOUNCE:
                        armed[i][f] = True


def med_scalar(name, per_env):
    v = torch.tensor([float(x) for x in per_env])
    if v.numel():
        print(f"[G] {name:34s} median {float(v.median()):.3f}  p90 {float(v.quantile(0.9)):.3f}  n={v.numel()}")


def med(name, per_env, scale=1.0):
    v = torch.tensor([float(torch.tensor(x).mean()) * scale for x in per_env
                      if not isinstance(x, float) and len(x) > 20])
    if v.numel():
        print(f"[G] {name:34s} median {float(v.median()):.3f}  p90 {float(v.quantile(0.9)):.3f}  n={v.numel()}")


secs = (STEPS - 100) / 50.0
med("forward speed, base x (m/s)", speed_b)
med("path speed, tangent (m/s)", speed_path)
med("vertical rate (m/s)", vz)
rate = [len(o) / secs for o in onsets]
med_scalar("footfalls/s (debounced)", rate)
duty = [(ls[0] + ls[1]) / (2.0 * STEPS) for ls in loaded_steps]
med_scalar("duty per foot", duty)

# step length and treads per step, from consecutive onsets of opposite feet
step_len, treads = [], []
for o in onsets:
    for a1, b1 in zip(o[:-1], o[1:]):
        if a1[1] == b1[1]:
            continue
        step_len.append(math.hypot(b1[2] - a1[2], b1[3] - a1[3]))
        treads.append(abs(b1[4] - a1[4]) / 0.12)
if step_len:
    t = torch.tensor(step_len)
    print(f"[G] step length opposite foot (m)      mean {float(t.mean()):.3f}  median {float(t.median()):.3f}  p90 {float(t.quantile(0.9)):.3f}  n={t.numel()}")
if treads:
    t = torch.tensor(treads)
    print(f"[G] treads gained per step             mean {float(t.mean()):.2f}  median {float(t.median()):.2f}  frac>=1 {float((t >= 0.9).float().mean()):.2f}")
print("[G] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
