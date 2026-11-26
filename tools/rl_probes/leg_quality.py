"""Measure leg smoothness and landing hardness on the spiral.

Two complaints to quantify separately:

* "legs shaky" -- high-frequency joint chatter. Measured as the RMS leg joint
  velocity per joint group, the RMS step-to-step joint acceleration, the policy
  action rate and the swing foot's vertical acceleration (a foot that is being
  yanked around in the air, not carried).
* "stomps onto the stairs" -- the downward speed the foot carries into the
  tread and the normal-force spike it produces. Measured at debounced contact
  onsets: the foot's vertical speed one frame before the onset, and the peak
  normal force over the five frames after it.
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

TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ.get("CKPT_OVERRIDE", "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_299_gait.pt")
N, STEPS = 6, 2500
LOAD_N, DEBOUNCE = 50.0, 5

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005), (-1.504, -0.182, 2.017, 4.8331),
    (1.361, -0.666, 2.857, 7.3987), (-1.192, 0.935, 3.817, 10.3309), (1.500, -0.213, 5.017, 13.9960),
)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
if "Spiral" in TASK:
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

groups = {}
for name, key in (("hip", "hip"), ("knee", "knee"), ("ankle", "ankle")):
    groups[name] = [i for i, n in enumerate(robot.joint_names) if key in n.lower()]
leg_idx = sorted({i for idx in groups.values() for i in idx})
print("[Q] leg joints", [robot.joint_names[i] for i in leg_idx])

dt = u.step_dt
armed = [[True, True] for _ in range(N)]
unloaded = [[0, 0] for _ in range(N)]
history = [[] for _ in range(N)]  # (step, vel_z[2], force[2])
desc, peak, skid, hard = [[] for _ in range(N)], [[] for _ in range(N)], [[] for _ in range(N)], [[] for _ in range(N)]
jvel = {k: [[] for _ in range(N)] for k in groups}
jacc, arate, swing_acc = [[] for _ in range(N)], [[] for _ in range(N)], [[] for _ in range(N)]
knee_hist = [[] for _ in range(N)]
vib, zc = [[] for _ in range(N)], [[] for _ in range(N)]
vib_c, vib_s = [[] for _ in range(N)], [[] for _ in range(N)]
knee_idx = groups.get("knee", [])
prev_vel = None
prev_action = None
prev_vz = None
prev_knee = None

obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS):
        action = policy(obs)
        obs, *_ = env.step(action)
        done = (u.reset_terminated | u.reset_time_outs).tolist()
        force = sensor.data.net_forces_w[:, feet, 2].cpu()
        vel_z = robot.data.body_lin_vel_w[:, feet, 2].cpu()
        vel_x = robot.data.body_lin_vel_w[:, feet, 0].cpu()
        jv = robot.data.joint_vel.cpu()
        act = action.detach().cpu()
        for i in range(N):
            history[i].append((step, vel_z[i].tolist(), force[i].tolist()))
            history[i] = history[i][-8:]
            if done[i]:
                armed[i] = [True, True]
                unloaded[i] = [0, 0]
                continue
            if step > 100:
                for k, idx in groups.items():
                    if idx:
                        jvel[k][i].append(float(jv[i, idx].square().mean().sqrt()))
                if prev_vel is not None:
                    jacc[i].append(float(((jv[i, leg_idx] - prev_vel[i, leg_idx]) / dt).square().mean().sqrt()))
                if prev_action is not None:
                    arate[i].append(float((act[i] - prev_action[i]).square().mean().sqrt()))
                if knee_idx:
                    kh = knee_hist[i]
                    kh.append(jv[i, knee_idx].clone())
                    kh[:] = kh[-3:]
                    if len(kh) == 3:
                        # high-pass residual: what is left after removing the
                        # local mean of the knee velocity (a vibration measure,
                        # independent of how fast the leg is moving)
                        resid = kh[2] - 0.5 * (kh[0] + kh[1])
                        vib[i].append(float(resid.square().mean().sqrt()))
                        for leg in range(min(len(knee_idx), 2)):
                            if force[i, leg] > 50.0:
                                vib_c[i].append(float(resid[leg].abs()))
                            else:
                                vib_s[i].append(float(resid[leg].abs()))
                    if prev_knee is not None:
                        zc[i].append(float(((jv[i, knee_idx] > 0) != (prev_knee[i] > 0)).float().mean()))
                if prev_vz is not None:
                    swing = (force[i] < 1.0)
                    if bool(swing.any()):
                        acc = ((vel_z[i] - prev_vz[i]) / dt)[swing]
                        swing_acc[i].append(float(acc.square().mean().sqrt()))
            for f in range(2):
                if force[i, f] > LOAD_N:
                    if armed[i][f]:
                        back = [h for h in history[i][:-1] if h[0] >= step - 5]
                        if back:
                            desc[i].append(-back[-1][1][f])
                            skid[i].append(abs(float(vel_x[i, f])))
                        peak[i].append(float(force[i, f]))
                        hard[i].append(1.0 if force[i, f] > 1000.0 else 0.0)
                        armed[i][f] = False
                    else:
                        if peak[i]:
                            peak[i][-1] = max(peak[i][-1], float(force[i, f]))
                            hard[i][-1] = 1.0 if peak[i][-1] > 1000.0 else 0.0
                    unloaded[i][f] = 0
                else:
                    unloaded[i][f] += 1
                    if unloaded[i][f] >= DEBOUNCE:
                        armed[i][f] = True
        prev_vel = jv
        prev_action = act
        prev_vz = vel_z
        prev_knee = jv[:, knee_idx].clone() if knee_idx else None


def report(name, per_env, scale=1.0, fmt=".3f", reducer="mean"):
    vals = []
    for x in per_env:
        if len(x) > 20:
            t = torch.tensor(x)
            vals.append(float(t.mean() if reducer == "mean" else t.median()) * scale)
    if vals:
        v = torch.tensor(vals)
        print(f"[Q] {name:34s} median {float(v.median()):{fmt}}  p90 {float(v.quantile(0.9)):{fmt}}  n={v.numel()}")


for k in groups:
    report(f"{k} joint vel RMS (rad/s)", jvel[k])
report("leg joint acc RMS (rad/s^2)", jacc)
report("action rate RMS (norm units)", arate)
report("swing foot z acc RMS (m/s^2)", swing_acc, fmt=".0f")
report("knee vel high-pass RMS (rad/s)", vib, fmt=".3f")
report("knee vel sign flips/s", zc, scale=50.0, fmt=".2f")
report("knee high-pass |res| in stance", vib_c, fmt=".3f")
report("knee high-pass |res| in swing", vib_s, fmt=".3f")
report("touchdown descent speed (m/s)", desc, fmt=".3f", reducer="median")
report("touchdown peak force (N)", peak, fmt=".0f", reducer="median")
report("touchdown hard-landing frac", hard, fmt=".2f", reducer="mean")
report("touchdown foot x speed (m/s)", skid, fmt=".3f", reducer="median")
print("[Q] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
