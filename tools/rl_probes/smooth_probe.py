"""Where does the jerk and the hard landing actually come from?

Reports, for the current checkpoint and the straight-stairs deliverable:

* leg-joint acceleration and the knee's high-frequency angle residual,
* base linear/angular acceleration (whole-body jitter as opposed to legs),
* touchdown: peak contact force and vertical/horizontal sole speed at the
  instant the foot first loads,
* and the per-step charge of every smoothness and impact term, so the weights
  can be set from what they cost rather than guessed.
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
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ.get("CKPT_OVERRIDE",
                      "/workspace/humanoid/outputs/rl/pioneer_humanoid_spiral_foothold/2026-11-18_16-04-30/model_299.pt")
N, STEPS, WARMUP = int(os.environ.get("SMOOTH_N", "8")), int(os.environ.get("SMOOTH_STEPS", "1200")), 60
SMOOTH_TERMS = ("action_rate_l2", "dof_acc_l2", "joint_vel_l2", "feet_impact",
                "feet_slide", "feet_swing_height", "action_l2", "contact_mismatch",
                "feet_swing_behind", "shank_backward_rotation", "knee_flexion_airborne")

_BASE_SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005),
    (-1.504, -0.182, 2.017, 4.8331), (1.361, -0.666, 2.857, 7.3987),
)
SLOTS = tuple(_BASE_SLOTS[i % len(_BASE_SLOTS)] for i in range(N))

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
if "Spiral" in TASK:
    ev = u.event_manager.get_term_cfg("reset_base")
    ev.params["slots"] = SLOTS
    ev.params["xy_jitter"] = 0.03
    ev.params["yaw_jitter"] = 0.03


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
sensor = u.scene["contact_forces"]
feet_cfg = SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True)
feet_cfg.resolve(u.scene)
fid = list(feet_cfg.body_ids)
leg_idx = [i for i, n in enumerate(robot.joint_names)
           if any(k in n.lower() for k in ("hip", "knee", "ankle"))]

V, BV, BAV, F, Z = [], [], [], [], []
charges = {}
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        act = policy(obs)
        obs, rew, *_ = env.step(act)
        if step >= WARMUP:
            V.append(robot.data.joint_vel[:, leg_idx].clone())
            BV.append(robot.data.root_lin_vel_w.clone())
            BAV.append(robot.data.root_ang_vel_w.clone())
            F.append(sensor.data.net_forces_w[:, fid, :].norm(dim=-1).clone())
            Z.append(robot.data.body_pos_w[:, fid, 2].clone())
            for name in SMOOTH_TERMS:
                if name not in charges:
                    try:
                        charges[name] = []
                        tc = u.reward_manager.get_term_cfg(name)
                    except Exception:  # noqa: BLE001
                        charges[name] = None
                        continue
                if charges.get(name) is not None:
                    tc = u.reward_manager.get_term_cfg(name)
                    charges[name].append(float(tc.func(u, **tc.params).mean()) * tc.weight)

V = torch.stack(V)
BV = torch.stack(BV)
BAV = torch.stack(BAV)
F = torch.stack(F)
Z = torch.stack(Z)
dt = u.step_dt

acc = (V[1:] - V[:-1]) / dt
bacc = (BV[1:] - BV[:-1]) / dt
bav_acc = (BAV[1:] - BAV[:-1]) / dt
print(f"[S] {os.path.basename(os.path.dirname(CKPT))}/{os.path.basename(CKPT)}  "
      f"{V.shape[0]} steps", flush=True)
print(f"[S] leg joint acc      mean {float(acc.abs().mean()):7.1f}  p99 {float(acc.abs().flatten().quantile(0.99)):7.1f} rad/s^2",
      flush=True)
print(f"[S] base lin acc       mean {float(bacc.abs().mean()):7.2f}  p99 {float(bacc.abs().flatten().quantile(0.99)):7.2f} m/s^2",
      flush=True)
print(f"[S] base ang acc       mean {float(bav_acc.abs().mean()):7.2f}  p99 {float(bav_acc.abs().flatten().quantile(0.99)):7.2f} rad/s^2",
      flush=True)

planted = F > 20.0
# Touchdown = the step a *real* swing ends (airborne for >= 10 steps), so a
# mid-air contact blip is not mistaken for a landing. The previous version took
# any plant->load edge and reported the sole moving UP at "touchdown", which is
# how you know it was measuring chatter.
descent, peak_after = [], []
for e in range(N):
    for f_ in range(2):
        air = (~planted[:, e, f_]).tolist()
        run = 0
        for i, flag in enumerate(air):
            if flag:
                run += 1
                continue
            if run >= 10 and i >= 3:                      # a real swing just ended
                vz = (float(Z[i, e, f_]) - float(Z[i - 3, e, f_])) / (3 * dt)
                descent.append(vz)
                peak_after.append(float(F[i:i + 5, e, f_].max()))
            run = 0
if descent:
    dz = torch.tensor(descent)
    pf = torch.tensor(peak_after)
    print(f"[S] touchdown descent  mean {float(dz.mean()):+7.3f}  p10 {float(dz.quantile(0.1)):+7.3f} m/s "
          f"({len(descent)} landings)", flush=True)
    print(f"[S] peak force at landing  mean {float(pf.mean()):6.0f}  p90 {float(pf.quantile(0.9)):6.0f}  "
          f"max {float(pf.max()):6.0f} N", flush=True)
print(f"[S] foot force         mean {float(F.mean()):6.0f}  p99 {float(F.flatten().quantile(0.99)):6.0f}  "
      f"max {float(F.max()):6.0f} N", flush=True)

print("[S] smoothness/impact term charge per step (weight x raw):", flush=True)
for name, vals in charges.items():
    if vals:
        t = torch.tensor(vals)
        print(f"[S]   {name:26s} mean {float(t.mean()):+9.5f}  p90 {float(t.quantile(0.9)):+9.5f}  "
              f"sum {float(t.sum()):+9.2f}", flush=True)
print("[S] OK", flush=True)
sys.stdout.flush()
env.close()
simulation_app.close()
