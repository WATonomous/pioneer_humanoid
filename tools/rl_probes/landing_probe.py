"""What does the foot actually do at the instant it lands?

``feet_impact_contact`` charges ``(-vel_z)**2`` on frames where the contact
sensor already reports contact. The physics solver applies the contact impulse
inside that same substep, so by the time the flag is set the normal velocity has
already been cancelled: the term reads ~0 for a hard slam and a soft landing
alike. That is consistent with a 20x weight increase (peak force 2466 -> 4145 N)
producing no change at all.

This probe measures the *approach* instead, on the frames before contact:

* vertical and horizontal speed 1 and 3 control steps before first contact,
* peak contact force in the first 5 steps of contact,
* the same numbers for the straight-stairs reference model, so the spiral gait
  can be judged against the one the user already accepts.

Env vars: CKPT_OVERRIDE, TASK_OVERRIDE, LAND_N, LAND_STEPS.
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

TASK = os.environ.get("TASK_OVERRIDE",
                      "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ.get("CKPT_OVERRIDE",
                      "/workspace/humanoid/outputs/rl/pioneer_humanoid_spiral_foothold/2026-11-18_19-08-38/model_299.pt")
N = int(os.environ.get("LAND_N", "8"))
STEPS = int(os.environ.get("LAND_STEPS", "1200"))
WARMUP = 60
AIR_MIN = 10          # a real swing, in control steps (0.2 s)

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

F, VZ, VXY, VFWD = [], [], [], []
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        act = policy(obs)
        obs, rew, *_ = env.step(act)
        if step >= WARMUP:
            F.append(sensor.data.net_forces_w[:, fid, :].norm(dim=-1).clone())
            v = robot.data.body_lin_vel_w[:, fid, :].clone()
            VZ.append(v[..., 2])
            VXY.append(v[..., :2].norm(dim=-1))
            # horizontal component along the root's heading: +forward, -backward
            q = robot.data.root_quat_w
            fwd = torch.stack([1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                               2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])], dim=-1)
            VFWD.append((v[..., :2] * fwd.unsqueeze(1)).sum(dim=-1))

F = torch.stack(F)
VZ = torch.stack(VZ)
VXY = torch.stack(VXY)
VFWD = torch.stack(VFWD)
dt = u.step_dt
planted = F > 20.0
T = F.shape[0]

vz_prev, vxy_prev, peak, air_len, dropped, fwd_prev = [], [], [], [], [], []
for e in range(N):
    for f_ in range(2):
        p = planted[:, e, f_].tolist()
        run = 0
        for i, flag in enumerate(p):
            if not flag:
                run += 1
                continue
            if run >= AIR_MIN and i >= 4:
                # i is the first loaded step; look back before the impulse
                vz_prev.append(float(VZ[i - 1, e, f_]))
                vxy_prev.append(float(VXY[i - 1, e, f_]))
                fwd_prev.append(float(VFWD[i - 1, e, f_]))
                peak.append(float(F[i:i + 5, e, f_].max()))
                air_len.append(run)
                dropped.append(float(VZ[i - 3, e, f_]))
            run = 0

if not vz_prev:
    print("[L] no landings detected", flush=True)
    sys.exit(0)

vz_p = torch.tensor(vz_prev)
vxy_p = torch.tensor(vxy_prev)
pk = torch.tensor(peak)
dr = torch.tensor(dropped)
print(f"[L] {os.path.basename(os.path.dirname(CKPT))}/{os.path.basename(CKPT)}  "
      f"{len(vz_prev)} landings", flush=True)
print(f"[L] approach vz (1 step before contact)  mean {float(vz_p.mean()):+7.3f}  "
      f"p10 {float(vz_p.quantile(0.1)):+7.3f}  min {float(vz_p.min()):+7.3f} m/s", flush=True)
print(f"[L] approach |vxy|                       mean {float(vxy_p.mean()):7.3f}  "
      f"p90 {float(vxy_p.quantile(0.9)):7.3f} m/s", flush=True)
print(f"[L] vz 3 steps before contact            mean {float(dr.mean()):+7.3f} m/s", flush=True)
print(f"[L] peak force in first 5 contact steps  mean {float(pk.mean()):6.0f}  "
      f"p90 {float(pk.quantile(0.9)):6.0f}  max {float(pk.max()):6.0f} N", flush=True)
print(f"[L] airborne length before landing       mean {sum(air_len) / len(air_len):.1f} steps", flush=True)
fwd_p = torch.tensor(fwd_prev)
print(f"[L] approach forward vel (root frame)    mean {float(fwd_p.mean()):+7.3f}  "
      f"p10 {float(fwd_p.quantile(0.1)):+7.3f}  p90 {float(fwd_p.quantile(0.9)):+7.3f} m/s", flush=True)
print(f"[L] share landing while moving backward  "
      f"{100.0 * float((fwd_p < -0.2).float().mean()):.0f}%", flush=True)
print(f"[L] corr(|vxy| approach, peak force)     "
      f"{float(torch.corrcoef(torch.stack([vxy_p, pk]))[0, 1]):+.2f}", flush=True)
print(f"[L] share of landings with vz < -0.5     "
      f"{100.0 * float((vz_p < -0.5).float().mean()):.0f}%", flush=True)
print(f"[L] OK", flush=True)
sys.stdout.flush()
env.close()
simulation_app.close()
