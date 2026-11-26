"""What does the swing leg do right after it leaves the ground?

The complaint is a backward kick after push-off. This aligns every real swing
(an airborne episode lasting >= 0.1 s) at its lift-off step and prints the leg's
mean trajectory in the robot's own frame for the following 0.4 s:

* fore/aft -- the foot's body-x relative to the root (negative = behind the
  body, which is the kick),
* knee angle -- deep flexion is what throws the shank backwards,
* lift -- the sole's height above where it left the tread.

Also reports how much of the swing the knee spends on its clip rail, since a
command pinned at the rail means the shape is a limit, not a choice.
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
                      "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_299_gain05.pt")
N, STEPS, WARMUP = int(os.environ.get("KICK_N", "8")), int(os.environ.get("KICK_STEPS", "2400")), 60
HORIZON = 20          # steps after lift-off to tabulate (0.4 s at 50 Hz)
AIR_N = 20.0          # foot force below this counts as airborne
# 0.2 s. This gait has many 0.02-0.06 s contact blips; admitting them as
# "swings" made 43% of episodes look like they moved backwards when most
# were a planted foot barely twitching.
MIN_AIR = int(os.environ.get("KICK_MIN_AIR", "10"))

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
rid = [robot.body_names.index(f) for f in ("l_foot", "r_foot")]
kid = [robot.joint_names.index(k) for k in ("Knee_L", "Knee_R")]
print(f"[K] bodies {robot.body_names}", flush=True)

TH, SH, OM = [], [], []
thid = [robot.body_names.index(f) for f in ("l_thigh", "r_thigh")]
shid = [robot.body_names.index(f) for f in ("l_shank_foot", "r_shank_foot")]
ankid = [robot.body_names.index(f) for f in ("l_ankle_cross", "r_ankle_cross")]
FP, RP, YAW, Q, F = [], [], [], [], []
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        act = policy(obs)
        obs, *_ = env.step(act)
        if step >= WARMUP:
            FP.append(robot.data.body_pos_w[:, rid, :].clone())
            RP.append(robot.data.root_pos_w[:, :2].clone())
            YAW.append(robot.data.heading_w.clone())
            Q.append(robot.data.joint_pos[:, kid].clone())
            BP = robot.data.body_pos_w
            yaw_t = robot.data.heading_w
            thigh_vec = BP[:, shid, :] - BP[:, thid, :]
            shank_vec = BP[:, ankid, :] - BP[:, shid, :]
            def _pitch(v):
                fwd = torch.cos(yaw_t).unsqueeze(-1) * v[..., 0] + torch.sin(yaw_t).unsqueeze(-1) * v[..., 1]
                return torch.atan2(fwd, -v[..., 2])
            TH.append(_pitch(thigh_vec))
            SH.append(_pitch(shank_vec))
            OM.append(robot.data.body_ang_vel_w[:, shid, :].clone())
            F.append(sensor.data.net_forces_w[:, fid, :].norm(dim=-1).clone())

FP = torch.stack(FP)    # (T, N, 2, 3)
RP = torch.stack(RP)    # (T, N, 2)
YAW = torch.stack(YAW)  # (T, N)
Q = torch.stack(Q)      # (T, N, 2)
TH = torch.stack(TH)    # (T, N, 2)
SH = torch.stack(SH)    # (T, N, 2)
OM = torch.stack(OM)
F = torch.stack(F)      # (T, N, 2)
T = F.shape[0]
airborne = F < AIR_N

# align on real lift-offs
samples = []   # list of (body_x, knee, lift) per episode, HORIZON long
episodes = 0
for e in range(N):
    for f_ in range(2):
        air = airborne[:, e, f_]
        i = 0
        while i < T:
            if not bool(air[i]):
                i += 1
                continue
            j = i
            while j < T and bool(air[j]):
                j += 1
            if j - i >= MIN_AIR and i + HORIZON < T:   # a real swing, full horizon
                episodes += 1
                z0 = float(FP[i, e, f_, 2])
                bx, kn, zs = [], [], []
                thv, shv, sw = [], [], []
                for t in range(i, i + HORIZON):
                    dx = float(FP[t, e, f_, 0]) - float(RP[t, e, 0])
                    dy = float(FP[t, e, f_, 1]) - float(RP[t, e, 1])
                    yaw = float(YAW[t, e])
                    bx.append(dx * torch.cos(torch.tensor(yaw)).item() + dy * torch.sin(torch.tensor(yaw)).item())
                    kn.append(float(Q[t, e, f_]))
                    zs.append(float(FP[t, e, f_, 2]) - z0)
                    thv.append(float(TH[t, e, f_]))
                    shv.append(float(SH[t, e, f_]))
                    if t == i:
                        y0 = float(YAW[i, e])
                        x0, y0p = float(FP[i, e, f_, 0]), float(FP[i, e, f_, 1])
                    sw.append((float(FP[t, e, f_, 0]) - x0) * torch.cos(torch.tensor(y0)).item()
                              + (float(FP[t, e, f_, 1]) - y0p) * torch.sin(torch.tensor(y0)).item())
                samples.append((bx, kn, zs, thv, shv, sw))
            i = j

print(f"[K] {os.path.basename(CKPT)}  {episodes} swings aligned at lift-off", flush=True)
if episodes:
    bx = torch.tensor([s[0] for s in samples])   # (E, H)
    kn = torch.tensor([s[1] for s in samples])
    zs = torch.tensor([s[2] for s in samples])
    th = torch.tensor([s[3] for s in samples])
    sh = torch.tensor([s[4] for s in samples])
    sw = torch.tensor([s[5] for s in samples])
    print("[K]  step   body_x(m)   knee(rad)   lift(m)   thigh(deg)  shank(deg)  world_fwd(m)", flush=True)
    for t in range(0, HORIZON, 2):
        print(f"[K] {t:5d}   {float(bx[:, t].mean()):+8.3f}   {float(kn[:, t].mean()):+8.3f}   "
              f"{float(zs[:, t].mean()):8.3f}   {float(th[:, t].mean()) * 57.2958:+9.1f}"
              f"   {float(sh[:, t].mean()) * 57.2958:+9.1f}   {float(sw[:, t].mean()):+9.3f}", flush=True)
    min_bx = float(bx.min(dim=1).values.mean())
    arg = int(bx.mean(dim=0).argmin())
    print(f"[K] most rearward foot: mean {min_bx:+.3f} m behind root, "
          f"typical at step {arg} ({arg * 0.02:.2f} s after lift-off)", flush=True)
    print(f"[K] deepest knee during swing: {float(kn.min(dim=1).values.mean()):+.3f} rad  "
          f"(rail is -1.10)", flush=True)
    # Smoothness of the knee's rise: from lift-off to its deepest flexion the
    # angle should fall monotonically (bend), with no stutter back and forth.
    d1 = kn[:, 1:] - kn[:, :-1]                       # (E, H-1) rad/step, <0 = flexing
    peak_i = kn.argmin(dim=1)                          # step of deepest flexion
    idx = torch.arange(d1.shape[1]).unsqueeze(0)
    rising = idx < peak_i.unsqueeze(1)                 # steps before the peak
    back = ((d1 > 0.004) & rising).float().sum(dim=1)  # extension steps during the rise
    d2 = (d1[:, 1:] - d1[:, :-1]).abs()
    print(f"[K] knee rise: peak at {float(peak_i.float().mean()) * 0.02:.2f} s, "
          f"stutter steps before peak {float(back.mean()):.2f}/swing "
          f"({100.0 * float((back > 0).float().mean()):.0f}% of swings have any), "
          f"mean |d2 knee| {float(d2.mean()):.4f} rad/step^2, p95 {float(d2.flatten().quantile(0.95)):.4f}",
          flush=True)
    rail = 0.50 * float((Q < -0.80).float().mean())
    print(f"[K] knee below -0.80 rad: {rail:.2f} of all steps", flush=True)

lat_x = -torch.sin(YAW)
lat_y = torch.cos(YAW)
om_lat = OM[..., 0] * lat_x.unsqueeze(-1) + OM[..., 1] * lat_y.unsqueeze(-1)
dpitch = (SH[1:] - SH[:-1]) / u.step_dt
a = om_lat[:-1].flatten()
b = dpitch.flatten()
corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
print(f"[K] sign check: corr(omega_lat, d(shank pitch)/dt) = {corr:+.3f}", flush=True)
print(f"[K] omega_lat mean {float(a.mean()):+.3f} rad/s   dpitch/dt mean {float(b.mean()):+.3f} rad/s",
      flush=True)
for lbl, t in (("+0.06 s", 3), ("+0.10 s", 5), ("+0.20 s", 10), ("+0.40 s", 18)):
    print(f"[K] world-forward at {lbl}: mean {float(sw[:, t].mean()):+.3f} m  "
          f"p10 {float(sw[:, t].quantile(0.1)):+.3f}  "
          f"{100.0 * float((sw[:, t] < 0).float().mean()):5.1f}% go backwards", flush=True)
print("[K] OK", flush=True)
sys.stdout.flush()
env.close()
simulation_app.close()
