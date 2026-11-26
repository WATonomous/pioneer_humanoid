"""Two complaints, measured: the foot *hovers* over the next tread then drops, and
the back foot pushes off its *ball* instead of the flat sole.

Per real swing (>= MIN_AIR steps airborne), per foot:
* swing duration, and how long the foot is "hovering": airborne with speed below
  HOVER_V (m/s) -- the foot is neither travelling nor descending,
* the time between reaching its peak height and touching down, and the speed it
  descends at in the last 0.1 s (a hover-then-drop shows as a long, flat peak
  followed by a fast fall),
* sole tilt (degrees between the foot's down-axis and gravity) over the last 0.2 s
  of the stance before lift-off. Flat sole ~0; heel-lift/ball push-off grows
  toward lift-off. The tilt is split into its two foot-frame components so the
  pitch axis can be told from the roll axis.

Env vars: CKPT_OVERRIDE (required), TASK_OVERRIDE, STEP_N, STEP_STEPS.

    isaaclab.sh -p tools/onshape_course/step_probe.py   # with CKPT_OVERRIDE=<model.pt>
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
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ["CKPT_OVERRIDE"]  # checkpoint to probe
N, STEPS, WARMUP = int(os.environ.get("STEP_N", "8")), int(os.environ.get("STEP_STEPS", "2400")), 60
MIN_AIR = 10
HOVER_V = 0.15
AIR_N = 20.0
PRE = 10              # steps of stance before lift-off to examine (0.2 s)

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
env.clip_actions = agent_cfg.clip_actions
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)

robot = u.scene["robot"]
sensor = u.scene["contact_forces"]
s_cfg = SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True)
s_cfg.resolve(u.scene)
sid = list(s_cfg.body_ids)
rid = [robot.body_names.index(n) for n in ("l_foot", "r_foot")]
print(f"[ST] id check: sensor ids {sid}  robot ids {rid}  "
      f"({'SAME' if sid == rid else 'DIFFERENT -- earlier probes that indexed robot data with sensor ids were wrong'})",
      flush=True)

jid = [robot.joint_names.index(n) for n in ("Ankle_P_L", "Ankle_P_R")]
lim = robot.data.joint_pos_limits[0].cpu()[jid]          # (2, 2) lower/upper
AP = []
F, Z, V, TILT = [], [], [], []
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        obs, *_ = env.step(policy(obs))
        if step < WARMUP:
            continue
        F.append(sensor.data.net_forces_w[:, sid, :].norm(dim=-1).clone())
        Z.append(robot.data.body_pos_w[:, rid, 2].clone())
        V.append(robot.data.body_lin_vel_w[:, rid, :].norm(dim=-1).clone())
        q = robot.data.body_quat_w[:, rid, :]
        g = robot.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, 2, -1)
        gl = quat_apply_inverse(q.reshape(-1, 4), g.reshape(-1, 3)).view(N, 2, 3)
        TILT.append(gl.clone())
        AP.append(robot.data.joint_pos[:, jid].clone())

F = torch.stack(F)
Z = torch.stack(Z)
V = torch.stack(V)
AP = torch.stack(AP)
TILT = torch.stack(TILT)     # (T, N, 2, 3): gravity in the foot frame (unit down-vector = (0,0,-1) when flat)
dt = u.step_dt
T = F.shape[0]
air = F < AIR_N

dur, hover, peak2land, fall_v, pre_tilt, pre_a, pre_b = [], [], [], [], [], [], []
ap_pre, ap_lo, ap_hi = [], [], []
plateau, hi_time, hpk = [], [], []
n_sw = 0
for e in range(N):
    for f_ in range(2):
        a_ = air[:, e, f_].tolist()
        i = 0
        while i < T:
            if not a_[i]:
                i += 1
                continue
            j = i
            while j < T and a_[j]:
                j += 1
            if j - i >= MIN_AIR and i >= PRE and j < T:
                n_sw += 1
                z = Z[i:j, e, f_]
                v = V[i:j, e, f_]
                dur.append((j - i) * dt)
                hover.append(float((v < HOVER_V).float().sum()) * dt)
                pk = int(z.argmax())
                hh = z - z[-1]                       # height above where it lands
                hp = float(hh.max())
                hpk.append(hp)
                if hp > 0.02:
                    plateau.append(float((hh > 0.8 * hp).float().sum()) * dt)   # time within 20% of its peak height
                    hi_time.append(float((hh > 0.5 * hp).float().sum()) * dt)   # time above half its peak height
                peak2land.append((j - i - pk) * dt)
                k = max(j - 5, i)
                fall_v.append(float(Z[k, e, f_] - Z[j - 1, e, f_]) / max((j - 1 - k) * dt, dt))
                # stance tilt in the PRE steps before lift-off (foot frame gravity xy)
                tl = TILT[i - PRE:i, e, f_, :]
                pre_tilt.append(torch.rad2deg(torch.asin(tl[:, :2].norm(dim=-1).clamp(max=1.0))))
                ap_pre.append(AP[i - PRE:i, e, f_].cpu())
                ap_lo.append(float(lim[f_, 0]))
                ap_hi.append(float(lim[f_, 1]))
                pre_a.append(torch.rad2deg(torch.asin(tl[:, 0].clamp(-1.0, 1.0))))
                pre_b.append(torch.rad2deg(torch.asin(tl[:, 1].clamp(-1.0, 1.0))))
            i = j

if n_sw == 0:
    print("[ST] no swings", flush=True)
    sys.exit(0)

d = torch.tensor(dur)
h = torch.tensor(hover)
p2l = torch.tensor(peak2land)
fv = torch.tensor(fall_v)
print(f"[ST] {os.path.basename(os.path.dirname(CKPT))}/{os.path.basename(CKPT)}  {n_sw} swings", flush=True)
print(f"[ST] swing duration       mean {float(d.mean()):.2f} s  p90 {float(d.quantile(0.9)):.2f}", flush=True)
print(f"[ST] hover (airborne, speed < {HOVER_V} m/s)  mean {float(h.mean()):.2f} s = "
      f"{100.0 * float(h.sum() / d.sum()):.0f}% of swing   p90 {float(h.quantile(0.9)):.2f} s", flush=True)
if plateau:
    pl = torch.tensor(plateau)
    ht = torch.tensor(hi_time)
    print(f"[ST] peak height above landing tread  mean {float(torch.tensor(hpk).mean()):.3f} m", flush=True)
    print(f"[ST] time within 20% of peak height   mean {float(pl.mean()):.2f} s  p90 {float(pl.quantile(0.9)):.2f}", flush=True)
    print(f"[ST] time above half of peak height   mean {float(ht.mean()):.2f} s  "
          f"= {100.0 * float(ht.sum() / d[: len(ht)].sum()):.0f}% of swing", flush=True)
print(f"[ST] peak height -> touchdown  mean {float(p2l.mean()):.2f} s  "
      f"({100.0 * float(p2l.sum() / d.sum()):.0f}% of swing is after the peak)", flush=True)
print(f"[ST] descent speed, last 0.1 s  mean {float(fv.mean()):+.3f}  p90 {float(fv.quantile(0.9)):+.3f} m/s", flush=True)
PT = torch.stack(pre_tilt)
PA = torch.stack(pre_a)
PB = torch.stack(pre_b)
print("[ST] stance sole tilt before lift-off (deg; 0 = flat)   t-0.20  -0.14  -0.08  -0.02", flush=True)
print("[ST]   total tilt        " + "  ".join(f"{float(PT[:, k].mean()):6.1f}" for k in (0, 3, 6, 9)), flush=True)
print("[ST]   foot-x component  " + "  ".join(f"{float(PA[:, k].mean()):+6.1f}" for k in (0, 3, 6, 9)), flush=True)
print("[ST]   foot-y component  " + "  ".join(f"{float(PB[:, k].mean()):+6.1f}" for k in (0, 3, 6, 9)), flush=True)
print(f"[ST] swings whose sole tilts > 10 deg in the last 0.1 s of stance: "
      f"{100.0 * float((PT[:, 5:].max(dim=1).values > 10.0).float().mean()):.0f}%", flush=True)
AQ = torch.stack(ap_pre)
lo = torch.tensor(ap_lo)
hi = torch.tensor(ap_hi)
print(f"[ST] Ankle_P joint limits (rad): {float(lo.min()):+.2f} .. {float(hi.max()):+.2f}", flush=True)
print("[ST] Ankle_P angle before lift-off (rad)  t-0.20  -0.14  -0.08  -0.02: " +
      "  ".join(f"{float(AQ[:, k].mean()):+6.2f}" for k in (0, 3, 6, 9)), flush=True)
near = ((AQ[:, 5:] - lo.unsqueeze(1)).abs().min(dim=1).values < 0.05) | ((AQ[:, 5:] - hi.unsqueeze(1)).abs().min(dim=1).values < 0.05)
print(f"[ST] ankle within 0.05 rad of a limit in the last 0.1 s of stance: {100.0 * float(near.float().mean()):.0f}% of swings", flush=True)
print("[ST] OK", flush=True)
sys.stdout.flush()
env.close()
simulation_app.close()
