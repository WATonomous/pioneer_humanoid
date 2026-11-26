"""Does capping the knee target's slew rate remove the swing chatter?

Measured on the shipped checkpoint (target_probe.py): the knee's commanded
target flips between its two clip rails about ten times inside a ~12-step swing
-- +/-0.7 rad per 20 ms step, i.e. ~35 rad/s of commanded knee speed against a
3.67 rad/s motor -- and the joint reverses direction 17.1x/s against the
straight deliverable's 5.0. That burst is the visible clunk.

This caps how far the target may move per control step:

    tgt_t = clip(tgt_raw_t, tgt_{t-1} - lim, tgt_{t-1} + lim)

with ``lim`` in radians of *joint target* per step, converted to normalised
action units per joint via that joint's action scale. Every limit tested is
below the motor's own velocity limit, so it only removes motion the hardware
could not follow anyway.

Reports, per limit: the target's own reversal rate, the knee joint's reversal
rate and acceleration, the climb rate (root height gain per second of sim time)
and falls -- so the cost of the lag is visible next to the smoothing.

Written like target_probe.py (one reset, no repeated env.reset) because a
second reset outside inference mode is what hung the earlier sweep probes.
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
                      "/workspace/humanoid/outputs/rl/pioneer_humanoid_spiral_foothold/2026-11-16_22-26-35/model_299.pt")
N, STEPS, WARMUP = 4, 900, 60
LIMIT_RAD = float(os.environ.get("SLEW_RAD", "99"))

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005),
    (-1.504, -0.182, 2.017, 4.8331), (1.361, -0.666, 2.857, 7.3987),
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
at = u.action_manager.get_term("joint_pos")
names = list(at._joint_names)
jid = [robot.joint_names.index(n) for n in names]


def _vec(v, n):
    t = torch.as_tensor(v, dtype=torch.float32)
    if t.dim() == 0:
        return t.expand(n).clone()
    if t.dim() == 2:
        t = t[0]
    return t.reshape(-1)[:n].clone()


scale = _vec(at._scale, len(names)).to(u.device).abs().clamp_min(1e-6)
lim = (LIMIT_RAD / scale).clamp_max(1e6)
# SLEW_SCOPE=knee limits only the knees. The all-joint sweep crippled the hips
# and ankles too, so its "the flick is load-bearing" reading is confounded: the
# robot fell because every joint was rate-limited, not because the knee's
# command had to stay fast.
if os.environ.get("SLEW_SCOPE", "all").lower() == "knee":
    wide = torch.full_like(lim, 1e6)
    for i in (names.index("Knee_L"), names.index("Knee_R")):
        wide[i] = lim[i]
    lim = wide
knee = [names.index("Knee_L"), names.index("Knee_R")]
leg_idx = [i for i, n in enumerate(names) if any(k in n.lower() for k in ("hip", "knee", "ankle"))]
dt = u.step_dt

# Foot load lives in the contact sensor's own body order, not the robot's.
feet_cfg = SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True)
feet_cfg.resolve(u.scene)
feet_ids = list(feet_cfg.body_ids)

tgt_hist, q_hist, v_hist, f_hist = [], [], [], []
climb, falls = 0.0, 0
alive = torch.zeros(N, device=u.device)
prev_act = torch.zeros(N, u.action_manager.total_action_dim, device=u.device)
prev_z = None
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        raw = policy(obs)
        act = torch.max(torch.min(raw, prev_act + lim), prev_act - lim)
        prev_act = act
        obs, *_ = env.step(act)
        done = u.reset_terminated | u.reset_time_outs
        keep = (~done).float()
        z = robot.data.root_pos_w[:, 2]
        falls += int(done.sum().item())
        if step >= WARMUP:
            tgt_hist.append(at.processed_actions.clone())
            q_hist.append(robot.data.joint_pos[:, jid].clone())
            v_hist.append(robot.data.joint_vel[:, jid].clone())
            # foot load, for the touchdown-bounce count that is the visible clunk
            f_hist.append(u.scene["contact_forces"].data.net_forces_w[:, feet_ids, :].norm(dim=-1).clone())
            alive += keep
            if prev_z is not None:
                climb += float(((z - prev_z) * keep).sum())
        prev_z = z.clone()

tgt = torch.stack(tgt_hist)
q = torch.stack(q_hist)
v = torch.stack(v_hist)
F = torch.stack(f_hist)


def taps_per_s(F, thr=20.0):
    """Air episodes shorter than 0.1 s, per second per foot: touchdown bounces."""
    planted = F > thr
    short = 0
    for e in range(F.shape[1]):
        for f_ in range(F.shape[2]):
            run = 0
            for flag in (~planted[:, e, f_]).tolist():
                if flag:
                    run += 1
                elif run:
                    if run * u.step_dt < 0.10:
                        short += 1
                    run = 0
    return short / (F.shape[0] * F.shape[1] * F.shape[2] * u.step_dt)


def rev_per_s(x):
    d = x[1:] - x[:-1]
    s = torch.sign(d)
    return float(((s[1:] * s[:-1]) < 0).float().mean()) / dt


k = knee
kk = tgt[:, :, k]
lo = kk.amin(dim=(0, 1), keepdim=True)
hi = kk.amax(dim=(0, 1), keepdim=True)
rail = ((kk - lo).abs() < 1e-3) | ((kk - hi).abs() < 1e-3)
jacc = float(((v[1:, :, leg_idx] - v[:-1, :, leg_idx]) / dt).square().mean().sqrt())
climb_rate = climb / max(float(alive.sum()), 1.0) / dt

print(f"[S] lim_rad {LIMIT_RAD:7.3f}  tgt_dRMS {float((kk[1:] - kk[:-1]).abs().mean()):7.3f}  "
      f"tgt_rev/s {rev_per_s(kk):6.2f}  knee_q_rev/s {rev_per_s(q[:, :, k]):6.2f}  "
      f"rail {float(rail.float().mean()):5.2f}  leg_acc {jacc:7.1f}  "
      f"climb {climb_rate:+.3f} m/s  falls {falls:4d}  mean|a| {float(prev_act.abs().mean()):6.2f}  "
      f"taps/s {taps_per_s(F):5.2f}")
sys.stdout.flush()
env.close()
simulation_app.close()
