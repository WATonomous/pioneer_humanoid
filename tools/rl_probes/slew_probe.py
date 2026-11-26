"""Does a slew-rate limit on the commanded joint target fix the leg chatter?

The baseline measurement says the policy's normalised action moves 1.06 RMS per
20 ms step. On the knee that is scale 0.3 rad, i.e. the *target* jumps 0.3 rad
every control step -- a commanded knee speed of ~15 rad/s against a 3.67 rad/s
motor velocity limit. The motor cannot follow that, so it saturates and the leg
reads as a buzz rather than a stride.

A slew limiter caps how far the target may move per control step without adding
lag when the motion is genuinely slow:

    a_t = clip(a_raw_t, a_{t-1} - lim, a_{t-1} + lim)

with `lim` per joint = LIMIT_RAD_PER_STEP / action_scale[joint], so LIMIT is in
radians of actual joint target per 20 ms (2.0 rad/s at 0.04). A real stride at
0.10 m/s needs well under that, so the limit should be invisible to the gait and
only bite on the buzz.

Reported per limit: action delta, leg joint acceleration, knee velocity sign
flips, climb rate (root height gain/s) and falls.
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
N, STEPS = 4, 700
LIMITS = [float(v) for v in os.environ.get("SLEW_LIMITS", "99,0.08,0.05,0.035,0.025").split(",")]

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005),
    (-1.504, -0.182, 2.017, 4.8331), (1.361, -0.666, 2.857, 7.3987),
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

groups = {}
for name, key in (("hip", "hip"), ("knee", "knee"), ("ankle", "ankle")):
    groups[name] = [i for i, n in enumerate(robot.joint_names) if key in n.lower()]
leg_idx = sorted({i for idx in groups.values() for i in idx})
knee_idx = groups["knee"]
dt = u.step_dt

# per-joint normalised-action limit that corresponds to LIMIT rad/step
at = u.action_manager.get_term("joint_pos")
_names = list(at._joint_names)


def _vec(v, n):
    t = torch.as_tensor(v, dtype=torch.float32)
    if t.dim() == 0:
        return t.expand(n).clone()
    if t.dim() == 2:
        t = t[0]
    return t.reshape(-1)[:n].clone()


scale = _vec(at._scale, len(_names)).to(u.device).abs().clamp_min(1e-6)
print("[L] action joints", len(at._joint_names), "scale", scale.tolist())
print("[L] lim_rad  action_dRMS  leg_acc  knee_flip/s  climb_m/s  falls  len_s")


def run(limit_rad):
    lim = (limit_rad / scale).clamp_max(1e6)
    arate, jacc, flips = [], [], []
    climb, falls = 0.0, 0
    with torch.inference_mode():
        # reset INSIDE inference mode: a second reset outside it tries to write
        # in place into tensors the previous run created as inference tensors.
        obs, _ = env.reset()
        prev_act = torch.zeros(N, u.action_manager.total_action_dim, device=u.device)
        prev_vel = prev_knee = prev_z = None
        alive = torch.zeros(N, device=u.device)
        for step in range(STEPS):
            raw = policy(obs)
            act = torch.max(torch.min(raw, prev_act + lim), prev_act - lim)
            delta = (act - prev_act).square().mean().sqrt()
            prev_act = act
            obs, *_ = env.step(act)
            done = u.reset_terminated | u.reset_time_outs
            keep = (~done).float()
            jv = robot.data.joint_vel
            z = robot.data.root_pos_w[:, 2]
            if step > 60:
                arate.append(float(delta))
                if prev_vel is not None:
                    jacc.append(float(((jv[:, leg_idx] - prev_vel[:, leg_idx]) / dt).square().mean().sqrt()))
                if prev_knee is not None:
                    flips.append(float(((jv[:, knee_idx] > 0) != (prev_knee[:, knee_idx] > 0)).float().mean()))
                alive += keep
                if prev_z is not None:
                    climb += float(((z - prev_z) * keep).sum())
            falls += int(done.sum().item())
            if step > 60:
                prev_vel = jv.clone()
                prev_knee = jv[:, knee_idx].clone()
            prev_z = z.clone()
    return dict(arate=float(torch.tensor(arate).mean()) if arate else 0.0,
                jacc=float(torch.tensor(jacc).mean()) if jacc else 0.0,
                flips=float(torch.tensor(flips).mean()) * 50.0 if flips else 0.0,
                climb=climb / max(float(alive.sum()), 1.0) / dt,
                falls=falls, len_s=float(alive.mean()) * dt)


for limit in LIMITS:
    r = run(limit)
    print(f"[L] {limit:8.3f}  {r['arate']:10.3f}  {r['jacc']:7.1f}  {r['flips']:10.2f}  "
          f"{r['climb']:+9.3f}  {r['falls']:5d}  {r['len_s']:5.1f}")
    sys.stdout.flush()
print("[L] OK")
env.close()
simulation_app.close()
