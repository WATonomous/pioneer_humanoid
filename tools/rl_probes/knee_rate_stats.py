"""How fast does the knee's target actually need to move?

The slew sweep says which limits the *trained* policy survives, but not how much
of the command each limit has to throw away. That matters for picking a value to
train against: a limit that clips 1 % of steps is a small nudge, one that clips
40 % is a different gait.

Reports the per-step knee target change (rad/step and rad/s) as quantiles, and
the share of steps each candidate limit would clamp, measured both on the raw
policy output and on the clipped target the joint actually receives.
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
                      "/workspace/humanoid/outputs/rl/pioneer_humanoid_spiral_foothold/2026-11-16_22-26-35/model_299.pt")
N, STEPS, WARMUP = 4, 900, 60
CANDIDATES = (6.0, 8.0, 12.0, 16.0, 20.0, 30.0)   # rad/s

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005),
    (-1.504, -0.182, 2.017, 4.8331), (1.361, -0.666, 2.857, 7.3987),
)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
# The limit must be off for this measurement: we want the raw demand.
if hasattr(cfg.actions.joint_pos, "max_rate"):
    cfg.actions.joint_pos.max_rate = None
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

at = u.action_manager.get_term("joint_pos")
names = list(at._joint_names)
knee = [names.index("Knee_L"), names.index("Knee_R")]
dt = u.step_dt

raw_d, clip_d = [], []
prev_raw = prev_clip = None
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        act = policy(obs)
        obs, *_ = env.step(act)
        if step >= WARMUP:
            # the joint's target after the affine transform and clip, before any
            # rate limit (this cfg has none)
            proc = at.processed_actions[:, knee]
            raw = at.raw_actions[:, knee]
            if prev_raw is not None:
                raw_d.append((raw - prev_raw).abs())
                clip_d.append((proc - prev_clip).abs())
            prev_raw = raw.clone()
            prev_clip = proc.clone()

D = torch.stack(clip_d).flatten()     # clipped target -- what the joint receives
R = torch.stack(raw_d).flatten()      # policy's own action units
print(f"[K] {os.path.basename(CKPT)}  {D.numel()} knee samples @ {1.0 / dt:.0f} Hz", flush=True)
for label, x, unit in (("target (post-clip)", D, "rad"), ("raw action", R, "action")):
    q = [float(x.quantile(p)) for p in (0.5, 0.9, 0.99, 0.999)]
    print(f"[K] {label:20s} mean {float(x.mean()):7.4f}  "
          f"p50 {q[0]:7.4f}  p90 {q[1]:7.4f}  p99 {q[2]:7.4f}  p99.9 {q[3]:7.4f}  max {float(x.max()):7.4f} {unit}"
          f"   -> {float(x.mean()) / dt:6.2f} rad/s mean", flush=True)

print("[K] share of steps each candidate limit would clamp:", flush=True)
for rate in CANDIDATES:
    lim = rate * dt
    frac = float((D > lim).float().mean())
    print(f"[K]   {rate:5.1f} rad/s ({lim:.3f} rad/step)  clamps {100.0 * frac:5.1f}% of knee steps", flush=True)

print("[K] OK", flush=True)
sys.stdout.flush()
env.close()
simulation_app.close()
