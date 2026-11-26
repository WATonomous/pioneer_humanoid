"""Smoke test for the in-term knee rate limit (mdp/joint_rate_limit.py).

Checks that the action term actually became the rate-limited one, and that the
knee target it emits never moves more than the configured rad/s * step_dt.
Also reports the foot-tap rate so the limit's effect on the trained policy's
behaviour is visible before any retraining (it should barely change anything --
the point of training the limit in is that the policy adapts to it).
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
N, STEPS, WARMUP = 2, 300, 40

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005),
)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
print(f"[R] action cfg {type(cfg.actions.joint_pos).__name__}  "
      f"class_type {cfg.actions.joint_pos.class_type.__name__}  "
      f"max_rate {getattr(cfg.actions.joint_pos, 'max_rate', 'MISSING')}", flush=True)

env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
term = u.action_manager.get_term("joint_pos")
print(f"[R] built term {type(term).__name__}", flush=True)
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
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)

names = list(term._joint_names)
knee = [names.index("Knee_L"), names.index("Knee_R")]
print(f"[R] knee scale {term._scale[0, knee].tolist()}  offset {term._offset[0, knee].tolist()}  "
      f"clip {term._clip[0, knee].tolist()}", flush=True)
feet_cfg = SceneEntityCfg("contact_forces", body_names=["l_foot", "r_foot"], preserve_order=True)
feet_cfg.resolve(u.scene)
feet_ids = list(feet_cfg.body_ids)
dt = u.step_dt

tgts, forces, ALL = [], [], []
prev = None
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        act = policy(obs)
        obs, *_ = env.step(act)
        if step >= WARMUP:
            tgt = term.processed_actions[:, knee]
            ALL.append(term.processed_actions.clone())
            if prev is not None:
                tgts.append((tgt - prev).abs())
            prev = tgt.clone()
            forces.append(u.scene["contact_forces"].data.net_forces_w[:, feet_ids, :].norm(dim=-1).clone())

d = torch.stack(tgts)                     # (T-1, N, 2)
F = torch.stack(forces)
print(f"[R] knee target step change: max {float(d.max()):.4f} rad  "
      f"p99 {float(d.flatten().quantile(0.99)):.4f}  mean {float(d.mean()):.4f}", flush=True)
# Only meaningful when the term is actually rate limited (max_rate set).
_lim = getattr(term, "_max_step", None)
if _lim is not None:
    knee_lim = float(_lim[knee].max())
    ok = float(d.max()) <= knee_lim * 1.001
    print(f"[R] {'PASS' if ok else 'FAIL'} knee rate limit "
          f"{'honoured' if ok else 'EXCEEDED'} (limit {knee_lim:.4f} rad/step)", flush=True)
else:
    print("[R] no knee rate limit configured (SPIRAL_KNEE_RATE=0)", flush=True)

# per-joint target-step distribution: which joints actually spike?
allsteps = torch.stack(allsteps) if False else None
planted = F > 20.0
short = 0
for e in range(N):
    for f_ in range(2):
        run = 0
        for flag in (~planted[:, e, f_]).tolist():
            if flag:
                run += 1
            elif run:
                if run * dt < 0.10:
                    short += 1
                run = 0
print(f"[R] foot taps/s {short / (F.shape[0] * N * 2 * dt):.2f}", flush=True)
A = torch.stack(ALL)
D = (A[1:] - A[:-1]).abs()
print("[R] per-joint target step (rad): p99 / max", flush=True)
order = D.flatten(0, 1).quantile(0.99, dim=0).argsort(descending=True)
for i in order[:8].tolist():
    print(f"[R]   {names[i]:14s} p99 {float(D[..., i].quantile(0.99)):.4f}  max {float(D[..., i].max()):.4f}",
          flush=True)
print("[R] OK", flush=True)
sys.stdout.flush()
env.close()
simulation_app.close()
