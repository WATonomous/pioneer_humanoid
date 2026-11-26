"""Where does the leg jitter actually live: the command, or the joint?

The knee's target sits on one of its two clip rails on 71.5% of steps, which
has been read as "a two-state relay". That reading is not yet justified: a knee
pinned straight for a whole stance and tucked for a whole swing is *also* on a
rail 70% of the time, and that is a gait, not a buzz. What separates them is
the *duration* of each rail visit -- 2 transitions per stride (fine) versus
rail-to-rail flips several times a second (the buzz) -- and whether the joint
itself reverses at the same rate.

So this measures, per leg joint, on both the spiral checkpoint and the
straight deliverable:

* the step-to-step change of the commanded target (rad/step) and how many times
  per second its direction reverses,
* the fraction of steps on the extreme of the target's own range (the rails),
  and the run length of each rail visit in steps (a stride is ~50-100 steps),
* the same reversal and high-pass numbers on the joint position/velocity, so a
  noisy command can be told apart from a noisy joint.
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
N, STEPS = 4, 900
WARMUP = 60

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
dt = u.step_dt

tgt_hist = []
q_hist = []
v_hist = []
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS + WARMUP):
        act = policy(obs)
        obs, *_ = env.step(act)
        if step >= WARMUP:
            tgt_hist.append(at.processed_actions.clone())
            q_hist.append(robot.data.joint_pos[:, jid].clone())
            v_hist.append(robot.data.joint_vel[:, jid].clone())

tgt = torch.stack(tgt_hist)   # (T, N, J)
q = torch.stack(q_hist)
v = torch.stack(v_hist)


def rate_of_reversals(x):
    """Direction changes per second of a per-step series."""
    d = x[1:] - x[:-1]
    s = torch.sign(d)
    flip = (s[1:] * s[:-1]) < 0
    return float(flip.float().mean()) * (1.0 / dt)


print(f"[T] task {TASK.split('-')[1] if '-' in TASK else TASK}  ckpt {os.path.basename(CKPT)}  "
      f"{STEPS} steps @ {1.0/dt:.0f} Hz")
print("[T] joint      tgt_dRMS  tgt_rev/s  rail_frac  rail_run  q_rev/s  q_hpRMS  v_hpRMS")
for k, name in enumerate(names):
    t = tgt[:, :, k]
    d = (t[1:] - t[:-1]).abs()
    lo = t.min()
    hi = t.max()
    span = float(hi - lo)
    rail = ((t - lo).abs() < 1e-3) | ((t - hi).abs() < 1e-3)
    # run length of consecutive rail steps (a stride is 50-100 control steps)
    flat = rail.reshape(-1)
    runs, cur = [], 0
    for flag in flat.tolist():
        if flag:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    run_mean = sum(runs) / len(runs) if runs else 0.0
    qh = q[:, :, k] - q[:, :, k].mean()
    hp_q = float((qh[2:] - 2 * qh[1:-1] + qh[:-2]).abs().mean())
    vh = v[:, :, k]
    hp_v = float((vh[2:] - 2 * vh[1:-1] + vh[:-2]).abs().mean())
    print(f"[T] {name:10s} {float(d.mean()):8.3f}  {rate_of_reversals(t):9.2f}  "
          f"{float(rail.float().mean()):9.2f}  {run_mean:8.1f}  {rate_of_reversals(q[:, :, k]):7.2f}  "
          f"{hp_q:7.3f}  {hp_v:7.3f}   span {span:.2f}")

print(f"[T] knee target sample, left knee, env 0, last 60 steps:")
knee = names.index("Knee_L")
print("[T] " + " ".join(f"{float(x):+.2f}" for x in tgt[-60:, 0, knee]))
print("[T] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
