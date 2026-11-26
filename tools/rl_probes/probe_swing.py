"""Do the terrain-gated swing rewards actually fire while walking?

Roll the policy out and record, per control step: the two terrain-contact
sensors' air/contact clocks, their filtered force, and the raw value of every
terrain-gated reward term. If ``max air time`` is 0 the sensor never sees a
foot leave the staircase and the swing-clearance / knee-tuck / air-time rewards
are dead weight in the cfg.
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

TASK = os.environ.get("TASK_OVERRIDE",
                      "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ.get("CKPT_OVERRIDE",
                      "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_299_gait.pt")
N, STEPS = 3, 800

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005), (-1.504, -0.182, 2.017, 4.8331),
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

terms = ("foot_swing_clearance", "knee_swing_flexion", "feet_air_time",
         "foot_backswing_cost", "foot_flat", "feet_impact", "dof_vel",
         "contact_no_vel", "action_rate_l2", "dof_acc_l2", "foot_low_fly_in")
sensors = [u.scene.sensors["left_foot_terrain_contact"], u.scene.sensors["right_foot_terrain_contact"]]

stats = {t: [] for t in terms}
max_air = 0.0
max_contact = 0.0
detached = 0
steps = 0
low_force_steps = 0
peak_force = 0.0
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS):
        action = policy(obs)
        obs, *_ = env.step(action)
        force = torch.cat([s.data.net_forces_w.norm(dim=-1) for s in sensors], dim=1)
        air = torch.cat([s.data.current_air_time for s in sensors], dim=1)
        contact = torch.cat([s.data.current_contact_time for s in sensors], dim=1)
        peak_force = max(peak_force, float(force.max()))
        if step > 60:
            steps += 1
            max_air = max(max_air, float(air.max()))
            max_contact = max(max_contact, float(contact.max()))
            detached += int((air > 0).sum())
            low_force_steps += int((force < 1.0).any(dim=1).sum())
            for t in terms:
                tc = u.reward_manager.get_term_cfg(t)
                try:
                    stats[t].append(float(tc.func(u, **tc.params).mean()))
                except Exception as exc:  # noqa: BLE001
                    stats[t].append(float("nan"))
                    if step == 61:
                        print(f"[W] {t} failed: {type(exc).__name__}: {exc}")

print(f"[W] steps {steps}  peak filtered force {peak_force:.0f} N")
print(f"[W] max air time {max_air:.3f} s   max contact time {max_contact:.2f} s")
print(f"[W] foot-steps with air time > 0: {detached} / {steps * 2}")
print(f"[W] env-steps with a foot below 1 N: {low_force_steps} / {steps}")
for t in terms:
    v = torch.tensor([x for x in stats[t] if x == x])
    if v.numel():
        print(f"[W] {t:22s} mean {float(v.mean()):+.5f}  max {float(v.max()):+.5f}  "
              f"nonzero frac {float((v != 0).float().mean()):.2f}")
print("[W] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
