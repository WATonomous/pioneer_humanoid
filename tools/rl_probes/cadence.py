"""Step cadence and sway on a course, with the arms swinging or frozen.

Answers "is the shuffle the scripted arm swing / the 1 s gait clock, or the
terrain": counts foot-contact onsets per second and per clock cycle, and the
torso sway frequency.
"""

import math
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

SPIRAL = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
STRAIGHT = "Isaac-Locomotion-StairsStraightFoothold-PioneerHumanoid-Play-v0"
CKPT_OVERRIDE = os.environ.get("CKPT_OVERRIDE")
CKPTS = {
    "spiral": "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_399.pt",
    "straight": "/workspace/humanoid/models/pioneer_humanoid_stairs_foothold/model_599.pt",
}
SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005), (-1.504, -0.182, 2.017, 4.8331),
    (1.361, -0.666, 2.857, 7.3987), (-1.192, 0.935, 3.817, 10.3309), (1.500, -0.213, 5.017, 13.9960),
)

COURSE = sys.argv[1]          # spiral | straight
FREEZE = len(sys.argv) > 2 and sys.argv[2] == "freeze"
N, STEPS = 6, 1500

TASK = SPIRAL if COURSE == "spiral" else STRAIGHT
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
if FREEZE:
    cfg.actions.arm_swing.amplitude = 0.0
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
if COURSE == "spiral":
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
runner.load(CKPT_OVERRIDE or CKPTS[COURSE], load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)
robot = u.scene["robot"]
sensor = u.scene.sensors["contact_forces"]
feet = [sensor.body_names.index(n) for n in ("l_foot", "r_foot")]
shoulder = robot.joint_names.index("l_shoulder_pitch")

onsets = torch.zeros(N, device=u.device)
contact_steps = torch.zeros(N, 2, device=u.device)
prev = torch.zeros(N, 2, dtype=torch.bool, device=u.device)
roll, pitch, shoulder_pos = [], [], []
clock_cycles = 0.0
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS):
        obs, *_ = env.step(policy(obs))
        done = u.reset_terminated | u.reset_time_outs
        in_contact = sensor.data.net_forces_w[:, feet, 2] > 1.0
        rising = in_contact & ~prev
        onsets += rising.sum(dim=1).float()
        contact_steps += in_contact.float()
        prev = torch.where(done[:, None], torch.zeros_like(prev), in_contact)
        if step > 100:
            g = robot.data.projected_gravity_b
            roll.append(math.degrees(math.asin(min(1.0, float(g[:, 1].abs().mean())))))
            pitch.append(math.degrees(math.asin(min(1.0, float(g[:, 0].abs().mean())))))
            shoulder_pos.append(float(robot.data.joint_pos[:, shoulder].std()))
            clock_cycles += 1.0 / 1.0 / 50.0  # one cycle per second, dt = 0.02

secs = (STEPS - 100) / 50.0
onset_rate = float((onsets / secs).mean())
duty = float((contact_steps / STEPS).mean())
print(f"[CAD] {COURSE}{' arms-frozen' if FREEZE else ''}: "
      f"contacts/s {onset_rate:.2f} (per leg {(onsets / secs).mean(dim=0).tolist()}) | "
      f"duty {duty:.2f} | clock cycles {clock_cycles:.1f} | shoulder std {float(torch.tensor(shoulder_pos).mean()):.3f} rad")
print(f"[CAD] torso tilt mean roll {float(torch.tensor(roll).mean()):.1f} deg pitch {float(torch.tensor(pitch).mean()):.1f} deg")
print("[CAD] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
