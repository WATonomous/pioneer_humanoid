"""Measure the straight course's terrain height along its centreline.

Teleports the robot along +Y at x = 4.163 and reads the height scanner, so the
numbers come from the same collision mesh the robot walks on. Prints the flat
ground-level stretches (safe spawn slots) and writes the full profile to
assets/staircase_onshape/straight_stairs_centreline.npz.

    isaaclab.sh -p tools/onshape_course/measure_centreline.py
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-Locomotion-StairsStraightFoothold-PioneerHumanoid-Play-v0"
X = 4.163
Y0, Y1, DY = -3.0, 20.0, 0.05

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.log_dir = "/workspace/humanoid/models/pioneer_humanoid_stairs_foothold"

env = gym.make(TASK, cfg=cfg).unwrapped
env.reset()

robot = env.scene["robot"]
sensor = env.scene["height_scanner"]
ids = torch.arange(env.num_envs, device=env.device)
default = robot.data.default_root_state.clone()

ys = np.arange(Y0, Y1 + 1e-9, DY)
heights = []
for y in ys:
    pose = default[:, :7].clone()
    pose[:, 0] = X
    pose[:, 1] = float(y)
    pose[:, 2] = 2.5
    pose[:, 3:7] = math_utils.quat_from_euler_xyz(
        torch.zeros(1, device=env.device), torch.zeros(1, device=env.device),
        torch.full((1,), 0.5 * np.pi, device=env.device))
    robot.write_root_pose_to_sim(pose, ids)
    robot.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device), ids)
    env.sim.step()
    sensor.update(dt=0.02, force_recompute=True)
    hits = sensor.data.ray_hits_w[0]
    z = torch.nan_to_num(hits[:, 2], nan=-99.0, posinf=-99.0, neginf=-99.0)
    # rays near the teleport point only
    xy = sensor.data.pos_w[0, :2]
    d = (hits[:, :2] - torch.tensor([X, float(y)], device=env.device)).norm(dim=-1)
    near = d < 0.10
    heights.append(float(z[near].median()) if near.any() else float("nan"))

heights = np.array(heights)
os.makedirs("/workspace/humanoid/assets/staircase_onshape", exist_ok=True)
np.savez("/workspace/humanoid/assets/staircase_onshape/straight_stairs_centreline.npz",
         y=ys, z=heights)
print(f"[COURSE] sampled {len(ys)} points, z range {np.nanmin(heights):.3f} .. {np.nanmax(heights):.3f}")

# flat runs at ground level (z < 2 cm) long enough to stand on (>= 0.6 m)
flat = np.abs(heights) < 0.02
runs, start = [], None
for i, f in enumerate(flat):
    if f and start is None:
        start = i
    if (not f or i == len(flat) - 1) and start is not None:
        end = i if not f else i + 1
        if ys[end - 1] - ys[start] >= 0.6:
            runs.append((ys[start], ys[end - 1]))
        start = None
print(f"[COURSE] ground-level runs >= 0.6 m: {[(round(a,2), round(b,2)) for a, b in runs]}")
print("[COURSE] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
