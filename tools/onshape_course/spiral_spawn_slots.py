"""Measure spawn slots (tread poses) along the Onshape spiral staircase.

The extracted geometry (``assets/staircase_onshape/spiral_geometry.json``) has
exact tread heights (a 0.12 m riser per angular step) but its tread *angles*
carry ~0.8 rad of offset and a few centimetres of per-tread noise, so spawning
from the analytic helicoid puts the robot across a riser. This tool measures
the poses off the merged collider instead, the same way the straight course's
spawn slots were measured off its centreline:

  1. spawn on the marker (the known-good pose) and read the base-level height
     scan, which sees the tread under the robot and the first riser ahead;
  2. the first riser's world position gives the helicoid's angular phase;
  3. for each requested tread, place the robot at the tread centre on the
     walking circle, let it settle, and read the tread top back from the scan.

The output is a Python literal ready to paste into
``onshape_spiral_foothold_env_cfg.SPIRAL_SPAWN_SLOTS``.

    isaaclab.sh -p tools/onshape_course/spiral_spawn_slots.py --treads 4 9 16 24 34
"""

import argparse
import json
import math
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 2))
sys.path.insert(0, os.path.join(_REPO, "src/simulation/humanoid_rl/humanoid_rl/scripts"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
parser.add_argument("--treads", type=int, nargs="+", default=[4, 9, 16, 24, 34])
parser.add_argument("--settle_steps", type=int, default=60)
args_cli = parser.parse_args()

app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_rl_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

GEOMETRY = os.path.join(_REPO, "assets", "staircase_onshape", "spiral_geometry.json")
STANDING = 0.845


def settle(env, steps: int) -> tuple[float, float]:
    """Run the scene at rest and return (root z, median tread hit z)."""
    env.reset()
    zero = torch.zeros(1, env.action_space.shape[-1], device=env.device)
    for _ in range(steps):
        env.step(zero)
    robot = env.scene["robot"]
    hits = env.scene["height_scanner"].data.ray_hits_w[0]
    z = hits[:, 2]
    finite = z[torch.isfinite(z)]
    return float(robot.data.root_pos_w[0, 2]), float(finite.median()) if finite.numel() else float("nan")


def main() -> None:
    with open(GEOMETRY) as handle:
        geometry = json.load(handle)
    centre = tuple(float(v) for v in geometry["centre"])
    radius = float(geometry["walking_radius"])
    step = float(geometry["angular_step"])
    riser = float(geometry["riser"])
    marker = tuple(float(v) for v in geometry["marker"])

    # One env for every measurement: the reset event reads its slot list from
    # the term's params at each reset, so swapping the slot needs no rebuild
    # (building a second env in one process is not supported here).
    marker_slot = (marker[0] + 0.25, marker[1], STANDING, math.pi / 2)
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    cfg.scene.env_spacing = 0.0
    cfg.observations.policy.enable_corruption = False
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    term = env.event_manager.get_term_cfg("reset_base")
    term.params["slots"] = (marker_slot,)
    term.params["xy_jitter"] = 0.0
    term.params["yaw_jitter"] = 0.0

    root_z, tread = settle(env, args_cli.settle_steps)
    print(f"[SLOTS] marker root z {root_z:.3f} tread {tread:.3f}")
    if abs(root_z - STANDING) > 0.03:
        print(f"[SLOTS] WARNING marker spawn does not stand at {STANDING} m (got {root_z:.3f})")

    # Angular phase of the helicoid from the first riser ahead of the marker:
    # a radial tread edge is a line of constant angle, so the world angle of
    # the raised cells is the first tread's edge.
    hits = env.scene["height_scanner"].data.ray_hits_w[0]
    raised = hits[hits[:, 2] > tread + 0.5 * riser]
    if raised.numel() == 0:
        raise SystemExit("[SLOTS] no riser found ahead of the marker")
    angles = torch.atan2(raised[:, 1] - centre[1], raised[:, 0] - centre[0])
    theta_first = float(angles.min())
    print(f"[SLOTS] first tread edge at {theta_first:+.4f} rad "
          f"(marker at {math.atan2(marker_slot[1] - centre[1], marker_slot[0] - centre[0]):+.4f})")

    slots = [marker_slot]
    for tread_index in args_cli.treads:
        theta = theta_first + (tread_index + 0.5) * step
        x = centre[0] + radius * math.cos(theta)
        y = centre[1] + radius * math.sin(theta)
        yaw = theta + math.pi / 2.0  # the path's tangent at this angle
        z_guess = STANDING + riser * (tread_index + 1)
        term.params["slots"] = ((x, y, z_guess, yaw),)
        root_z, tread_z = settle(env, args_cli.settle_steps)
        expected = riser * (tread_index + 1)
        print(f"[SLOTS] tread {tread_index}: theta {theta:+.4f} guess z {z_guess:.3f} "
              f"-> tread {tread_z:.3f} (expected {expected:.3f}), root {root_z:.3f}")
        slots.append((round(x, 3), round(y, 3), round(tread_z + STANDING, 3), round(yaw, 4)))

    print("[SLOTS] paste into onshape_spiral_foothold_env_cfg.SPIRAL_SPAWN_SLOTS:")
    print("SPIRAL_SPAWN_SLOTS = (")
    for slot in slots:
        print(f"    ({slot[0]:.3f}, {slot[1]:.3f}, {slot[2]:.3f}, {slot[3]:.4f}),")
    print(")")
    sys.stdout.flush()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
