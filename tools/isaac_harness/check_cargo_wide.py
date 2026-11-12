"""Check supported wide-profile spawns and real-arm corner lifts; no policy training."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for package in ("src/robot_learning", "src/pioneer_humanoid", "src/simulation/isaac_scenes"):
    sys.path.insert(0, str(ROOT / package))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, default=ROOT / "outputs/nasa_cargo/wide_range_check.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import numpy as np
from humanoid_robot_learning.sim.cargo_env import CargoEnv, FPS
from humanoid_robot_learning.sim.cargo_expert import ArmKinematics, CargoExpert
from humanoid_robot_learning.sim.cargo_randomization import sample_reset
from humanoid_isaac_scenes.nasa_cargo.scene import TARGET_XY, TABLE_TOP_Z


def main():
    env = CargoEnv(args.device, cameras=False, randomization="wide")
    kin = ArmKinematics()
    results = []
    for i, (x, y) in enumerate(((.475, .115), (.475, .15), (.525, .115), (.525, .15))):
        rng, params = sample_reset(71000+i, "wide")
        params["pickup_offset_xy_m"] = [x-.5, y-.14]
        params["yaw_rad"] = -.04 if i % 2 == 0 else .04
        with patch("humanoid_robot_learning.sim.cargo_randomization.sample_reset", return_value=(rng, params)):
            env.reset(71000+i)
        settled = env.cargo.data.root_pos_w[0].cpu().numpy().copy()
        assert np.linalg.norm(settled[:2] - [x, y]) < .005, (x, y, settled)
        assert abs(env.status["bottom_height"][0].item() - TABLE_TOP_Z) < .01
        assert bool(env.status["upright"][0].item())
        q = env.robot.data.joint_pos[0, env.arm_ids].cpu().numpy()
        expert = CargoExpert(kin, env.robot.data.root_pos_w[0].cpu().numpy(), q,
                             settled, env.cargo.data.root_quat_w[0].cpu().numpy(), TARGET_XY, rng)
        for _ in range(int(expert.duration * FPS) + 1):
            action = expert.action(1/FPS, env.cargo.data.root_pos_w[0].cpu().numpy(), env.tip_pose()[0])
            env.command(action)
            env.advance()
            if expert.phase == "transfer":
                break
        row = {"requested_xy": [x, y], "settled_spawn_xyz": settled.tolist(),
               "lifted": bool(env.task.lifted[0].item()),
               "cargo_rise_m": float(env.cargo.data.root_pos_w[0, 2].item()-settled[2]),
               "max_ik_error_m": expert.max_error}
        results.append(row)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"profile": "wide", "device": args.device,
                                           "corners": results}, indent=2) + "\n")
        print(f"WIDE_CORNER {row}", flush=True)
        assert row["lifted"] and row["cargo_rise_m"] > .08, row
    print("CARGO_WIDE_REACH_PASS corners=4", flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
os._exit(0)
