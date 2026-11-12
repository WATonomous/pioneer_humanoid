"""Check moving tray collisions, goal metrics and real-arm corner transfers; no policy training."""
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
parser.add_argument("--report", type=Path, default=ROOT / "outputs/nasa_cargo/tray_range_check.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import numpy as np
from humanoid_robot_learning.sim.cargo_env import CargoEnv, FPS
from humanoid_robot_learning.sim.cargo_expert import ArmKinematics, CargoExpert
from humanoid_robot_learning.sim.cargo_randomization import sample_reset, tray_expert_yaw
from humanoid_isaac_scenes.nasa_cargo.scene import TARGET_XY, TABLE_TOP_Z


def main():
    env = CargoEnv(args.device, cameras=False, randomization="tray")
    kin = ArmKinematics()
    results = []
    for i, (x, y) in enumerate(((.25, .49), (.25, .53), (.45, .49), (.45, .53))):
        rng, params = sample_reset(72000+i, "tray")
        params["target_xy_m"] = [x, y]
        params["expert_place_yaw_rad"] = tray_expert_yaw([x, y])
        with patch("humanoid_robot_learning.sim.cargo_randomization.sample_reset", return_value=(rng, params)):
            env.reset(72000+i)
        settled = env.cargo.data.root_pos_w[0].cpu().numpy().copy()
        assert np.linalg.norm(settled[:2] - [.5, .14]) < .005, (x, y, settled)
        assert abs(env.status["bottom_height"][0].item() - TABLE_TOP_Z) < .01
        assert bool(env.status["upright"][0].item())
        for name in env.target_parts:
            part = env.scene[name]
            expected = part.data.default_root_state[0, :2].cpu().numpy() + np.array([x, y]) - TARGET_XY
            np.testing.assert_allclose(part.data.root_pos_w[0, :2].cpu().numpy(), expected, atol=1e-4)
        # Check that the goal follows the actual floor, rejecting a different goal.
        from humanoid_isaac_scenes.nasa_cargo.task import placement_metrics
        test_state = env.cargo.data.root_state_w.clone()
        test_state[:, :2] = test_state.new_tensor([x, y])
        assert placement_metrics(test_state, env.scene.env_origins, env.target_xy)["settled"].item()
        assert not placement_metrics(test_state, env.scene.env_origins, env.target_xy + [.6, 0])["inside_target"].item()
        q = env.robot.data.joint_pos[0, env.arm_ids].cpu().numpy()
        expert = CargoExpert(kin, env.robot.data.root_pos_w[0].cpu().numpy(), q,
                             settled, env.cargo.data.root_quat_w[0].cpu().numpy(), env.target_xy, rng, retreat_height=.10,
                             target_yaw=params["expert_place_yaw_rad"])
        failure = None
        previous = None
        try:
            for _ in range(int(expert.duration * FPS) + 1):
                action = expert.action(1/FPS, env.cargo.data.root_pos_w[0].cpu().numpy(), env.tip_pose()[0])
                env.command(action)
                env.advance()
                if previous != expert.phase:
                    print(f"TRAY_PHASE target={(x,y)} phase={expert.phase} cargo={env.cargo.data.root_pos_w[0].cpu().tolist()}", flush=True)
                    previous = expert.phase
        except RuntimeError as exc:
            failure = str(exc)
        row = {"target_xy": [x, y], "expert_place_yaw_rad": params["expert_place_yaw_rad"], "failure": failure, "settled_spawn_xyz": settled.tolist(),
               "lifted": bool(env.task.lifted[0].item()),
               "success": bool(env.status["success"][0].item()),
               "final_cargo_xyz": env.cargo.data.root_pos_w[0].cpu().tolist(),
               "max_ik_error_m": expert.max_error}
        results.append(row)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"profile": "tray", "device": args.device,
                                           "corners": results}, indent=2) + "\n")
        print(f"TRAY_CORNER {row}", flush=True)
    # This checks reach/physics, not a perfect scripted-expert success rate.
    assert all(r["lifted"] and r["failure"] is None for r in results), results
    for x in (.25, .45):
        assert any(r["success"] for r in results if r["target_xy"][0] == x), results
    print("CARGO_TRAY_REACH_PASS corners=4", flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
os._exit(0)
