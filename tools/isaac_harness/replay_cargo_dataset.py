"""Independently replay recorded commands in physics; no scripted expert or attachment."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
for package in ("src/robot_learning", "src/pioneer_humanoid", "src/simulation/isaac_scenes"):
    sys.path.insert(0, str(ROOT / package))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/nasa_cargo/synthetic_50")
parser.add_argument("--report", type=Path, default=ROOT / "outputs/nasa_cargo/audit/physics_replay.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import numpy as np
import pyarrow.parquet as pq
from humanoid_robot_learning.sim.cargo_env import CargoEnv


def main():
    manifest = [json.loads(line) for line in (args.dataset_root / "collection_manifest.jsonl").read_text().splitlines()]
    table = pq.read_table(sorted((args.dataset_root / "data").glob("chunk-*/file-*.parquet")))
    ids = np.array(table["episode_index"])
    actions = np.stack(table["action"].to_pylist())
    original_states = np.stack(table["observation.state"].to_pylist())
    env = CargoEnv(args.device, cameras=False)
    assert env.contract() == json.loads((args.dataset_root / "cargo_contract.json").read_text())
    results = []
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for row in manifest:
        env.reset(row["seed"])
        errors = []
        grip = []
        cargo = []
        success = False
        selected_actions = actions[ids == row["episode"]]
        for action, original in zip(selected_actions, original_states[ids == row["episode"]]):
            measured = env.observe()["observation.state"]
            errors.append(float(np.abs(measured[:6] - original[:6]).max()))
            grip.append(float(measured[6]))
            env.command(action)
            success = env.advance()
            cargo.append(env.cargo.data.root_pos_w[0].cpu().numpy().copy())
        final = env.observe()["observation.state"]
        cargo = np.array(cargo)
        tip = env.tip_pose()[0]
        result = {
            "episode": row["episode"], "seed": row["seed"],
            "success": bool(success), "ever_lifted": bool(env.task.lifted[0].item()),
            "inside_target": bool(env.status["inside_target"][0].item()),
            "settled_s": float(env.task.settled_s[0].item()),
            "final_measured_gripper_closure": float(final[6]),
            "released": bool(final[6] < .1),
            "max_recorded_arm_state_deviation_rad": max(errors),
            "max_cargo_rise_m": float(cargo[:, 2].max() - cargo[0, 2]),
            "max_cargo_step_m": float(np.linalg.norm(np.diff(cargo, axis=0), axis=1).max()),
            "final_cargo_position": cargo[-1].tolist(),
            "final_tip_height_above_cargo_origin_m": float(tip[2] - cargo[-1, 2]),
        }
        results.append(result)
        args.report.write_text(json.dumps({"episodes": results}, indent=2) + "\n")
        print(f"CARGO_REPLAY {json.dumps(result)}", flush=True)
    passed = all(r["success"] and r["released"] for r in results)
    print(f"CARGO_REPLAY_COMPLETE episodes={len(results)} successful={sum(r['success'] and r['released'] for r in results)}", flush=True)
    return 0 if passed else 2


try:
    code = main()
except Exception:
    traceback.print_exc()
    code = 1
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
