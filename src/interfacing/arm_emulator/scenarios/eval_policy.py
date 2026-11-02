"""A checkpoint in closed loop through the real deploy runner's adapters and joint_command: the rehearsal.

    python3 scenarios/eval_policy.py --policy <run>/checkpoints/last/pretrained_model --episodes 40 --seed 1000 \
        --gripper-force-scale 2 --inference-delay 0.05

Reads images over ROS, feedback from the motors, and publishes through RealArmPolicyRunner, with its checks on
every action; a take ends on success, a timeout, or the first action the runner would refuse. A success rate
here doesn't clear a policy for the real arm: the placeholders in motor_models.yaml still apply.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from arm_emulator import config  # noqa: F401 -- sys.path for the arm scripts and scenes
from harness import HOME_DEG, Client, Stack

_SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SRC / "robot_learning"))
SCHEMA = _SRC / "robot_learning" / "config" / "pioneer_emulator_dataset_schema.yaml"
HOME_SECONDS = 4.0


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def run_take(runner, client: Client, stack: Stack, policy_fn, reset, timeout_s: float, delay_s: float) -> tuple[bool, str]:
    """One take; -> (success, why it ended)."""
    from humanoid_robot_learning.deploy.real_arm import action_failures

    reset()
    start = time.monotonic()
    first = True
    while time.monotonic() - start < timeout_s:
        t0 = time.monotonic()
        if stack.joint_command_says("MIT FAULT"):
            return False, "joint_command fault"
        snapshot = runner.observation()
        if snapshot is None:
            return False, "lost data: " + ", ".join(runner.missing())
        obs, q_deg = snapshot
        action = np.asarray(policy_fn(obs), dtype=np.float64).reshape(-1)
        failures = action_failures(action, q_deg, runner.names, runner.start_tol_deg if first else runner.max_gap_deg)
        if failures:
            return False, "refused: " + "; ".join(failures)
        first = False
        if delay_s:
            time.sleep(delay_s)  # the robot computer's inference time, if this machine's is shorter
        runner.publish(action)
        if client.status.get("success"):
            return True, "lifted"
        time.sleep(max(0.0, runner.period - (time.monotonic() - t0)))
    return False, "timeout"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=1000, help="layouts the training demos didn't use")
    p.add_argument("--timeout", type=float, default=30.0, help="s per take")
    p.add_argument("--device", default="cpu")
    p.add_argument("--task", default=None)
    p.add_argument("--schema", default=str(SCHEMA))
    p.add_argument("--inference-delay", type=float, default=0.0, help="s added before each action")
    p.add_argument("--gripper-force-scale", type=float, default=None)
    args = p.parse_args()

    from arm_feedback import ArmFeedbackState
    from arm_emulator.config import REPO_CALIBRATION
    from humanoid_robot_learning.deploy.real_arm import RealArmPolicyRunner, camera_topics, make_policy_fn
    from humanoid_robot_learning.recording.schema import load_yaml
    from humanoid_robot_learning.utils.checkpoint import training_task
    from rclpy.node import Node

    schema = load_yaml(Path(args.schema))
    task = args.task or training_task(args.policy)
    if task is None:
        raise SystemExit("pass --task: the training dataset's task was not found")
    policy_fn, sizes, policy = make_policy_fn(args.policy, args.device, task)
    cameras = camera_topics(schema, sizes)
    extra = ["--gripper-force-scale", str(args.gripper_force_scale)] if args.gripper_force_scale is not None else []
    stack = Stack(cameras=",".join(cameras), seed=args.seed, extra_args=extra)
    client = Client(stack)
    node = Node("real_arm_policy_runner")
    client.add_node(node)
    runner = RealArmPolicyRunner(node, policy_fn, ArmFeedbackState(REPO_CALIBRATION), fps=float(schema["fps"]),
                                 cameras=cameras, live=True)
    wins = 0
    try:
        client.measured()
        failures = [f for f in runner.preflight() if "already has a publisher" not in f]  # the harness itself
        if failures:
            raise SystemExit("[PREFLIGHT] not starting:\n  " + "\n  ".join(failures))
        for episode in range(args.episodes):
            def reset(seed: int = args.seed + episode) -> None:
                client.control(f"reset_objects {seed}")
                client.ramp(client.measured(), HOME_DEG, HOME_SECONDS, grip_from=client.closure(), grip_to=0.0)
                client.hold(0.5, HOME_DEG)
                policy.reset()

            ok, why = run_take(runner, client, stack, policy_fn, reset, args.timeout, args.inference_delay)
            wins += ok
            print(f"take {episode + 1}/{args.episodes}: {why}  ({wins} lifted)", flush=True)
            if why == "joint_command fault":
                print("[STOP] joint_command halted: restart it, as on the real arm.", flush=True)
                break
        low, high = wilson_interval(wins, episode + 1)
        print(f"{wins}/{episode + 1} lifted = {wins / (episode + 1):.0%} (95% interval {low:.0%}-{high:.0%}). "
              f"Late ticks {client.status.get('late_ticks')} of {client.status.get('steps')}; "
              f"breaches {client.status.get('breaches')}", flush=True)
    finally:
        client.close()
        stack.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
