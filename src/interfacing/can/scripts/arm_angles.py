#!/usr/bin/env python3
"""Live real-arm joint readout in degrees -- the real-arm counterpart of encoder_test.py.

Subscribes to /interfacing/motorFeedback and prints every joint's angle in the command frame
(= URDF frame, what ArmPose carries): cmd = zero_offset + motor / direction, from
arm_calibration.yaml. The gripper is shown as its position, 0 open .. 1 closed.

Read-only: it creates no publisher, so it cannot command a motor. Run it next to anything else
(joint_command, teleop, arm_roundtrip.py) to watch the arm.

Run (joint_command or interfacing container, can_node running)::

    python3 /opt/humanoid_scripts/arm_angles.py                # every joint + gripper
    python3 /opt/humanoid_scripts/arm_angles.py --ids 14,11    # just shoulder pitch, elbow pitch
    python3 /opt/humanoid_scripts/arm_angles.py --raw          # also motor-frame degrees

"---" = no feedback from that motor in the last 0.5 s. "!" = outside its calibrated limits.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from common_msgs.msg import MotorFeedback

import joint_config as jc

POLL_PERIOD = 0.02  # 50 Hz
STALE_AFTER_S = 0.5
SHORT = {"shoulder": "sh", "elbow": "el", "wrist": "wr", "gripper": "grip"}


def _label(name: str) -> str:
    group, _, joint = name.partition(".")
    return "grip" if group == "gripper" else f"{SHORT.get(group, group)}.{joint}"


def _select(joint_map: dict[int, dict], spec: str | None) -> dict[int, dict]:
    """The joints to show: ArmPose slot order, then anything else (the gripper)."""
    ordered = sorted(joint_map.items(), key=lambda kv: (kv[1]["slot"] is None, kv[1]["slot"] or 0, kv[0]))
    if not spec:
        return dict(ordered)
    try:
        wanted = [int(item.strip()) for item in spec.split(",") if item.strip()]
    except ValueError:
        raise SystemExit(f"--ids: expected comma-separated integers, got {spec!r}")
    unknown = [i for i in wanted if i not in joint_map]
    if unknown:
        raise SystemExit(f"--ids: {unknown} not in the calibration. Known IDs: {sorted(joint_map)}")
    return {i: joint_map[i] for i in wanted}


def _cell(info: dict, motor_deg: float | None, raw: bool) -> str:
    label = _label(info["name"])
    if motor_deg is None:
        return f"{label}:{'---':>8} " + (" " * 9 if raw else "")
    cmd = jc.motor_to_cmd_deg(info, motor_deg)
    if info["name"].startswith("gripper"):
        cell = f"{label}:{jc.gripper_position(info, cmd):+7.2f} "
    else:
        cell = f"{label}:{cmd:+7.1f}d"
    out = info["limit_range"] and not info["lower"] <= cmd <= info["upper"]
    cell += "!" if out else " "
    if raw:
        cell += f"[{motor_deg:+7.1f}]"
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration", default=None,
                        help="arm_calibration.yaml (default: auto, as joint_command)")
    parser.add_argument("--side", default="left", help="arm side in the calibration (default left)")
    parser.add_argument("--ids", default=None, help="comma-separated CAN ids to watch (default: all)")
    parser.add_argument("--raw", action="store_true", help="also show motor-frame degrees")
    args = parser.parse_args()

    calibration = jc.find_calibration(args.calibration)
    joints = _select(jc.load_joint_map(calibration, args.side), args.ids)

    lock = threading.Lock()
    latest: dict[int, tuple[float, float]] = {}  # motor_id -> (motor deg, monotonic time)

    def on_feedback(msg: MotorFeedback) -> None:
        motor_id = int(msg.motor_id)
        if motor_id in joints:
            with lock:
                latest[motor_id] = (float(msg.position), time.monotonic())

    rclpy.init()
    # Subscription only -- no publisher, so this cannot move the arm.
    node = Node("arm_angles")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
    node.create_subscription(MotorFeedback, "/interfacing/motorFeedback", on_feedback, qos)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print()
    print("=" * 62)
    print(" REAL ARM JOINT ANGLES (command frame, deg)")
    print("=" * 62)
    print(f"Calibration: {calibration}")
    print("Joints: " + ", ".join(f"{_label(info['name'])}=ID{mid}" for mid, info in joints.items()))
    print()
    print("Read-only: nothing is published. Press CTRL+C to stop.")
    print()

    try:
        while rclpy.ok():
            now = time.monotonic()
            with lock:
                fresh = {i: deg for i, (deg, t) in latest.items() if now - t <= STALE_AFTER_S}
            cells = [_cell(info, fresh.get(mid), args.raw) for mid, info in joints.items()]
            print("\r" + " ".join(cells), end="", flush=True)
            time.sleep(POLL_PERIOD)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("\nStopped. Nothing was published.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
