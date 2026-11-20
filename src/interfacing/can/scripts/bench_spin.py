#!/usr/bin/env python3
"""Bench test: spin ONE loose AK motor at a slow, fixed speed, then stop it.

FOR A MOTOR CLAMPED ON THE BENCH WITH NOTHING ON ITS SHAFT. NOT FOR ARM JOINTS.
This sends servo-mode VELOCITY_LOOP straight to /interfacing/motorCMD, so it skips
joint_command's clamp, speed limit and watchdog. To move an arm joint, use joint_command.

Guards:
  - only ids in ALLOWED_IDS, and never an id listed in arm_calibration.yaml
  - refuses if anything else publishes on /interfacing/motorCMD (e.g. joint_command)
  - needs live feedback from the motor, error_code 0, before it starts
  - speed capped at MAX_SPEED_RAD_S, run time at MAX_DURATION_S
  - aborts on stale feedback, a motor fault, or a measured speed far above the target
  - always ends with speed 0 then DISABLE, including on Ctrl+C

The AK servo velocity loop has no CAN timeout (can/README.md): if this process is killed hard
or the PC hangs mid-run, the motor keeps turning. Keep the hardware E-stop in reach.

Run in the interfacing container, with can_node up:
  python3 /root/ament_ws/src/interfacing/can/scripts/bench_spin.py --id 67 --speed 0.1 --duration 5
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from common_msgs.msg import MotorCmd, MotorFeedback

ALLOWED_IDS = {67}          # bench motors only; add an id here on purpose, never an arm joint
MAX_SPEED_RAD_S = 0.2       # output shaft
MAX_DURATION_S = 30.0
SEND_HZ = 50.0
STALE_S = 0.5               # abort if no feedback for this long
OVERSPEED_FACTOR = 3.0      # abort if measured speed > this x target (e.g. wrong gear/pole pairs)
CMD_TOPIC = "/interfacing/motorCMD"
FB_TOPIC = "/interfacing/motorFeedback"

CALIBRATION_PATHS = [
    Path("/calibration/arm_calibration.yaml"),
    Path(__file__).resolve().parents[2] / "joint_command" / "config" / "arm_calibration.yaml",
]


def arm_ids() -> set:
    """Every can_id in arm_calibration.yaml (both arms), so an arm joint can never be spun."""
    import yaml
    for path in CALIBRATION_PATHS:
        if path.exists():
            ids = set()

            def walk(node):
                if isinstance(node, dict):
                    if "can_id" in node:
                        ids.add(int(node["can_id"]))
                    for v in node.values():
                        walk(v)

            walk(yaml.safe_load(path.read_text()))
            return ids
    raise FileNotFoundError("arm_calibration.yaml not found; refusing to run without it")


def rad_s_to_erpm(rad_s: float, gear_ratio: float, pole_pairs: int) -> float:
    output_rpm = rad_s * 60.0 / (2.0 * math.pi)
    return output_rpm * gear_ratio * pole_pairs


class BenchSpin(Node):
    def __init__(self, motor_id: int) -> None:
        super().__init__("bench_spin")
        self.motor_id = motor_id
        self._lock = threading.Lock()
        self._fb: MotorFeedback | None = None
        self._fb_time = 0.0
        self._pub = self.create_publisher(MotorCmd, CMD_TOPIC, 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=50)
        self.create_subscription(MotorFeedback, FB_TOPIC, self._on_fb, qos)

    def _on_fb(self, msg: MotorFeedback) -> None:
        if int(msg.motor_id) == self.motor_id:
            with self._lock:
                self._fb, self._fb_time = msg, time.monotonic()

    def feedback(self):
        with self._lock:
            return self._fb, time.monotonic() - self._fb_time

    def other_publishers(self) -> list:
        return sorted({i.node_name for i in self.get_publishers_info_by_topic(CMD_TOPIC)
                       if i.node_name != self.get_name()})

    def send(self, control_type: int, erpm: float = 0.0) -> None:
        msg = MotorCmd()
        msg.motor_id = self.motor_id
        msg.control_type = control_type
        msg.velocity = float(erpm)
        self._pub.publish(msg)


def stop(node: BenchSpin) -> None:
    """Speed 0 for 1 s, then DISABLE a few times (one lost frame must not leave it on)."""
    end = time.monotonic() + 1.0
    while time.monotonic() < end:
        node.send(MotorCmd.VELOCITY_LOOP, 0.0)
        time.sleep(1.0 / SEND_HZ)
    for _ in range(3):
        node.send(MotorCmd.DISABLE)
        time.sleep(0.05)
    print("Stopped: speed 0, then DISABLE sent.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--id", type=int, required=True, help=f"motor id, one of {sorted(ALLOWED_IDS)}")
    p.add_argument("--speed", type=float, default=0.1, help="output shaft rad/s (sign = direction)")
    p.add_argument("--duration", type=float, default=5.0, help="seconds")
    p.add_argument("--gear-ratio", type=float, default=9.0, help="AK10-9 / AK80-9: 9")
    p.add_argument("--pole-pairs", type=int, default=21, help="check the motor's datasheet")
    args = p.parse_args()

    if args.id not in ALLOWED_IDS:
        sys.exit(f"Refusing: id {args.id} is not a bench motor (allowed: {sorted(ALLOWED_IDS)}).")
    if args.id in arm_ids():
        sys.exit(f"Refusing: id {args.id} is an arm joint in arm_calibration.yaml.")
    if not 0 < abs(args.speed) <= MAX_SPEED_RAD_S:
        sys.exit(f"Refusing: |speed| must be in (0, {MAX_SPEED_RAD_S}] rad/s.")
    if not 0 < args.duration <= MAX_DURATION_S:
        sys.exit(f"Refusing: duration must be in (0, {MAX_DURATION_S}] s.")

    erpm = rad_s_to_erpm(args.speed, args.gear_ratio, args.pole_pairs)
    target_deg_s = math.degrees(abs(args.speed))

    rclpy.init()
    node = BenchSpin(args.id)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    started = False
    try:
        time.sleep(1.0)  # let DDS discover publishers and feedback
        others = node.other_publishers()
        if others:
            print(f"Refusing: other nodes publish on {CMD_TOPIC}: {others}. Stop them first.")
            return 1

        deadline = time.monotonic() + 3.0
        fb, age = node.feedback()
        while fb is None and time.monotonic() < deadline:
            time.sleep(0.05)
            fb, age = node.feedback()
        if fb is None:
            print(f"Refusing: no feedback from motor {args.id} in 3 s (is can_node up, motor powered?).")
            return 1
        if int(fb.error_code) != 0:
            print(f"Refusing: motor {args.id} reports error_code {fb.error_code}.")
            return 1

        print(f"Motor {args.id}: at {fb.position:.1f} deg, {fb.temperature} C, no fault.")
        print(f"Plan: {args.speed:+.3f} rad/s ({target_deg_s:.1f} deg/s at the output) "
              f"= {erpm:+.0f} ERPM, for {args.duration:.1f} s, then stop.")
        print("Motor clamped, nothing on the shaft, hand on the E-stop?")
        try:
            answer = input("Type 'spin' to start: ").strip()
        except EOFError:
            answer = ""
        if answer != "spin":
            print("Not started.")
            return 0

        started = True
        start_pos, start_t = fb.position, time.monotonic()
        last_print = start_t
        reason = "done"
        while time.monotonic() - start_t < args.duration:
            node.send(MotorCmd.VELOCITY_LOOP, erpm)
            time.sleep(1.0 / SEND_HZ)
            fb, age = node.feedback()
            if age > STALE_S:
                reason = f"no feedback for {age:.2f} s"
                break
            if int(fb.error_code) != 0:
                reason = f"motor fault, error_code {fb.error_code}"
                break
            elapsed = time.monotonic() - start_t
            if elapsed > 1.0:
                measured = abs(fb.position - start_pos) / elapsed
                if measured > OVERSPEED_FACTOR * target_deg_s:
                    reason = (f"too fast: {measured:.1f} deg/s vs {target_deg_s:.1f} target "
                              f"(check --gear-ratio / --pole-pairs)")
                    break
            if time.monotonic() - last_print >= 1.0:
                last_print = time.monotonic()
                print(f"  t={elapsed:4.1f}s  pos={fb.position:8.1f} deg  "
                      f"avg={abs(fb.position - start_pos) / max(elapsed, 1e-3):5.1f} deg/s  "
                      f"{fb.temperature} C")
        print(f"Stopping ({reason}).")
        fb, _ = node.feedback()
        if fb is not None:
            el = time.monotonic() - start_t
            print(f"Moved {fb.position - start_pos:+.1f} deg in {el:.1f} s "
                  f"({abs(fb.position - start_pos) / max(el, 1e-3):.1f} deg/s, target {target_deg_s:.1f}).")
    except KeyboardInterrupt:
        print("\nCtrl+C.")
    finally:
        if started:
            stop(node)
        # Shut down first: destroying a node while rclpy.spin runs aborts the process.
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
    return 0


if __name__ == "__main__":
    sys.exit(main())
