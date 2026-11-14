#!/usr/bin/env python3
"""Record joint angle / velocity / torque telemetry from the live ROS pipeline.

Writes a run folder (telemetry.csv + run.json) for tools/gl40_telemetry_plot.py::

    /interfacing/motorCMD      MotorCmd, MOTOR frame       -> sp_deg (what was really sent)
    /arm/joint_targets         ArmPose, command frame, deg -> sp_raw_deg (what was asked for)
    /interfacing/motorFeedback MotorFeedback, MOTOR frame  -> pos_deg (converted below)

Both setpoints are recorded: their difference is the clamp and velocity limiter at work.
Feedback is converted to the command frame (cmd = zero_offset + motor / direction). Rows are
sampled on a fixed grid (--rate) so every motor shares one timebase.

Examples (inside the interfacing or joint_command container, ROS sourced)::

  python3 telemetry_record.py --duration 20 --label wrist-mit-40deg
  ros2 run can telemetry_record.py --label ros-clamp-test
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from common_msgs.msg import ArmPose, MotorCmd, MotorFeedback

from joint_config import find_calibration, load_joint_map, load_actuator_limits, max_torque_by_joint
from telemetry import RunFolder

# GL II status nibble (MIT feedback). Servo feedback uses the DBC's own error codes.
MIT_STATUS = {
    0: "Disable", 1: "Enable", 8: "Over-voltage", 9: "Under-voltage", 10: "Over-current",
    11: "MOS over-temp", 12: "Winding over-temp", 13: "Comms loss", 14: "Overload",
}


class TelemetryRecorder(Node):
    def __init__(self, joint_map: Dict[int, dict], log: RunFolder, motors: Optional[List[int]],
                 rate_hz: float):
        super().__init__("telemetry_recorder")
        self.joint_map = joint_map
        self.log = log
        self.motors = motors
        self.rate_hz = rate_hz
        self.setpoints: Dict[int, float] = {}      # moderated, from MotorCmd
        self.mit_cmd: Dict[int, tuple] = {}        # (feed-forward N.m, kp) of the last MIT frame
        self.requested: Dict[int, float] = {}      # raw, from ArmPose
        self.feedback: Dict[int, MotorFeedback] = {}
        self.seen: set = set()
        self.rows = 0
        self.first_pose_time: Optional[float] = None
        # Phase label for every row (set by arm_roundtrip.py).
        self.phase = "stream"

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=50)
        self.create_subscription(ArmPose, "/arm/joint_targets", self._on_pose, qos)
        self.create_subscription(MotorCmd, "/interfacing/motorCMD", self._on_cmd, qos)
        self.create_subscription(MotorFeedback, "/interfacing/motorFeedback", self._on_feedback,
                                 qos)
        self.create_timer(1.0 / rate_hz, self._sample)

    def _on_pose(self, msg: ArmPose) -> None:
        if self.first_pose_time is None:
            self.first_pose_time = time.monotonic()
        slots = list(msg.shoulder.position[:3]) + list(msg.elbow.position[:2]) + \
            list(msg.wrist.position[:1])
        for motor_id, info in self.joint_map.items():
            slot = info["slot"]
            if slot is not None and slot < len(slots):
                self.requested[motor_id] = float(slots[slot])

    def _on_cmd(self, msg: MotorCmd) -> None:
        # Only these carry a position setpoint (the rest leave it at 0).
        if msg.control_type not in (MotorCmd.MIT_CONTROL, MotorCmd.POSITION_LOOP,
                                    MotorCmd.POSITION_VELOCITY):
            return
        motor_id = int(msg.motor_id)
        info = self.joint_map.get(motor_id, {})
        direction = info.get("direction", 1.0) or 1.0
        zero = info.get("zero_offset", 0.0)
        # MIT_CONTROL carries radians; every servo mode carries degrees. Both are motor frame.
        deg = math.degrees(msg.position) if msg.control_type == 0 else float(msg.position)
        if msg.control_type == 0 and msg.kp == 0.0:
            # No stiffness: a poke or fault/seed frame, not a setpoint.
            self.mit_cmd.pop(motor_id, None)
            return
        self.setpoints[motor_id] = zero + deg / direction
        if msg.control_type == 0:
            self.mit_cmd[motor_id] = (float(msg.torque), float(msg.kp))
        else:
            self.mit_cmd.pop(motor_id, None)

    def _on_feedback(self, msg: MotorFeedback) -> None:
        self.feedback[int(msg.motor_id)] = msg
        self.seen.add(int(msg.motor_id))

    def _sample(self) -> None:
        for motor_id, fb in sorted(self.feedback.items()):
            if self.motors is not None and motor_id not in self.motors:
                continue
            info = self.joint_map.get(motor_id, {})
            direction = info.get("direction", 1.0) or 1.0
            zero = info.get("zero_offset", 0.0)
            # Motor frame -> command frame, the inverse of what joint_command applies.
            pos_cmd_deg = zero + float(fb.position) / direction
            torque = float(getattr(fb, "torque", 0.0))
            current = float(fb.current)
            status = MIT_STATUS.get(int(fb.error_code), str(int(fb.error_code)))
            self.log.row(
                motor_id=motor_id,
                joint=info.get("name", f"motor{motor_id}"),
                phase=self.phase,
                sp_deg=self.setpoints.get(motor_id),
                sp_raw_deg=self.requested.get(motor_id),
                pos_deg=pos_cmd_deg,
                # Units differ (ERPM / drive units): recorded raw; the plotter uses d(position)/dt.
                vel_dps=None,
                tau_nm=torque if torque else None,
                current_a=current if current else None,
                drive_c=int(fb.temperature),
                status=status,
                ff_nm=self.mit_cmd.get(motor_id, (None, None))[0],
                kp=self.mit_cmd.get(motor_id, (None, None))[1],
            )
            self.rows += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--label", default="ros", help="name for the run folder")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to record; 0 = until Ctrl-C (default)")
    ap.add_argument("--rate", type=float, default=50.0, help="sampling rate, Hz (default 50)")
    ap.add_argument("--calibration", "--mapping", dest="mapping", help="path to arm_calibration.yaml")
    ap.add_argument("--actuators", "--safety-limits", dest="actuators", help="path to arm_actuators.yaml")
    ap.add_argument("--arm-side", default="left")
    ap.add_argument("--motors", help="comma-separated motor ids to record (default: all seen)")
    ap.add_argument("--no-log", action="store_true", help="print only, write nothing")
    args = ap.parse_args(argv)

    mapping_path = find_calibration(args.mapping)
    joint_map = load_joint_map(mapping_path, args.arm_side)
    velocity_max, mit_limits = load_actuator_limits(args.actuators)
    motors = [int(x) for x in args.motors.split(",")] if args.motors else None

    log = RunFolder(
        label=args.label,
        source="ros",
        enabled=not args.no_log,
        meta={
            "tool": "telemetry_record.py",
            "arm_side": args.arm_side,
            "calibration": str(mapping_path),
            "rate_hz": args.rate,
            "frame": "command frame (degrees) for both sp_deg and pos_deg",
            "joints": {str(mid): info["name"] for mid, info in sorted(joint_map.items())},
            "limits": {
                "per_joint_deg": {
                    info["name"]: [info["lower"], info["upper"]]
                    for info in joint_map.values() if info["limit_range"]
                },
                "velocity_max_dps_per_joint": velocity_max,
                # One ceiling for whole-run plots: the largest configured.
                "velocity_max_dps": max(velocity_max.values()) if velocity_max else None,
                "mit": mit_limits,
                "max_torque_nm": min((v["max_torque_nm"] for v in mit_limits.values()
                                      if v.get("max_torque_nm") is not None), default=None),
                "max_torque_nm_by_joint": max_torque_by_joint(args.actuators),
                "max_track_err_deg": min((v["max_track_err_deg"] for v in mit_limits.values()
                                          if v.get("max_track_err_deg") is not None),
                                         default=None),
            },
        },
    )
    print(f"mapping: {mapping_path}")
    print(log.describe())

    rclpy.init()
    node = TelemetryRecorder(joint_map, log, motors, args.rate)
    stop_at = time.monotonic() + args.duration if args.duration > 0 else None
    stopping = {"now": False}

    def on_signal(signum, _frame):
        stopping["now"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        while rclpy.ok() and not stopping["now"]:
            rclpy.spin_once(node, timeout_sec=0.05)
            if stop_at is not None and time.monotonic() >= stop_at:
                break
    finally:
        seen = sorted(node.seen)
        log.note(motors_seen=seen, rows_written=node.rows)
        run_dir = log.close("completed")
        node.destroy_node()
        rclpy.shutdown()
        print(f"\nrecorded {node.rows} rows from motors {seen}")
        if not seen:
            print("WARNING: no feedback received at all -- is can_node running?")
        if run_dir is not None:
            print(f"Telemetry: {run_dir}")
            print(f"  plot with: uv run --with matplotlib --with numpy "
                  f"tools/gl40_telemetry_plot.py {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
