"""The emulator's ROS face, standing where can_node stands: MotorCmd in, MotorFeedback out.

    /interfacing/motorCMD        in   one reply per command, as a drive answers
    /interfacing/motorFeedback   out
    /arm_emulator/control        in   std_msgs/String: "reset_objects [seed]" or "fault kind:motor_id[:value]"
    /arm_emulator/status         out  std_msgs/String, JSON, 10 Hz: time, success, timing, loads
    /arm_emulator/qpos           out  std_msgs/Float64MultiArray: the whole pose, for camera.py and the scripted operator

Physics runs in another process (run_emulator.py) and everything here crosses a pipe. Sharing one Python
interpreter, the 1 kHz loop left too little time for joint_command's 350 commands a second, so commands
were lost and its feedback watchdog halted the arm.
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from multiprocessing.connection import Connection

from rclpy.node import Node
from rclpy.qos import QoSProfile

from arm_emulator import faults, safety_guard
from common_msgs.msg import MotorCmd, MotorFeedback
from std_msgs.msg import Float64MultiArray, String

GRAPH_CHECK_PERIOD_S = 2.0


class BridgeNode(Node):
    def __init__(self, physics: Connection) -> None:
        super().__init__("arm_emulator")
        self.physics = physics
        self.received: Counter[int] = Counter()  # commands per motor
        self.replied: Counter[int] = Counter()   # feedback frames per motor
        self.refusal: str | None = None
        self.ready = threading.Event()
        self.physics_done = threading.Event()
        # Same depth as can_node: joint_command sends one burst per tick, three times over on a fault.
        self.create_subscription(MotorCmd, "/interfacing/motorCMD", self._on_command, QoSProfile(depth=100))
        # Deeper than can_node's 10: replies come in bursts of 7 and should never overwrite each other here.
        self.feedback_pub = self.create_publisher(MotorFeedback, "/interfacing/motorFeedback", 100)
        self.create_subscription(String, "/arm_emulator/control", self._on_control, 10)
        self.status_pub = self.create_publisher(String, "/arm_emulator/status", 10)
        self.qpos_pub = self.create_publisher(Float64MultiArray, "/arm_emulator/qpos", 10)
        threading.Thread(target=self._read_physics, daemon=True, name="emulator-physics-reader").start()
        # Not a ROS timer: a graph query can stall during discovery, and must not delay a command's reply.
        threading.Thread(target=self._watch_graph, daemon=True, name="emulator-graph-guard").start()

    def _watch_graph(self) -> None:
        while True:
            time.sleep(GRAPH_CHECK_PERIOD_S)
            self.check_graph()

    def check_graph(self) -> None:
        failures = safety_guard.graph_failures(
            self.get_node_names(),
            [i.node_name for i in self.get_publishers_info_by_topic("/interfacing/motorFeedback")],
            self.get_name())
        if failures and self.refusal is None:
            self.refusal = "; ".join(failures)
            self.get_logger().error(f"REFUSING TO RUN: {self.refusal}")

    def _on_command(self, msg: MotorCmd) -> None:
        if self.refusal is not None:
            return
        self.received[int(msg.motor_id)] += 1
        self.physics.send(("command", int(msg.motor_id), int(msg.control_type), float(msg.position), float(msg.velocity),
                           float(msg.kp), float(msg.kd), float(msg.torque)))

    def _on_control(self, msg: String) -> None:
        parts = msg.data.split()
        try:
            if parts[:1] == ["reset_objects"]:
                self.physics.send(("reset_objects", int(parts[1]) if len(parts) > 1 else None))
            elif parts[:1] == ["fault"] and len(parts) == 2:
                self.physics.send(("fault", faults.parse(parts[1])))
                self.get_logger().warn(f"injected fault {parts[1]}")
            else:
                self.get_logger().error(f"unknown control {msg.data!r}")
        except ValueError as e:
            self.get_logger().error(f"bad control {msg.data!r}: {e}")

    def _read_physics(self) -> None:
        """Publish what the physics process sends, as it arrives."""
        try:
            while True:
                kind, *rest = self.physics.recv()
                if kind == "feedback":
                    self._publish_feedback(*rest)
                elif kind == "qpos":
                    self.qpos_pub.publish(Float64MultiArray(data=rest[0]))
                elif kind == "status":
                    self._publish_status(rest[0])
                elif kind == "ready":
                    self.ready.set()
        except (EOFError, OSError):
            self.physics_done.set()

    def _publish_feedback(self, reply) -> None:
        out = MotorFeedback()
        out.motor_id = reply.motor_id
        out.position = float(reply.position)
        out.velocity = float(reply.velocity)
        out.current = float(reply.current)
        out.torque = float(reply.torque)
        out.temperature = reply.temperature
        out.error_code = reply.error_code
        self.feedback_pub.publish(out)
        self.replied[reply.motor_id] += 1

    def _publish_status(self, status: dict) -> None:
        status.update(refusal=self.refusal, commands_received=dict(self.received), replies_sent=dict(self.replied))
        self.status_pub.publish(String(data=json.dumps(status)))
