"""Starts the real stack against the emulator for a scenario, and talks to it as a client would.

Everything here goes through ArmPose on /arm/left/joint_targets, like teleop and deploy. Needs ROS: run
inside the emulator container, where the safety guard's environment is set.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HOME_DEG = [0.0, 0.0, 0.0, -90.0, 0.0, 0.0]
SETTLE_S = 0.6
# The binary itself: `ros2 run` would leave it running when only the wrapper gets the signal.
JOINT_COMMAND = ["/opt/watonomous/joint_command/lib/joint_command/joint_command_node", "--ros-args",
                 "-p", "control_type:=0", "-p", "arms:=[left]"]
# Each Stack gets its own domain: a stopped node lingers on the graph for a while, and the next
# emulator's safety guard would rightly refuse to share it.
_next_domain = itertools.count(int(os.environ.get("ROS_DOMAIN_ID", "42")) + 1)


class Stack:
    """The emulator and the real joint_command_node as subprocesses, plus a client node."""

    def __init__(self, faults: list[str] | None = None, cameras: str = "none", seed: int = 0, scene: str = "rubiks_cube",
                 joint_command: bool = True, extra_args: list[str] | None = None) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="arm_emulator_"))
        self.domain = next(_next_domain)
        self.processes: list[subprocess.Popen] = []
        command = [sys.executable, "-m", "arm_emulator.run_emulator", "--scene", scene, "--cameras", cameras, "--seed", str(seed), *(extra_args or [])]
        for fault in faults or []:
            command += ["--fault", fault]
        self.emulator_log = self.dir / "emulator.log"
        self._start(command, self.emulator_log)
        self.joint_command_log = self.dir / "joint_command.log"
        # As on the arm, where can_node comes up before joint_command.
        self._wait_for_emulator()
        if joint_command:
            self._start(JOINT_COMMAND, self.joint_command_log)

    def _wait_for_emulator(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while "running." not in self.emulator_log.read_text():
            if self.processes[0].poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("the emulator did not start:\n" + self.emulator_log.read_text()[-2000:])
            time.sleep(0.1)

    def _start(self, command: list[str], log: Path) -> None:
        with log.open("w") as out:
            self.processes.append(subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
                                                   env={**os.environ, "ROS_DOMAIN_ID": str(self.domain), "PYTHONUNBUFFERED": "1"}))

    def logs(self) -> str:
        return "".join(f"--- {p.name}\n{p.read_text()}" for p in (self.emulator_log, self.joint_command_log) if p.exists())

    def joint_command_says(self, text: str) -> bool:
        return self.joint_command_log.exists() and text in self.joint_command_log.read_text()

    def stop(self) -> None:
        for process in reversed(self.processes):
            process.send_signal(2)  # SIGINT: joint_command's shutdown path
        for process in reversed(self.processes):
            try:
                process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class Client:
    """A ROS client of the stack: sends poses, reads feedback through the same reader the policy deploy uses."""

    def __init__(self, stack: Stack, name: str = "scenario_client") -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        from arm_feedback import ArmFeedbackState
        from arm_emulator.config import REPO_CALIBRATION
        from common_msgs.msg import ArmPose, MotorCmd, MotorFeedback
        from humanoid_robot_learning_shim import vector_to_arm_pose
        from std_msgs.msg import String

        self._rclpy = rclpy
        rclpy.init(domain_id=stack.domain)
        self.node = Node(name)
        self.feedback = ArmFeedbackState(REPO_CALIBRATION)
        self._vector_to_pose, self._ArmPose = vector_to_arm_pose, ArmPose
        self.status: dict = {}
        self.commands: dict[int, object] = {}
        self.pose_pub = None  # created on the first send: a publisher here would fail the deploy runner's preflight
        self._settled = False
        self.control_pub = self.node.create_publisher(String, "/arm_emulator/control", 10)
        self._String = String
        self.node.create_subscription(MotorFeedback, "/interfacing/motorFeedback", self.feedback.on_feedback, 20)
        self.node.create_subscription(MotorCmd, "/interfacing/motorCMD", self._on_command, 100)
        self.node.create_subscription(String, "/arm_emulator/status", self._on_status, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _on_command(self, msg) -> None:
        self.commands[int(msg.motor_id)] = msg

    def _on_status(self, msg) -> None:
        self.status = json.loads(msg.data)

    def control(self, text: str) -> None:
        self.control_pub.publish(self._String(data=text))

    def measured(self, timeout: float = 15.0) -> np.ndarray:
        """Six measured joint angles in degrees, command frame; raises if any motor stays silent."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.feedback.snapshot()
            if snapshot is not None:
                if not self._settled:
                    # joint_command's first reply is from its constructor; its control timer starts a moment
                    # later. A pose inside that gap would trip the feedback watchdog, as no operator's does.
                    self._settled = True
                    time.sleep(SETTLE_S)
                    continue
                return snapshot[1]
            time.sleep(0.02)
        raise TimeoutError(f"no feedback from {self.feedback.missing()}")

    def closure(self) -> float:
        snapshot = self.feedback.snapshot()
        return float(snapshot[0][6]) if snapshot is not None else float("nan")

    def send(self, q_deg, grip: float = 0.0) -> None:
        if self.pose_pub is None:
            self.pose_pub = self.node.create_publisher(self._ArmPose, "/arm/left/joint_targets", 10)
        msg = self._vector_to_pose(self._ArmPose, np.append(q_deg, grip), unit="deg", name="scenario")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.pose_pub.publish(msg)

    def hold(self, seconds: float, q_deg, grip: float = 0.0, stop=None) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not (stop and stop()):
            self.send(q_deg, grip)
            time.sleep(0.02)

    def ramp(self, q_from, q_to, seconds: float, grip_from: float = 0.0, grip_to: float | None = None, stop=None) -> None:
        q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
        grip_to = grip_from if grip_to is None else grip_to
        start = time.monotonic()
        while (t := (time.monotonic() - start) / seconds) < 1.0 and not (stop and stop()):
            s = t * t * (3.0 - 2.0 * t)
            self.send(q_from + s * (q_to - q_from), grip_from + s * (grip_to - grip_from))
            time.sleep(0.02)

    def add_node(self, node) -> None:
        """Spin another node of this process (it joins the same context)."""
        self._executor.add_node(node)

    def close(self) -> None:
        self._executor.shutdown()
        self.node.destroy_node()
        self._rclpy.shutdown()
