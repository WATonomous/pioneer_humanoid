#!/usr/bin/env python3
"""Log how old messages are when they arrive: arrival time minus the header stamp.

Meant for the live robot: on the elevation map (the default topic) this is the sensor -> map
latency. Works on any message with a header. The message type is detected from the topic
(or set msg_type).

Usage:
    ros2 run terrain_mapping_bringup latency_logger.py
    ros2 run terrain_mapping_bringup latency_logger.py --ros-args -p topic:=/camera/camera/color/image_raw
"""
import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message


class LatencyLogger(Node):
    def __init__(self):
        super().__init__("latency_logger")
        self.declare_parameter("topic", "/elevation_mapping_node/elevation_map_raw")
        self.declare_parameter("msg_type", "")
        self.declare_parameter("report_period_s", 1.0)
        self.declare_parameter("csv_path", "")

        self._topic = self.get_parameter("topic").value
        msg_type = self.get_parameter("msg_type").value
        period = self.get_parameter("report_period_s").value
        csv_path = self.get_parameter("csv_path").value

        self._samples_ms = []
        self._csv_file = None
        self._csv_writer = None
        if csv_path:
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["arrival_time_ns", "latency_ms"])

        if msg_type:
            self._subscribe(msg_type)
        else:
            self.get_logger().info(f"Waiting for {self._topic} to appear to detect its type")
            self._detect_timer = self.create_timer(0.5, self._detect_type)
        self.create_timer(period, self._report)

    def _detect_type(self):
        """Look up the topic's message type once it shows up on the network."""
        for name, types in self.get_topic_names_and_types():
            if name == self._topic:
                self._detect_timer.cancel()
                self._subscribe(types[0])
                return

    def _subscribe(self, msg_type):
        self.create_subscription(
            get_message(msg_type), self._topic, self._on_msg, qos_profile_sensor_data
        )
        self.get_logger().info(f"Measuring latency on {self._topic} ({msg_type})")

    def _on_msg(self, msg):
        now_ns = self.get_clock().now().nanoseconds
        stamp = msg.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        latency_ms = (now_ns - stamp_ns) / 1e6
        self._samples_ms.append(latency_ms)
        if self._csv_writer:
            self._csv_writer.writerow([now_ns, f"{latency_ms:.3f}"])

    def _report(self):
        if not self._samples_ms:
            self.get_logger().warn("No messages received in this period")
            return
        s = sorted(self._samples_ms)
        n = len(s)
        mean = sum(s) / n
        p95 = s[min(n - 1, int(0.95 * n))]
        self.get_logger().info(
            f"n={n} mean={mean:.1f} ms p95={p95:.1f} ms max={s[-1]:.1f} ms"
        )
        self._samples_ms.clear()

    def close(self):
        if self._csv_file:
            self._csv_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = LatencyLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
