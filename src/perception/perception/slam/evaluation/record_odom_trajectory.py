#!/usr/bin/env python3

"""Write a ROS 2 Odometry topic to a trajectory in TUM format."""

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class OdomTrajectoryRecorder(Node):
    def __init__(self) -> None:
        super().__init__("odom_trajectory_recorder")
        self.declare_parameter("odom_topic", "/rtabmap/odom")
        self.declare_parameter("output_path", "frontend_poses.txt")

        odom_topic = self.get_parameter("odom_topic").value
        output_path = self.get_parameter("output_path").value
        self._output = open(output_path, "w", encoding="utf-8", buffering=1)
        self._output.write("# timestamp tx ty tz qx qy qz qw\n")
        self.create_subscription(
            Odometry, odom_topic, self._record_pose, qos_profile_sensor_data
        )

    def _record_pose(self, message: Odometry) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self._output.write(
            f"{stamp:.9f} "
            f"{position.x:.9f} {position.y:.9f} {position.z:.9f} "
            f"{orientation.x:.9f} {orientation.y:.9f} "
            f"{orientation.z:.9f} {orientation.w:.9f}\n"
        )

    def destroy_node(self) -> None:
        self._output.close()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OdomTrajectoryRecorder()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
