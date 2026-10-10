#!/usr/bin/env python3

"""Record raw odometry and its latest available map-frame correction."""

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_geometry_msgs import do_transform_pose


class OdomTrajectoryRecorder(Node):
    def __init__(self) -> None:
        super().__init__("odom_trajectory_recorder")
        self.declare_parameter("odom_topic", "/rtabmap/odom")
        self.declare_parameter("output_path", "frontend_poses.txt")
        self.declare_parameter("corrected_output_path", "live_corrected_poses.txt")
        self.declare_parameter("map_frame", "map")

        odom_topic = self.get_parameter("odom_topic").value
        output_path = self.get_parameter("output_path").value
        self._output = open(output_path, "w", encoding="utf-8", buffering=1)
        self._output.write("# timestamp tx ty tz qx qy qz qw\n")
        self._corrected = open(
            self.get_parameter("corrected_output_path").value,
            "w", encoding="utf-8", buffering=1,
        )
        self._corrected.write("# timestamp tx ty tz qx qy qz qw\n")
        self._map_frame = self.get_parameter("map_frame").value
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._missing_correction = 0
        self.create_subscription(
            Odometry, odom_topic, self._record_pose, qos_profile_sensor_data
        )

    def _record_pose(self, message: Odometry) -> None:
        self._write_pose(self._output, message.header.stamp, message.pose.pose)
        try:
            # Apply the correction available NOW to this exact odometry pose.
            # Do not wait for later backend updates or retrospectively rewrite
            # earlier samples. Looking up map->camera at Time() instead would
            # risk pairing a different camera pose with this message timestamp.
            correction = self._tf_buffer.lookup_transform(
                self._map_frame, message.header.frame_id, Time()
            )
        except TransformException:
            self._missing_correction += 1
            if self._missing_correction == 1:
                self.get_logger().warning(
                    "No map-to-odom correction yet; skipping corrected samples "
                    "until TF becomes available. Raw odometry is still recorded."
                )
            return
        pose = do_transform_pose(message.pose.pose, correction)
        self._write_pose(self._corrected, message.header.stamp, pose)

    @staticmethod
    def _write_pose(output, stamp, pose) -> None:
        position = pose.position
        orientation = pose.orientation
        output.write(
            f"{stamp.sec}.{stamp.nanosec:09d} "
            f"{position.x:.9f} {position.y:.9f} {position.z:.9f} "
            f"{orientation.x:.9f} {orientation.y:.9f} "
            f"{orientation.z:.9f} {orientation.w:.9f}\n"
        )

    def destroy_node(self) -> None:
        self.get_logger().info(
            f"Skipped {self._missing_correction} corrected samples without TF."
        )
        self._output.close()
        self._corrected.close()
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
