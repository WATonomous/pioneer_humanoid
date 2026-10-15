#!/usr/bin/env python3

"""Record RTAB-Map odometry after the currently available map correction."""

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener


class LiveTrajectoryRecorder(Node):
    def __init__(self) -> None:
        super().__init__("live_trajectory_recorder")
        self.declare_parameter("output_path", "trajectory.txt")
        self.declare_parameter("odom_topic", "/rtabmap/odom")
        self.declare_parameter("map_frame", "map")
        self._output = open(self.get_parameter("output_path").value, "w", buffering=1)
        self._output.write("# timestamp tx ty tz qx qy qz qw\n")
        self._map_frame = self.get_parameter("map_frame").value
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self._record,
            qos_profile_sensor_data,
        )

    def _record(self, message: Odometry) -> None:
        pose = message.pose.pose
        # RTAB-Map publishes an identity pose while visual odometry is lost.
        # Do not write that placeholder into the evaluated trajectory.
        if (
            abs(pose.position.x) < 1e-12
            and abs(pose.position.y) < 1e-12
            and abs(pose.position.z) < 1e-12
            and abs(pose.orientation.x) < 1e-12
            and abs(pose.orientation.y) < 1e-12
            and abs(pose.orientation.z) < 1e-12
            and abs(pose.orientation.w - 1.0) < 1e-12
        ):
            return
        try:
            transform = self._tf.lookup_transform(
                self._map_frame, message.header.frame_id, Time()
            )
        except TransformException:
            return
        pose = do_transform_pose(pose, transform)
        p, q = pose.position, pose.orientation
        stamp = message.header.stamp
        self._output.write(
            f"{stamp.sec}.{stamp.nanosec:09d} {p.x:.9f} {p.y:.9f} {p.z:.9f} "
            f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n"
        )

    def destroy_node(self) -> None:
        self._output.close()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = LiveTrajectoryRecorder()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
