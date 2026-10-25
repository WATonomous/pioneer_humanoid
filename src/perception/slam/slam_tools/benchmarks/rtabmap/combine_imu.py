#!/usr/bin/env python3
"""Combine separate D455 gyro and accelerometer messages for Madgwick."""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class ImuCombiner(Node):
    def __init__(self):
        super().__init__("imu_combiner")
        self.declare_parameter("gyro_topic", "/gyro")
        self.declare_parameter("accel_topic", "/accel")
        self.declare_parameter("output_topic", "/rtabmap/imu_raw")
        self.accel = None
        self.publisher = self.create_publisher(
            Imu, self.get_parameter("output_topic").value, qos_profile_sensor_data
        )
        self.create_subscription(
            Imu,
            self.get_parameter("accel_topic").value,
            self.on_accel,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            self.get_parameter("gyro_topic").value,
            self.on_gyro,
            qos_profile_sensor_data,
        )

    def on_accel(self, message):
        self.accel = message

    def on_gyro(self, message):
        if self.accel is None:
            return
        combined = Imu()
        combined.header = message.header
        combined.orientation_covariance[0] = -1.0
        combined.angular_velocity = message.angular_velocity
        combined.angular_velocity_covariance = message.angular_velocity_covariance
        combined.linear_acceleration = self.accel.linear_acceleration
        combined.linear_acceleration_covariance = self.accel.linear_acceleration_covariance
        self.publisher.publish(combined)


def main():
    rclpy.init()
    node = ImuCombiner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
