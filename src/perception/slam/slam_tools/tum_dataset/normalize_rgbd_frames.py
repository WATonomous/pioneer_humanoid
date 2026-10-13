#!/usr/bin/env python3

"""Relay legacy RGB-D bag topics with ROS 2-compatible frame IDs."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


class RGBDFrameNormalizer(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_frame_normalizer")
        self.declare_parameter("input_rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("input_depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("input_camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("frame_id", "camera_color_optical_frame")

        rgb_topic = self.get_parameter("input_rgb_topic").value
        depth_topic = self.get_parameter("input_depth_topic").value
        camera_info_topic = self.get_parameter("input_camera_info_topic").value
        self._frame_id = self.get_parameter("frame_id").value
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._rgb_pub = self.create_publisher(
            Image, "/eval/rgb/image", output_qos
        )
        self._depth_pub = self.create_publisher(
            Image, "/eval/depth/image", output_qos
        )
        self._info_pub = self.create_publisher(
            CameraInfo, "/eval/rgb/camera_info", output_qos
        )

        self.create_subscription(
            Image,
            rgb_topic,
            self._relay_rgb,
            input_qos,
        )
        self.create_subscription(
            Image,
            depth_topic,
            self._relay_depth,
            input_qos,
        )
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._relay_info,
            input_qos,
        )

    def _relay_rgb(self, message: Image) -> None:
        message.header.frame_id = self._frame_id
        self._rgb_pub.publish(message)

    def _relay_depth(self, message: Image) -> None:
        # Aligned depth and RGB must share the configured optical frame.
        message.header.frame_id = self._frame_id
        self._depth_pub.publish(message)

    def _relay_info(self, message: CameraInfo) -> None:
        message.header.frame_id = self._frame_id
        self._info_pub.publish(message)


def main() -> None:
    rclpy.init()
    node = RGBDFrameNormalizer()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
