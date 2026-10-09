#!/usr/bin/env python3

"""Relay legacy RGB-D bag topics with ROS 2-compatible frame IDs."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class RGBDFrameNormalizer(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_frame_normalizer")
        self._frame_id = "openni_rgb_optical_frame"

        self._rgb_pub = self.create_publisher(
            Image, "/eval/camera/rgb/image", qos_profile_sensor_data
        )
        self._depth_pub = self.create_publisher(
            Image, "/eval/camera/depth/image", qos_profile_sensor_data
        )
        self._info_pub = self.create_publisher(
            CameraInfo, "/eval/camera/rgb/camera_info", qos_profile_sensor_data
        )

        self.create_subscription(
            Image,
            "/camera/rgb/image_color",
            self._relay_rgb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/camera/depth/image",
            self._relay_depth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/rgb/camera_info",
            self._relay_info,
            qos_profile_sensor_data,
        )

    def _relay_rgb(self, message: Image) -> None:
        message.header.frame_id = self._frame_id
        self._rgb_pub.publish(message)

    def _relay_depth(self, message: Image) -> None:
        # TUM's depth image is registered to the RGB image. Giving both the
        # same optical frame avoids invalid ROS 1 frame names during playback.
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
