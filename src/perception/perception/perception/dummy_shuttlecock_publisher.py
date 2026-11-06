#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool


class DummyShuttlecockPublisher(Node):
    def __init__(self):
        super().__init__("synthetic_shuttlecock_publisher")

        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('point_topic', '/perception/shuttlecock/detections_2d')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('visible_topic', '/perception/shuttlecock/visible')

        self.fx = self.fy = 500.0
        self.cx, self.cy = 320.0, 240.0
        self.u, self.v = 370.0, 290.0
        # 2.0m, will be read back as uint16 mm by the real node
        self.depth_mm = 2000
        self.bridge = CvBridge()
        self.info_pub = self.create_publisher(
            CameraInfo, self._param('camera_info_topic'), 10
        )
        self.det_pub = self.create_publisher(
            PointStamped, self._param('point_topic'), 10
        )
        self.depth_pub = self.create_publisher(
            Image, self._param('depth_topic'), 10)

        self.timer = self.create_timer(
            0.5, self._publish_all
        )  # 2Hz is plenty for a test

        self.visible_pub = self.create_publisher(
            Bool, self._param('visible_topic'), 10
        )

    def _param(self, name):
        return self.get_parameter(name).value

    def _publish_all(self):
        stamp = self.get_clock().now().to_msg()

        visible_msg = Bool()
        visible_msg.data = True
        self.visible_pub.publish(visible_msg)

        # Build and Publish CameraInfo
        info_msg = CameraInfo()
        info_msg.header.stamp = stamp
        info_msg.header.frame_id = "camera_link"
        info_msg.width = 640
        info_msg.height = 480
        info_msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]
        self.info_pub.publish(info_msg)

        # Build and publish Detection2D using stamp + self.u/self.v
        det_msg = PointStamped()
        det_msg.header.stamp = stamp
        det_msg.header.frame_id = "camera_link"
        det_msg.point.x = self.u
        det_msg.point.y = self.v
        det_msg.point.z = 0.0
        self.det_pub.publish(det_msg)

        # Build and publish Image (depth) using stamp + self.depth_mm
        depth_array = np.full((480, 640), self.depth_mm, dtype=np.uint16)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_array, encoding="16UC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "camera_link"
        self.depth_pub.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DummyShuttlecockPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
