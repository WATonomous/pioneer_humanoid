#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer
import numpy as np
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Bool


class ShuttlecockLocalizer(Node):
    def __init__(self):
        super().__init__('shuttlecock_localizer_node')
        self.bridge = CvBridge()
        self.intrinsics = None

        self.declare_parameter('point_topic', '/perception/shuttlecock/detections_2d')
        self.declare_parameter('visible_topic', '/perception/shuttlecock/visible')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('output_topic', '/perception/shuttlecock/detections_3d')
        self.declare_parameter('sync_slop', 0.015)
        self.declare_parameter('sync_queue_size', 60)

        self.create_subscription(
            CameraInfo,
            self._param('camera_info_topic'),
            self._store_intrinsics,
            qos_profile_sensor_data,
        )

        self.pub = self.create_publisher(
            PointStamped,
            self._param('output_topic'),
            10
        )

        det_sub = message_filters.Subscriber(
            self,
            PointStamped,
            self._param('point_topic')
        )
        visible_sub = message_filters.Subscriber(
            self,
            Bool,
            self._param('visible_topic')
        )
        depth_sub = message_filters.Subscriber(
            self,
            Image,
            self._param('depth_topic'),
            qos_profile=qos_profile_sensor_data
        )

        self.ts = ApproximateTimeSynchronizer(
            [det_sub, visible_sub, depth_sub],
            queue_size=self._param('sync_queue_size'),
            slop=self._param('sync_slop'),
            allow_headerless=True,
        )
        self.ts.registerCallback(self._synchronized_callback)

    def _param(self, name):
        return self.get_parameter(name).value

    def _store_intrinsics(self, msg: CameraInfo):
        k = msg.k
        self.intrinsics = (k[0], k[4], k[2], k[5])
        self.get_logger().info(
            f"Got intrinsics: fx={k[0]:.1f} fy={k[4]:.1f} "
            f"cx={k[2]:.1f} cy={k[5]:.1f}"
        )

    def _synchronized_callback(
            self,
            det_msg: PointStamped,
            visible_msg: Bool,
            depth_msg: Image):
        if self.intrinsics is None:
            self.get_logger().warn(
                "Waiting for camera intrinsics...",
                throttle_duration_sec=2.0)
            return

        if not visible_msg.data:
            return

        u = det_msg.point.x
        v = det_msg.point.y

        try:
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {str(e)}")
            return

        z_cam = self._sample_depth(depth_img, u, v, depth_msg.encoding)

        if z_cam is None:
            self.get_logger().warn(
                "Birdie detected, but depth patch is invalid "
                "(too many 0s/NaNs).",
                throttle_duration_sec=1.0,
            )
            return

        fx, fy, cx, cy = self.intrinsics

        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy

        point_msg = PointStamped()
        point_msg.header.stamp = det_msg.header.stamp
        point_msg.header.frame_id = depth_msg.header.frame_id
        point_msg.point = Point(x=x_cam, y=y_cam, z=z_cam)
        self.pub.publish(point_msg)

        self.get_logger().info(
            f"Successfully synced Detection (Time: "
            f"{det_msg.header.stamp.sec}) "
            f"with Depth Image (Time: "
            f"{depth_msg.header.stamp.sec}) "
            f"Birdie Position: X={x_cam:.3f}m, "
            f"Y={y_cam:.3f}m, Z={z_cam:.3f}m"
        )

    def _sample_depth(self, depth_img, u: float, v: float, encoding: str):
        r = 2
        h, w = depth_img.shape[:2]
        ui, vi = int(round(u)), int(round(v))

        u_min = max(0, ui - r)
        u_max = min(w, ui + r + 1)
        v_min = max(0, vi - r)
        v_max = min(h, vi + r + 1)

        patch = depth_img[v_min:v_max, u_min:u_max]

        if patch.size == 0:
            return None

        valid_mask = (patch > 0) & (~np.isnan(patch))
        valid_values = patch[valid_mask]

        if len(valid_values) < 0.5 * patch.size:
            return None

        median = float(np.median(valid_values))

        return median if encoding == '32FC1' else median / 1000.0


def main(args=None):
    rclpy.init(args=args)
    node = ShuttlecockLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()