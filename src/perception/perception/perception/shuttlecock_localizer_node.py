#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# standard for tracking fast objects such as a birdie
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
        # holding (fx, fy, cx, cy) until CameraInfo arrives
        self.intrinsics = None

        self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self._store_intrinsics,
            qos_profile_sensor_data,
        )

        self.pub = self.create_publisher(
            PointStamped,
            '/perception/shuttlecock/detections_3d',
            10
        )
        visible_sub = message_filters.Subscriber(
            self,
            Bool,
            "/tracknetv3/shuttle_visible"
        )

        det_sub = message_filters.Subscriber(
            self,
            PointStamped,
            '/tracknetv3/shuttle_point'
        )
        depth_sub = message_filters.Subscriber(
            self,
            Image,
            '/camera/depth/image_raw',
            qos_profile=qos_profile_sensor_data
        )

        self.ts = ApproximateTimeSynchronizer(
            [det_sub, visible_sub, depth_sub],
            queue_size=10,
            slop=0.015,
            allow_headerless=True,
        )

        self.ts.registerCallback(self._synchronized_callback)

    def _store_intrinsics(self, msg: CameraInfo):
        # 3x3 intrinsic matrix
        k = msg.k
        self.intrinsics = (k[0], k[4], k[2], k[5])  # fx, fy, cx, cy
        self.get_logger().info(
            f"Got intrinsics: fx={k[0]:.1f} fy={k[4]:.1f} "
            f"cx={k[2]:.1f} cy={k[5]:.1f}"
        )

    def _synchronized_callback(
            self,
            det_msg: PointStamped,
            visible_msg: Bool,
            depth_msg: Image):
        # Checking if intrinsics have been calculated first
        if self.intrinsics is None:
            self.get_logger().warn(
                "Waiting for camera intrinsics...",
                throttle_duration_sec=2.0)
            return

        if not visible_msg.data:
            return

        # extract 2d coordinates
        u = det_msg.point.x
        v = det_msg.point.y

        # convert ROS2 image message to numpy array
        try:
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {str(e)}")
            return

        # sample depth array around u,v patch
        z_cam = self._sample_depth(depth_img, u, v)

        # bail if depth sample returns none (invalid patch)
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
        point_msg.header.frame_id = "camera_link"
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

    def _sample_depth(self, depth_img, u: float, v: float):
        # median-sample around a small patch around (u.v). Returns depth in
        # meters or none if the patch is invalid
        r = 2  # patch radius
        h, w = depth_img.shape[:2]
        ui, vi = int(round(u)), int(round(v))

        # slice out depth image
        u_min = max(0, ui - r)
        u_max = min(w, ui + r + 1)
        v_min = max(0, vi - r)
        v_max = min(h, vi + r + 1)

        patch = depth_img[v_min:v_max, u_min:u_max]

        # if requested pixel is outside of the frame
        if patch.size == 0:
            return None

        # keep valid numbers only
        valid_mask = (patch > 0) & (~np.isnan(patch))
        valid_values = patch[valid_mask]

        # reject if less then half of the pixels in the matrix
        if len(valid_values) < 0.5 * patch.size:
            return None

        # take the median and convert from mm to m
        median_depth_mm = np.median(valid_values)

        return float(median_depth_mm / 1000)


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
