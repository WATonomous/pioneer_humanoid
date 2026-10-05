import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import numpy as np


class DummyPublisherNode(Node):
    def __init__(self):
        super().__init__('dummy_publisher_node')

        self.bridge = CvBridge()

        self.rgb_pub = self.create_publisher(
            Image,
            '/camera/color/image_raw',
            qos_profile_sensor_data
        )

        self.depth_pub = self.create_publisher(
            Image,
            '/camera/depth/image_raw',
            qos_profile_sensor_data
        )

        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            '/camera/depth/camera_info',
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(1.0, self.publish_dummy_data)

        self.get_logger().info('Dummy camera publisher started (1 Hz)')

    def publish_dummy_data(self):
        now = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = now
        header.frame_id = 'camera_link'

        # Create a simple scene with geometric shapes at different depths
        rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_image = np.full(
            (480, 640), 3000, dtype=np.uint16)  # Background at 3000mm

        # Add a large rectangle (closer object) - like a wall or box
        cv2.rectangle(rgb_image, (150, 120), (490, 360),
                      (100, 150, 200), -1)  # Filled rectangle
        cv2.rectangle(depth_image, (150, 120), (490, 360),
                      1200, -1)  # 1200mm depth

        # Add a smaller circle (closest object) - like a ball
        cv2.circle(rgb_image, (320, 240), 80,
                   (50, 200, 50), -1)  # Green circle
        # 800mm depth (closest)
        cv2.circle(depth_image, (320, 240), 80, 800, -1)

        # Add another rectangle (medium distance)
        cv2.rectangle(rgb_image, (50, 300), (200, 450),
                      (200, 100, 100), -1)  # Reddish rectangle
        cv2.rectangle(depth_image, (50, 300), (200, 450),
                      1800, -1)  # 1800mm depth

        # Optional: Add some gradient/texture to make it more realistic
        noise = np.random.randint(-20, 20, rgb_image.shape, dtype=np.int16)
        rgb_image = np.clip(
            rgb_image.astype(
                np.int16) +
            noise,
            0,
            255).astype(
            np.uint8)

        # Add slight depth variation (noise)
        depth_noise = np.random.randint(-30,
                                        30,
                                        depth_image.shape,
                                        dtype=np.int16)
        depth_image = np.clip(
            depth_image.astype(
                np.int32) +
            depth_noise,
            0,
            65535).astype(
            np.uint16)

        rgb_msg = self.bridge.cv2_to_imgmsg(
            rgb_image, encoding='bgr8'
        )
        rgb_msg.header = header
        self.rgb_pub.publish(rgb_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(
            depth_image, encoding='mono16'
        )
        depth_msg.header = header
        self.depth_pub.publish(depth_msg)

        info = CameraInfo()
        info.header = header
        info.height = 480
        info.width = 640
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        fx = 525.0
        fy = 525.0
        cx = 320.0
        cy = 240.0

        info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        info.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        self.camera_info_pub.publish(info)


def main(args=None):
    rclpy.init(args=args)
    node = DummyPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
