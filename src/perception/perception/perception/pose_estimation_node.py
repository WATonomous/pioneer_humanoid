#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from mmpose.apis import init_model, inference_topdown


class PoseEstimation(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            qos_profile_sensor_data
        )
        self.model = init_model(
            "/usr/local/lib/python3.10/dist-packages/mmpose/.mim/configs/"
            "body_2d_keypoint/rtmpose/coco/"
            "rtmpose-s_8xb256-420e_coco-256x192.py",
            "/root/models/rtmpose-s.pth",
            device="cpu",
        )

        self.pub = self.create_publisher(
            PoseArray,
            "/perception/pose_estimation",
            10
        )

        self.get_logger().info("Pose Estimation Node started")

    def image_callback(self, msg):
        # converting the ros2 message to a numpy array
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        pose_array = PoseArray()
        pose_array.header = msg.header

        results = inference_topdown(self.model, frame)
        keypoints = results[0].pred_instances.keypoints[0]
        pose_array.header = msg.header

        for kp in keypoints:
            pose = Pose()
            pose.position.x = float(kp[0])
            pose.position.y = float(kp[1])
            pose.position.z = 0.0
            pose_array.poses.append(pose)

        self.pub.publish(pose_array)
        self.get_logger().info(f'Publishing {len(keypoints)} keypoints')


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
