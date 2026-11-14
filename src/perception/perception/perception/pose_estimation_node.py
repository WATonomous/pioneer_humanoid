#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from rtmlib import RTMPose
from ultralytics import YOLO


class PoseEstimation(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")
        self.bridge = CvBridge()

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', '/perception/pose_estimation')
        self.declare_parameter('person_weights', '/data/weights/yolo26n.pt')
        self.declare_parameter('pose_onnx', '/opt/models/rtmpose-s_body7.onnx')
        self.declare_parameter('device', 'cuda:0')  # Person detector; set by launch
        self.declare_parameter('half', True)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('person_conf', 0.50)
        self.declare_parameter('pose_device', 'cpu')

        self.device = self._param('device')
        self.half = self._param('half')
        self.imgsz = self._param('imgsz')
        self.person_conf = self._param('person_conf')

        self.person_model = YOLO(self._param('person_weights'))
        self.pose_model = RTMPose(
            onnx_model=self._param('pose_onnx'),
            model_input_size=(192, 256),
            backend='onnxruntime',
            device=self._param('pose_device')
        )

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self._detect_person(dummy)
            self.pose_model(dummy, bboxes=[[0, 0, 640, 480]])

        self.sub = self.create_subscription(
            Image,
            self._param('image_topic'),
            self.image_callback,
            qos_profile_sensor_data
        )
        self.pub = self.create_publisher(
            PoseArray,
            self._param('output_topic'),
            10
        )

        self.get_logger().info(f"Pose Estimation up: "
                               f"person_weights={self._param('person_weights')} "
                               f"device={self.device} pose_onnx={self._param('pose_onnx')}")

    def _param(self, name):
        return self.get_parameter(name).value

    def _detect_person(self, frame):
        boxes = self.person_model.predict(
            frame, imgsz=self.imgsz, conf=self.person_conf, classes=[0],
            device=self.device, half=self.half, verbose=False
        )[0].boxes

        if len(boxes) == 0:
            return None
        return boxes.xyxy[int(boxes.conf.argmax())].float().tolist()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        box = self._detect_person(frame)
        if box is None:
            return

        keypoints, scores = self.pose_model(frame, bboxes=[box])

        pose_array = PoseArray()
        pose_array.header = msg.header
        for (x, y), score in zip(keypoints[0], scores[0]):
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(score)
            pose_array.poses.append(pose)

        self.pub.publish(pose_array)
        self.get_logger().debug(f'Publishing {len(pose_array.poses)} keypoints, box={box}')


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
