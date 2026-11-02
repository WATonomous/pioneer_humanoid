"""TrackNet node: camera image in, the frame's top shuttle hotspots out.

Subscribes  /camera/color/image_raw   (sensor_msgs/Image)
Publishes   /shuttle/hotspots         (vision_msgs/Detection2DArray)
            one detection per hotspot, strongest first: bbox.center = pixel (u, v),
            results[0].hypothesis.score = TrackNet confidence 0-1. Empty for the first 7 frames
            and when nothing is found. The header (time, camera frame) is copied from the image.

Parameters  model_path   default: <package share>/models/TrackNet_best.pt
            top_k        hotspots per frame (3)
            min_score    weakest hotspot to report (0.05)
"""
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from perception.live_tracker import LiveTracker


class TrackNetNode(Node):
    def __init__(self):
        super().__init__('tracknet_node')
        default_model = os.path.join(get_package_share_directory('perception'), 'models', 'TrackNet_best.pt')
        model_path = self.declare_parameter('model_path', default_model).value
        top_k = self.declare_parameter('top_k', 3).value
        min_score = self.declare_parameter('min_score', 0.05).value

        self.tracker = LiveTracker(model_path, top_k=top_k, min_score=min_score)
        self.bridge = CvBridge()
        # Queue of 1: if TrackNet falls behind, skip to the newest frame instead of lagging
        camera_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, '/camera/color/image_raw', self.on_image, camera_qos)
        self.pub = self.create_publisher(Detection2DArray, '/shuttle/hotspots', 10)
        self.get_logger().info(f'TrackNet on {self.tracker.device}, model {model_path}')

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hotspots = self.tracker.push(frame) or []

        out = Detection2DArray(header=msg.header)
        for u, v, score in hotspots:
            det = Detection2D(header=msg.header)
            det.bbox.center.position.x, det.bbox.center.position.y = u, v
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = 'shuttle', score
            det.results.append(hyp)
            out.detections.append(det)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrackNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
