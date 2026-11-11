"""TrackNet node: camera image in, the shuttle's 2D pixel out.

Publishes the topics shuttlecock_localizer_node already subscribes to (as on main and in #501), which adds depth.

Subscribes  image_topic        /camera/color/image_raw    (sensor_msgs/Image)
            camera_info_topic  /camera/color/camera_info  (sensor_msgs/CameraInfo)
Publishes   point_topic        /tracknetv3/shuttle_point    (geometry_msgs/PointStamped)
                x, y = shuttle pixel (u, v), z = TrackNet confidence 0-1; header copied from the image.
                Only published for frames where the gated track matched a hotspot.
            visible_topic      /tracknetv3/shuttle_visible  (std_msgs/Bool)
                every frame: whether point_topic got a point for it
            hotspots_topic     /perception/tracknet/hotspots              (vision_msgs/Detection2DArray)
                debug: TrackNet's raw top hotspots before gating, strongest first

TrackNet's top-k hotspots are filtered by HotspotGate: only one that fits the shuttle's followed
flight is published, so a stronger blob on the wall is ignored. A new track needs 3 hotspots moving
steadily within 4 frames, so the first ~3 frames of a flight are not published.

Parameters  model_path   default: <package share>/models/TrackNet_best.pt
            top_k        hotspots per frame (3)
            min_score    weakest hotspot to consider (0.05)
            fps          camera frame rate (30)
            camera_pitch_deg  how far the camera looks up from level (0); sets gravity's direction
                         in the camera frame for the gate's flight prediction
            use_gate     false = publish TrackNet's strongest hotspot ungated (for comparing) (true)
"""
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from perception.hotspot_gate import HotspotGate
from perception.live_tracker import LiveTracker
from perception.shuttle_ekf import Camera


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class TrackNetNode(Node):
    def __init__(self):
        super().__init__('tracknet_node')
        model_path = self.declare_parameter('model_path', '').value or \
            os.path.join(get_package_share_directory('perception'), 'models', 'TrackNet_best.pt')
        top_k = self.declare_parameter('top_k', 3).value
        min_score = self.declare_parameter('min_score', 0.05).value
        self.fps = self.declare_parameter('fps', 30.).value
        self.use_gate = self.declare_parameter('use_gate', True).value
        self.pitch = self.declare_parameter('camera_pitch_deg', 0.).value
        image_topic = self.declare_parameter('image_topic', '/camera/color/image_raw').value
        info_topic = self.declare_parameter('camera_info_topic', '/camera/color/camera_info').value
        point_topic = self.declare_parameter('point_topic', '/tracknetv3/shuttle_point').value
        visible_topic = self.declare_parameter('visible_topic', '/tracknetv3/shuttle_visible').value
        hotspots_topic = self.declare_parameter('hotspots_topic', '/perception/tracknet/hotspots').value

        self.tracker = LiveTracker(model_path, top_k=top_k, min_score=min_score)
        self.gate = None                  # created once the camera's calibration arrives
        self.bridge = CvBridge()
        # Queue of 1: if TrackNet falls behind, skip to the newest frame instead of lagging
        camera_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, info_topic, self.on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, self.on_image, camera_qos)
        self.point_pub = self.create_publisher(PointStamped, point_topic, 10)
        self.visible_pub = self.create_publisher(Bool, visible_topic, 10)
        self.hotspots_pub = self.create_publisher(Detection2DArray, hotspots_topic, 10)
        self.get_logger().info(f'TrackNet on {self.tracker.device}, model {model_path}, gate {self.use_gate}')

    def on_camera_info(self, msg):
        if self.gate is None:
            fx, _, cx, _, fy, cy = msg.k[:6]
            camera = Camera(fx, fy, cx, cy, msg.width, msg.height, self.pitch)
            self.gate = HotspotGate(camera, fps=self.fps)
            self.get_logger().info(f'Camera {msg.width}x{msg.height}, fx {fx:.0f}: gate ready')

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hotspots = self.tracker.push(frame) or []      # always push: the model needs every frame
        self.hotspots_pub.publish(self.hotspots_msg(msg.header, hotspots))

        if self.use_gate:
            if self.gate is None:
                self.get_logger().warn('No camera_info yet', throttle_duration_sec=2.)
                return
            best = self.gate.update(seconds(msg.header.stamp), hotspots)
        else:
            best = hotspots[0] if hotspots else None

        self.visible_pub.publish(Bool(data=best is not None))
        if best is not None:
            out = PointStamped(header=msg.header)
            out.point.x, out.point.y, out.point.z = float(best[0]), float(best[1]), float(best[2])
            self.point_pub.publish(out)

    @staticmethod
    def hotspots_msg(header, hotspots):
        out = Detection2DArray(header=header)
        for u, v, score in hotspots:
            det = Detection2D(header=header)
            det.bbox.center.position.x, det.bbox.center.position.y = u, v
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = 'shuttle', score
            det.results.append(hyp)
            out.detections.append(det)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = TrackNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():  # Ctrl+C may already have shut ROS down
            rclpy.shutdown()


if __name__ == '__main__':
    main()
