import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
import numpy as np
from ultralytics import YOLO


class ShuttlecockDetector(Node):
    def __init__(self):
        super().__init__('shuttlecock_detector_node')
        self.bridge = CvBridge()

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('point_topic', '/perception/shuttlecock/detections_2d')
        self.declare_parameter('visible_topic', '/perception/shuttlecock/detection_visible')
        # update to '/data/weights/yolo26n_v2.pt'
        self.declare_parameter('weights', '/data/weights/yolo26n.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('imgsz', 1280)
        self.declare_parameter('conf_threshold', 0.60)
        self.declare_parameter('class_ids', [0])
        self.declare_parameter('half', True)

        self.device = self._param('device')
        self.imgsz = self._param('imgsz')
        self.conf = self._param('conf_threshold')
        self.class_ids = list(self._param('class_ids'))
        self.half = self._param('half')

        self.model = YOLO(self._param('weights'))
        # Warm up with a blank frame
        dummy = np.zeros([480, 640, 3], dtype=np.uint8)  # Matches rgb_camera.color_profile in perception.launch.py
        for _ in range(2):
            self.model.predict(dummy, imgsz=self.imgsz, conf=self.conf, classes=self.class_ids,
                               device=self.device, half=self.half, verbose=False)

        self.point_pub = self.create_publisher(PointStamped, self._param('point_topic'), 10)
        self.visible_pub = self.create_publisher(Bool, self._param('visible_topic'), 10)
        self.create_subscription(Image, self._param('image_topic'), self._image_callback, qos_profile_sensor_data)

        self.get_logger().info(f"Detector up: weights={self._param('weights')} device={self.device} imgsz={self.imgsz}")

    def _param(self, name):
        return self.get_parameter(name).value

    def _image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=self.class_ids,
                                    device=self.device, half=self.half, verbose=False)[0]
        boxes = result.boxes

        visible = len(boxes) > 0
        self.visible_pub.publish(Bool(data=visible))
        if not visible:
            return

        best = int(boxes.conf.argmax())
        u, v = boxes.xywh[best][:2].tolist()

        out = PointStamped()
        out.header = msg.header
        out.point.x = float(u)
        out.point.y = float(v)
        out.point.z = float(boxes.conf[best])
        self.point_pub.publish(out)

        latency_ms = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e6
        self.get_logger().debug(f'u={u:.0f} v={v:.0f} conf={out.point.z:.2f} latency={latency_ms:.1f} ms')


def main(args=None):
    rclpy.init(args=args)
    node = ShuttlecockDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
