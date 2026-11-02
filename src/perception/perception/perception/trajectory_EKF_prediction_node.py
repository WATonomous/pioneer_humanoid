"""Shuttle EKF node: TrackNet hotspots (+ depth) in, the shuttle's 3D state and predicted impact out.

The maths is in shuttle_ekf.py; this node only moves messages in and out.

Subscribes  /shuttle/hotspots                        (vision_msgs/Detection2DArray, from tracknet_node)
            /camera/color/camera_info                (sensor_msgs/CameraInfo: focal length, image centre)
            /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image: depth, pixel-aligned with the
                                                      colour image; optional, used when available)
Publishes   /shuttle/state     (nav_msgs/Odometry) position (m) and velocity (m/s) in the camera frame
                               (x right, y down, z forward), with covariance; only while tracked
            /shuttle/impact    (common_msgs/PerceptionImpactEstimation) when the shuttle is predicted
                               to come within impact_radius of the camera

Parameters  camera_pitch_deg  how far the camera looks up from level (0), sets gravity's direction
            fps               camera frame rate (30)
            impact_radius     m (2.0)
            look_ahead        s, how far ahead to predict the impact (2.0)
"""
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from common_msgs.msg import PerceptionImpactEstimation
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

from perception.shuttle_ekf import Camera, ShuttleEKF


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def patch_depths(depth_img, u, v):
    """Valid readings (m) in the 5x5 patch around (u, v)."""
    to_m = 0.001 if depth_img.dtype == np.uint16 else 1.  # RealSense: uint16 mm; simulators: float m
    ui, vi = int(round(u)), int(round(v))
    patch = depth_img[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3].astype(float)
    return patch[(patch > 0) & np.isfinite(patch)] * to_m


def depth_reader(depth_img, earlier_imgs, margin=0.3):
    """depth_at(u, v) for ShuttleEKF.

    A shuttle covers only a few pixels of the 5x5 patch, and TrackNet's pixel can be a few pixels off,
    so most of the patch shows what is behind it: take the closest reading, not the median. The camera
    is fixed, so the usual depth at that pixel (median over earlier images) is the background; the
    closest reading counts as the shuttle only if it is at least margin (m) in front of it. Otherwise
    the sensor missed the shuttle and saw the wall or floor: no reading."""
    def depth_at(u, v):
        now = patch_depths(depth_img, u, v)
        if not now.size:
            return None
        behind = [np.median(d) for d in (patch_depths(img, u, v) for img in earlier_imgs) if d.size]
        closest = float(now.min())
        if behind and closest > np.median(behind) - margin:
            return None
        return closest
    return depth_at


class EKFPredictionNode(Node):
    def __init__(self):
        super().__init__('trajectory_EKF_prediction_node')
        self.pitch = self.declare_parameter('camera_pitch_deg', 0.).value
        self.fps = self.declare_parameter('fps', 30.).value
        self.impact_radius = self.declare_parameter('impact_radius', 2.).value
        self.look_ahead = self.declare_parameter('look_ahead', 2.).value

        self.ekf = None                   # created once the camera's calibration arrives
        self.depths = deque(maxlen=15)    # recent (time, depth image): to match each frame's hotspots,
        #                                   and as the background (half a second at 30 fps)
        self.bridge = CvBridge()
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.on_camera_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, '/shuttle/hotspots', self.on_hotspots, 10)
        self.state_pub = self.create_publisher(Odometry, '/shuttle/state', 10)
        self.impact_pub = self.create_publisher(PerceptionImpactEstimation, '/shuttle/impact', 10)

    def on_camera_info(self, msg):
        if self.ekf is None:
            fx, _, cx, _, fy, cy = msg.k[:6]
            camera = Camera(fx, fy, cx, cy, msg.width, msg.height, self.pitch)
            self.ekf = ShuttleEKF(camera, fps=self.fps)
            self.get_logger().info(f'Camera {msg.width}x{msg.height}, fx {fx:.0f}: EKF ready')

    def on_depth(self, msg):
        self.depths.append((seconds(msg.header.stamp), self.bridge.imgmsg_to_cv2(msg, 'passthrough')))

    def on_hotspots(self, msg):
        if self.ekf is None:
            self.get_logger().warn('No camera_info yet', throttle_duration_sec=2.)
            return
        time = seconds(msg.header.stamp)
        hotspots = [(d.bbox.center.position.x, d.bbox.center.position.y, d.results[0].hypothesis.score)
                    for d in msg.detections]
        # Depth image taken at the same moment (within half a frame), if any
        depth_at = None
        if self.depths:
            t, img = min(self.depths, key=lambda d: abs(d[0] - time))
            if abs(t - time) <= 0.5 / self.fps:
                depth_at = depth_reader(img, [d for _, d in self.depths if d is not img])

        self.ekf.update(time, hotspots, depth_at)
        if not self.ekf.visible():
            return

        x, P = self.ekf.x, self.ekf.P
        state = Odometry(header=msg.header)
        p, v = state.pose.pose.position, state.twist.twist.linear
        p.x, p.y, p.z = x[:3]
        v.x, v.y, v.z = x[3:]
        state.pose.pose.orientation.w = 1.
        cov = np.zeros((2, 6, 6))  # Odometry covariances are 6x6 (xyz + rotation); rotation unused
        cov[0, :3, :3], cov[1, :3, :3] = P[:3, :3], P[3:, 3:]
        state.pose.covariance, state.twist.covariance = cov[0].ravel(), cov[1].ravel()
        self.state_pub.publish(state)

        impact = self.ekf.predict_impact(self.impact_radius, self.look_ahead)
        if impact is not None:
            secs, position, velocity = impact
            out = PerceptionImpactEstimation(header=msg.header)
            out.time_to_impact = float(secs)
            out.position_of_impact.x, out.position_of_impact.y, out.position_of_impact.z = position
            # Hit back along the incoming path
            out.hitback_direction.x, out.hitback_direction.y, out.hitback_direction.z = \
                -velocity / (np.linalg.norm(velocity) + 1e-9)
            out.uncertainty = float(np.trace(P))
            self.impact_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EKFPredictionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():  # Ctrl+C may already have shut ROS down
            rclpy.shutdown()
