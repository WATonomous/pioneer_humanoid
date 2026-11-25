#!/usr/bin/env python3
"""
Dynamic baseline stereo depth node.

Computes depth from two wrist-mounted cameras using robot kinematics
to determine the camera poses at runtime.

Pipeline:
  1. Receive images from left/right wrist cameras
  2. Get joint states and compute FK to get each camera's pose
  3. Compute relative pose between cameras (dynamic baseline)
  4. Rectify images for this baseline
  5. Run SGBM stereo matching
  6. Publish depth image and optionally point cloud

Subscribes:
  - /wrist_left/image_raw (sensor_msgs/Image)
  - /wrist_right/image_raw (sensor_msgs/Image)
  - /joint_states (sensor_msgs/JointState) or ArmPose messages

Publishes:
  - /wrist_stereo/depth (sensor_msgs/Image)
  - /wrist_stereo/points (sensor_msgs/PointCloud2, optional)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, JointState
from std_msgs.msg import Header
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

import cv2
import numpy as np
import yaml
import os
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import threading


@dataclass
class CameraCalibration:
    """Camera intrinsics and hand-eye calibration."""
    K: np.ndarray           # 3x3 camera matrix
    D: np.ndarray           # distortion coefficients
    image_size: Tuple[int, int]
    T_cam_wrist: np.ndarray  # 4x4 transform: camera frame in wrist frame


def load_intrinsics(filepath: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Load camera intrinsics from YAML file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    K = np.array(data['camera_matrix']['data']).reshape(3, 3)
    D = np.array(data['distortion_coefficients']['data'])
    size = (data['image_width'], data['image_height'])

    return K, D, size


def load_hand_eye(filepath: str) -> np.ndarray:
    """Load hand-eye calibration (camera pose in wrist frame)."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Load from quaternion
    t = np.array([
        data['translation']['x'],
        data['translation']['y'],
        data['translation']['z']
    ])

    q = np.array([
        data['rotation']['x'],
        data['rotation']['y'],
        data['rotation']['z'],
        data['rotation']['w']
    ])

    R = Rotation.from_quat(q).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


class SimpleFK:
    """Simple forward kinematics from URDF."""

    def __init__(self, urdf_path: str):
        self.joints = {}
        self._parse_urdf(urdf_path)

    def _parse_urdf(self, urdf_path: str):
        """Parse URDF for joint information."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for joint in root.findall('joint'):
            name = joint.get('name')
            jtype = joint.get('type')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')

            origin = joint.find('origin')
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin is not None:
                if origin.get('xyz'):
                    xyz = [float(x) for x in origin.get('xyz').split()]
                if origin.get('rpy'):
                    rpy = [float(x) for x in origin.get('rpy').split()]

            axis = [0, 0, 1]
            axis_elem = joint.find('axis')
            if axis_elem is not None and axis_elem.get('xyz'):
                axis = [float(x) for x in axis_elem.get('xyz').split()]

            self.joints[name] = {
                'type': jtype,
                'parent': parent,
                'child': child,
                'origin_xyz': xyz,
                'origin_rpy': rpy,
                'axis': axis
            }

    def _rpy_to_matrix(self, rpy) -> np.ndarray:
        """Roll-pitch-yaw to rotation matrix."""
        return Rotation.from_euler('xyz', rpy).as_matrix()

    def _axis_angle_to_matrix(self, axis, angle: float) -> np.ndarray:
        """Axis-angle to rotation matrix."""
        return Rotation.from_rotvec(np.array(axis) * angle).as_matrix()

    def get_transform(self, base_link: str, target_link: str,
                      joint_positions: Dict[str, float]) -> Optional[np.ndarray]:
        """Compute transform from base_link to target_link."""
        chain = self._find_chain(base_link, target_link)
        if chain is None:
            return None

        T = np.eye(4)
        for joint_name in chain:
            joint = self.joints[joint_name]

            # Fixed transform from joint origin
            T_origin = np.eye(4)
            T_origin[:3, :3] = self._rpy_to_matrix(joint['origin_rpy'])
            T_origin[:3, 3] = joint['origin_xyz']
            T = T @ T_origin

            # Joint motion
            if joint['type'] in ('revolute', 'continuous'):
                angle = joint_positions.get(joint_name, 0.0)
                R_joint = self._axis_angle_to_matrix(joint['axis'], angle)
                T_joint = np.eye(4)
                T_joint[:3, :3] = R_joint
                T = T @ T_joint
            elif joint['type'] == 'prismatic':
                d = joint_positions.get(joint_name, 0.0)
                T_joint = np.eye(4)
                T_joint[:3, 3] = np.array(joint['axis']) * d
                T = T @ T_joint

        return T

    def _find_chain(self, base_link: str, target_link: str):
        """Find joint chain from base to target using parent traversal."""
        parent_map = {}
        for jname, jinfo in self.joints.items():
            parent_map[jinfo['child']] = (jname, jinfo['parent'])

        chain = []
        current = target_link
        while current != base_link:
            if current not in parent_map:
                return None
            joint_name, parent = parent_map[current]
            chain.append(joint_name)
            current = parent

        chain.reverse()
        return chain


class StereoDepthNode(Node):
    """Dynamic baseline stereo depth from wrist cameras."""

    def __init__(self):
        super().__init__('stereo_depth_node')

        # Declare parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('left_image_topic', '/wrist_left/image_raw')
        self.declare_parameter('right_image_topic', '/wrist_right/image_raw')
        self.declare_parameter('left_intrinsics', 'config/left_intrinsics.yaml')
        self.declare_parameter('right_intrinsics', 'config/right_intrinsics.yaml')
        self.declare_parameter('left_hand_eye', 'config/left_hand_eye.yaml')
        self.declare_parameter('right_hand_eye', 'config/right_hand_eye.yaml')
        self.declare_parameter('urdf_file', '')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('left_wrist_link', 'link6l')
        self.declare_parameter('right_wrist_link', 'link6')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('depth_topic', '/wrist_stereo/depth')
        self.declare_parameter('points_topic', '/wrist_stereo/points')
        self.declare_parameter('publish_points', True)
        self.declare_parameter('min_depth', 0.05)
        self.declare_parameter('max_depth', 0.5)
        self.declare_parameter('num_disparities', 128)
        self.declare_parameter('block_size', 5)

        # Rolling shutter compensation - pause before capture
        self.declare_parameter('require_stationary', True)   # Only capture when arm is still
        self.declare_parameter('velocity_threshold', 0.01)   # rad/s - below this = stationary
        self.declare_parameter('settle_time', 0.05)          # seconds to wait after stopping

        # Load config file if provided
        config_file = self.get_parameter('config_file').value
        if config_file and os.path.exists(config_file):
            self._load_config(config_file)

        # Get parameters
        self.left_image_topic = self.get_parameter('left_image_topic').value
        self.right_image_topic = self.get_parameter('right_image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.points_topic = self.get_parameter('points_topic').value
        self.publish_points = self.get_parameter('publish_points').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.base_link = self.get_parameter('base_link').value
        self.left_wrist_link = self.get_parameter('left_wrist_link').value
        self.right_wrist_link = self.get_parameter('right_wrist_link').value

        # Rolling shutter parameters
        self.require_stationary = self.get_parameter('require_stationary').value
        self.velocity_threshold = self.get_parameter('velocity_threshold').value
        self.settle_time = self.get_parameter('settle_time').value

        # Load calibrations
        self.left_calib = self._load_camera_calib(
            self.get_parameter('left_intrinsics').value,
            self.get_parameter('left_hand_eye').value
        )
        self.right_calib = self._load_camera_calib(
            self.get_parameter('right_intrinsics').value,
            self.get_parameter('right_hand_eye').value
        )

        # Load URDF for FK
        urdf_file = self.get_parameter('urdf_file').value
        if urdf_file and os.path.exists(urdf_file):
            self.fk = SimpleFK(urdf_file)
            self.get_logger().info(f'Loaded URDF: {urdf_file}')
        else:
            self.fk = None
            self.get_logger().warn('No URDF provided - stereo will not work without FK')

        # Initialize SGBM
        num_disp = self.get_parameter('num_disparities').value
        block_size = self.get_parameter('block_size').value
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.bridge = CvBridge()

        # State
        self.left_image = None
        self.right_image = None
        self.left_stamp = None
        self.right_stamp = None
        self.joint_positions: Dict[str, float] = {}
        self.joint_velocities: Dict[str, float] = {}
        self.last_joint_positions: Dict[str, float] = {}
        self.last_joint_time: Optional[float] = None
        self.stationary_since: Optional[float] = None  # Time when arm became stationary
        self.lock = threading.Lock()

        # Cached rectification maps (recomputed when baseline changes significantly)
        self.rect_maps = None
        self.last_baseline = None
        self.Q = None  # Disparity-to-depth matrix

        # Subscribers
        self.left_sub = self.create_subscription(
            Image, self.left_image_topic, self.left_image_cb, qos_profile_sensor_data
        )
        self.right_sub = self.create_subscription(
            Image, self.right_image_topic, self.right_image_cb, qos_profile_sensor_data
        )
        self.joint_sub = self.create_subscription(
            JointState, self.get_parameter('joint_state_topic').value,
            self.joint_cb, 10
        )

        # Publishers
        self.depth_pub = self.create_publisher(
            Image, self.depth_topic, qos_profile_sensor_data
        )
        if self.publish_points:
            self.points_pub = self.create_publisher(
                PointCloud2, self.points_topic, qos_profile_sensor_data
            )

        # Processing timer
        self.create_timer(1.0 / 30.0, self.process_stereo)

        self.get_logger().info('Stereo depth node started')
        self.get_logger().info(f'  Left camera: {self.left_image_topic}')
        self.get_logger().info(f'  Right camera: {self.right_image_topic}')
        self.get_logger().info(f'  Depth output: {self.depth_topic}')

    def _load_config(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Map config to parameters
        if 'left_camera' in config:
            self.set_parameters([
                rclpy.parameter.Parameter('left_image_topic',
                    rclpy.Parameter.Type.STRING,
                    config['left_camera'].get('image_topic', '/wrist_left/image_raw')),
            ])
        # ... similar for other config sections

    def _load_camera_calib(self, intrinsics_file: str,
                           hand_eye_file: str) -> Optional[CameraCalibration]:
        """Load camera calibration files."""
        try:
            K, D, size = load_intrinsics(intrinsics_file)
            T_cam_wrist = load_hand_eye(hand_eye_file)
            return CameraCalibration(K=K, D=D, image_size=size, T_cam_wrist=T_cam_wrist)
        except Exception as e:
            self.get_logger().error(f'Failed to load calibration: {e}')
            return None

    def left_image_cb(self, msg: Image):
        with self.lock:
            try:
                self.left_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.left_stamp = msg.header.stamp
            except Exception as e:
                self.get_logger().error(f'Left image conversion failed: {e}')

    def right_image_cb(self, msg: Image):
        with self.lock:
            try:
                self.right_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.right_stamp = msg.header.stamp
            except Exception as e:
                self.get_logger().error(f'Right image conversion failed: {e}')

    def joint_cb(self, msg: JointState):
        with self.lock:
            current_time = self.get_clock().now().nanoseconds / 1e9

            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]

                    # Compute velocity from position change if not provided
                    if i < len(msg.velocity) and msg.velocity[i] != 0.0:
                        self.joint_velocities[name] = abs(msg.velocity[i])
                    elif self.last_joint_time is not None and name in self.last_joint_positions:
                        dt = current_time - self.last_joint_time
                        if dt > 0:
                            dpos = abs(msg.position[i] - self.last_joint_positions[name])
                            self.joint_velocities[name] = dpos / dt

                    self.last_joint_positions[name] = msg.position[i]

            self.last_joint_time = current_time

            # Check if arm is stationary
            if self.joint_velocities:
                max_vel = max(self.joint_velocities.values())
                if max_vel < self.velocity_threshold:
                    if self.stationary_since is None:
                        self.stationary_since = current_time
                else:
                    self.stationary_since = None

    def get_camera_poses(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get left and right camera poses in base frame."""
        if self.fk is None or not self.joint_positions:
            return None

        # Get wrist poses in base frame
        T_base_left_wrist = self.fk.get_transform(
            self.base_link, self.left_wrist_link, self.joint_positions
        )
        T_base_right_wrist = self.fk.get_transform(
            self.base_link, self.right_wrist_link, self.joint_positions
        )

        if T_base_left_wrist is None or T_base_right_wrist is None:
            return None

        # Get camera poses: T_base_cam = T_base_wrist @ T_wrist_cam
        # Note: hand-eye gives T_cam_wrist, so we need inverse
        T_left_wrist_cam = np.linalg.inv(self.left_calib.T_cam_wrist)
        T_right_wrist_cam = np.linalg.inv(self.right_calib.T_cam_wrist)

        T_base_left_cam = T_base_left_wrist @ T_left_wrist_cam
        T_base_right_cam = T_base_right_wrist @ T_right_wrist_cam

        return T_base_left_cam, T_base_right_cam

    def compute_rectification(self, T_left: np.ndarray, T_right: np.ndarray):
        """Compute stereo rectification for current camera poses."""
        # Relative pose: right camera in left camera frame
        T_left_right = np.linalg.inv(T_left) @ T_right

        R = T_left_right[:3, :3]
        t = T_left_right[:3, 3]

        baseline = np.linalg.norm(t)

        # Check if baseline changed significantly
        if self.last_baseline is not None:
            if abs(baseline - self.last_baseline) < 0.001:  # 1mm threshold
                return  # Use cached rectification

        self.last_baseline = baseline
        self.get_logger().info(f'Recomputing rectification, baseline={baseline:.4f}m')

        # Stereo rectification
        image_size = self.left_calib.image_size

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.left_calib.K, self.left_calib.D,
            self.right_calib.K, self.right_calib.D,
            image_size, R, t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0  # Crop to valid pixels only
        )

        # Compute rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.left_calib.K, self.left_calib.D, R1, P1,
            image_size, cv2.CV_32FC1
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.right_calib.K, self.right_calib.D, R2, P2,
            image_size, cv2.CV_32FC1
        )

        self.rect_maps = (map1_left, map2_left, map1_right, map2_right)
        self.Q = Q

    def process_stereo(self):
        """Main processing loop."""
        with self.lock:
            if self.left_image is None or self.right_image is None:
                return
            if not self.joint_positions:
                return

            # Rolling shutter protection: skip if arm is moving
            if self.require_stationary:
                current_time = self.get_clock().now().nanoseconds / 1e9
                if self.stationary_since is None:
                    # Arm is moving - skip this frame
                    return
                time_stationary = current_time - self.stationary_since
                if time_stationary < self.settle_time:
                    # Arm just stopped - wait for settle_time before capturing
                    return

            left_img = self.left_image.copy()
            right_img = self.right_image.copy()
            stamp = self.left_stamp

        # Check calibration
        if self.left_calib is None or self.right_calib is None:
            self.get_logger().warn_throttle(
                self.get_clock(), 5000, 'Missing camera calibration'
            )
            return

        # Get camera poses
        poses = self.get_camera_poses()
        if poses is None:
            self.get_logger().warn_throttle(
                self.get_clock(), 5000, 'Could not compute camera poses'
            )
            return

        T_left, T_right = poses

        # Compute/update rectification
        self.compute_rectification(T_left, T_right)

        if self.rect_maps is None:
            return

        # Rectify images
        map1_l, map2_l, map1_r, map2_r = self.rect_maps
        left_rect = cv2.remap(left_img, map1_l, map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, map1_r, map2_r, cv2.INTER_LINEAR)

        # Convert to grayscale for SGBM
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Convert disparity to depth
        # depth = baseline * fx / disparity
        fx = self.left_calib.K[0, 0]
        baseline = self.last_baseline

        with np.errstate(divide='ignore', invalid='ignore'):
            depth = baseline * fx / disparity
            depth[disparity <= 0] = 0
            depth[depth < self.min_depth] = 0
            depth[depth > self.max_depth] = 0
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # Publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(
            (depth * 1000).astype(np.uint16),  # Convert to mm
            encoding='mono16'
        )
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = 'wrist_stereo'
        self.depth_pub.publish(depth_msg)

        # Publish point cloud if enabled
        if self.publish_points and self.Q is not None:
            points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

            # Filter valid points
            mask = (depth > self.min_depth) & (depth < self.max_depth)
            points = points_3d[mask].reshape(-1, 3).astype(np.float32)

            if points.shape[0] > 0:
                self._publish_point_cloud(points, stamp)

    def _publish_point_cloud(self, points: np.ndarray, stamp):
        """Publish point cloud message."""
        header = Header()
        header.stamp = stamp
        header.frame_id = 'wrist_stereo'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points.tolist())
        self.points_pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
