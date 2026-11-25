#!/usr/bin/env python3
"""
Hand-eye calibration tool for wrist-mounted cameras using ChArUco board.

Computes the transform from camera frame to wrist link frame (T_wrist_camera).
Uses the classic AX=XB hand-eye calibration with ChArUco board poses.

For each capture, we need:
  - Camera pose relative to ChArUco board (from image detection)
  - Wrist pose relative to robot base (from joint states + FK)

The ChArUco board should be fixed in the world while the robot moves its wrist around it.

Usage:
  ros2 run wrist_stereo calibrate_hand_eye --ros-args \
    -p camera_topic:=/wrist_left/image_raw \
    -p intrinsics_file:=config/left_intrinsics.yaml \
    -p output_file:=config/left_hand_eye.yaml \
    -p urdf_file:=/path/to/urdf \
    -p wrist_link:=link6l \
    -p arm_side:=left

Controls:
  SPACE - capture current frame + robot pose
  c     - run hand-eye calibration
  s     - save calibration to file
  q     - quit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

import cv2
import numpy as np
import yaml
import os
from datetime import datetime
from typing import Optional, Tuple, List


def load_intrinsics(filepath: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Load camera intrinsics from YAML file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    K = np.array(data['camera_matrix']['data']).reshape(3, 3)
    D = np.array(data['distortion_coefficients']['data'])
    size = (data['image_width'], data['image_height'])

    return K, D, size


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues vector to 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to Rodrigues vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


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

    def _rpy_to_matrix(self, rpy: List[float]) -> np.ndarray:
        """Convert roll-pitch-yaw to rotation matrix."""
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        return R

    def _axis_angle_to_matrix(self, axis: List[float], angle: float) -> np.ndarray:
        """Convert axis-angle to rotation matrix."""
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def get_transform(self, base_link: str, target_link: str,
                      joint_positions: dict) -> Optional[np.ndarray]:
        """Compute transform from base_link to target_link."""
        chain = self._find_chain(base_link, target_link)
        if chain is None:
            return None

        T = np.eye(4)
        for joint_name in chain:
            joint = self.joints[joint_name]

            T_origin = np.eye(4)
            T_origin[:3, :3] = self._rpy_to_matrix(joint['origin_rpy'])
            T_origin[:3, 3] = joint['origin_xyz']
            T = T @ T_origin

            if joint['type'] == 'revolute' or joint['type'] == 'continuous':
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

    def _find_chain(self, base_link: str, target_link: str) -> Optional[List[str]]:
        """Find joint chain from base_link to target_link."""
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


class HandEyeCalibrator(Node):
    def __init__(self):
        super().__init__('hand_eye_calibrator')

        # Parameters
        self.declare_parameter('camera_topic', '/wrist_left/image_raw')
        self.declare_parameter('intrinsics_file', 'config/left_intrinsics.yaml')
        self.declare_parameter('output_file', 'config/left_hand_eye.yaml')
        self.declare_parameter('urdf_file', '')
        self.declare_parameter('wrist_link', 'link6l')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('arm_side', 'left')
        self.declare_parameter('min_captures', 10)
        self.declare_parameter('joint_state_topic', '/joint_states')

        # ChArUco board parameters (matching your printed board)
        self.declare_parameter('squares_x', 9)
        self.declare_parameter('squares_y', 6)
        self.declare_parameter('square_length', 0.025)   # 25mm
        self.declare_parameter('marker_length', 0.018)   # 18mm
        self.declare_parameter('dictionary', 'DICT_6X5_100')

        self.camera_topic = self.get_parameter('camera_topic').value
        self.intrinsics_file = self.get_parameter('intrinsics_file').value
        self.output_file = self.get_parameter('output_file').value
        self.urdf_file = self.get_parameter('urdf_file').value
        self.wrist_link = self.get_parameter('wrist_link').value
        self.base_link = self.get_parameter('base_link').value
        self.arm_side = self.get_parameter('arm_side').value
        self.min_captures = self.get_parameter('min_captures').value
        self.joint_state_topic = self.get_parameter('joint_state_topic').value

        self.squares_x = self.get_parameter('squares_x').value
        self.squares_y = self.get_parameter('squares_y').value
        self.square_length = self.get_parameter('square_length').value
        self.marker_length = self.get_parameter('marker_length').value
        dict_name = self.get_parameter('dictionary').value

        # Load camera intrinsics
        self.K, self.D, self.image_size = load_intrinsics(self.intrinsics_file)
        self.get_logger().info(f'Loaded intrinsics from {self.intrinsics_file}')

        # Set up ArUco dictionary and ChArUco board
        dict_map = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
            'DICT_6X5_100': cv2.aruco.DICT_6X6_100,  # Mapping for your board
        }

        if dict_name not in dict_map:
            self.get_logger().warn(f'Unknown dictionary {dict_name}, using DICT_6X6_100')
            dict_name = 'DICT_6X6_100'

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_map[dict_name])
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        # Initialize FK
        if self.urdf_file:
            self.fk = SimpleFK(self.urdf_file)
            self.get_logger().info(f'Loaded URDF from {self.urdf_file}')
        else:
            self.fk = None
            self.get_logger().warn('No URDF provided - hand-eye calibration will not work!')

        self.bridge = CvBridge()
        self.current_frame = None
        self.current_joint_positions = {}

        # Calibration data
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        self.captures = []

        # Result
        self.R_cam2gripper = None
        self.t_cam2gripper = None
        self.calibrated = False

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, self.joint_state_topic, self.joint_callback, 10
        )

        self.get_logger().info('ChArUco Hand-Eye Calibrator started')
        self.get_logger().info(f'  Camera: {self.camera_topic}')
        self.get_logger().info(f'  Wrist link: {self.wrist_link}')
        self.get_logger().info(f'  Board: {self.squares_x}x{self.squares_y}')
        self.get_logger().info('Fix the ChArUco board in the workspace, then move the arm to different poses')
        self.get_logger().info('Press SPACE to capture, C to calibrate, S to save, Q to quit')

    def image_callback(self, msg: Image):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def joint_callback(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def get_wrist_pose(self) -> Optional[np.ndarray]:
        """Get current wrist pose in base frame using FK."""
        if self.fk is None:
            self.get_logger().warn('No FK available')
            return None

        if not self.current_joint_positions:
            self.get_logger().warn('No joint positions received')
            return None

        T = self.fk.get_transform(self.base_link, self.wrist_link, self.current_joint_positions)
        return T

    def detect_charuco_pose(self, frame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Detect ChArUco board and compute pose relative to camera."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) < 4:
            return None

        # Interpolate ChArUco corners
        charuco_corners, charuco_ids, _, _ = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.charuco_board
        )

        if charuco_corners is None or len(charuco_corners) < 4:
            return None

        # Estimate pose of the ChArUco board
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.charuco_board, self.K, self.D, None, None
        )

        if not success:
            return None

        return rvec.flatten(), tvec.flatten()

    def capture_frame(self):
        """Capture current frame with ChArUco board and robot pose."""
        if self.current_frame is None:
            self.get_logger().warn('No frame available')
            return False

        # Get ChArUco board pose in camera frame
        result = self.detect_charuco_pose(self.current_frame)
        if result is None:
            self.get_logger().warn('ChArUco board not detected (need at least 4 markers visible)')
            return False

        rvec_target, tvec_target = result

        # Get wrist pose in base frame
        T_wrist = self.get_wrist_pose()
        if T_wrist is None:
            self.get_logger().warn('Could not get wrist pose - check URDF and joint states')
            return False

        # Store poses
        R_target = rodrigues_to_matrix(rvec_target)
        self.R_target2cam.append(R_target)
        self.t_target2cam.append(tvec_target)

        R_wrist = T_wrist[:3, :3]
        t_wrist = T_wrist[:3, 3]
        self.R_gripper2base.append(R_wrist)
        self.t_gripper2base.append(t_wrist)

        self.captures.append(self.current_frame.copy())

        self.get_logger().info(f'Captured frame {len(self.captures)}/{self.min_captures}')
        return True

    def run_calibration(self):
        """Run hand-eye calibration using OpenCV's calibrateHandEye."""
        if len(self.captures) < self.min_captures:
            self.get_logger().warn(
                f'Need at least {self.min_captures} captures, have {len(self.captures)}'
            )
            return False

        self.get_logger().info('Running hand-eye calibration...')

        R_gripper2base = [R for R in self.R_gripper2base]
        t_gripper2base = [t.reshape(3, 1) for t in self.t_gripper2base]
        R_target2cam = [R for R in self.R_target2cam]
        t_target2cam = [t.reshape(3, 1) for t in self.t_target2cam]

        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, 'Tsai'),
            (cv2.CALIB_HAND_EYE_PARK, 'Park'),
            (cv2.CALIB_HAND_EYE_HORAUD, 'Horaud'),
            (cv2.CALIB_HAND_EYE_ANDREFF, 'Andreff'),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, 'Daniilidis'),
        ]

        best_R = None
        best_t = None
        best_error = float('inf')
        best_method = None

        for method, name in methods:
            try:
                R, t = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method
                )

                error = self._compute_calibration_error(R, t)
                self.get_logger().info(f'  {name}: error = {error:.6f}')

                if error < best_error:
                    best_error = error
                    best_R = R
                    best_t = t
                    best_method = name

            except Exception as e:
                self.get_logger().warn(f'  {name} failed: {e}')

        if best_R is not None:
            self.R_cam2gripper = best_R
            self.t_cam2gripper = best_t.flatten()
            self.calibrated = True

            self.get_logger().info(f'Best method: {best_method}')
            self.get_logger().info(f'Translation (camera in wrist frame): {self.t_cam2gripper}')

            rvec = matrix_to_rodrigues(self.R_cam2gripper)
            angle = np.linalg.norm(rvec)
            axis = rvec / angle if angle > 1e-6 else np.array([0, 0, 1])
            self.get_logger().info(
                f'Rotation: {np.degrees(angle):.2f} deg around [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]'
            )
            return True
        else:
            self.get_logger().error('All calibration methods failed')
            return False

    def _compute_calibration_error(self, R_cam2gripper, t_cam2gripper) -> float:
        """Compute consistency error for the calibration."""
        errors = []
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

        for i in range(len(self.captures)):
            for j in range(i + 1, len(self.captures)):
                T_gi = np.eye(4)
                T_gi[:3, :3] = self.R_gripper2base[i]
                T_gi[:3, 3] = self.t_gripper2base[i]

                T_gj = np.eye(4)
                T_gj[:3, :3] = self.R_gripper2base[j]
                T_gj[:3, 3] = self.t_gripper2base[j]

                A = np.linalg.inv(T_gi) @ T_gj

                T_ti = np.eye(4)
                T_ti[:3, :3] = self.R_target2cam[i]
                T_ti[:3, 3] = self.t_target2cam[i]

                T_tj = np.eye(4)
                T_tj[:3, :3] = self.R_target2cam[j]
                T_tj[:3, 3] = self.t_target2cam[j]

                B = T_ti @ np.linalg.inv(T_tj)

                X = T_cam2gripper
                AX = A @ X
                XB = X @ B

                error = np.linalg.norm(AX - XB, 'fro')
                errors.append(error)

        return np.mean(errors) if errors else float('inf')

    def save_calibration(self):
        """Save hand-eye calibration to YAML file."""
        if not self.calibrated:
            self.get_logger().warn('No calibration to save')
            return False

        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(self.R_cam2gripper)
        quat = r.as_quat()

        data = {
            'frame_id': self.wrist_link,
            'child_frame_id': f'{self.arm_side}_wrist_camera',
            'translation': {
                'x': float(self.t_cam2gripper[0]),
                'y': float(self.t_cam2gripper[1]),
                'z': float(self.t_cam2gripper[2]),
            },
            'rotation': {
                'x': float(quat[0]),
                'y': float(quat[1]),
                'z': float(quat[2]),
                'w': float(quat[3]),
            },
            'rotation_matrix': {
                'rows': 3,
                'cols': 3,
                'data': self.R_cam2gripper.flatten().tolist()
            },
            'calibration_date': datetime.now().isoformat(),
            'num_captures': len(self.captures),
            'arm_side': self.arm_side,
            'wrist_link': self.wrist_link,
        }

        with open(self.output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        self.get_logger().info(f'Saved hand-eye calibration to {self.output_file}')
        return True

    def run_interactive(self):
        """Run interactive calibration with OpenCV window."""
        cv2.namedWindow('ChArUco Hand-Eye Calibration', cv2.WINDOW_NORMAL)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.current_frame is None:
                continue

            display = self.current_frame.copy()
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

            # Detect and draw markers
            marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)

            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

                # Try to get pose
                charuco_corners, charuco_ids, _, _ = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.charuco_board
                )

                if charuco_corners is not None and len(charuco_corners) >= 4:
                    cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

                    # Draw pose if we can estimate it
                    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners, charuco_ids, self.charuco_board, self.K, self.D, None, None
                    )
                    if success:
                        cv2.drawFrameAxes(display, self.K, self.D, rvec, tvec, 0.05)

                    status = f"DETECTED ({len(charuco_corners)} corners) - Press SPACE"
                    color = (0, 255, 0)
                else:
                    status = f"Partial detection ({len(marker_ids)} markers) - need more"
                    color = (0, 165, 255)
            else:
                status = "Searching for ChArUco board..."
                color = (0, 0, 255)

            # Joint status
            if self.current_joint_positions:
                joint_status = f"Joints: {len(self.current_joint_positions)} received"
            else:
                joint_status = "Joints: waiting..."

            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, joint_status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Captures: {len(self.captures)}/{self.min_captures}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.calibrated:
                cv2.putText(display, "CALIBRATED - Press S to save",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('ChArUco Hand-Eye Calibration', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                self.capture_frame()
            elif key == ord('c'):
                self.run_calibration()
            elif key == ord('s'):
                self.save_calibration()
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrator()

    try:
        node.run_interactive()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
