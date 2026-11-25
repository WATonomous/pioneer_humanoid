#!/usr/bin/env python3
"""
Intrinsic calibration tool for HBVCAM wrist cameras using ChArUco board.

Captures ChArUco board images and computes camera intrinsics (K, D).
Run separately for each camera (left/right wrist).

ChArUco is better than plain checkerboard because it works even if
part of the board is occluded.

Usage:
  ros2 run wrist_stereo calibrate_intrinsics --ros-args \
    -p camera_topic:=/wrist_left/image_raw \
    -p output_file:=config/left_intrinsics.yaml

Controls:
  SPACE - capture current frame
  c     - run calibration on captured frames
  s     - save calibration to file
  q     - quit
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import yaml
import os
from datetime import datetime


class IntrinsicCalibrator(Node):
    def __init__(self):
        super().__init__('intrinsic_calibrator')

        # Parameters
        self.declare_parameter('camera_topic', '/wrist_left/image_raw')
        self.declare_parameter('output_file', 'left_intrinsics.yaml')
        self.declare_parameter('min_captures', 15)

        # ChArUco board parameters (from your printed board)
        # "ChArUco 9x6 DICT_6X5_100 square 25 mm marker 18 mm"
        self.declare_parameter('squares_x', 9)
        self.declare_parameter('squares_y', 6)
        self.declare_parameter('square_length', 0.025)   # 25mm in meters
        self.declare_parameter('marker_length', 0.018)   # 18mm in meters
        self.declare_parameter('dictionary', 'DICT_6X5_100')

        self.camera_topic = self.get_parameter('camera_topic').value
        self.output_file = self.get_parameter('output_file').value
        self.min_captures = self.get_parameter('min_captures').value

        self.squares_x = self.get_parameter('squares_x').value
        self.squares_y = self.get_parameter('squares_y').value
        self.square_length = self.get_parameter('square_length').value
        self.marker_length = self.get_parameter('marker_length').value
        dict_name = self.get_parameter('dictionary').value

        self.bridge = CvBridge()
        self.current_frame = None
        self.image_size = None

        # Set up ArUco dictionary and ChArUco board
        # Map string to OpenCV constant
        dict_map = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
            'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
            'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
            'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
            'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
            'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
            'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
            'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
            # Note: DICT_6X5_100 doesn't exist - it's likely DICT_6X6_100
            # The "6X5" in your board name refers to board dimensions, not dictionary
            'DICT_6X5_100': cv2.aruco.DICT_6X6_100,  # Assuming this mapping
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

        # Calibration data
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.captures = []

        # Calibration results
        self.K = None
        self.D = None
        self.calibrated = False

        # Subscribe to camera
        self.sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )

        self.get_logger().info(f'ChArUco Intrinsic Calibrator started')
        self.get_logger().info(f'  Camera topic: {self.camera_topic}')
        self.get_logger().info(f'  Board: {self.squares_x}x{self.squares_y}, square={self.square_length*1000:.0f}mm, marker={self.marker_length*1000:.0f}mm')
        self.get_logger().info(f'  Output: {self.output_file}')
        self.get_logger().info('Press SPACE to capture, C to calibrate, S to save, Q to quit')

    def image_callback(self, msg: Image):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.image_size is None:
                self.image_size = (self.current_frame.shape[1], self.current_frame.shape[0])
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def detect_charuco(self, frame):
        """Detect ChArUco corners in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers first
        marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)

        charuco_corners = None
        charuco_ids = None

        if marker_ids is not None and len(marker_ids) > 0:
            # Interpolate ChArUco corners from detected markers
            charuco_corners, charuco_ids, _, _ = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.charuco_board
            )

        return marker_corners, marker_ids, charuco_corners, charuco_ids, gray

    def capture_frame(self):
        """Capture current frame if ChArUco board is detected."""
        if self.current_frame is None:
            self.get_logger().warn('No frame available')
            return False

        frame = self.current_frame.copy()
        marker_corners, marker_ids, charuco_corners, charuco_ids, gray = self.detect_charuco(frame)

        if charuco_corners is not None and len(charuco_corners) >= 4:
            self.all_charuco_corners.append(charuco_corners)
            self.all_charuco_ids.append(charuco_ids)
            self.captures.append(frame.copy())
            self.get_logger().info(f'Captured frame {len(self.captures)}/{self.min_captures} ({len(charuco_corners)} corners)')
            return True
        else:
            corners_found = len(charuco_corners) if charuco_corners is not None else 0
            self.get_logger().warn(f'Not enough corners detected ({corners_found}), need at least 4')
            return False

    def run_calibration(self):
        """Run OpenCV camera calibration."""
        if len(self.captures) < self.min_captures:
            self.get_logger().warn(
                f'Need at least {self.min_captures} captures, have {len(self.captures)}'
            )
            return False

        self.get_logger().info('Running calibration...')

        # ChArUco calibration
        ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            self.all_charuco_corners,
            self.all_charuco_ids,
            self.charuco_board,
            self.image_size,
            None, None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )

        if ret:
            self.K = K
            self.D = D
            self.calibrated = True

            self.get_logger().info('Calibration successful!')
            self.get_logger().info(f'  RMS reprojection error: {ret:.4f} pixels')
            self.get_logger().info(f'  fx={K[0,0]:.2f}, fy={K[1,1]:.2f}')
            self.get_logger().info(f'  cx={K[0,2]:.2f}, cy={K[1,2]:.2f}')
            return True
        else:
            self.get_logger().error('Calibration failed')
            return False

    def save_calibration(self):
        """Save calibration to YAML file."""
        if not self.calibrated:
            self.get_logger().warn('No calibration to save - run calibration first')
            return False

        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data = {
            'camera_name': os.path.basename(self.output_file).replace('.yaml', ''),
            'image_width': self.image_size[0],
            'image_height': self.image_size[1],
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': self.K.flatten().tolist()
            },
            'distortion_model': 'rational_polynomial',
            'distortion_coefficients': {
                'rows': 1,
                'cols': len(self.D.flatten()),
                'data': self.D.flatten().tolist()
            },
            'calibration_date': datetime.now().isoformat(),
            'num_captures': len(self.captures),
            'charuco_board': {
                'squares_x': self.squares_x,
                'squares_y': self.squares_y,
                'square_length_m': self.square_length,
                'marker_length_m': self.marker_length
            }
        }

        with open(self.output_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        self.get_logger().info(f'Saved calibration to {self.output_file}')
        return True

    def run_interactive(self):
        """Run interactive calibration with OpenCV window."""
        cv2.namedWindow('ChArUco Calibration', cv2.WINDOW_NORMAL)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.current_frame is None:
                continue

            display = self.current_frame.copy()
            marker_corners, marker_ids, charuco_corners, charuco_ids, _ = self.detect_charuco(display)

            # Draw detected markers
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

            # Draw ChArUco corners
            if charuco_corners is not None and len(charuco_corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                status = f"DETECTED ({len(charuco_corners)} corners) - Press SPACE to capture"
                color = (0, 255, 0)
            else:
                status = "Searching for ChArUco board..."
                color = (0, 0, 255)

            # Draw status
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, f"Captures: {len(self.captures)}/{self.min_captures}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.calibrated:
                cv2.putText(display, "CALIBRATED - Press S to save",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('ChArUco Calibration', display)
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
    node = IntrinsicCalibrator()

    try:
        node.run_interactive()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
