"""Publishes the real wrist camera, rectified to the 640x480 pinhole the sim wrist cameras render.

    ~/image_rect   sensor_msgs/Image, rgb8, 640x480 (what policies and recordings should use)
    ~/camera_info  sensor_msgs/CameraInfo for image_rect: the pinhole K, zero distortion

Run with `ros2 launch wrist_camera wrist_camera.launch.py` (topics under /wrist_cam/), and view it
with `ros2 run rqt_image_view rqt_image_view /wrist_cam/image_rect`. See ../README.md.
"""
import array
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from wrist_cam import WristCam, open_camera  # src/perception/camera_calibration (pip install -e)


class WristCameraNode(Node):
    def __init__(self):
        super().__init__('wrist_cam')
        device = self.declare_parameter('device', '/dev/video0').value
        fps = self.declare_parameter('fps', 30).value
        self.frame_id = self.declare_parameter('frame_id', 'wrist_cam_optical').value

        self.cam = WristCam()
        self.cap = open_camera(int(device) if str(device).isdigit() else device, self.cam, fps)
        # Reliable (plain depth): the recorder subscribes reliable, and a best-effort publisher
        # would never connect to it. Reliable still serves best-effort viewers.
        self.image_pub = self.create_publisher(Image, '~/image_rect', 10)
        self.info_pub = self.create_publisher(CameraInfo, '~/camera_info', 10)
        self.info = self._camera_info()

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        w, h = self.cam.size
        self.get_logger().info(f'{device}: {self.cam.raw_size[0]}x{self.cam.raw_size[1]} -> '
                               f'{w}x{h} pinhole, f = {self.cam.K[0, 0]:g} px')

    def _camera_info(self) -> CameraInfo:
        info = CameraInfo()
        info.header.frame_id = self.frame_id
        info.width, info.height = self.cam.size
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [float(v) for v in self.cam.K.ravel()]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [info.k[0], 0.0, info.k[2], 0.0, 0.0, info.k[4], info.k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _capture_loop(self):
        # cap.read() blocks until the next frame, so it runs off the executor thread.
        failures = 0
        while self.running and rclpy.ok():
            ok, raw = self.cap.read()
            stamp = self.get_clock().now().to_msg()
            if not ok:
                failures += 1
                if failures == 30:
                    self.get_logger().error('no frames from the camera (unplugged, or in use?)')
                time.sleep(0.05)  # a missing camera fails instantly; don't spin
                continue
            failures = 0
            rgb = cv2.cvtColor(self.cam.rectify(raw), cv2.COLOR_BGR2RGB)

            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.height, msg.width = rgb.shape[:2]
            msg.encoding = 'rgb8'
            msg.step = msg.width * 3
            # array, not bytes: rclpy checks a bytes value element by element (~37 ms per frame).
            msg.data = array.array('B', rgb.tobytes())
            self.info.header.stamp = stamp
            self.image_pub.publish(msg)
            self.info_pub.publish(self.info)

    def destroy_node(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WristCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
