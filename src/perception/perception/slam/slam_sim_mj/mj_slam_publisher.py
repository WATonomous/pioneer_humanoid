#!/usr/bin/env python3
"""Publish a moving simulated RealSense D455 from MuJoCo as ROS 2 topics.

Feeds RTAB-Map from simulation instead of a physical camera. Topic names, encodings
and frame ids match what `realsense2_camera` publishes on real hardware, so the SLAM
side cannot tell which one it is talking to -- the same launch file and the same
imu_filter_madgwick node from src/perception/perception/slam/slam-quickstart.md work
against either.

    source /opt/ros/jazzy/setup.bash
    MUJOCO_GL=egl .venv/bin/python mj_slam_publisher.py --path corridor

Everything runs in ONE process: ROS 2 Jazzy and the MuJoCo wheel are both Python
3.12, so the renderer and rclpy live in the same interpreter and no image data has to
cross a process or container boundary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imu import imu_sample  # noqa: E402
from mj_camera import MjCamera  # noqa: E402
from orbit_path import corridor_pose, orbit_pose, rect_perimeter  # noqa: E402

import rclpy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, Imu  # noqa: E402

# Copied from the real driver so the SLAM side needs no knowledge of the source.
TOPIC_COLOR = "/camera/camera/color/image_raw"
TOPIC_DEPTH = "/camera/camera/depth/image_rect_raw"
TOPIC_INFO = "/camera/camera/color/camera_info"
TOPIC_IMU = "/camera/camera/imu"

# Images and IMU are stamped in the OPTICAL frame (+Z boresight, +X right, +Y down).
# The fixed rotation to camera_link is published by the launch file.
OPTICAL_FRAME = "camera_color_optical_frame"

SCENE_DEFAULT = Path(__file__).resolve().parent / "scene" / "room.xml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=SCENE_DEFAULT,
                   help="Scene MJCF. Swap this for a downloaded room; nothing else "
                        "in the pipeline knows about the scene.")
    p.add_argument("--path", default="corridor", choices=["corridor", "orbit"],
                   help="corridor: walk a building loop once, which is what produces "
                        "a readable floor plan. orbit: circle a point -- the tighter "
                        "SLAM test trajectory, loop closure fires every lap.")
    p.add_argument("--period", type=float, default=100.0,
                   help="Seconds per lap. Higher is slower and easier on odometry. "
                        "100 s over the 37.9 m loop is 0.38 m/s, which runs "
                        "reliably. Odometry quality falls off sharply above ~0.5 m/s, "
                        "and does so without logging obvious errors, so lower this "
                        "only deliberately.")
    p.add_argument("--laps", type=float, default=1.0,
                   help="Stop after this many laps. One lap is what a person walking "
                        "a building actually does, and one solid loop is enough.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--rate", type=float, default=30.0, help="Camera rate (Hz).")
    p.add_argument("--imu_decim", type=int, default=7,
                   help="IMU samples per camera frame. Expressed as a ratio rather "
                        "than an absolute rate so the two can never be set to values "
                        "that do not divide. 7 x 30 Hz = 210 Hz, near the real "
                        "D455's 200 Hz gyro.")
    p.add_argument("--height_m", type=float, default=1.5,
                   help="Camera height above the floor. KEEP IN SYNC with the launch "
                        "file's camera_height, or the 2D grid filters at the wrong datum.")
    p.add_argument("--half_x", type=float, default=6.0)
    p.add_argument("--half_y", type=float, default=4.0)
    p.add_argument("--corner", type=float, default=1.2)
    p.add_argument("--center_x", type=float, default=0.0)
    p.add_argument("--center_y", type=float, default=0.0)
    p.add_argument("--scan_yaw", type=float, default=0.35,
                   help="Head sweep left/right (rad). A person glances around while "
                        "walking, and that is what fills in the walls instead of a "
                        "narrow band straight ahead.")
    p.add_argument("--scan_pitch", type=float, default=0.12,
                   help="Head sweep up/down (rad). Keep modest -- large values stare "
                        "at blank ceiling and starve feature matching.")
    p.add_argument("--record", type=Path, default=None,
                   help="Also write the camera's point of view to this MP4. Encoding "
                        "runs in the publish loop, so check the reported camera rate "
                        "afterwards -- a run that could not keep up is a run whose "
                        "timestamps no longer match its motion.")
    p.add_argument("--radius", type=float, default=2.0, help="--path orbit only.")
    p.add_argument("--target_height", type=float, default=0.6, help="--path orbit only.")
    return p.parse_args()


def make_camera_info(k: np.ndarray, width: int, height: int) -> CameraInfo:
    """Built from the matrix the renderer actually used, never recomputed from the
    constructor arguments, so intrinsics cannot silently disagree with the images."""
    msg = CameraInfo()
    msg.width, msg.height = width, height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5  # a pinhole render has no lens distortion
    msg.k = [float(v) for v in k.reshape(-1)]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def main() -> None:
    args = parse_args()

    decim = args.imu_decim
    imu_rate = args.rate * decim

    center = np.array([args.center_x, args.center_y, 0.0])
    look_at = center + np.array([0.0, 0.0, args.target_height])

    if args.path == "corridor":
        def pose_fn(t: float):
            return corridor_pose(
                t, center, half_x=args.half_x, half_y=args.half_y, corner=args.corner,
                height=args.height_m, period=args.period,
                scan_yaw=args.scan_yaw, scan_pitch=args.scan_pitch,
            )
    else:
        def pose_fn(t: float):
            return orbit_pose(
                t, center, radius=args.radius, height=args.height_m,
                period=args.period, target=look_at,
            )

    camera = MjCamera(args.scene, args.width, args.height)
    bridge = CvBridge()

    writer = None
    if args.record:
        import imageio  # bundled ffmpeg; the system has none and needs no sudo

        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.record), fps=args.rate, quality=8,
                                    macro_block_size=1)

    rclpy.init()
    node = rclpy.create_node("mj_slam_publisher")

    # RELIABLE is compatible with both reliable and best-effort subscribers.
    # BEST_EFFORT here would be silently incompatible with rtabmap's default
    # reliable subscription: the topics would list, and no data would ever arrive.
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)
    pub_color = node.create_publisher(Image, TOPIC_COLOR, qos)
    pub_depth = node.create_publisher(Image, TOPIC_DEPTH, qos)
    pub_info = node.create_publisher(CameraInfo, TOPIC_INFO, qos)
    pub_imu = node.create_publisher(Imu, TOPIC_IMU, qos)

    k = camera.intrinsic_matrix
    hfov = 2.0 * np.degrees(np.arctan(0.5 * args.width / k[0, 0]))
    node.get_logger().info(
        f"scene={args.scene.name} {args.width}x{args.height} "
        f"fx={k[0, 0]:.2f} -> HFOV {hfov:.1f} deg"
    )
    if args.path == "corridor":
        lap = rect_perimeter(args.half_x, args.half_y, args.corner)
        node.get_logger().info(
            f"corridor loop: {lap:.1f} m/lap at {lap / args.period:.2f} m/s, "
            f"{args.laps:g} lap(s) = {args.laps * args.period:.0f}s"
        )
    node.get_logger().info(f"camera {args.rate:g} Hz, imu {imu_rate:g} Hz")

    imu_dt = 1.0 / imu_rate
    duration = args.laps * args.period
    sim_time = 0.0
    step = 0
    frames = 0
    start_wall = time.perf_counter()

    try:
        while rclpy.ok() and sim_time < duration:
            stamp = node.get_clock().now().to_msg()

            # Pose advances by SIMULATION time, never wall-clock. If rendering ever
            # runs slower than real time, a wall-clock path teleports the camera
            # between frames and visual odometry loses tracking immediately.
            position, quat = pose_fn(sim_time)

            omega, accel = imu_sample(pose_fn, sim_time)
            imu_msg = Imu()
            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = OPTICAL_FRAME
            imu_msg.angular_velocity.x = float(omega[0])
            imu_msg.angular_velocity.y = float(omega[1])
            imu_msg.angular_velocity.z = float(omega[2])
            imu_msg.linear_acceleration.x = float(accel[0])
            imu_msg.linear_acceleration.y = float(accel[1])
            imu_msg.linear_acceleration.z = float(accel[2])
            # -1 in the first covariance slot is the ROS convention for "this field is
            # not provided". Orientation is left unset on purpose: madgwick's job is to
            # estimate it from gyro+accel, exactly as it does on the real camera.
            imu_msg.orientation_covariance[0] = -1.0
            pub_imu.publish(imu_msg)

            if step % decim == 0:
                camera.set_pose(position, quat)
                rgb, depth_m = camera.render()

                # 16UC1 millimetres is what realsense2_camera actually puts on this
                # topic. Matching it keeps the SLAM side identical between sim and
                # hardware, and halves the bytes per frame against 32FC1.
                depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

                color_msg = bridge.cv2_to_imgmsg(np.ascontiguousarray(rgb), "rgb8")
                depth_msg = bridge.cv2_to_imgmsg(np.ascontiguousarray(depth_mm), "16UC1")
                info_msg = make_camera_info(k, args.width, args.height)

                # One render of one camera produced all three, so they share a stamp
                # exactly. That is why the launch file can set approx_sync:=false.
                for msg in (color_msg, depth_msg, info_msg):
                    msg.header.stamp = stamp
                    msg.header.frame_id = OPTICAL_FRAME

                pub_color.publish(color_msg)
                pub_depth.publish(depth_msg)
                pub_info.publish(info_msg)
                if writer is not None:
                    writer.append_data(rgb)
                frames += 1

            sim_time += imu_dt
            step += 1

            # Pace against an ABSOLUTE schedule, not `imu_dt - elapsed`. The pose
            # advances by simulation time while the stamps come from the wall clock,
            # so the two must not drift apart: per-iteration sleeps each round up by
            # the OS timer granularity, and at 210 Hz that compounded into a measured
            # 28.4 Hz against a 30 Hz target -- a 5% shortfall that reads to the IMU
            # filter as the camera turning 5% slower than the images show.
            deadline = start_wall + (step + 1) * imu_dt
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass

    if writer is not None:
        writer.close()
        node.get_logger().info(f"wrote {args.record}")

    elapsed = time.perf_counter() - start_wall
    node.get_logger().info(
        f"done: {frames} frames in {elapsed:.1f}s "
        f"({frames / elapsed:.1f} Hz camera, sim time {sim_time:.1f}s)"
    )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
