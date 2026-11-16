#!/usr/bin/env python3
"""Score the live elevation map against the bag acceptance checks.

Run next to bag_pipeline.launch.py. When the bag's /clock stops, it prints a JSON report and
writes a top-down PNG plus the raw height grid (.npy) of the last map.

Checks:
  map       floor: the largest near-level plane at least 0.5 m under the camera, with its tilt
            and flatness; objects: blobs 4 to 60 cm above that plane (step, box, ramp) with their
            height and edge smear; nothing above camera + max_height_range
  odometry  a turn in place: yaw near 360 deg, small end-to-start distance
  gravity   angle between measured gravity and the odom z axis; the same angle in camera_link is
            the control (what odom would read without the IMU alignment)
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformListener

FLOOR_INLIER_M = 0.05
FLOOR_FLAT_M = 0.02
FLOOR_MAX_TILT_DEG = 10.0
RANSAC_ITERS = 300
FLOOR_BELOW_CAMERA_M = 0.5    # max_height_range clips wall tops into flat planes above the camera
OBJECT_MIN_M = 0.04
OBJECT_MAX_M = 0.6
OBJECT_MIN_AREA_M2 = 0.04
IMU_HZ = 200
IMU_EVERY = 20              # sample one IMU message in this many
IMU_SAMPLES_PER_S = IMU_HZ // IMU_EVERY


def grid_layer(msg: GridMap, layer: str) -> np.ndarray:
    """GridMap layer as a (rows, cols) array; grid_map stores data column-major."""
    data = msg.data[msg.layers.index(layer)]
    cols, rows = data.layout.dim[0].size, data.layout.dim[1].size
    return np.asarray(data.data, dtype=np.float32).reshape(cols, rows).T


def fit_floor(heights: np.ndarray, resolution: float, camera_z: float):
    rows, cols = np.nonzero(np.isfinite(heights))
    if rows.size < 3:
        return None
    centre = (np.array(heights.shape) - 1) / 2.0
    design = np.column_stack([(rows - centre[0]) * resolution, (cols - centre[1]) * resolution, np.ones(rows.size)])
    z = heights[rows, cols].astype(np.float64)
    max_slope = math.tan(math.radians(FLOOR_MAX_TILT_DEG))
    rng = np.random.default_rng(0)
    best_count, best = 0, None
    for _ in range(RANSAC_ITERS):
        pick = rng.choice(z.size, 3, replace=False)
        try:
            coef = np.linalg.solve(design[pick], z[pick])
        except np.linalg.LinAlgError:
            continue
        if math.hypot(coef[0], coef[1]) > max_slope or coef[2] > camera_z - FLOOR_BELOW_CAMERA_M:
            continue
        inliers = np.abs(design @ coef - z) <= FLOOR_INLIER_M
        if inliers.sum() > best_count:
            best_count, best = int(inliers.sum()), inliers
    if best is None:
        return None
    for _ in range(3):
        coef = np.linalg.lstsq(design[best], z[best], rcond=None)[0]
        best = np.abs(design @ coef - z) <= FLOOR_INLIER_M
    plane = np.fromfunction(
        lambda r, c: coef[0] * (r - centre[0]) * resolution + coef[1] * (c - centre[1]) * resolution + coef[2],
        heights.shape)
    return coef, plane, design[best] @ coef - z[best]


def objects_above(above: np.ndarray, resolution: float) -> list:
    mask = ((above > OBJECT_MIN_M) & (above < OBJECT_MAX_M)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centre = (np.array(above.shape) - 1) / 2.0
    objects = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] * resolution ** 2 < OBJECT_MIN_AREA_M2:
            continue
        blob = labels == i
        h = above[blob]
        top = float(np.percentile(h, 75))
        boundary = blob & ~cv2.erode(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        smeared = int(((h > 0.2 * top) & (h < 0.8 * top)).sum())
        row, col = centroids[i][1], centroids[i][0]
        objects.append({
            'height': round(top, 3),
            'area_m2': round(int(blob.sum()) * resolution ** 2, 2),
            'edge_smear': round(smeared / max(int(boundary.sum()), 1), 2),
            'centre_m': [round((row - centre[0]) * resolution, 2), round((col - centre[1]) * resolution, 2)],
        })
    return sorted(objects, key=lambda o: -o['area_m2'])


def analyze(heights: np.ndarray, resolution: float, camera_z: float, camera_z_max: float) -> dict:
    """camera_z: camera height at the end of the run; camera_z_max: highest camera height seen.

    max_height_range is measured from the camera at capture time, so no cell should sit more than
    max_height_range above camera_z_max.
    """
    valid = heights[np.isfinite(heights)]
    if valid.size == 0:
        return {'coverage': 0.0}
    report = {
        'coverage': round(valid.size / heights.size, 3),
        'max_above_highest_camera': round(float(valid.max() - camera_z_max), 3),
        'floor': None,
        'objects': [],
    }
    fit = fit_floor(heights, resolution, camera_z)
    if fit is None:
        return report
    coef, plane, residuals = fit
    report['floor'] = {
        'below_camera': round(camera_z - float(coef[2]), 3),
        'area_m2': round(residuals.size * resolution ** 2, 2),
        'tilt_deg': round(math.degrees(math.atan(math.hypot(coef[0], coef[1]))), 2),
        'within_2cm': round(float(np.mean(np.abs(residuals) <= FLOOR_FLAT_M)), 3),
    }
    report['objects'] = objects_above(heights - plane, resolution)
    return report


def rotate(q, v: np.ndarray) -> np.ndarray:
    u = np.array([q.x, q.y, q.z])
    return v + 2.0 * q.w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def top_down_png(heights: np.ndarray, path: str) -> None:
    finite = np.isfinite(heights)
    lo, hi = np.percentile(heights[finite], [1, 99]) if finite.any() else (0.0, 1.0)
    scaled = np.clip((heights - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((np.nan_to_num(scaled) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~finite] = (40, 40, 40)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, f'{lo:.2f} m .. {hi:.2f} m', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(path, img)


class MapChecker(Node):
    def __init__(self, map_topic: str, imu_topic: str):
        super().__init__('terrain_map_checker', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.map_msg = None
        self.poses = []  # (x, y, z, yaw) of camera_link in odom, sampled at 10 Hz
        self.imu_samples = []  # (stamp, frame_id, accel); resolved against TF after the run
        self.imu_count = 0
        self.tf_buffer = Buffer(cache_time=Duration(seconds=300))
        TransformListener(self.tf_buffer, self)
        self.create_subscription(GridMap, map_topic, self.on_map, 10)
        self.create_subscription(Imu, imu_topic, self.on_imu, qos_profile_sensor_data)
        self.create_timer(0.1, self.sample_pose)

    def on_map(self, msg: GridMap) -> None:
        self.map_msg = msg

    def on_imu(self, msg: Imu) -> None:
        self.imu_count += 1
        if self.imu_count % IMU_EVERY:
            return
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.imu_samples.append((msg.header.stamp, msg.header.frame_id, accel))

    def sample_pose(self) -> None:
        try:
            t = self.tf_buffer.lookup_transform('odom', 'camera_link', rclpy.time.Time())
        except Exception:
            return
        p, q = t.transform.translation, t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.poses.append((p.x, p.y, p.z, yaw))

    def odometry_report(self) -> dict:
        if len(self.poses) < 2:
            return {'pose_samples': len(self.poses)}
        xyz = np.array([p[:3] for p in self.poses])
        yaws = np.unwrap([p[3] for p in self.poses])
        return {
            'pose_samples': len(self.poses),
            'yaw_traversed_deg': round(float(np.degrees(yaws[-1] - yaws[0])), 1),
            'end_to_start_m': round(float(np.linalg.norm(xyz[-1] - xyz[0])), 3),
            'camera_z_range_m': [round(float(xyz[:, 2].min()), 3), round(float(xyz[:, 2].max()), 3)],
            'max_xy_excursion_m': round(float(np.max(np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1))), 3),
        }

    def gravity_report(self) -> dict:
        report = {}
        for frame in ('odom', 'camera_link'):
            errors = []
            for stamp, imu_frame, accel in self.imu_samples:
                try:
                    t = self.tf_buffer.lookup_transform(frame, imu_frame, stamp)
                except Exception:
                    continue
                up = rotate(t.transform.rotation, accel)
                errors.append(math.degrees(math.acos(np.clip(up[2] / np.linalg.norm(up), -1.0, 1.0))))
            report[frame] = {
                'samples': len(errors),
                'median_deg': round(float(np.median(errors)), 2),
                'p95_deg': round(float(np.percentile(errors, 95)), 2),
                'first_s_median_deg': round(float(np.median(errors[:IMU_SAMPLES_PER_S])), 2),
                'last_s_median_deg': round(float(np.median(errors[-IMU_SAMPLES_PER_S:])), 2),
            } if errors else {'samples': 0, 'imu_seen': len(self.imu_samples)}
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--topic', default='/elevation_mapping_node/elevation_map_raw')
    parser.add_argument('--layer', default='elevation')
    parser.add_argument('--imu', default='/camera/camera/imu')
    parser.add_argument('--idle', type=float, default=5.0, help='wall seconds of stalled /clock that end the run')
    parser.add_argument('--timeout', type=float, default=180.0, help='wall seconds before giving up')
    parser.add_argument('--out', default='/tmp/terrain_map_check')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = MapChecker(args.topic, args.imu)
    deadline = time.monotonic() + args.timeout
    sim_now, sim_changed = None, time.monotonic()
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = node.get_clock().now().nanoseconds
        if now != sim_now:
            sim_now, sim_changed = now, time.monotonic()
        if node.map_msg is not None and time.monotonic() - sim_changed > args.idle:
            break

    if node.map_msg is None:
        raise SystemExit(f'no map received on {args.topic} within {args.timeout:.0f} s')
    heights = grid_layer(node.map_msg, args.layer)
    camera_z = node.poses[-1][2] if node.poses else 0.0
    camera_z_max = max(p[2] for p in node.poses) if node.poses else 0.0
    os.makedirs(args.out, exist_ok=True)
    png = os.path.join(args.out, f'{args.layer}_top_down.png')
    top_down_png(heights, png)
    np.save(os.path.join(args.out, f'{args.layer}.npy'), heights)
    report = {
        'map': analyze(heights, node.map_msg.info.resolution, camera_z, camera_z_max),
        'odometry': node.odometry_report(),
        'gravity_error': node.gravity_report(),
        'png': png,
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
