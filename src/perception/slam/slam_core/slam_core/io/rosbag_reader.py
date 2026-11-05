"""Reader for our D455 recordings: the ROS 2 bags (.db3) that slam_core.io.realsense records.

A .db3 is an SQLite database (tables `topics` and `messages`), so this reads it with
Python's sqlite3; no ROS or extra packages needed. For now it reads the IMU and the
factory camera-to-IMU extrinsics. Images are read with slam_core.io.realsense (play=...).

Export a recording's IMU next to a vo.py run (so the backend can use it without the
multi-GB recording):
  python -m slam_core.io.rosbag_reader ../data/bags/demo3.db3 ../prototype/out/demo3
writes imu.txt and imu_extrinsics.txt into the output folder.

Recording layout (librealsense): gyro and accel each have an imu/data topic
(sensor_msgs/Imu, values in SI units) and an imu/metadata topic (a string whose
"timestamp" is the camera's global clock in ms, the same clock as Frame.stamp). The
data messages' own header stamps only count from the start of the recording, so the
metadata is what lines the IMU up with the camera frames.
"""
import sqlite3
import struct

import numpy as np

from slam_core.types import ImuSample
from slam_core.types.pose import inv_T, make_T

GYRO = "/device_0/sensor_2/Gyro_0"
ACCEL = "/device_0/sensor_2/Accel_0"
IR = "/device_0/sensor_0/Infrared_1"


def _open(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _topic_id(db, name):
    row = db.execute("select id from topics where name=?", (name,)).fetchone()
    if row is None:
        raise ValueError(f"topic {name} not in the recording (recorded with imu off?)")
    return row[0]


# std_msgs/String in CDR: 4-byte encapsulation header, uint32 length (incl. NUL), bytes
def _string(data):
    n = struct.unpack_from("<I", data, 4)[0]
    return data[8:8 + n - 1].decode(errors="replace")


def _fields(text):
    return dict(kv.split("=", 1) for kv in text.split(";") if "=" in kv)


# sensor_msgs/Imu in CDR -> (angular_velocity, linear_acceleration). Offsets are relative
# to the payload after the 4-byte encapsulation header; doubles are 8-byte aligned.
def _imu_vectors(data):
    n = struct.unpack_from("<I", data, 4 + 8)[0]   # header: stamp (8 bytes), then frame_id
    o = (12 + n + 7) // 8 * 8
    v = struct.unpack_from("<4d9d3d9d3d9d", data, 4 + o)   # orientation, cov, gyro, cov, accel, cov
    return v[13:16], v[25:28]


# One IMU stream as (stamps in s on the camera's global clock, (N, 3) values).
def _read_stream(db, prefix, accel):
    data_id = _topic_id(db, prefix + "/imu/data")
    meta_id = _topic_id(db, prefix + "/imu/metadata")
    stamps = {bt: float(_fields(_string(d))["timestamp"]) / 1000.0
              for bt, d in db.execute("select timestamp, data from messages where topic_id=?", (meta_id,))}
    t, vals = [], []
    for bt, d in db.execute("select timestamp, data from messages where topic_id=? order by timestamp", (data_id,)):
        if bt in stamps:   # metadata and data of one sample share the bag timestamp
            gyro, acc = _imu_vectors(d)
            t.append(stamps[bt])
            vals.append(acc if accel else gyro)
    t, vals = np.array(t), np.array(vals, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(t, kind="stable")
    return t[order], vals[order]


def read_imu(path):
    """All IMU samples of a recording as ImuSamples at the gyro rate (accel interpolated
    onto the gyro timestamps), on the same clock as the camera frames."""
    with _open(path) as db:
        tg, gyro = _read_stream(db, GYRO, accel=False)
        ta, acc = _read_stream(db, ACCEL, accel=True)
    if len(tg) < 2 or len(ta) < 2:
        raise ValueError(f"{path}: not enough IMU samples ({len(tg)} gyro, {len(ta)} accel)")
    # The first accel sample can be stamped seconds early (before the camera's global
    # clock locks), so drop anything well before the gyro starts.
    keep = ta > tg[0] - 0.1
    ta, acc = ta[keep], acc[keep]
    # keep gyro samples inside the accel range, so interpolation never extrapolates
    inside = (tg >= ta[0]) & (tg <= ta[-1])
    tg, gyro = tg[inside], gyro[inside]
    acc_i = np.column_stack([np.interp(tg, ta, acc[:, k]) for k in range(3)])
    return [ImuSample(float(s), g, a) for s, g, a in zip(tg, gyro, acc_i)]


def read_imu_extrinsics(path):
    """T_cam_imu: the pose of the IMU in the left IR camera frame (p_cam = T_cam_imu @ p_imu),
    from the factory calibration stored in the recording.

    The recording stores the transform the other way (camera -> IMU, "tf/ref_0"), with
    a column-major rotation; checked against pyrealsense's get_extrinsics_to on demo3.
    """
    with _open(path) as db:
        ir = _fields(_string(db.execute("select data from messages where topic_id=?",
                                        (_topic_id(db, IR + "/tf/ref_0"),)).fetchone()[0]))
        imu = _fields(_string(db.execute("select data from messages where topic_id=?",
                                         (_topic_id(db, GYRO + "/tf/ref_0"),)).fetchone()[0]))
    if any(abs(float(x)) > 1e-9 for x in ir["translation"].split(",")):
        raise ValueError("expected the left IR camera to be the recording's reference frame")
    R = np.array([float(x) for x in imu["rotation"].split(",")]).reshape(3, 3).T   # column-major
    t = np.array([float(x) for x in imu["translation"].split(",")])
    return inv_T(make_T(R, t))   # stored: p_imu = R p_cam + t, we want camera <- IMU


def write_imu(out_dir, samples, T_cam_imu):
    """imu.txt (one sample per line) and imu_extrinsics.txt (T_cam_imu) into out_dir."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "imu.txt"), "w") as fh:
        fh.write("# stamp gx gy gz ax ay az   (s, rad/s, m/s^2; IMU frame = left IR optical axes: "
                 "x right, y down, z forward; accel interpolated onto gyro stamps)\n")
        for s in samples:
            fh.write(f"{s.stamp:.6f} {s.gyro[0]:.6f} {s.gyro[1]:.6f} {s.gyro[2]:.6f} "
                     f"{s.accel[0]:.6f} {s.accel[1]:.6f} {s.accel[2]:.6f}\n")
    with open(os.path.join(out_dir, "imu_extrinsics.txt"), "w") as fh:
        fh.write("# T_cam_imu: pose of the IMU in the left IR camera frame (p_cam = T_cam_imu @ p_imu), "
                 "D455 factory calibration, 4x4 row-major\n")
        for row in T_cam_imu:
            fh.write(" ".join(f"{v:.6f}" for v in row) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export a recording's IMU to imu.txt + imu_extrinsics.txt")
    ap.add_argument("recording", help=".db3 recorded by slam_core.io.realsense")
    ap.add_argument("out", help="output folder (e.g. the vo.py run folder)")
    args = ap.parse_args()

    samples = read_imu(args.recording)
    T = read_imu_extrinsics(args.recording)
    write_imu(args.out, samples, T)

    t = np.array([s.stamp for s in samples])
    g = np.array([s.gyro for s in samples])
    a = np.array([s.accel for s in samples])
    first = t < t[0] + 1.0
    print(f"{len(samples)} samples over {t[-1] - t[0]:.1f} s (~{(len(t) - 1) / (t[-1] - t[0]):.0f} Hz), "
          f"{t[0]:.3f} .. {t[-1]:.3f}")
    print(f"first second: |accel| {np.linalg.norm(a[first], axis=1).mean():.2f} m/s^2 (still = ~9.81), "
          f"|gyro| {np.linalg.norm(g[first], axis=1).mean():.3f} rad/s (still = ~0)")
    print(f"IMU in camera frame: t = {np.round(T[:3, 3], 4)} m")
    print(f"wrote {args.out}/imu.txt and imu_extrinsics.txt")
