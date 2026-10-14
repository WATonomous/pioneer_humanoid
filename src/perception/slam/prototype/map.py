#!/usr/bin/env python3
import argparse
import os

import numpy as np

from slam_core.types.pose import from_tum
from viz import R_WORLD_ODOM


# turns every step-th depth pixel in [zmin, zmax] into a 3D point in the camera frame
# (optical: x right, y down, z forward), plus the IR brightness at that pixel
def depth_to_points(depth, image, K, step=2, zmin=0.3, zmax=3.0):
    z = depth[::step, ::step]
    v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    ok = (z > zmin) & (z < zmax)
    z, u, v = z[ok], u[ok] * step, v[ok] * step

    # pinhole back-projection: x = (u - cx) / fx * z, same for y
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.column_stack([x, y, z]).astype(np.float32), image[::step, ::step][ok].astype(np.float32)


# moves points by the 4x4 pose T
def transform(T, pts):
    return pts @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)


# keeps one point per voxel cube: the mean position and brightness of the points inside it
def voxel_downsample(pts, vals, voxel):
    if len(pts) == 0:
        return pts, vals

    # gives every point the integer index of its voxel, then groups points that share one
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()

    # sums positions and brightness per voxel, then divides by the count to get the mean
    out = np.zeros((len(counts), 3), np.float64)
    np.add.at(out, inv, pts)
    v = np.bincount(inv, weights=vals, minlength=len(counts))
    return (out / counts[:, None]).astype(np.float32), (v / counts).astype(np.float32)


# writes the points as a binary PLY with the brightness as gray RGB (opens in MeshLab / CloudCompare)
def write_ply(path, pts, gray):
    g = np.clip(gray, 0, 255).astype(np.uint8)
    data = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    data["red"] = data["green"] = data["blue"] = g
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        fh.write(data.tobytes())


# builds the whole map: each keyframe's depth placed at its pose (T_odom_kf, in keyframe id order).
# points come out in the z-up world frame, or in the odom frame with in_world=False
def build_map(images, poses, step, zmin, zmax, voxel, in_world=True):
    K = images["K"]
    world = np.eye(4)
    if in_world:
        world[:3, :3] = R_WORLD_ODOM
    all_pts, all_val = [], []
    for i, T in enumerate(poses):
        # depth was saved as uint16 mm, so convert back to meters
        depth = images[f"depth_{i}"].astype(np.float32) / 1000.0
        pts, val = depth_to_points(depth, images[f"image_{i}"], K, step, zmin, zmax)
        # downsamples each keyframe first, which saves memory
        pts, val = voxel_downsample(transform(world @ T, pts), val, voxel)
        all_pts.append(pts)
        all_val.append(val)

    # downsamples again so overlapping keyframes don't double up
    return voxel_downsample(np.concatenate(all_pts), np.concatenate(all_val), voxel)


# writes an MCAP with the optimized result on the same topics vo.py --mcap records
# (/vo/map, /vo/trajectory, /vo/keyframes, /tf), timestamped just after the run's last frame.
# open it in Foxglove together with the vo.py recording: playback shows the live,
# drifting map, then it snaps to the corrected one
def write_reveal(path, run, images, poses, args):
    import foxglove
    from foxglove import messages as m
    from slam_core.types.pose import rot_to_quat

    # stamps everything 1 s after the last frame so it plays right after the live recording
    frames = np.loadtxt(os.path.join(run, "frames.txt"), ndmin=2)
    t_end = frames[-1, 0] + 1.0
    stamp, t_ns = m.Timestamp.from_epoch_secs(t_end), int(t_end * 1e9)

    # builds the map in the odom frame, since that's what the live recording uses
    pts, gray = build_map(images, poses, args.step, args.zmin, args.zmax, args.voxel, in_world=False)
    traj = np.loadtxt(os.path.join(run, "trajectory_after.txt"), ndmin=2)
    loops = np.loadtxt(os.path.join(run, "loops.txt"), ndmin=2) if os.path.exists(os.path.join(run, "loops.txt")) else []

    # small helpers to turn numpy values into Foxglove message parts
    def p3(t):
        return m.Point3(x=float(t[0]), y=float(t[1]), z=float(t[2]))

    def pose(T):
        q = rot_to_quat(T[:3, :3])
        return m.Pose(position=m.Vector3(x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3])),
                      orientation=m.Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))

    # the corrected map as a point cloud (x, y, z, intensity as 4 floats per point)
    writer = foxglove.open_mcap(path, allow_overwrite=True)
    F = m.PackedElementFieldNumericType.Float32
    fields = [m.PackedElementField(name=n, offset=4 * i, type=F) for i, n in enumerate(("x", "y", "z", "intensity"))]
    foxglove.log("/vo/map", m.PointCloud(timestamp=stamp, frame_id="odom", point_stride=16, fields=fields,
                                         data=np.column_stack([pts, gray]).astype(np.float32).tobytes()), log_time=t_ns)

    # the corrected trajectory as a blue line
    line = m.LinePrimitive(type=m.LinePrimitiveLineType.LineStrip, thickness=3, scale_invariant=True,
                           points=[p3(t) for t in traj[:, 1:4]], color=m.Color(r=0.2, g=0.6, b=1.0, a=1.0))
    foxglove.log("/vo/trajectory", m.SceneUpdate(entities=[m.SceneEntity(timestamp=stamp, frame_id="odom",
                                                                          id="trajectory", lines=[line])]), log_time=t_ns)

    # the corrected keyframes as green cubes, with a yellow line between each loop pair
    cubes = [m.CubePrimitive(pose=pose(T), size=m.Vector3(x=0.04, y=0.04, z=0.04), color=m.Color(r=0.3, g=1.0, b=0.3, a=1.0))
             for T in poses]
    lines = []
    if len(loops):
        lp = [p3(poses[int(i)][:3, 3]) for row in loops for i in (row[0], row[1])]
        lines.append(m.LinePrimitive(type=m.LinePrimitiveLineType.LineList, thickness=4, scale_invariant=True,
                                     points=lp, color=m.Color(r=1.0, g=0.9, b=0.0, a=1.0)))
    foxglove.log("/vo/keyframes", m.SceneUpdate(entities=[m.SceneEntity(timestamp=stamp, frame_id="odom",
                                                                         id="keyframes", cubes=cubes, lines=lines)]),
                 log_time=t_ns)

    # moves the camera to its optimized final pose too, and clears the live depth cloud
    T_last = from_tum(traj[-1, 1:8])
    q = rot_to_quat(T_last[:3, :3])
    tf = m.FrameTransform(timestamp=stamp, parent_frame_id="odom", child_frame_id="camera",
                          translation=m.Vector3(x=float(T_last[0, 3]), y=float(T_last[1, 3]), z=float(T_last[2, 3])),
                          rotation=m.Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))
    foxglove.log("/tf", m.FrameTransforms(transforms=[tf]), log_time=t_ns)
    foxglove.log("/vo/cloud", m.PointCloud(timestamp=stamp, frame_id="camera", point_stride=16, fields=fields, data=b""),
                 log_time=t_ns)
    writer.close()
    print(f"wrote {path}: optimized map ({len(pts)} points), trajectory, {len(poses)} keyframes, {len(loops)} loops")


# reads a TUM pose file (kf_vo.txt or kf_after.txt) into a list of 4x4 poses, dropping the stamps
def read_poses(path):
    with open(path) as fh:
        return [from_tum([float(x) for x in line.split()[1:8]]) for line in fh if line.strip()]


# serves the map on ws://localhost:port until Ctrl-C (display frame: world)
def show_foxglove(pts, gray, path, kf_poses, port=8765):
    import time
    import foxglove
    from foxglove import messages as m
    server = foxglove.start_server(name="slam map", port=port)

    # packs the map as 4 floats per point (x, y, z, intensity)
    F = m.PackedElementFieldNumericType.Float32
    fields = [m.PackedElementField(name=n, offset=4 * i, type=F) for i, n in enumerate(("x", "y", "z", "intensity"))]
    data = np.column_stack([pts, gray]).astype(np.float32).tobytes()

    # the trajectory (every frame) as a line and keyframes as cubes, both already in world
    def p3(t):
        return m.Point3(x=float(t[0]), y=float(t[1]), z=float(t[2]))
    line = m.LinePrimitive(type=m.LinePrimitiveLineType.LineStrip, thickness=3, scale_invariant=True,
                           points=[p3(t) for t in path], color=m.Color(r=0.2, g=0.6, b=1.0, a=1.0))
    cubes = [m.CubePrimitive(pose=m.Pose(position=m.Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                                         orientation=m.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
                             size=m.Vector3(x=0.04, y=0.04, z=0.04), color=m.Color(r=0.3, g=1.0, b=0.3, a=1.0))
             for t in kf_poses]
    print(f"foxglove: connect to ws://localhost:{port}; 3D panel display frame 'world', turn on /map and /trajectory. Ctrl-C to quit")

    # Foxglove only offers frames it has seen in /tf, so publish one: world -> map (identity)
    ident = m.FrameTransform(parent_frame_id="world", child_frame_id="map",
                             translation=m.Vector3(x=0.0, y=0.0, z=0.0), rotation=m.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))

    # re-sends everything every second so a client that connects later still gets it
    try:
        while True:
            ts = m.Timestamp.from_epoch_secs(time.time())
            foxglove.log("/tf", m.FrameTransforms(transforms=[ident]))
            foxglove.log("/map", m.PointCloud(timestamp=ts, frame_id="world", point_stride=16, fields=fields, data=data))
            ent = m.SceneEntity(timestamp=ts, frame_id="world", id="trajectory", lines=[line], cubes=cubes)
            foxglove.log("/trajectory", m.SceneUpdate(entities=[ent]))
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="vo.py output directory")
    ap.add_argument("--poses", choices=["vo", "after"], default="vo",
                    help="vo: raw VO keyframe poses (kf_vo.txt); after: pose_graph.py output (kf_after.txt)")
    ap.add_argument("--voxel", type=float, default=0.03, help="m, map resolution")
    ap.add_argument("--step", type=int, default=2, help="use every Nth depth pixel")
    ap.add_argument("--zmin", type=float, default=0.3, help="m")
    ap.add_argument("--zmax", type=float, default=3.0, help="m (D455 depth gets noisy beyond ~3 m)")
    ap.add_argument("--foxglove", action="store_true", help="also show the map in Foxglove")
    ap.add_argument("--mcap", help="write the 'reveal' MCAP for a demo video (needs pose_graph.py's output)")
    args = ap.parse_args()

    # loads the keyframe images and the chosen poses, and checks they come from the same run
    images = np.load(os.path.join(args.run, "keyframe_images.npz"))
    poses = read_poses(os.path.join(args.run, "kf_vo.txt" if args.poses == "vo" else "kf_after.txt"))
    n_kf = sum(1 for k in images.files if k.startswith("depth_"))
    if len(poses) != n_kf:
        raise SystemExit(f"{len(poses)} poses but {n_kf} keyframe images: rerun vo.py (and pose_graph.py)")

    # builds the map and saves it as map.ply in the run directory
    pts, gray = build_map(images, poses, args.step, args.zmin, args.zmax, args.voxel)
    out = os.path.join(args.run, "map.ply")
    write_ply(out, pts, gray)
    ext = pts.max(0) - pts.min(0) if len(pts) else np.zeros(3)
    print(f"{n_kf} keyframes -> {len(pts)} points at {args.voxel * 100:.0f} cm, "
          f"extent {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} m (world: x forward, y left, z up)")
    print(f"wrote {out}")

    # the reveal MCAP only makes sense with the optimized poses
    if args.mcap:
        if args.poses != "after":
            raise SystemExit("--mcap shows the optimized result: run pose_graph.py, then use --poses after")
        write_reveal(args.mcap, args.run, images, poses, args)

    if args.foxglove:
        # uses the same poses as the map: the optimized trajectory with --poses after, else the raw VO
        traj = os.path.join(args.run, "trajectory_after.txt" if args.poses == "after" else "trajectory_vo.txt")
        # rotates odom positions into the z-up world frame
        to_world = lambda P: P @ R_WORLD_ODOM.T   # noqa: E731
        path = to_world(np.loadtxt(traj, ndmin=2)[:, 1:4]) if os.path.exists(traj) else np.zeros((0, 3))
        kf_pos = to_world(np.array([T[:3, 3] for T in poses]))
        show_foxglove(pts, gray, path, kf_pos)


if __name__ == "__main__":
    main()
