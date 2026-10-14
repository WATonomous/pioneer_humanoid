import numpy as np

from slam_core.types.pose import rot_to_quat

# rotates odom (optical: x right, y down, z forward) into world (x forward, y left, z up) so the scene shows upright
R_WORLD_ODOM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)


# live 3D view of vo.py in Foxglove, sent to a websocket, an MCAP file, or both
class FoxgloveViz:
    # server: live websocket for the Foxglove app. mcap: also record everything to this file
    def __init__(self, port=8765, cloud_every=3, cloud_step=4, map_voxel=0.05, server=True, mcap=None):
        import foxglove
        from foxglove import messages as m
        self.fg, self.m = foxglove, m
        self.server = foxglove.start_server(name="slam vo prototype", port=port) if server else None
        self.writer = foxglove.open_mcap(mcap, allow_overwrite=True) if mcap else None
        self.cloud_every = cloud_every   # send the point cloud every N frames (it's the heavy one)
        self.cloud_step = cloud_step     # and only every Nth pixel in x and y
        self.n = 0
        self.path = []                   # positions of every tracked frame, odom frame
        self.n_kf = -1
        self.map_voxel = map_voxel       # live map resolution (map.py makes the fine one)
        self.map_pts = np.zeros((0, 3), np.float32)
        self.map_val = np.zeros(0, np.float32)
        if server:
            print(f"foxglove: connect to ws://localhost:{port}")
        if mcap:
            print(f"foxglove: recording to {mcap}")

    # splits a 4x4 pose into the Vector3 and Quaternion Foxglove messages use
    def _parts(self, T):
        m = self.m
        qx, qy, qz, qw = (float(v) for v in rot_to_quat(T[:3, :3]))
        x, y, z = (float(v) for v in T[:3, 3])
        return m.Vector3(x=x, y=y, z=z), m.Quaternion(x=qx, y=qy, z=qz, w=qw)

    # 4x4 pose -> Pose message
    def _pose(self, T):
        t, q = self._parts(T)
        return self.m.Pose(position=t, orientation=q)

    # 4x4 pose -> FrameTransform message from parent to child
    def _tf(self, stamp, parent, child, T):
        t, q = self._parts(T)
        return self.m.FrameTransform(timestamp=stamp, parent_frame_id=parent, child_frame_id=child,
                                     translation=t, rotation=q)

    # sends one frame's worth of data
    # f: the Frame, res: vo.process(f), vo: the VO, vis: the debug image (BGR)
    def update(self, f, res, vo, vis):
        m, log = self.m, self.fg.log
        stamp = m.Timestamp.from_epoch_secs(f.stamp)
        t_ns = int(f.stamp * 1e9)
        self.n += 1

        # publishes world -> odom (the fixed upright rotation) and odom -> camera (the current VO pose)
        world = np.eye(4)
        world[:3, :3] = R_WORLD_ODOM
        log("/tf", m.FrameTransforms(transforms=[self._tf(stamp, "world", "odom", world),
                                                 self._tf(stamp, "odom", "camera", res.pose_w)]),
            log_time=t_ns)

        # adds the position to the trajectory while tracking is good and redraws it as a blue line
        if res.state in ("ok", "recovered", "init", "reinit"):
            self.path.append(res.pose_w[:3, 3].copy())
        if len(self.path) >= 2:
            pts = [m.Point3(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.path]
            line = m.LinePrimitive(type=m.LinePrimitiveLineType.LineStrip, thickness=3, scale_invariant=True,
                                   points=pts, color=m.Color(r=0.2, g=0.6, b=1.0, a=1.0))
            log("/vo/trajectory", m.SceneUpdate(entities=[m.SceneEntity(timestamp=stamp, frame_id="odom",
                                                                        id="trajectory", lines=[line])]),
                log_time=t_ns)

        # when VO made a new keyframe, redraws the keyframes and adds its depth to the live map
        if len(vo.refs) - 1 != self.n_kf:
            self.n_kf = len(vo.refs) - 1
            self._keyframes(stamp, vo, t_ns)
            self._add_to_map(stamp, vo.refs[-1], f.K, t_ns)

        # sends the current depth cloud only every few frames since it's the heaviest message
        if self.n % self.cloud_every == 0 and f.depth is not None:
            self._cloud(stamp, f, t_ns)

        # sends the debug image as JPEG, which is ~20x smaller than raw
        import cv2
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            log("/vo/image", m.CompressedImage(timestamp=stamp, frame_id="camera", data=jpg.tobytes(),
                                               format="jpeg"), log_time=t_ns)

    # draws every keyframe as a cube (green, or red after a tracking gap) and a yellow line for each loop
    def _keyframes(self, stamp, vo, t_ns):
        m = self.m
        kfs = vo.keyframes
        cubes = [m.CubePrimitive(pose=self._pose(kf.T_odom_kf), size=m.Vector3(x=0.04, y=0.04, z=0.04),
                                 color=m.Color(r=1.0, g=0.3, b=0.3, a=1.0) if kf.tracking_gap
                                 else m.Color(r=0.3, g=1.0, b=0.3, a=1.0))
                 for kf in kfs]
        lines = []
        if vo.loops:
            # each loop is a line from the new keyframe to the old one it matched
            pts = []
            for new_id, old_id, *_ in vo.loops:
                for i in (new_id, old_id):
                    t = kfs[i].T_odom_kf[:3, 3]
                    pts.append(m.Point3(x=float(t[0]), y=float(t[1]), z=float(t[2])))
            lines.append(m.LinePrimitive(type=m.LinePrimitiveLineType.LineList, thickness=4, scale_invariant=True,
                                         points=pts, color=m.Color(r=1.0, g=0.9, b=0.0, a=1.0)))
        ent = m.SceneEntity(timestamp=stamp, frame_id="odom", id="keyframes", cubes=cubes, lines=lines)
        self.fg.log("/vo/keyframes", m.SceneUpdate(entities=[ent]), log_time=t_ns)

    # adds a keyframe's depth (at its raw VO pose) to the live map and publishes the whole map.
    # this map is never corrected by loops; map.py's reveal MCAP shows the corrected one
    def _add_to_map(self, stamp, ref, K, t_ns):
        from map import depth_to_points, transform, voxel_downsample
        m = self.m
        pts, val = depth_to_points(ref.depth, ref.image, K, step=4)
        pts = transform(ref.kf.T_odom_kf, pts)
        self.map_pts, self.map_val = voxel_downsample(np.concatenate([self.map_pts, pts]),
                                                      np.concatenate([self.map_val, val]), self.map_voxel)
        F = m.PackedElementFieldNumericType.Float32
        fields = [m.PackedElementField(name=n, offset=4 * i, type=F) for i, n in enumerate(("x", "y", "z", "intensity"))]
        data = np.column_stack([self.map_pts, self.map_val]).astype(np.float32).tobytes()
        self.fg.log("/vo/map", m.PointCloud(timestamp=stamp, frame_id="odom", point_stride=16,
                                            fields=fields, data=data), log_time=t_ns)

    # sends the current frame's depth as a point cloud attached to the camera
    def _cloud(self, stamp, f, t_ns):
        m = self.m
        s = self.cloud_step

        # takes every s-th pixel that has depth
        z = f.depth[::s, ::s]
        h, w = z.shape
        v, u = np.mgrid[0:h, 0:w]
        u, v = u * s, v * s
        K = f.K
        ok = z > 0
        z, u, v = z[ok], u[ok], v[ok]

        # back-projects with the pinhole model and packs x, y, z, brightness as 16 bytes per point
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        gray = f.image[::s, ::s][ok].astype(np.float32)
        pts = np.column_stack([x, y, z, gray]).astype(np.float32)
        F = m.PackedElementFieldNumericType.Float32
        fields = [m.PackedElementField(name=n, offset=4 * i, type=F) for i, n in enumerate(("x", "y", "z", "intensity"))]
        self.fg.log("/vo/cloud", m.PointCloud(timestamp=stamp, frame_id="camera", point_stride=16,
                                              fields=fields, data=pts.tobytes()), log_time=t_ns)

    # finishes the MCAP file and stops the websocket
    def close(self):
        if self.writer:
            self.writer.close()
        if self.server:
            self.server.stop()
