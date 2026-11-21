# Pose-graph backend

`PoseGraphBackend` in `pose_graph.py` maintains an incremental GTSAM iSAM2 pose graph, with
optional IMU factors.

For each `Keyframe`, it adds a pose variable. The first pose gets a prior; each later
pose gets a `BetweenFactorPose3` from the relative VO pose and its covariance. A later
keyframe's initial estimate is the previous keyframe's *optimized* pose plus VO's relative
step, so it stays in the same frame as the graph (VO's own odometry pose drifts away from
that frame once the IMU or a loop corrects the trajectory). A `LoopClosure` adds another
between factor, wrapped in a Huber or Cauchy robust noise model (Cauchy by default). After
each loop, 20 extra iSAM2 updates let the trajectory finish bending: one update is a single
solver step, and iSAM2 only relinearizes every 10th update by default (`relinearizeSkip = 1`
here).

```python
backend = PoseGraphBackend(loop_kernel="cauchy")
backend.add_keyframe(keyframe)
poses = backend.add_loop_constraint(loop_closure)
```

`add_keyframe` returns the new optimized pose. `add_loop_constraint` returns the
optimized poses for all keyframes. The pose and covariance conventions are defined
in `slam_core.types`. Both methods raise `ValueError` for a duplicate keyframe or loop, an
unseen keyframe, or keyframe stamps that do not increase.

## IMU factors (optional)

Pass the IMU recording with each keyframe:

```python
from slam_core.backend.imu_fusion.imu import read_imu

imu = read_imu("imu.txt")            # (stamp, gyro, accel) for the whole recording
backend = PoseGraphBackend()
for keyframe in keyframes:
    backend.add_keyframe(keyframe, imu)
```

With IMU data each keyframe also gets a velocity and an IMU bias variable. The first keyframe
gets loose priors on both. Each later keyframe gets an `ImuFactor` from the IMU readings
between the two keyframe stamps (preintegrated, `imu_fusion/preintegration.py`), and a bias
random-walk factor between the two bias variables. The IMU noise values are the Kalibr
results for our D455 and are used as given.

- **Every keyframe or none.** Mixing keyframes with and without IMU data raises `ValueError`.
- **The IMU must cover every keyframe interval** (from the previous keyframe's stamp to this
  one's), or `add_keyframe` raises `ValueError` before anything is added, so you can retry.
- **Camera-IMU extrinsic.** `T_cam_imu` (constructor argument, default `T_CAM_IMU` in
  `preintegration.py`) is the IMU's pose in the left IR camera frame. It is derived (the
  factory color-to-IR transform times the Kalibr color-camera result), not measured directly.
- **Solver.** The IMU graph uses QR factorization. The default Cholesky raised
  `IndeterminateSystemException` on a long real run, where the bias random-walk sigmas
  (about 1e-6 per keyframe) sit next to the 0.02 bias prior.

### Gravity and the leveled frame

The IMU factors assume gravity points along +y in the graph's frame. VO's frame is just the
first camera pose, which is usually tilted. Either:

- pass `R_odom_kf0`, the leveling rotation, to the constructor; or
- leave it `None`: the first `gravity_keyframes` keyframes (default 4, at least 3) are held
  back, gravity is estimated from them (`imu_fusion/gravity.py`), and they are inserted
  leveled. Yaw cannot be observed from gravity, so keyframe 0's yaw and position stay pinned.

While keyframes are held back:

- `add_keyframe` returns VO's own **unleveled** pose, a provisional value. From the
  `gravity_keyframes`-th call on, poses are in the **leveled frame**, which is rotated
  about keyframe 0 by the camera's initial tilt (about 15 degrees in the synthetic test).
  So the returned poses jump frame at that point. Do not store the early return values:
  call `optimized_poses()` after the `gravity_keyframes`-th keyframe for a consistent
  trajectory.
- `optimized_pose(id)` raises `KeyError` for a held-back keyframe.
- A loop that touches a held-back keyframe is queued and added as soon as that keyframe is
  inserted. `add_loop_constraint` returns the current optimized poses meanwhile (empty until
  the first keyframes are in).

The gravity estimate reads the IMU array passed with the first held-back keyframe, which
suits replaying a recording. A live stream would need an IMU buffer, which does not exist yet.

## Replay and tests

```bash
python slam_tools/backend/replay_vo_keyframes.py keyframes.txt kf_vo.txt \
    --loops loops.txt --imu imu.txt --output trajectory.txt
cd slam_core && pytest tests -q      # test_pose_graph_loops.py, test_pose_graph_imu.py
```

`--imu` is optional. `--t-cam-imu FILE` (needs `--imu`) gives the IMU's 4x4 pose in the camera frame
the keyframes are in, as a text file; the default is the left IR camera calibration, so use it when the
VO ran on another camera (for example the color camera). The IMU tests use synthetic data: a known camera motion with a pitch
drift added to the VO and an ideal IMU at the Kalibr mount.

## Not handled

This module currently handles pose-graph optimization only. It does not detect loops,
consume ROS topics, publish TF, or save and reload a graph. Those are separate backend tasks.
For the IMU path:

- The Kalibr camera-IMU time offset (about -2.8 ms) is not applied to the IMU stamps.
- Velocity and bias estimates are not exposed through a public method.
- The IMU noise values have not been tuned on real recordings.
- A `tracking_gap` keyframe is treated like any other: it relies on its large covariance.

The runtime needs NumPy and the GTSAM Python bindings. GTSAM is an optional extra in
this package's `pyproject.toml`; install it with `pip install -e ".[backend]"`.
