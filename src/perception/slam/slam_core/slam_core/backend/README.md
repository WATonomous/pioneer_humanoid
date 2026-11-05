# Pose-graph backend

`PoseGraphBackend` in `pose_graph.py` maintains an incremental GTSAM iSAM2 pose graph.

For each `Keyframe`, it adds a pose variable. The first pose gets a prior; each later
pose gets a `BetweenFactorPose3` from the relative VO pose and its covariance. The VO
odometry pose is used as the initial estimate. A `LoopClosure` adds another between
factor, wrapped in a Huber or Cauchy robust noise model.

```python
backend = PoseGraphBackend(loop_kernel="huber")
backend.add_keyframe(keyframe)
poses = backend.add_loop_constraint(loop_closure)
```

`add_keyframe` returns the new optimized pose. `add_loop_constraint` returns the
optimized poses for all keyframes. The pose and covariance conventions are defined
in `slam_core.types`.

This module currently handles pose-graph optimization only. It does not detect loops,
consume ROS topics, publish TF, add IMU preintegration factors, or save and reload a
graph. Those are separate backend tasks.

The runtime needs NumPy and the GTSAM Python bindings. GTSAM is not currently listed
in this package's `pyproject.toml`, so it must be provided by the environment.
