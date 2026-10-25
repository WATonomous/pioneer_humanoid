# RTAB-Map bag runner

`run_rtabmap.sh` replays a ROS 2 bag, launches RTAB-Map, and records the live map-corrected trajectory to `trajectory.txt`. It replays only the supplied RGB, depth, camera-info, and optional IMU topics, plus `/tf_static`; dynamic `/tf` is intentionally omitted to avoid conflicts with RTAB-Map's own transforms.

Run it from the RTAB-Map directory inside the ROS 2 container:

```bash
source /opt/ros/humble/setup.bash
./run_rtabmap.sh BAG OUTPUT RGB_TOPIC DEPTH_TOPIC CAMERA_INFO_TOPIC FRAME_ID
```

The output directory contains `trajectory.txt`, `rtabmap.db`, and logs. The RGB, depth, and camera-info topics must exist in the bag. `FRAME_ID` should be the camera frame used by the bag and TF tree.

## Common topic layouts

Converted TUM RGB-D bag:

```bash
./run_rtabmap.sh \
  /work/pioneer_humanoid/datasets/tum/ros2 \
  /work/pioneer_humanoid/datasets/tum/rtabmap_output \
  /camera/rgb/image_color \
  /camera/depth/image \
  /camera/rgb/camera_info \
  openni_rgb_optical_frame
```

RealSense ROS 2 bag:

```bash
./run_rtabmap.sh \
  /work/ros_bags/d455_bag/d455_bag_0.mcap \
  /work/ros_bags/d455_bag/rtabmap_output \
  /camera/camera/color/image_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/color/camera_info \
  camera_color_optical_frame \
  /camera/camera/gyro/sample \
  /camera/camera/accel/sample
```

The final two arguments are the separate gyro and accelerometer topics. The
runner combines them and passes the filtered orientation from Madgwick to
RTAB-Map. Do not pass the raw combined `/camera/camera/imu` topic unless it
already contains a valid orientation.

Playback runs at real-time speed by default. Override it with the
`PLAYBACK_RATE` environment variable, for example:

```bash
PLAYBACK_RATE=0.25 ./run_rtabmap.sh \
  /work/ros_bags/d455_bag/d455_bag_0.mcap \
  /work/ros_bags/d455_bag/rtabmap_output \
  /camera/camera/color/image_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/color/camera_info \
  camera_color_optical_frame \
  /camera/camera/gyro/sample \
  /camera/camera/accel/sample
```

For a bag with different names, inspect them first:

```bash
ros2 bag info /path/to/bag
```

The bag must be a ROS 2 bag directory containing `metadata.yaml` and either `.db3` or `.mcap` storage. ROS 1 `.bag` files must be converted first.
