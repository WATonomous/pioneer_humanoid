# RTAB-Map bag runner

`run_rtabmap.sh` replays a ROS 2 bag, launches RTAB-Map, and records the live map-corrected trajectory to `trajectory.txt`. It replays all bag topics so `/tf`, `/tf_static`, and optional IMU data remain available.

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
  /work/pioneer_humanoid/datasets/d455/ros2 \
  /work/pioneer_humanoid/datasets/d455/rtabmap_output \
  /camera/camera/color/image_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/color/camera_info \
  camera_color_optical_frame
```

For a bag with different names, inspect them first:

```bash
ros2 bag info /path/to/bag
```

The bag must be a ROS 2 bag directory containing `metadata.yaml` and either `.db3` or `.mcap` storage. ROS 1 `.bag` files must be converted first.
