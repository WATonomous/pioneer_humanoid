# Terrain mapping

This module builds a GPU elevation map (one height per 4 cm cell) from recorded RealSense D455 bags with
[elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) on ROS 2 Humble.
It is perception only. It never starts `interfacing`, CAN, or `joint_command`.

## Layout

```
src/terrain_mapping/
├── terrain_mapping_bringup/      # our package: launch, configs, RViz, check_map.py
└── third_party/
    ├── elevation_mapping_cupy/   # vendored upstream (do not edit by hand)
    ├── patches/                  # Humble fixes applied on top of upstream
    └── vendor_elevation_mapping_cupy.sh
docker/terrain_mapping/           # Dockerfile and the Fast DDS profile
modules/docker-compose.terrain_mapping.yaml
```

## Build the container

1. Put the module in `watod-config.local.sh`:

   ```bash
   ACTIVE_MODULES="terrain_mapping"
   ```

2. Build and start it:

   ```bash
   ./watod up -d
   ```

   The first build downloads about 10.5 GB (the CUDA 12.8 image and PyTorch) and unpacks to about 21 GB.

3. Open a shell in it:

   ```bash
   ./watod -t terrain_mapping
   ```

The image starts from `nvidia/cuda:12.8.1-devel-ubuntu22.04`, not `robot_base`, because `robot_base` has no CUDA.
Blackwell GPUs (RTX 50xx) need CUDA 12.8 or newer.
The host needs an NVIDIA driver of version 570 or newer. On a GeForce card with an older driver, the NVIDIA container runtime refuses to start the container.
CI does not build this module. It is on the skip list in `.github/templates/docker_context/docker_context.sh` with the other GPU modules.

To check the GPU stack, run upstream's tests inside the container. They include a CuPy kernel compile and the synthetic demo launch:

```bash
cd $AMENT_WS
colcon build --symlink-install --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select elevation_mapping_cupy && colcon test-result
```

All 17 test targets (82 test cases) pass on an RTX 5070 Laptop GPU.

## Rebuild after editing source

`src/terrain_mapping` is mounted into the container at `$AMENT_WS/src/terrain_mapping` (`/home/bolty/ament_ws`). The image already has a build in `/opt/watonomous`.
To use your edits, build an overlay inside the container:

```bash
cd $AMENT_WS
colcon build --symlink-install
source install/setup.bash
```

Rebuild the image (`./watod build terrain_mapping`) only after you change the Dockerfile.

## Run on a bag

The bags live in `Terrain Mapping Bags/` at the repo root. Git ignores that folder, and the container sees it read-only at `/bags`.
To use another folder, set `TERRAIN_BAGS_DIR` in `watod-config.local.sh`.
If the folder does not exist, `./watod up` fails. To start the container without bags, point `TERRAIN_BAGS_DIR` at any existing folder.

Bag 002 (office, 26.6 s) is the default:

```bash
ros2 launch terrain_mapping_bringup bag_pipeline.launch.py
```

Bag 001 (16.4 s):

```bash
ros2 launch terrain_mapping_bringup bag_pipeline.launch.py bag:='/bags/001-D455(360_turn_v1)'
```

Launch arguments are `bag`, `rate` (playback speed, default `1.0`), and `rviz` (default `true`).
The launch file starts every node on the bag's clock (`use_sim_time`) and plays the bag with `--clock` after 5 s.

RViz opens on your X display. If it cannot open the display, run `xhost +SI:localuser:$(id -un)` on the host.

## Score a run

Start the checker in a second shell before the bag ends:

```bash
ros2 run terrain_mapping_bringup check_map.py --out /tmp/terrain_map_check
```

It stops when the bag's clock stops, then prints a JSON report and writes `elevation_top_down.png` and `elevation.npy`.

| Field | Meaning |
|---|---|
| `map.floor` | Lowest connected flat patch of at least 0.25 m² (height spread under ±3 cm), or `null`. Gives its height, size, spread, and how many cells stand more than 0.3 m above it. |
| `map.max_above_highest_camera` | Highest cell minus the highest camera position. `max_height_range` keeps this at or below 0.3 m. |
| `odometry` | Yaw turned, end-to-start distance, and camera height range from TF `odom -> camera_link`. |
| `gravity_error` | Angle between measured gravity and the z axis of `odom`. The same angle in `camera_link` is the control, the value `odom` would show with no gravity alignment. |

Results on an RTX 5070 Laptop GPU:

| | Bag 002 | Bag 001 |
|---|---|---|
| Point cloud, odometry, map rate | 30 Hz, 27 to 29 Hz, 5 Hz | 91 Hz, 29 Hz, 5 Hz |
| Yaw turned (gyro integral) | -354° (-350°) | -357° (-350°) |
| End-to-start distance | 0.16 m | 0.06 m (RTAB-Map reference: 0.11 m) |
| Camera height range | 0.19 m | 0.57 m (reference: 0.56 m) |
| Gravity error in `odom` (control) | 1.9° median (5.1°) | 3.0° median (9.7°) |
| Floor patch | 0.29 m² at -0.65 m, spread 1.4 cm | 0.29 m² at -0.20 m (likely a desk top, not floor) |
| Highest cell above highest camera | 0.29 m | 0.28 m |

## Pipeline

```
/camera/camera/depth/image_rect_raw ─> depth_image_proc ─> /camera/depth/points ─> elevation_mapping_cupy ─> /elevation_mapping_node/elevation_map_{raw,filter}
/camera/camera/color/image_raw ──────┐
/camera/camera/aligned_depth_to_color ┼─> rtabmap rgbd_odometry ─> TF odom -> camera_link
/camera/camera/imu ─> Madgwick filter ┘
```

Four details are easy to break:

- **Gravity alignment comes from `initial_pose`.** `rgbd_odometry` (rtabmap 0.23.13) only aligns `odom` with gravity from IMU samples older than its first frame. A frame reaches ROS about 38 ms after capture and an IMU sample about 1 ms after, so the first frame always predates the IMU and `odom` keeps the camera's starting tilt (5.6° on bag 002). The launch file takes the median of the bag's first 0.25 s of accelerometer data (the camera is still then), computes the camera's roll and pitch, and passes them as `initial_pose`. The median ignores two glitched samples at the start of bag 001. Upstream rtabmap master handles this case. Drop the workaround once a fixed release reaches Humble.
- **Madgwick gives the IMU an orientation.** The D455 IMU publishes angular velocity and acceleration only. `rgbd_odometry` ignores an IMU without orientation. With it, odometry uses the IMU to predict motion.
- **Large messages go over shared memory.** With Fast DDS's default transports, 2.7 MB color frames arrived at under 1 Hz instead of 30 Hz. They do not fit the 512 KB default shared-memory segment, so they fall back to UDP, and the host's `net.core.rmem_max` (208 KB) drops fragments. `docker/terrain_mapping/fastdds_shm.xml` gives every node in the container a shared-memory transport with 32 MB segments, and the compose file sets `shm_size: 1gb`. `ROS_LOCALHOST_ONLY=1` also adds a 127.0.0.1-only UDP transport. Same-host traffic still uses shared memory.
- **`depth_image_proc` needs an explicit `camera_info` remap.** On Humble it derives the info topic from the unremapped image name (`/camera_info`), so the launch file remaps it too.

## Frame tree

```
odom                       (rgbd_odometry, gravity-aligned, origin = camera at bag start)
└── camera_link            (rgbd_odometry)
    ├── base_link          (static placeholder, identity)
    ├── camera_depth_frame -> camera_depth_optical_frame        (bag /tf_static)
    ├── camera_color_frame -> camera_color_optical_frame        (bag /tf_static)
    └── camera_gyro_frame  -> camera_imu_frame -> camera_imu_optical_frame
```

`elevation_mapping_cupy` uses `map_frame: odom` and `base_frame: base_link`.
Because `odom` starts at the camera, map heights are relative to the camera's starting height. The floor is negative (about -0.65 m in bag 002).

`base_link` is a placeholder. On the robot, the state estimator publishes `odom -> base_link` and the URDF camera mount gives `base_link -> camera_link`.
Delete the `base_link_placeholder` node in `bag_pipeline.launch.py` at that point.

## Configuration

`terrain_mapping_bringup/config/d455_elevation.yaml` is layered on top of upstream's `config/core/core_param.yaml`.
`terrain_mapping_bringup/config/d455_plugins.yaml` replaces upstream's plugin chain. Upstream's chain is tuned for an excavator, and its `near_base_height_filter` deletes anything taller than 0.5 m within 4 m.
All values are untuned first guesses.

| Parameter | Value | Why |
|---|---|---|
| `resolution` | 0.04 m | Fine enough for steps and cables. |
| `map_length` | 6 m | The D455 depth is usable to about 3 m on each side. A 10 m map added no floor on bag 002. |
| `min_valid_distance` | 0.3 m | Closer D455 depth is unreliable. |
| `max_height_range` | 0.3 m | Measured above the camera at capture time. Drops the ceiling. |
| `use_initializer_at_start` | false | The initializer paints a flat patch at `base_link` height, which is camera height here. |

## Upstream code

`third_party/elevation_mapping_cupy` is upstream `ros2` branch commit `d9e1c9fe` (2026-10-04). It holds only the `elevation_mapping_cupy` and `elevation_map_msgs` packages and the MIT license.
Upstream targets Jazzy. Each Humble fix is a file in `third_party/patches/`:

- `0001-humble-rosbag2-topic-metadata.patch` fixes the `save_map` and `load_map` services. Without it, upstream's save/load test fails on Humble.
- `0002-drop-semantic-sensor.patch` drops the `semantic_sensor` test and dependency. That package is not vendored.

To move to a newer upstream commit, change `UPSTREAM_COMMIT` in `vendor_elevation_mapping_cupy.sh` and run the script. The script replaces the vendored folder and applies the patches.
Never edit `third_party/elevation_mapping_cupy` by hand. Add a patch instead. The repo's autopep8 lint skips `src/terrain_mapping/third_party` for the same reason.

`ros2_numpy` has no Humble release. The Dockerfile builds its `humble` branch at commit `d665add`.

## Known limitations

- The bags are handheld. There is no robot TF, no leg odometry, and `base_link` is a placeholder. The camera height changes during each turn (0.2 m in bag 002, 0.5 m in bag 001).
- The floor is mostly out of view. In bag 002 the camera sits about 0.65 m above the floor, inside a cubicle, and sees the floor only from about 1.5 m out and between desks. The largest connected floor patch is about 0.3 m².
- Odometry comes from the camera alone. Gravity is fixed once at the start, so roll and pitch can drift, and nothing closes the loop at the end of the turn.
- Replay is not deterministic. Two runs of the same bag give slightly different maps because the nodes run in real time.
