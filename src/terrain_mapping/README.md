# Terrain mapping

This module builds a GPU elevation map (one height per 4 cm cell) from recorded RealSense D455 bags with
[elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) on ROS 2 Humble.
It is perception only. It never starts `interfacing`, CAN, or `joint_command`.

## Layout

```
src/terrain_mapping/
├── terrain_core/                 # ROS-free Python core (depth preprocessing, tests); see its README
├── terrain_mapping_bringup/      # our package: launch, configs, RViz, check_map.py
└── third_party/
    ├── elevation_mapping_cupy/   # generated during the Docker build (gitignored)
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

To check the GPU stack, run upstream's synthetic demo inside the container. It compiles the CuPy kernels and maps a simulated depth camera:

```bash
ros2 launch elevation_mapping_cupy synthetic_depth_demo.launch.py launch_rviz:=false
ros2 topic hz /elevation_mapping_node/elevation_map   # second shell; about 10 Hz
```

## Rebuild after editing source

`src/terrain_mapping` is mounted into the container at `$AMENT_WS/src/terrain_mapping` (`/home/bolty/ament_ws`). The image already has a build in `/opt/watonomous`.
To use your edits, build an overlay inside the container:

```bash
cd $AMENT_WS
colcon build --symlink-install
source install/setup.bash
```

Rebuild the image (`./watod build terrain_mapping`) after changing the Dockerfile, vendor script, or upstream patches.

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

The Nov 6 hallway bags (step, box, ramp; see the results under Score a run) are in `Terrain Mapping Bags/Nov6_rosbags/`. `2026-11-06_02-09-28` is the longest complete one:

```bash
ros2 launch terrain_mapping_bringup bag_pipeline.launch.py bag:=/bags/Nov6_rosbags/2026-11-06_02-09-28
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
| `map.floor` | The largest near-level plane (under 10° tilt) at least 0.5 m below the camera, fitted to the cells within ±5 cm of it. Gives `below_camera` (the camera's height above it), `area_m2`, `tilt_deg`, and `within_2cm` (the share of those cells within 2 cm of the plane). |
| `map.objects` | Blobs 4 to 60 cm above the floor plane and at least 0.04 m², largest first. `height` is the blob's 75th percentile (its top surface). `edge_smear` counts cells between 20 and 80 % of that height per boundary cell: near 0 for a sharp box, higher for a smeared edge, and high for a ramp by design. `centre_m` is the offset from the camera. |
| `map.max_above_highest_camera` | Highest cell minus the highest camera position. `max_height_range` keeps this at or below 0.3 m. |
| `odometry` | Yaw turned, end-to-start distance, and camera height range from TF `odom -> camera_link`. |
| `gravity_error` | Angle between measured gravity and the z axis of `odom`. The same angle in `camera_link` is the control, the value `odom` would show with no gravity alignment. `first_s_median_deg` and `last_s_median_deg` cover the first and last second: a high first value means a wrong starting tilt, a rise means drift. The accelerometer also reads hand acceleration, so the error reads high while the camera moves. |

Results on an RTX 5070 Laptop GPU with `MAX_DEPTH` at 2.5 m. Each value is the median of 3 runs; repeat runs agree to within 0.3° and 1 cm.
Object errors are against tape measurements: step 14 cm, box 39 cm, ramp 0.90 x 0.81 m at about 9° (high end about 13 to 14 cm).

| Nov 6 bag | Floor below camera | Floor tilt (no depth cap) | Floor cells within 2 cm (no cap) | Floor area | Step error | Box error | Ramp top (75th pct) |
|---|---|---|---|---|---|---|---|
| 02-02-46 | 1.54 m | 2.5° (3.1°) | 76 % (73 %) | 7.9 m² | +1.6 cm | | |
| 02-03-37 | 1.68 m | 0.7° (1.5°) | 74 % (63 %) | 5.1 m² | +0.7 cm | | |
| 02-04-00 | 1.58 m | 0.2° (0.2°) | 77 % (79 %) | 5.5 m² | +3.8 cm | +3.1 cm | 14.1 cm |
| 02-04-18 | 1.58 m | 1.5° (1.8°) | 73 % (73 %) | 5.8 m² | +1.2 cm | -0.7 cm | 12.0 cm |
| 02-04-34 | 1.52 m | 1.8° (2.2°) | 77 % (77 %) | 5.0 m² | | +3.2 cm | 11.9 cm |
| 02-06-11 | 1.43 m | 2.9° (2.5°) | 78 % (70 %) | 15.8 m² | -0.8 cm | | |
| 02-09-28 | 1.66 m | 2.2° (2.5°) | 84 % (61 %) | 22.0 m² | | +3.7 cm | 11.3 cm |

The camera was handheld at about 1.65 m, which matches the floor heights.
The floor still tilts 0.2 to 2.9°, against a 1° target. The tilt is already present in the first, still depth frame of each bag (0.7 to 1.3°) and matches the gravity error in `odom` (1.4 to 3.8°), so it most likely comes from the D455's accelerometer calibration, not from the map settings.

| Old bag | 002 | 001 |
|---|---|---|
| Point cloud, odometry, map rate | 30 Hz, 27 to 29 Hz, 5 Hz | 91 Hz, 29 Hz, 5 Hz |
| Yaw turned (gyro integral) | -354° (-350°) | -354° (-350°) |
| End-to-start distance | 0.14 m | 0.06 m (RTAB-Map reference: 0.11 m) |
| Camera height range | 0.20 m | 0.56 m (reference: 0.56 m) |
| Gravity error in `odom` (control) | 2.1° median (5.1°) | 2.9° median (9.7°) |
| Floor | 1.5 m² of floor under 2.5 m depth; plane tilt not stable between runs | none within 2.5 m depth (the camera sees desks, not floor) |
| Highest cell above highest camera | 0.30 m | 0.29 m |

## Record a bag

Record D455 bags for terrain mapping and VSLAM using the [recording guide](recording.md).

## Pipeline

```
/camera/camera/depth/image_rect_raw ─> rtabmap_util point_cloud_xyz (depth ≤ MAX_DEPTH) ─> /camera/depth/points ─> elevation_mapping_cupy ─> /elevation_mapping_node/elevation_map_{raw,filter}
/camera/camera/color/image_raw ──────┐
/camera/camera/aligned_depth_to_color ┼─> rtabmap rgbd_odometry ─> TF odom -> camera_link
/camera/camera/imu ─> Madgwick filter ┘
```

Five details are easy to break:

- **Gravity alignment comes from `initial_pose`.** `rgbd_odometry` (rtabmap 0.23.13) only aligns `odom` with gravity from IMU samples older than its first frame. A frame reaches ROS about 38 ms after capture and an IMU sample about 1 ms after, so the first frame always predates the IMU and `odom` keeps the camera's starting tilt (5.6° on bag 002). The launch file takes the median of the bag's first 0.25 s of accelerometer data (the camera is still then), computes the camera's roll and pitch, and passes them as `initial_pose`. The median ignores two glitched samples at the start of bag 001. Upstream rtabmap master handles this case. Drop the workaround once a fixed release reaches Humble.
- **Madgwick gives the IMU an orientation.** The D455 IMU publishes angular velocity and acceleration only. `rgbd_odometry` ignores an IMU without orientation. With it, odometry uses the IMU to predict motion.
- **Large messages go over shared memory.** With Fast DDS's default transports, 2.7 MB color frames arrived at under 1 Hz instead of 30 Hz. They do not fit the 512 KB default shared-memory segment, so they fall back to UDP, and the host's `net.core.rmem_max` (208 KB) drops fragments. `docker/terrain_mapping/fastdds_shm.xml` gives every node in the container a shared-memory transport with 32 MB segments, and the compose file sets `shm_size: 1gb`. `ROS_LOCALHOST_ONLY=1` also adds a 127.0.0.1-only UDP transport. Same-host traffic still uses shared memory.
- **Depth stops at `MAX_DEPTH` (in the launch file).** Far D455 depth bends the floor, and glass walls reflect it into phantom points below the floor. A map built from the raw points of the Nov 6 bags has a 2.4 to 3.0° floor tilt with all depth and 0.4 to 1.0° with depth up to 2 m. `elevation_mapping_cupy` has `min_valid_distance` but no maximum, so `rtabmap_util`'s `point_cloud_xyz` cuts the depth with `max_depth` before the map sees it. It replaces `depth_image_proc`, whose Humble version also needed an explicit `camera_info` remap.
- **Some bags start with a stale image stamp.** In 6 of the 10 Nov 6 bags, the first color or aligned-depth message carries a stamp 4 to 189 s older than the next one. `rgbd_odometry` drops that frame during its exact-stamp sync, and the logs show no TF or tracking errors, so the pipeline does not filter it.

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
The Nov 6 bags were used to test the settings below; the depth cap was the only change that improved the floor (see `MAX_DEPTH` under Pipeline). Upstream drift compensation, visibility cleanup, edge sharpening, and two odometry settings (`Optimizer/GravitySigma` 0.1, `Vis/MaxDepth` 4 m) made no change past run-to-run spread, so they keep their defaults.

| Parameter | Value | Why |
|---|---|---|
| `resolution` | 0.04 m | Fine enough for steps and cables. |
| `MAX_DEPTH` (launch file) | 2.5 m | Farther depth bends the floor. 2.0 m flattens it more but leaves only about 1.1 to 1.4 m of floor ahead of the camera and gives unstable maps on two bags. |
| `map_length` | 6 m | The depth cap keeps data within about 2.5 m of the camera. A 10 m map added no floor on bag 002. |
| `min_valid_distance` | 0.3 m | Closer D455 depth is unreliable. |
| `max_height_range` | 0.3 m | Measured above the camera at capture time. Drops the ceiling. |
| `use_initializer_at_start` | false | The initializer paints a flat patch at `base_link` height, which is camera height here. |

Three upstream features stay off until the camera height comes from the robot's TF (URDF camera mount and a state estimator) instead of a placeholder `base_link` at the camera:
`use_initializer_at_start`, `initialize_tf_offset` (it needs the real height of `base_link` above the floor), and the `near_base_height_filter` plugin (it compares absolute map height with a threshold around `base_link`).
Turn them back on when `base_link_placeholder` is deleted.

## Upstream code

The Dockerfile runs `third_party/vendor_elevation_mapping_cupy.sh` before `colcon build`.
The script fetches upstream `ros2` branch commit `d9e1c9fe42ea3d463519333a443da99f680adde8` (2026-10-04) into `third_party/elevation_mapping_cupy`, a generated, gitignored directory.
It keeps the `elevation_mapping_cupy` and `elevation_map_msgs` packages and the MIT license.
Upstream targets Jazzy. Each Humble fix is a file in `third_party/patches/`:

- `0001-humble-rosbag2-topic-metadata.patch` fixes the `save_map` and `load_map` services, which fail on Humble without it.
- `0002-drop-semantic-sensor.patch` drops the `semantic_sensor` dependency. That package is not vendored.
- `0003-drop-tests.patch` removes upstream's test targets and test dependencies. The script deletes upstream's test folders after the patches apply, so the vendored code holds no tests.

To move to a newer upstream commit, change `UPSTREAM_COMMIT` in `vendor_elevation_mapping_cupy.sh` and rebuild the image. The script replaces the generated folder, applies the patches, and deletes the tests.
For local inspection, run `src/terrain_mapping/third_party/vendor_elevation_mapping_cupy.sh` from the repo root.
Keep upstream changes in `third_party/patches/` so each image build reproduces them.

`ros2_numpy` has no Humble release. The Dockerfile builds its `humble` branch at commit `d665add`.

## Known limitations

- The bags are handheld. There is no robot TF, no leg odometry, and `base_link` is a placeholder. The camera height changes during each turn (0.2 m in bag 002, 0.5 m in bag 001).
- Bags 001 and 002 barely see the floor. The camera sits about 0.65 m up inside a cubicle and sees the floor only between desks.
- The floor tilts up to about 3° in the map. Single still frames already show about 1°, and the gravity error in `odom` is 1.4 to 3.8°, so the D455 IMU likely needs calibration (`rs-imu-calibration`). Glass walls also add phantom points below the floor within the depth cap.
- Odometry comes from the camera alone. Gravity is fixed once at the start, so roll and pitch can drift, and nothing closes the loop at the end of the turn.
- Replay is not deterministic. Two runs of the same bag give slightly different maps because the nodes run in real time. With the current checker, repeat runs agree to within about 0.3° of tilt and 1 cm of object height.
