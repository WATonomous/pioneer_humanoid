# Record a bag for terrain mapping and VSLAM

Bags 001 and 002 miss most of the floor. The camera was handheld about 0.65 m up, looked nearly level, and turned in place.
Record new bags this way so the map can be checked against the floor and against objects of known height.
These steps follow RealSense ROS driver 4.58.4, the version in the perception image. Nobody has run them end to end with a camera yet.

## Set up the camera and the scene

- Mount the D455 rigidly at the humanoid's real camera height and angle, about 1.4 to 1.5 m above the floor. Use a pole, a tripod, or a helmet mount. Bags 001 and 002 were handheld and moved up and down by 0.2 to 0.5 m. The Nov 6 bags were handheld at about 1.65 m, above the robot's head.
- Tilt the camera down about 30°. At 1.45 m, the floor is then in view from about 0.9 m out, and the image center meets the floor about 2.5 m ahead.
- Measure the camera's height above the floor with a tape. The floor should map at minus that height.
- Include an open stretch of floor, and place two or three boxes of measured height (for example 0.1 m and 0.2 m). A step, a curb, or a stair is better still.
- Avoid glass walls and whiteboards close to the path. They reflect the IR projector, and the reflections land in the map as phantom floor below the real one.

## Start the camera

Start a one-off perception container with a shell. Its normal launch, `perception.launch.py`, holds the camera with different settings (no IMU, 640x480 depth, other topic names).
Put `perception` in `ACTIVE_MODULES`, stop a running perception container, and run:

```bash
mkdir -p ~/terrain_bags
./watod run --rm -v $HOME/terrain_bags:/bags_out perception bash
```

In that shell, start the driver:

```bash
ros2 launch realsense2_camera rs_launch.py \
	depth_module.depth_profile:=848x480x30 \
	depth_module.infra_profile:=848x480x30 enable_infra1:=true \
	rgb_camera.color_profile:=848x480x30 \
	align_depth.enable:=true \
	enable_gyro:=true enable_accel:=true unite_imu_method:=2
```

Before you record, make sure the IR projector is on for every frame. In a second shell, run:

```bash
ros2 param get /camera/camera depth_module.emitter_enabled   # expect 1 (laser on)
ros2 param get /camera/camera depth_module.emitter_on_off    # expect False or 0
```

Do not record these bags with `src/perception/slam/config/d455.yaml`. Its `emitter: on_off` setting turns the projector off on every other frame. That helps visual SLAM, but it halves the usable depth frames.

## Record

Record the terrain-mapping inputs and the left infrared image (`infra1`) with its camera calibration for VSLAM.
They add up to about 100 MB/s, or about 6 GB per minute.

```bash
cd /bags_out
ros2 bag record -o '003-D455(scene_name)' \
	/tf_static /camera/camera/imu \
	/camera/camera/depth/image_rect_raw /camera/camera/depth/camera_info \
	/camera/camera/infra1/image_rect_raw /camera/camera/infra1/camera_info \
	/camera/camera/color/image_raw /camera/camera/color/camera_info \
	/camera/camera/aligned_depth_to_color/image_raw /camera/camera/aligned_depth_to_color/camera_info
```

1. Hold the camera still for 3 s. The launch file computes the starting tilt from the first 0.25 s of IMU data.
2. Walk slowly (about 0.3 m/s) across the floor and past the boxes. Turn slowly, at under about 45° per second.
3. Stop at the starting point, facing the starting direction. The true end-to-start distance is then 0, so `check_map.py` measures the odometry drift directly.
4. Stop recording after 30 to 90 s.

Move the bag folder from `~/terrain_bags` into `Terrain Mapping Bags/`. Check that each bag folder holds a `.db3` file next to `metadata.yaml`. Two Nov 6 bags reached the shared drive with only `metadata.yaml`.
Run `ros2 bag info` on the bag folder and check that both `/camera/camera/infra1/image_rect_raw` and `/camera/camera/infra1/camera_info` have messages.
Beside the bags, add a `notes.md` with tape-measured values:

```markdown
Camera: 1.45 m above the floor (lens centre), tilted 30° down, on a pole
Step: 0.14 m high, 0.73 x 0.28 m, about 2 m ahead of the start
Box: 0.39 m high, 0.39 x 0.36 m top
Bags: 003 = walk past the step and box and back; 004 = ...
```
Then run it with `bag:='/bags/003-D455(scene_name)'`.

For container setup and bag replay, see the [terrain-mapping README](README.md).
