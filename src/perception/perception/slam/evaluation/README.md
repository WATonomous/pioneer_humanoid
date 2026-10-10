# SLAM Evaluation

This directory contains a reproducible pipeline for running RTAB-Map on a TUM
RGB-D ROS bag and scoring its estimated trajectory against TUM ground truth with
[`evo`](https://github.com/MichaelGrupp/evo).

The RTAB-Map runner and trajectory scorer are intentionally separate:

1. Convert a TUM ROS 1 bag to ROS 2.
2. Run RTAB-Map and export frontend, live-corrected, and backend trajectories.
3. Score each trajectory against ground truth.

## Requirements

- Linux with Bash and ROS 2 Jazzy (native or inside Docker)
- `rtabmap_ros`
- Python 3 with `rclpy` and ROS 2 message packages
- ROS 2 Python TF packages: `tf2_ros` and `tf2_geometry_msgs`
- `rosbags-convert` from the [`rosbags`](https://gitlab.com/ternaris/rosbags)
  Python package
- `evo_ape` from the `evo` Python package
- ROS 2 bag playback with MCAP support and the Linux `setsid` command

The runner sources ROS from `/opt/ros/jazzy/setup.bash`. The converter currently
requires `rosbags-convert` at `/opt/evo-venv/bin/rosbags-convert`. The scorer
finds `evo_ape` on `PATH`, falling back to `/opt/evo-venv/bin/evo_ape`.
These dependencies must already be installed; the scripts do not install them.

## Input data

Download both files for the same TUM RGB-D sequence:

- The ROS 1 `.bag` recording
- Its `groundtruth.txt` trajectory

Do not mix ground truth from a different sequence. Both trajectory files use
the TUM format:

```text
timestamp tx ty tz qx qy qz qw
```

## Usage

Run the commands in your Linux ROS environment and open this directory:

```bash
source /opt/ros/jazzy/setup.bash
cd /path/to/pioneer_humanoid/src/perception/perception/slam/evaluation
```

Replace every `/path/to/...` placeholder below with your own path. Data and
results may be stored anywhere accessible to the process. If using Docker,
use paths inside the container and mount the relevant host directories.

### 1. Convert the ROS 1 bag

This conversion only needs to be performed once:

```bash
./convert_ros1_bag.sh \
  /path/to/input.bag \
  /path/to/converted_bag
```

The output is a ROS 2 MCAP bag directory. The destination must not already
exist. The converter migrates ROS 1 messages to ROS 2 Jazzy definitions while
retaining recorded sensor data and timestamps; it does not run SLAM.

### 2. Run RTAB-Map

```bash
./run_tum_rtabmap.sh \
  /path/to/converted_bag \
  /path/to/results
```

The second argument selects the output directory. The command above creates:

```text
/path/to/results/
├── frontend_poses.txt
├── live_corrected_poses.txt
├── backend_poses.txt
├── rtabmap.db
└── rtabmap_ros.log
```

- `frontend_poses.txt` records received `/rtabmap/odom` visual-odometry poses,
  before backend optimization. It does not guarantee one pose per camera frame.
- `backend_poses.txt` contains the globally optimized map-node trajectory
  exported from `rtabmap.db`.
- `live_corrected_poses.txt` applies the latest received `map -> odom` TF
  correction to each incoming odometry pose. It preserves that odometry
  timestamp and represents the estimate available to this recorder during
  playback. Corrections can cause jumps when a loop closes. Earlier rows are
  never rewritten using later optimization results.

Corrected samples are skipped until the map-to-odom transform is available;
the skipped count is logged. The corrected file can therefore have fewer poses
than the frontend file. This is a live estimate using the last available
correction, not an interpolation of the final optimized backend trajectory.

The output directory is optional:

```bash
./run_tum_rtabmap.sh ROS2_BAG [OUTPUT_DIRECTORY]
```

When omitted, the directory is named `<input-bag-basename>_rtabmap` beside the
input bag. The runner accepts a ROS 2 bag directory or a standalone `.mcap` file.

Running the command again replaces `rtabmap.db`, all three trajectory files, and the
log in that output directory.

### 3. Score a trajectory

Score the frontend:

```bash
./eval_trajectory.sh \
  /path/to/groundtruth.txt \
  /path/to/results/frontend_poses.txt
```

Score the live-corrected trajectory:

```bash
./eval_trajectory.sh \
  /path/to/groundtruth.txt \
  /path/to/results/live_corrected_poses.txt
```

Score the optimized backend:

```bash
./eval_trajectory.sh \
  /path/to/groundtruth.txt \
  /path/to/results/backend_poses.txt
```

The scorer uses evo's absolute pose error (APE) with SE(3) alignment (`-a`,
rotation and translation without scale correction). It reports positional RMSE
in metres and rotational RMSE in degrees for timestamp-matched poses. Smaller
values indicate lower error. Ground truth must correspond to the same recording
and camera pose convention. The scorer can also evaluate trajectories from
other SLAM systems if both files use the TUM text format.

## RTAB-Map settings

`run_tum_rtabmap.sh` currently uses these fixed benchmark settings:

- Bag playback rate: `0.25x`
- RTAB-Map detection rate: `10 Hz`
- Approximate RGB-D synchronization: maximum interval `0.05 s`
- Input: RGB, depth, and RGB camera calibration
- IMU: disabled

The slower playback provides more processing time between messages but does
not guarantee every frame is processed. This is an offline accuracy evaluation,
not a real-time performance test. Node counts printed by the runner are exported
trajectory counts, not evo's matched-pose counts.

The runner expects `/camera/rgb/image_color`, `/camera/depth/image`, and
`/camera/rgb/camera_info`. RGB and depth must already be geometrically aligned.
`normalize_rgbd_frames.py` replaces their frame IDs with
`openni_rgb_optical_frame`, republishes with reliable QoS, and exposes
`/eval/rgb/image`, `/eval/depth/image`, and `/eval/rgb/camera_info`. It does not
modify image pixels, align depth, or change timestamps. These topic and frame
settings are hardcoded for TUM; arbitrary camera bags need appropriate changes.

## Files

- `convert_ros1_bag.sh`: converts a ROS 1 `.bag` to a ROS 2 MCAP bag.
- `run_tum_rtabmap.sh`: runs RTAB-Map and exports all three trajectories.
- `eval_trajectory.sh`: compares one estimated trajectory against ground truth.
- `normalize_rgbd_frames.py`: adapts legacy TUM RGB-D messages for ROS 2.
- `record_odom_trajectory.py`: records raw and live-corrected odometry in TUM
  text format.
