#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
  FORCE=true
  shift
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 [--force] /path/to/bag[.bag|.mcap|directory] [groundtruth.txt]" >&2
  exit 1
fi

INPUT_PATH="$(realpath "$1")"
GROUND_TRUTH="${2:-}"
CONVERTER=/opt/evo-venv/bin/rosbags-convert

if [[ -f "$INPUT_PATH" && "$INPUT_PATH" == *.bag ]]; then
  BAG_STEM="${INPUT_PATH%.bag}"
  ROS2_DIR="${BAG_STEM}_ros2"
  ROS2_BAG="$ROS2_DIR/$(basename "$ROS2_DIR").mcap"

  if [[ ! -f "$ROS2_BAG" ]]; then
    if [[ ! -x "$CONVERTER" ]]; then
      echo "Error: ROS 1 bag detected, but $CONVERTER is unavailable." >&2
      exit 1
    fi

    echo "ROS 1 bag detected; converting it to ROS 2 (one time only)..." >&2
    "$CONVERTER" \
      --src "$INPUT_PATH" \
      --dst "$ROS2_DIR" \
      --dst-storage mcap \
      --dst-typestore ros2_jazzy >&2
  else
    echo "Reusing converted ROS 2 bag: $ROS2_BAG" >&2
  fi

  if [[ -z "$GROUND_TRUTH" ]]; then
    GROUND_TRUTH="$BAG_STEM/groundtruth.txt"
  fi
elif [[ -f "$INPUT_PATH" && "$INPUT_PATH" == *.mcap ]]; then
  ROS2_BAG="$INPUT_PATH"
  BAG_STEM="${INPUT_PATH%.mcap}"
elif [[ -d "$INPUT_PATH" ]]; then
  ROS2_BAG="$(find "$INPUT_PATH" -maxdepth 1 -type f -name '*.mcap' -print -quit)"
  BAG_STEM="$INPUT_PATH"
  if [[ -z "$ROS2_BAG" && -f "$INPUT_PATH/metadata.yaml" ]]; then
    ROS2_BAG="$INPUT_PATH"
  fi
else
  echo "Error: unsupported bag path: $INPUT_PATH" >&2
  exit 1
fi

if [[ -z "${ROS2_BAG:-}" ]]; then
  echo "Error: no ROS 2 MCAP file found in $INPUT_PATH" >&2
  exit 1
fi

if [[ -z "$GROUND_TRUTH" ]]; then
  echo "Error: ground truth was not inferred; pass groundtruth.txt as the second argument." >&2
  exit 1
fi
GROUND_TRUTH="$(realpath "$GROUND_TRUTH")"
if [[ ! -f "$GROUND_TRUTH" ]]; then
  echo "Error: ground truth not found: $GROUND_TRUTH" >&2
  exit 1
fi

OUTPUT_DIR="${BAG_STEM}_bag_eval"
DATABASE="$OUTPUT_DIR/rtabmap.db"
ESTIMATED_POSES="$OUTPUT_DIR/rtabmap_poses.txt"
LOG_FILE="$OUTPUT_DIR/rtabmap_ros.log"
mkdir -p "$OUTPUT_DIR"

if [[ "$FORCE" == true || ! -s "$ESTIMATED_POSES" ]]; then
  rm -f "$DATABASE" "$ESTIMATED_POSES"

  echo "Normalizing legacy ROS frame IDs..." >&2
  python3 "$SCRIPT_DIR/normalize_rgbd_frames.py" >>"$LOG_FILE" 2>&1 &
  RELAY_PID=$!

  echo "Launching RTAB-Map ROS 2 nodes..." >&2
  ros2 launch rtabmap_launch rtabmap.launch.py \
    use_sim_time:=true \
    frame_id:=openni_rgb_optical_frame \
    rgb_topic:=/eval/camera/rgb/image \
    depth_topic:=/eval/camera/depth/image \
    camera_info_topic:=/eval/camera/rgb/camera_info \
    approx_sync:=true \
    rgbd_sync:=true \
    publish_tf_odom:=false \
    rtabmap_viz:=false \
    rviz:=false \
    database_path:="$DATABASE" \
    args:="-d" \
    >"$LOG_FILE" 2>&1 &
  LAUNCH_PID=$!

  cleanup() {
    if kill -0 "$LAUNCH_PID" 2>/dev/null; then
      kill -INT "$LAUNCH_PID" 2>/dev/null || true
      for _ in {1..20}; do
        kill -0 "$LAUNCH_PID" 2>/dev/null || break
        sleep 0.25
      done
      if kill -0 "$LAUNCH_PID" 2>/dev/null; then
        kill -TERM "$LAUNCH_PID" 2>/dev/null || true
        for _ in {1..20}; do
          kill -0 "$LAUNCH_PID" 2>/dev/null || break
          sleep 0.25
        done
      fi
      if kill -0 "$LAUNCH_PID" 2>/dev/null; then
        kill -KILL "$LAUNCH_PID" 2>/dev/null || true
      fi
      wait "$LAUNCH_PID" 2>/dev/null || true
    fi
    if kill -0 "$RELAY_PID" 2>/dev/null; then
      kill -TERM "$RELAY_PID" 2>/dev/null || true
      wait "$RELAY_PID" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  sleep 4
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "Error: RTAB-Map exited before bag playback. See $LOG_FILE" >&2
    exit 1
  fi

  echo "Playing RGB-D topics from the ROS 2 bag..." >&2
  # /tf is intentionally excluded because it contains motion-capture ground
  # truth. The relay gives registered RGB and depth data one optical frame.
  ros2 bag play "$ROS2_BAG" \
    --clock \
    --topics \
      /camera/rgb/image_color \
      /camera/depth/image \
      /camera/rgb/camera_info \
      /camera/depth/camera_info >&2

  sleep 2
  cleanup
  trap - EXIT INT TERM

  if [[ ! -f "$DATABASE" ]]; then
    echo "Error: RTAB-Map did not create $DATABASE. See $LOG_FILE" >&2
    exit 1
  fi

  echo "Exporting the estimated camera trajectory..." >&2
  rtabmap-export \
    --poses \
    --poses_format 10 \
    --opt 2 \
    --output rtabmap \
    --output_dir "$OUTPUT_DIR" \
    "$DATABASE" >&2
else
  echo "Reusing existing bag trajectory: $ESTIMATED_POSES" >&2
fi

if command -v evo_ape >/dev/null 2>&1; then
  EVO_APE="$(command -v evo_ape)"
elif [[ -x /opt/evo-venv/bin/evo_ape ]]; then
  EVO_APE=/opt/evo-venv/bin/evo_ape
else
  echo "Error: evo_ape is not installed or available." >&2
  exit 1
fi

if ! POSITION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_POSES" -a 2>&1)"; then
  echo "$POSITION_OUTPUT" >&2
  exit 1
fi
POSITION_RMSE="$(printf '%s\n' "$POSITION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"

if ! ROTATION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_POSES" -a -r angle_deg 2>&1)"; then
  echo "$ROTATION_OUTPUT" >&2
  exit 1
fi
ROTATION_RMSE="$(printf '%s\n' "$ROTATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"

POSE_COUNT="$(awk '!/^#/ {count++} END {print count+0}' "$ESTIMATED_POSES")"

printf '\nPositional RMSE: %s m\n' "$POSITION_RMSE"
printf 'Rotational RMSE: %s deg\n' "$ROTATION_RMSE"
printf 'Evaluated poses: %s\n' "$POSE_COUNT"
printf 'RTAB-Map log: %s\n\n' "$LOG_FILE"
