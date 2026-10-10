#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 ROS2_BAG [OUTPUT_DIRECTORY]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_PATH="$(realpath "$1")"
BAG_NAME="$(basename "$BAG_PATH")"
OUTPUT_DIR="${2:-$(dirname "$BAG_PATH")/${BAG_NAME}_rtabmap}"

DATABASE="$OUTPUT_DIR/rtabmap.db"
FRONTEND_POSES="$OUTPUT_DIR/frontend_poses.txt"
CORRECTED_POSES="$OUTPUT_DIR/live_corrected_poses.txt"
BACKEND_POSES="$OUTPUT_DIR/backend_poses.txt"
LOG_FILE="$OUTPUT_DIR/rtabmap_ros.log"

if [[ ! -d "$BAG_PATH" && ! ( -f "$BAG_PATH" && "$BAG_PATH" == *.mcap ) ]]; then
  echo "Error: expected a ROS 2 bag directory or .mcap file: $BAG_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$DATABASE" "$FRONTEND_POSES" "$CORRECTED_POSES" "$BACKEND_POSES" "$LOG_FILE"

stop_process() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill -INT -- "-$pid" 2>/dev/null || true
  for _ in {1..20}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.25
  done
  kill -TERM -- "-$pid" 2>/dev/null || true
}

cleanup() {
  stop_process "${RTABMAP_PID:-0}"
  stop_process "${ODOM_RECORDER_PID:-0}"
  stop_process "${RELAY_PID:-0}"
  wait "${RTABMAP_PID:-0}" 2>/dev/null || true
  wait "${ODOM_RECORDER_PID:-0}" 2>/dev/null || true
  wait "${RELAY_PID:-0}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting TUM RGB-D relay..." >&2
setsid python3 "$SCRIPT_DIR/normalize_rgbd_frames.py" --ros-args \
  -p input_rgb_topic:=/camera/rgb/image_color \
  -p input_depth_topic:=/camera/depth/image \
  -p input_camera_info_topic:=/camera/rgb/camera_info \
  -p frame_id:=openni_rgb_optical_frame \
  >>"$LOG_FILE" 2>&1 &
RELAY_PID=$!

echo "Recording frontend and live-corrected trajectories..." >&2
setsid python3 "$SCRIPT_DIR/record_odom_trajectory.py" --ros-args \
  -p odom_topic:=/rtabmap/odom \
  -p output_path:="$FRONTEND_POSES" \
  -p corrected_output_path:="$CORRECTED_POSES" \
  -p map_frame:=map \
  >>"$LOG_FILE" 2>&1 &
ODOM_RECORDER_PID=$!

echo "Launching RTAB-Map..." >&2
setsid ros2 launch rtabmap_launch rtabmap.launch.py \
  use_sim_time:=true \
  frame_id:=openni_rgb_optical_frame \
  rgb_topic:=/eval/rgb/image \
  depth_topic:=/eval/depth/image \
  camera_info_topic:=/eval/rgb/camera_info \
  approx_sync:=true \
  approx_sync_max_interval:=0.05 \
  rgbd_sync:=true \
  topic_queue_size:=100 \
  sync_queue_size:=100 \
  qos:=1 \
  publish_tf_odom:=true \
  publish_tf_map:=true \
  map_frame_id:=map \
  rtabmap_viz:=false \
  rviz:=false \
  database_path:="$DATABASE" \
  args:="-d --Rtabmap/DetectionRate 10" \
  >>"$LOG_FILE" 2>&1 &
RTABMAP_PID=$!

sleep 4
if ! kill -0 "$RTABMAP_PID" 2>/dev/null || ! kill -0 "$ODOM_RECORDER_PID" 2>/dev/null || ! kill -0 "$RELAY_PID" 2>/dev/null; then
  echo "Error: a required process exited before playback. See $LOG_FILE" >&2
  exit 1
fi

echo "Playing TUM bag at 0.25x speed..." >&2
ros2 bag play "$BAG_PATH" --clock --rate 0.25 --topics \
  /camera/rgb/image_color \
  /camera/depth/image \
  /camera/rgb/camera_info >&2

sleep 2
cleanup
trap - EXIT INT TERM

if [[ ! -f "$DATABASE" ]]; then
  echo "Error: RTAB-Map did not create a database. See $LOG_FILE" >&2
  exit 1
fi

echo "Exporting optimized backend trajectory..." >&2
rtabmap-export \
  --poses \
  --poses_format 10 \
  --opt 2 \
  --output backend \
  --output_dir "$OUTPUT_DIR" \
  "$DATABASE" >&2

FRONTEND_COUNT="$(awk '!/^#/ {count++} END {print count+0}' "$FRONTEND_POSES")"
BACKEND_COUNT="$(awk '!/^#/ {count++} END {print count+0}' "$BACKEND_POSES")"
CORRECTED_COUNT="$(awk 'NF && !/^#/ {count++} END {print count+0}' "$CORRECTED_POSES")"

if [[ "$FRONTEND_COUNT" -eq 0 || "$BACKEND_COUNT" -eq 0 || "$CORRECTED_COUNT" -eq 0 ]]; then
  echo "Error: at least one trajectory is empty. See $LOG_FILE" >&2
  exit 1
fi

printf '\nFrontend trajectory: %s (%s poses)\n' "$FRONTEND_POSES" "$FRONTEND_COUNT"
printf 'Backend trajectory:  %s (%s poses)\n' "$BACKEND_POSES" "$BACKEND_COUNT"
printf 'Live-corrected trajectory: %s (%s poses)\n' "$CORRECTED_POSES" "$CORRECTED_COUNT"
printf 'RTAB-Map database:   %s\n' "$DATABASE"
printf 'RTAB-Map log:        %s\n\n' "$LOG_FILE"
