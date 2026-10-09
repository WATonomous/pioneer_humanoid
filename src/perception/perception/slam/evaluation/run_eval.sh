#!/usr/bin/env bash

source /opt/ros/jazzy/setup.bash

set -euo pipefail

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
  FORCE=true
  shift
fi

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 [--force] ROS2_BAG GROUND_TRUTH.txt CONFIG.yaml" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_PATH="$(realpath "$1")"
GROUND_TRUTH="$(realpath "$2")"
CONFIG="$(realpath "$3")"

config_value() {
  local key="$1"
  awk -F':[[:space:]]*' -v key="$key" \
    '$1 == key {sub(/[[:space:]]*#.*/, "", $2); print $2; exit}' "$CONFIG"
}

RGB_TOPIC="$(config_value rgb_topic)"
DEPTH_TOPIC="$(config_value depth_topic)"
CAMERA_INFO_TOPIC="$(config_value camera_info_topic)"
FRAME_ID="$(config_value frame_id)"
PLAYBACK_RATE="$(config_value playback_rate)"
PLAYBACK_RATE="${PLAYBACK_RATE:-1.0}"

for value in RGB_TOPIC DEPTH_TOPIC CAMERA_INFO_TOPIC FRAME_ID; do
  if [[ -z "${!value}" ]]; then
    echo "Error: '$value' is missing from $CONFIG" >&2
    exit 1
  fi
done

if [[ -d "$BAG_PATH" ]]; then
  BAG_NAME="$(basename "$BAG_PATH")"
  BAG_PARENT="$(dirname "$BAG_PATH")"
elif [[ -f "$BAG_PATH" && "$BAG_PATH" == *.mcap ]]; then
  BAG_NAME="$(basename "${BAG_PATH%.mcap}")"
  BAG_PARENT="$(dirname "$BAG_PATH")"
else
  echo "Error: expected a ROS 2 bag directory or .mcap file: $BAG_PATH" >&2
  exit 1
fi

OUTPUT_DIR="$BAG_PARENT/${BAG_NAME}_eval"
DATABASE="$OUTPUT_DIR/rtabmap.db"
ESTIMATED_POSES="$OUTPUT_DIR/rtabmap_poses.txt"
LOG_FILE="$OUTPUT_DIR/rtabmap_ros.log"
mkdir -p "$OUTPUT_DIR"

stop_process() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 0
  kill -INT -- "-$pid" 2>/dev/null || true
  for _ in {1..20}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.25
  done
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in {1..20}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.25
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
}

if [[ "$FORCE" == true || ! -s "$ESTIMATED_POSES" ]]; then
  rm -f "$DATABASE" "$ESTIMATED_POSES" "$LOG_FILE"

  echo "Starting RGB-D topic relay..." >&2
  setsid python3 "$SCRIPT_DIR/normalize_rgbd_frames.py" --ros-args \
    -p input_rgb_topic:="$RGB_TOPIC" \
    -p input_depth_topic:="$DEPTH_TOPIC" \
    -p input_camera_info_topic:="$CAMERA_INFO_TOPIC" \
    -p frame_id:="$FRAME_ID" \
    >>"$LOG_FILE" 2>&1 &
  RELAY_PID=$!

  echo "Launching RTAB-Map..." >&2
  setsid ros2 launch rtabmap_launch rtabmap.launch.py \
    use_sim_time:=true \
    frame_id:="$FRAME_ID" \
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
    rtabmap_viz:=false \
    rviz:=false \
    database_path:="$DATABASE" \
    args:="-d" \
    >>"$LOG_FILE" 2>&1 &
  RTABMAP_PID=$!

  cleanup() {
    stop_process "$RTABMAP_PID"
    stop_process "$RELAY_PID"
    wait "$RTABMAP_PID" 2>/dev/null || true
    wait "$RELAY_PID" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM

  sleep 4
  if ! kill -0 "$RTABMAP_PID" 2>/dev/null; then
    echo "Error: RTAB-Map exited before playback. See $LOG_FILE" >&2
    exit 1
  fi

  echo "Playing ROS 2 bag..." >&2
  ros2 bag play "$BAG_PATH" --clock --rate "$PLAYBACK_RATE" --topics \
    "$RGB_TOPIC" "$DEPTH_TOPIC" "$CAMERA_INFO_TOPIC" >&2

  sleep 2
  cleanup
  trap - EXIT INT TERM

  if [[ ! -f "$DATABASE" ]]; then
    echo "Error: RTAB-Map did not create a database. See $LOG_FILE" >&2
    exit 1
  fi

  echo "Exporting estimated trajectory..." >&2
  rtabmap-export \
    --poses \
    --poses_format 10 \
    --opt 2 \
    --output rtabmap \
    --output_dir "$OUTPUT_DIR" \
    "$DATABASE" >&2
else
  echo "Reusing existing trajectory: $ESTIMATED_POSES" >&2
fi

if [[ ! -s "$ESTIMATED_POSES" ]]; then
  echo "Error: no estimated trajectory was exported. See $LOG_FILE" >&2
  exit 1
fi

if command -v evo_ape >/dev/null 2>&1; then
  EVO_APE="$(command -v evo_ape)"
elif [[ -x /opt/evo-venv/bin/evo_ape ]]; then
  EVO_APE=/opt/evo-venv/bin/evo_ape
else
  echo "Error: evo_ape is unavailable." >&2
  exit 1
fi

POSITION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_POSES" -a 2>&1)" || {
  echo "$POSITION_OUTPUT" >&2
  exit 1
}
ROTATION_OUTPUT="$($EVO_APE tum "$GROUND_TRUTH" "$ESTIMATED_POSES" -a -r angle_deg 2>&1)" || {
  echo "$ROTATION_OUTPUT" >&2
  exit 1
}

POSITION_RMSE="$(printf '%s\n' "$POSITION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
ROTATION_RMSE="$(printf '%s\n' "$ROTATION_OUTPUT" | awk '$1 == "rmse" {print $2; exit}')"
POSE_COUNT="$(awk '!/^#/ {count++} END {print count+0}' "$ESTIMATED_POSES")"

printf '\nPositional RMSE: %s m\n' "$POSITION_RMSE"
printf 'Rotational RMSE: %s deg\n' "$ROTATION_RMSE"
printf 'Evaluated poses: %s\n' "$POSE_COUNT"
printf 'RTAB-Map log: %s\n\n' "$LOG_FILE"
