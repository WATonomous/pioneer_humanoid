#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
set -uo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
  echo "Usage: $0 ROS2_BAG OUTPUT_DIR RGB_TOPIC DEPTH_TOPIC CAMERA_INFO_TOPIC FRAME_ID [IMU_TOPIC | GYRO_TOPIC ACCEL_TOPIC]" >&2
  exit 1
fi

BAG="$(realpath "$1")"
OUT="$2"
RGB_TOPIC="$3"
DEPTH_TOPIC="$4"
CAMERA_INFO_TOPIC="$5"
FRAME_ID="$6"
IMU_TOPIC="${7:-}"
ACCEL_TOPIC="${8:-}"
DB="$OUT/rtabmap.db"
LOG="$OUT/rtabmap.log"
PLAYER_LOG="$OUT/player.log"
mkdir -p "$OUT"
rm -f "$DB" "$OUT/trajectory.txt" "$LOG" "$PLAYER_LOG"

cleanup() {
  for pid in "${PLAYER_PID:-}" "${RECORDER_PID:-}" "${COMBINER_PID:-}" "${FILTER_PID:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  [[ -n "${RTABMAP_PID:-}" ]] && kill -- "-$RTABMAP_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PLAYER_EXCLUDES=(/tf /tf_static)
if [[ -n "$ACCEL_TOPIC" ]]; then
  PLAYER_EXCLUDES=(/tf)
fi
ros2 bag play "$BAG" --clock --rate 0.1 --read-ahead-queue-size 1000 \
  --exclude-topics "${PLAYER_EXCLUDES[@]}" \
  --start-paused --disable-keyboard-controls >"$PLAYER_LOG" 2>&1 &
PLAYER_PID=$!
CLOCK_READY=false
for _ in {1..20}; do
  if [[ "$(ros2 topic type /clock 2>/dev/null || true)" == "rosgraph_msgs/msg/Clock" ]] && \
     timeout 2 ros2 topic echo /clock --once >/dev/null 2>&1; then
    CLOCK_READY=true
    break
  fi
  sleep 0.5
done
if [[ "$CLOCK_READY" != true ]]; then
  echo "Error: bag player did not publish /clock. See $PLAYER_LOG" >&2
  exit 1
fi
kill -0 "$PLAYER_PID" 2>/dev/null || { echo "Error: bag player exited. See $PLAYER_LOG" >&2; exit 1; }

if [[ -n "$ACCEL_TOPIC" ]]; then
  python3 "$(dirname "$0")/combine_imu.py" --ros-args \
    -p use_sim_time:=true -p gyro_topic:="$IMU_TOPIC" -p accel_topic:="$ACCEL_TOPIC" \
    -p output_topic:=/rtabmap/imu_raw >"$OUT/imu_combiner.log" 2>&1 &
  COMBINER_PID=$!
  ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args \
    -p use_sim_time:=true -p use_mag:=false -p publish_tf:=false \
    --remap /imu/data_raw:=/rtabmap/imu_raw \
    --remap /imu/data:=/rtabmap/imu_filtered >"$OUT/imu_filter.log" 2>&1 &
  FILTER_PID=$!
  IMU_TOPIC=/rtabmap/imu_filtered
fi

RTABMAP_ARGS=(
  use_sim_time:=true frame_id:="$FRAME_ID"
  rgb_topic:="$RGB_TOPIC" depth_topic:="$DEPTH_TOPIC"
  camera_info_topic:="$CAMERA_INFO_TOPIC" approx_sync:=true
  approx_sync_max_interval:=0.05 topic_queue_size:=100 queue_size:=100 qos:=1 rgbd_sync:=true
  odom_always_process_most_recent_frame:=false
  odom_args:="--Odom/ResetCountdown 5 --Odom/GuessMotion true"
  rtabmap_viz:=false rviz:=false publish_tf_odom:=true publish_tf_map:=true
  map_frame_id:=map database_path:="$DB"
  args:="-d --Rtabmap/DetectionRate 10"
)
if [[ -n "$IMU_TOPIC" ]]; then
  RTABMAP_ARGS+=(imu_topic:="$IMU_TOPIC" wait_imu_to_init:=true)
fi

setsid ros2 launch rtabmap_launch rtabmap.launch.py \
  "${RTABMAP_ARGS[@]}" >"$LOG" 2>&1 &
RTABMAP_PID=$!

READY=false
for _ in {1..30}; do
  NODES="$(ros2 node list 2>/dev/null || true)"
  if grep -qx '/rtabmap/rgbd_odometry' <<<"$NODES" && grep -qx '/rtabmap/rtabmap' <<<"$NODES"; then
    READY=true
    break
  fi
  sleep 1
done
if [[ "$READY" != true ]]; then
  echo "Error: RTAB-Map did not start. See $LOG" >&2
  exit 1
fi

python3 "$(dirname "$0")/record_live_trajectory.py" --ros-args \
  -p output_path:="$OUT/trajectory.txt" >"$OUT/trajectory_recorder.log" 2>&1 &
RECORDER_PID=$!

timeout 5 ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume "{}" >/dev/null 2>&1 || true
sleep 1
if ! grep -q 'Resuming play' "$PLAYER_LOG"; then
  echo "Error: bag playback did not resume. See $PLAYER_LOG" >&2
  exit 1
fi
if wait "$PLAYER_PID"; then
  PLAYER_PID=""
else
  PLAYER_RC=$?
  PLAYER_PID=""
  echo "Error: bag player failed (exit $PLAYER_RC). See $PLAYER_LOG" >&2
  exit "$PLAYER_RC"
fi
sleep 2
PLAYER_PID=""
cleanup
trap - EXIT INT TERM

echo "Playback diagnostics:"
for pattern in \
  'Message queue starved' \
  'not part of the same tree' \
  'Could not get odometry value' \
  'Registration failed' \
  'Odom: quality=0' \
  'Detected jump back in time' \
  'arrived out of order'; do
  count="$(grep -c "$pattern" "$PLAYER_LOG" "$LOG" 2>/dev/null || true)"
  count="$(printf '%s\n' "$count" | awk -F: '{sum += $NF} END {print sum + 0}')"
  [[ "$count" -gt 0 ]] && printf '  %s: %s\n' "$pattern" "$count"
done

if [[ ! -s "$OUT/trajectory.txt" || "$(grep -vc '^#' "$OUT/trajectory.txt")" -eq 0 ]]; then
  echo "Error: no live trajectory was recorded. See $OUT/trajectory_recorder.log" >&2
  exit 1
fi
printf 'Trajectory: %s\n' "$OUT/trajectory.txt"
