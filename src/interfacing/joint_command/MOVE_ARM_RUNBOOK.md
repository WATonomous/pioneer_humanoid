# Move / wave the arm — runbook

Drive the real arm through the `joint_command` safety layer: it seeds from live feedback
(no startup slam), then velocity-limits + smooths every command. Read-only visualization
is separate (`pioneer_bimanual_arm/live_arm_mjviser.py`).

> ⚠️ Moves real motors. Arm clear, hand on the E-stop. Position clamp **enabled**,
> `velocity_max` 10°/s; a joint found outside its limits is excluded, not clamped.

Names below assume the `watod_hy-*` project and container `watod_hy-jc-dry`.

## Setup (once)
```bash
# interfacing up (publishes /interfacing/motorFeedback off CAN)
docker compose --env-file modules/.env -p watod_hy \
  -f modules/docker-compose.interfacing.yaml --profile develop up -d interfacing

# joint_command container (host net so it discovers the interfacing topics)
docker rm -f watod_hy-jc-dry 2>/dev/null
docker run -d --name watod_hy-jc-dry --network host \
  -e ROS_DOMAIN_ID=0 -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v "$PWD/src/interfacing/joint_command:/root/ament_ws/src/joint_command" \
  --entrypoint tail ghcr.io/watonomous/humanoid/interfacing/joint_command:dev_main -F /dev/null

# build (first time, or after editing C++ / arm_actuators.yaml)
docker exec watod_hy-jc-dry bash -c 'source /opt/ros/humble/setup.bash; source /opt/watonomous/setup.bash; \
  cd /root/ament_ws && colcon build --packages-select joint_command --cmake-args -DCMAKE_BUILD_TYPE=Release'
```

## Start the node (once — keep it up)
Start the node **once** and leave it running. It holds state across targets (its rate-limiter
ramps from the last commanded pose to the next), so sending pose → pose → pose, or waving, all
work with one persistent node — no per-move restart. Only restart when:
- you **hand-moved the arm** while disabled (the node must re-seed from real feedback), or
- **discovery dropped** — `ros2 topic info /interfacing/motorCMD` shows `Publisher count: 0`.

```bash
docker exec watod_hy-jc-dry bash -c 'pkill -f "install/joint_command/lib"; sleep 2'
docker exec -d watod_hy-jc-dry bash -c 'source /opt/ros/humble/setup.bash; source /opt/watonomous/setup.bash; \
  source /root/ament_ws/install/setup.bash; \
  ros2 run joint_command joint_command_node --ros-args \
  --params-file /root/ament_ws/src/joint_command/config/joint_command.yaml \
  -p motor_cmd_topic:=/interfacing/motorCMD'
sleep 4
# MUST read Publisher count: 1 AND Subscription count: 1 before moving:
docker exec watod_hy-jc-dry bash -c 'source /opt/ros/humble/setup.bash; source /opt/watonomous/setup.bash; \
  ros2 topic info /interfacing/motorCMD'
```
Dry run (nothing reaches motors): same launch but `-p motor_cmd_topic:=/dry_run/motorCMD`.

## Move to a pose
Stream one target at 50 Hz; the node ramps to it and **holds** when you stop.
Slot order: `shoulder=[pitch(14), roll(13), yaw(12)]`, `elbow=[pitch(11), roll(104)]`, `wrist=[pitch(22)]`.
```bash
docker exec -d watod_hy-jc-dry bash -c 'source /opt/ros/humble/setup.bash; source /opt/watonomous/setup.bash; \
  source /root/ament_ws/install/setup.bash; \
  timeout 12 ros2 topic pub -r 50 /arm/joint_targets common_msgs/msg/ArmPose \
  "{is_left: true, shoulder: {position: [-124.0, -40.5, -165.4]}, elbow: {position: [-88.7, 14.1]}, wrist: {position: [0.0]}, include_hand_pose: false}"'
```

## Record poses by hand (free the arm)
Disabling the motors makes them limp — **support the arm first**. Then hand-position and read.
```bash
# free the arm
docker exec watod_hy-jc-dry bash -c 'pkill -f "install/joint_command/lib"; sleep 2'
docker exec watod_hy-interfacing-1 bash -c 'source /opt/watonomous/setup.bash; \
  for id in 10 11 12 13 14; do ros2 topic pub --once /interfacing/motorCMD common_msgs/msg/MotorCmd "{motor_id: $id, control_type: 8}"; done'

# read the held pose (14=sh_pitch 12=sh_roll 13=sh_yaw 10=el_pitch 11=el_roll)
docker exec watod_hy-interfacing-1 bash -c 'source /opt/watonomous/setup.bash; \
  timeout 2 ros2 topic echo /interfacing/motorFeedback 2>/dev/null | awk "/motor_id:/{id=\$2} /position:/{p[id]=\$2} \
  END{printf \"14=%s 12=%s 13=%s 10=%s 11=%s\n\", p[14],p[12],p[13],p[10],p[11]}"'
```
Values are in raw motor-frame degrees; convert via each joint's `zero_offset`/`direction` in
`arm_calibration.yaml` before using as an `/arm/joint_targets` target (cmd-frame).
Recorded poses go stale whenever calibration changes — re-capture after any recalibration
rather than reusing old numbers.

## Watch feedback (any time)
```bash
docker exec watod_hy-interfacing-1 bash -c 'source /opt/watonomous/setup.bash; \
  timeout 2 ros2 topic echo /interfacing/motorFeedback 2>/dev/null | awk "/motor_id:/{id=\$2} /position:/{p[id]=\$2} \
  END{printf \"14=%s 12=%s 13=%s 10=%s 11=%s\n\", p[14],p[12],p[13],p[10],p[11]}"'
```

## Stop
```bash
docker exec watod_hy-jc-dry bash -c 'pkill -f "install/joint_command/lib"'  # arm holds last pose
# (to also free the arm, send control_type 8 as in the record step)
```

## Notes
- **Speed / smoothing:** edit `config/arm_actuators.yaml` (`velocity_max`, `low_pass_alpha`,
  the `enable_*` flags), then rebuild (Setup) + restart the node. Currently 10°/s, clamp on.
- **Telemetry:** `tools/arm_roundtrip.sh --offset "..."` moves, records and plots into
  `outputs/gl40_bench/<run>/`.
- **MIT joints** (all six) have a torque / tracking / feedback watchdog that latches on a fault.
- **Didn't move?** `ros2 topic info /interfacing/motorCMD` shows `Publisher count: 0` → node
  lost discovery; restart it (safe — it re-seeds from live feedback). Discovery drops are
  mostly triggered by churning many short-lived `ros2 topic pub/echo/hz` processes on the host
  net — keep one long publisher and sample feedback sparingly to avoid it.
- **6-vs-5 cmds:** `ros2 echo/hz` may show only 5 motor ids — a subscriber-side first-in-burst
  drop; the node log `Publishing 6 MotorCmd ...` is authoritative.
