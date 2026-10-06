# CAN interfacing (`can` package)

ROS 2 bridge: `/interfacing/motorCMD` ↔ CAN ↔ `/interfacing/motorFeedback`.
[Electrical docs](https://watonomous.github.io/humanoid-docs/electrical/index.html) · [Interfacing docs](https://watonomous.github.io/humanoid-docs/interfacing/index.html)

## Arm bring-up

1. **Hardware**: battery + E-stop closed (motor power ≠ CAN power). CANable USB → host; CAN_H/CAN_L → arm (120 Ω term).
2. **Host setup (once)**: `./src/interfacing/can/scripts/can_udev.sh install` → `/dev/canable`.
   `watod-config.local.sh`: `ACTIVE_MODULES="interfacing"`, `MODE_OF_OPERATION="develop"`.
3. **Bring up**:
   ```bash
   ./watod build && ./watod up -d
   ./watod -t interfacing
   source /opt/watonomous/setup.bash
   ```
   `can.launch.py` starts `can_node` + SLCAN (`/dev/canable` → `can0` @ 1 Mbps).

**gs_usb / candleLight adapter** (`lsusb` shows `1d50:606f`): it is a native `can0`, so skip
`can_udev.sh`/slcand:
```bash
./src/interfacing/can/scripts/setup_socketcan.sh can0 1000000   # [--listen-only] to sniff only
ros2 run can can_node --ros-args --params-file $(ros2 pkg prefix can)/share/can/config/params.yaml \
  -p bustype:=socketcan
```

### Verify
```bash
ros2 node list                  # /can_node
candump can0                    # e.g. 0x290A–0x290E
ros2 topic echo /interfacing/motorFeedback common_msgs/msg/MotorFeedback --once
```

### Calibrate (`calibrate_arm.py`)
Per joint: confirm motor id → home zero → one end Enter → other end Enter → writes `zero_offset`/limits/`can_id`.
```bash
source /opt/watonomous/setup.bash
python3 /root/ament_ws/src/interfacing/can/scripts/calibrate_arm.py \
  --arm-side left --write-mapping --calibration /calibration/arm_calibration.yaml
```
Prompt: **Enter**=yes · id=correct id · **s**=skip · **q**=quit.

### MIT mode

`config/mit_profiles.yaml` sets each motor's `family`:
- **gl2**: GL40 wrist (22) / gripper (21) on a GL II drive. Standard frame on the node id,
  `MIT_ENTER` before commands, feedback on the master id (`mit_master_id`, default `0x000`)
  ([manual §5](https://www.cubemars.com/images/file/20241231/1735633965678815.pdf)).
- **ak**: AK10-9 / AK80-9 on V3.0 firmware. Extended frame on `0x800 | id`, KP-first payload, no
  special frames (`MIT_SET_ZERO` is refused; use `SET_ORIGIN`). Feedback is the servo frame, with
  `torque = current × kt`.

Per-joint gains live in `joint_command/config/arm_actuators.yaml`; `joint_command` refuses to
start if `quantised kp × mit_max_track_err (+ feed-forward) > mit_max_torque`. Gains are snapped
to the nearest 12-bit code. The gripper has no `ArmPose` slot, so nothing drives it yet.

**AK bring-up, per joint** (arm supported, hardware E-stop in reach):
1. `candump can0`: extended frames on `0000080<id>`, KP first, no `FF..FC` to the AK.
2. Servo-mode and zero-gain-MIT readings at one pose agree within 0.5°. AK80-9s (11, 12, 104)
   lose zero on every power cycle.
3. Hold with low gains (kp 2, kd 0.3) first, watching torque, tracking error and temperature.
4. Stop `joint_command` and record what the drive does on stream loss (the V3 manual documents
   no CAN timeout).
5. Hold 30 s, then ±5°: `tools/arm_roundtrip.sh --joints shoulder.pitch --offset "5,0,0,0,0,0"`.

### Angle benchmarks (`arm_roundtrip.py`)

`tools/arm_roundtrip.sh` runs `scripts/arm_roundtrip.py` in the `joint_command` container. It
publishes `ArmPose` like teleop, so the clamp, `velocity_max` and MIT watchdog all apply.

```bash
tools/arm_roundtrip.sh --joints elbow.roll --offset "0,0,0,0,5,0"    # one joint, +5 deg
tools/arm_roundtrip.sh --offset "3,1.5,5,5,5,10" --dwell 5            # all six at once
```

It holds at the origin, ramps out (cosine, ≤ `--vel` and `velocity_max`), dwells, returns and
rests. It refuses to start if something else publishes `/arm/joint_targets`, a moving joint is
silent or near a limit, a move exceeds `--max-delta`, or `joint_command`'s installed configs
differ from the repo's (rebuild after every calibration or limits edit). Ctrl-C ramps back, a
second Ctrl-C stops streaming, and lost feedback freezes in place. `--exercise-limits` sends an
over-limit request and checks that `joint_command` clamped it.

### Telemetry

Runs write `outputs/gl40_bench/<run>/` (`telemetry.csv`, `run.json`; `--no-log` opts out). Plot on
the host:

```bash
uv run --with matplotlib --with numpy tools/gl40_telemetry_plot.py outputs/gl40_bench/<run>
```

---

## Open arm tasks (onboarding / assignable)

Live joint mirror, mjlab sim parity, and interactive calibration are done — see
[../README.md](../README.md) for calibrate → visualize → move.

| Status | Task | Why |
|--------|------|-----|
| TODO | **VR teleop** — Quest → real motors via teleop + `joint_command` / CAN | End-to-end teleop UX |
| TODO (later) | **Isaac Lab sim-to-real** — `task_space_ik.py --publish-real-left-arm` (IK) + `reach` RL task driving the real arm | Validate IK/policy against real hardware |

---

## Topics / config

`/interfacing/motorCMD` (`MotorCmd`, ROS→CAN) · `/interfacing/motorFeedback` (`MotorFeedback`, CAN→ROS)

`config/params.yaml` defaults: `can_interface=can0` `device_path=/dev/canable` `bustype=slcan` `bitrate=1000000`

DBC: `src/interfacing/dbc/humanoid.dbc` · Debug: `candump can0`
