# CAN interfacing (`can` package)

ROS 2 bridge: `/interfacing/motorCMD` ↔ CAN ↔ `/interfacing/motorFeedback`.
[Electrical docs](https://watonomous.github.io/humanoid-docs/electrical/index.html) · [Interfacing docs](https://watonomous.github.io/humanoid-docs/interfacing/index.html)

Bring-up (connect, verify, calibrate) is in [../README.md](../README.md).

## Two encodings: servo mode vs MIT

`MotorCmd.control_type` picks how `can_node` builds the CAN frame:

| | Servo mode (`POSITION_LOOP`, `SET_ORIGIN`, disable, …) | MIT (`MIT_CONTROL`, enter / exit) |
|---|---|---|
| Frame layout + scaling | [`../dbc/humanoid.dbc`](../dbc/README.md), via `libdbcppp` | `config/mit_profiles.yaml` (each motor's family and ranges) + `src/mit_protocol.cpp` |
| Why | one id per motor, same scaling for every motor | scaling depends on the motor's range; GL II feedback shares one id |

Exceptions: the GL II MIT command uses the DBC's `MITControlCmd` layout (codes already scaled by
`mit_protocol.cpp`), and AK feedback is always the DBC's `ServoStatusFeedback`, even in MIT.
Each joint's `control_type` is set in `joint_command/config/arm_actuators.yaml`.

## gs_usb / candleLight adapter

`lsusb` shows `1d50:606f`: it is a native `can0`, so skip `can_udev.sh`/slcand:
```bash
./src/interfacing/can/scripts/setup_socketcan.sh can0 1000000   # [--listen-only] to sniff only
ros2 run can can_node --ros-args --params-file $(ros2 pkg prefix can)/share/can/config/params.yaml \
  -p bustype:=socketcan
```

## MIT mode

`config/mit_profiles.yaml` sets each motor's `family`:
- **gl2**: GL40 wrist (22) / gripper (21) on a GL II drive. Standard frame on the node id,
  `MIT_ENTER` before commands, feedback on the master id (`mit_master_id`, default `0x000`)
  ([manual §5](https://www.cubemars.com/images/file/20241231/1735633965678815.pdf)).
- **ak**: AK10-9 / AK80-9 on V3.0 firmware. Extended frame on `0x800 | id`, KP-first payload, no
  special frames (`MIT_SET_ZERO` is refused; use `SET_ORIGIN`). Feedback is the servo frame, with
  `torque = current × kt`.

Per-joint gains live in `joint_command/config/arm_actuators.yaml`; `joint_command` refuses to
start if `quantised kp × mit_max_track_err (+ feed-forward) > mit_max_torque`. Gains are snapped
to the nearest 12-bit code. The gripper is driven from `ArmPose.gripper_position`
([joint_command/README.md](../joint_command/README.md#gripper)).

**AK bring-up, per joint** (arm supported, hardware E-stop in reach):
1. `candump can0`: extended frames on `0000080<id>`, KP first, no `FF..FC` to the AK.
2. Servo-mode and zero-gain-MIT readings at one pose agree within 0.5°.
3. Hold with low gains (kp 2, kd 0.3) first, watching torque, tracking error and temperature.
4. Stop `joint_command` and record what the drive does on stream loss (the V3 manual documents
   no CAN timeout).
5. Hold 30 s, then ±5°: `tools/arm_roundtrip.sh --joints shoulder.pitch --offset "5,0,0,0,0,0"`.

## Angle benchmarks (`arm_roundtrip.py`)

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

## Telemetry

Runs write `outputs/gl40_bench/<run>/` (`telemetry.csv`, `run.json`; `--no-log` opts out). Plot on
the host:

```bash
uv run --with matplotlib --with numpy tools/gl40_telemetry_plot.py outputs/gl40_bench/<run>
```

---

## Topics / config

`/interfacing/motorCMD` (`MotorCmd`, ROS→CAN) · `/interfacing/motorFeedback` (`MotorFeedback`, CAN→ROS)

`config/params.yaml` defaults: `can_interface=can0` `device_path=/dev/canable` `bustype=slcan` `bitrate=1000000`

DBC: `src/interfacing/dbc/humanoid.dbc` · Debug: `candump can0`
