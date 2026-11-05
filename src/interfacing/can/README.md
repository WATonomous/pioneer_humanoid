# CAN interfacing (`can` package)

ROS 2 bridge: `/interfacing/motorCMD` ↔ CAN ↔ `/interfacing/motorFeedback`.
[Electrical docs](https://watonomous.github.io/humanoid-docs/electrical/index.html) · [Interfacing docs](https://watonomous.github.io/humanoid-docs/interfacing/index.html)

Bring-up (connect, verify, calibrate) is in [../ARM_BRINGUP.md](../ARM_BRINGUP.md).

## Cubemars Can ID configuration
0. Download required software. If you are using an AK motor, download AK Series V3.23 Upper computer (found here: https://www.cubemars.com/product/ak80-9-v3-0-robotic-actuator.html). If you are using GL40 motor, download the GL II Upper Computer V1.0 (found here: https://www.cubemars.com/product/gl40-II-kv82-5-gimbal-motor.html)

    Detatch the motor from the humanoid body. 
    >Tip: Be careful not to disassemble too much. You just need the motor USART socket to be in reach of your computer. Call a mechanical team member to help disassemble.

    Find the 3-wire USART to the R-LinkV3.0. Plug the end with the white connector into the Cubemars motor. For the other end:
    - If AK motor: Connect 3 Dupont jumper wires from USART header to R-Link headers. 
      - AK GND -> R-Link GND
      - AK TX -> R-Link RX
      - AK RX -> R-Link TX

    - If GL II Motor:
      - GL GND -> R-Link GND
      - GL TX -> R-Link TX
      - GL RX -> R-Link RX

    Find an XT-30(2+2) header that has a XT60 connector attached. Connect to CubeMars motor. Connect appropriate voltage supply to XT60
    - 51.2V supply for AK motors
    - 16V bench supply for GL's

  1. Open Cubemars Upper computer software. If you cannot read Chinese, You will need to toggle language to English. 
  > The UI is buggy, and will crop out UI config buttons. Using an external monitor is highly recommended. 

  2. Find COM Port that R-Link is connected to. Set baud rate to 921600. Connect. Motor should start streaming real-time data.
  3. On the "Basic Configuration" Panel on the left, find the Can ID. You will need to find the section below "Application configuration" section, but above the "Realtime Status" panel. Set Can ID to desired ID, then click write button. Then at the right most panel, verify that changes were saved by disconnect the serial port, reconnecting, and waiting a few seconds. Can ID should be the value you set.
  4. Test the motor encoder calibration. On the top-right "Control" panel, switch to MIT control. Click the set origin button to set current motor position to 0. Then set position and kp value to a small amount (ex. p=1rad, kp=1). Keep your hand on the E-stop. Test failing conditions (if motor does not fail these, then motor calibration is ok):
  - Motor does not move at all
  - Motor spins out of control
  > It is okay if the motor doesn't move to the specified position. This is due to friction, and it can easily be overcome through setting higher k_p values in the interfacing backend.
  5. If motor calibration tests failed: Disassemble the motor so that it is completely outside of its enclosure. Then run "Motor Idenficiation" and "Encoder Identification" in the left "Basic configuration" Panel. These calibrate the motor encoder (important for FoC drive — commutation sequence that makes motor spin as expected). Click "Write" after both calibration sequences. 


## Cubemars Can ID configuration
0. Download required software. If you are using an AK motor, download AK Series V3.23 Upper computer (found here: https://www.cubemars.com/product/ak80-9-v3-0-robotic-actuator.html). If you are using GL40 motor, download the GL II Upper Computer V1.0 (found here: https://www.cubemars.com/product/gl40-II-kv82-5-gimbal-motor.html)

    Detatch the motor from the humanoid body. 
    >Tip: Be careful not to disassemble too much. You just need the motor USART socket to be in reach of your computer. Call a mechanical team member to help disassemble.

    Find the 3-wire USART to the R-LinkV3.0. Plug the end with the white connector into the Cubemars motor. For the other end:
    - If AK motor: Connect 3 Dupont jumper wires from USART header to R-Link headers. 
      - AK GND -> R-Link GND
      - AK TX -> R-Link RX
      - AK RX -> R-Link TX

    - If GL II Motor:
      - GL GND -> R-Link GND
      - GL TX -> R-Link TX
      - GL RX -> R-Link RX

    Find an XT-30(2+2) header that has a XT60 connector attached. Connect to CubeMars motor. Connect appropriate voltage supply to XT60
    - 51.2V supply for AK motors
    - 16V bench supply for GL's

  1. Open Cubemars Upper computer software. If you cannot read Chinese, You will need to toggle language to English. 
  > The UI is buggy, and will crop out UI config buttons. Using an external monitor is highly recommended. 

  2. Find COM Port that R-Link is connected to. Set baud rate to 921600. Connect. Motor should start streaming real-time data.
  3. On the "Basic Configuration" Panel on the left, find the Can ID. You will need to find the section below "Application configuration" section, but above the "Realtime Status" panel. Set Can ID to desired ID, then click write button. Then at the right most panel, verify that changes were saved by disconnect the serial port, reconnecting, and waiting a few seconds. Can ID should be the value you set.
  4. Test the motor encoder calibration. On the top-right "Control" panel, switch to MIT control. Click the set origin button to set current motor position to 0. Then set position and kp value to a small amount (ex. p=1rad, kp=1). Keep your hand on the E-stop. Test failing conditions (if motor does not fail these, then motor calibration is ok):
  - Motor does not move at all
  - Motor spins out of control
  > It is okay if the motor doesn't move to the specified position. This is due to friction, and it can easily be overcome through setting higher k_p values in the interfacing backend.
  5. If motor calibration tests failed: Disassemble the motor so that it is completely outside of its enclosure. Then run "Motor Idenficiation" and "Encoder Identification" in the left "Basic configuration" Panel. These calibrate the motor encoder (important for FoC drive — commutation sequence that makes motor spin as expected). Click "Write" after both calibration sequences. 


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
start if `quantised kp × mit_max_track_err (+ feed-forward) > mit_peak_torque` (`mit_max_torque`
when no peak is set). Gains are snapped
to the nearest 12-bit code. The gripper is driven from `ArmPose.gripper_position`
([joint_command/README.md](../joint_command/README.md#gripper)).

## Both arms on one bus

Both arms' harnesses join at the one adapter, so they are one bus (`can0`) and one `can_node`.
That only works with:
- **Wiring:** one line, left end → adapter → right end, a 120 Ω terminator at each far end and the
  adapter's own off. Unpowered, CAN_H to CAN_L reads ≈ 60 Ω (≈ 40 Ω: three terminators; ≈ 120 Ω:
  one end unterminated).
- **Ids:** every motor's id unique and ≤ 127 (`motor_id` is `int8`). The GL40s all reply on master
  id 0 with only their id's last hex digit, so every GL40 needs a different last hex digit, and no
  AK may end in the same hex digit as a GL40 (`can_node` drops that AK's profile). Set the GL40
  master id to 0 and keep the drives' MIT ranges as in `mit_profiles.yaml`.
- Every motor listed in `config/mit_profiles.yaml`.

**AK bring-up, per joint** (arm supported, hardware E-stop in reach):
1. `candump can0`: extended frames on `0000080<id>`, KP first, no `FF..FC` to the AK.
2. Servo-mode and zero-gain-MIT readings at one pose agree within 0.5°.
3. Hold with low gains (kp 2, kd 0.3) first, watching torque, tracking error and temperature.
4. Stop `joint_command` and record what the drive does on stream loss (the V3 manual documents
   no CAN timeout).
5. Hold 30 s, then ±5°: `tools/arm_roundtrip.sh --joints shoulder.pitch --offset "5,0,0,0,0,0"`.

## Live joint angles (`arm_angles.py`)

Read-only readout of every joint in the command frame (degrees, from `arm_calibration.yaml`), like
the leader's `encoder_test.py`. It publishes nothing, so it is safe next to anything else:

```bash
python3 /opt/humanoid_scripts/arm_angles.py [--ids 14,11] [--raw]   # in the joint_command container
```

## Angle benchmarks (`arm_roundtrip.py`)

`tools/arm_roundtrip.sh` runs `scripts/arm_roundtrip.py` in the `joint_command` container. It
publishes `ArmPose` like teleop, so the clamp, `velocity_max` and MIT watchdog all apply.

```bash
tools/arm_roundtrip.sh --joints elbow.roll --offset "0,0,0,0,5,0"    # one joint, +5 deg
tools/arm_roundtrip.sh --offset "3,1.5,5,5,5,10" --dwell 5            # all six at once
tools/arm_roundtrip.sh --arm-side both --joints left.elbow.roll,right.elbow.roll \
  --offset "0,0,0,0,5,0,0,0,0,0,5,0"                                  # both arms, one run
```

It holds at the origin, ramps out (cosine, ≤ `--vel` and `velocity_max`), dwells, returns and
rests. `--arm-side` (default `left`) picks the arm and its topic, `/arm/<side>/joint_targets`.
`--arm-side both` drives both arms together: joints are prefixed `left.`/`right.` and
`--offset`/`--pose` take twelve values, the left arm's six then the right's.
It refuses to start if something else publishes that topic, a moving joint is
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

## Windows + WSL2 with a USB SLCAN adapter

This is a workaround for running the CAN container from Ubuntu 24.04 on WSL2 when Docker
Desktop's engine cannot create or share the SocketCAN interface with the container. It uses
Ubuntu's own Docker Engine inside WSL, with the container on WSL's host network. It does not
require native Linux, but it is not a substitute for checking the CAN bus and motor feedback.

The tested adapter was a DSD TECH SH-C31G (CANable 2.0 Pro firmware), forwarded into WSL as
`/dev/ttyACM0`. Windows USB forwarding and the basic WSL/Docker installations are prerequisites;
this section covers the WSL-specific setup and run steps.

### 1. Confirm the forwarded device

In Ubuntu WSL, check that the adapter is present and that the path expected by the CAN container
resolves to it:

```bash
ls -l /dev/ttyACM0 /dev/canable
```

In this setup `/dev/canable` was a symlink to `/dev/ttyACM0`. If it is missing, create it after
confirming the current device name:

```bash
sudo ln -s /dev/ttyACM0 /dev/canable
```

USB device names can change after detach/reboot; check the device before recreating the symlink.
If you previously ran `slcand` manually in WSL to test the adapter, stop that process before
starting the container so it does not contend for the serial adapter.

### 2. Start the Docker Engine inside WSL

In **terminal 1**, start a separate daemon with its own socket and data directory, using the `vfs`
storage driver:

```bash
sudo dockerd \
  --host=unix:///run/docker-wsl.sock \
  --data-root=/var/lib/docker-wsl-vfs \
  --exec-root=/run/docker-wsl-vfs \
  --pidfile=/run/docker-wsl-vfs.pid \
  --storage-driver=vfs
```

Leave this terminal running while using this engine. `vfs` was needed in the tested setup because
the initial `overlayfs` daemon could not start containers; it is slower and less storage-efficient.
The custom socket and data root keep this daemon separate from Docker Desktop. Commands below
explicitly select it with `DOCKER_HOST`; omitting that prefix may use a different Docker Engine.

### 3. Build and start only the CAN service

In **terminal 2**, from the repository root, build and start `interfacing` using the WSL daemon:

```bash
DOCKER_HOST=unix:///run/docker-wsl.sock ./watod build interfacing
DOCKER_HOST=unix:///run/docker-wsl.sock ./watod up -d interfacing
DOCKER_HOST=unix:///run/docker-wsl.sock ./watod logs --tail=30 interfacing
```

The successful run logged the setup script starting `slcand -o -c -s8`, bringing up `can0`, and
initializing the CAN core. `-s8` selects the adapter's 1 Mbps SLCAN rate. The initial image build
took about 14 minutes on the test machine.

### 4. Verify from inside the container

Open a shell in the running container:

```bash
DOCKER_HOST=unix:///run/docker-wsl.sock ./watod exec interfacing /bin/bash
```

At the container prompt, source the ROS environment before using ROS commands:

```bash
source /opt/watonomous/setup.bash
ros2 node list
ros2 topic list
ip -s -details link show can0
```

The tested setup showed `/can_node`, `/interfacing/motorCMD`, and
`/interfacing/motorFeedback`; `can0` was `UP`. `docker compose exec` does not run the container
entrypoint, so ROS may not be on `PATH` until `setup.bash` is sourced in the shell.

### What this verifies—and what it does not

This verifies that the WSL-local Docker Engine can run the CAN container, create `can0`, and
start the ROS CAN node. A feedback topic publisher only means the ROS publisher exists. In the
test, CAN RX/TX counters remained zero and `ros2 topic echo --once /interfacing/motorFeedback`
received no message. Therefore, actual bus wiring, motor communication, and feedback were **not**
verified. `can state ERROR-ACTIVE` and an `UP` interface do not prove that motors are connected;
SLCAN may also report `bitrate 0` in `ip link` even though the setup log shows `-s8`.

Do not start `joint_command`, enable motor power, or send joint targets based only on these checks.
Follow [../ARM_BRINGUP.md](../ARM_BRINGUP.md) and the team's approved bench procedure to verify
wiring, termination, calibration, feedback, and the actuator configuration before any powered
test. Keep the arm secured and the hardware E-stop within reach for every powered test.
