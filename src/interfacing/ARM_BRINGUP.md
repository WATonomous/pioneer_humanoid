# Real arm bring-up: connect → calibrate → visualize → move

Follow top to bottom. Package details: [can/README.md](can/README.md) (CAN bridge, MIT mode,
benchmarks), [joint_command/README.md](joint_command/README.md) (command node, configs).

## 0. Connect
1. **Hardware**: battery + E-stop closed (motor power ≠ CAN power). CANable USB → host; CAN_H/CAN_L → arm (120 Ω term).
> ⚠️ If connecting voltage source XT60 connectors to the motors, ensure you DONT connect the 51.2V battery to the GL40 batteries -- it will immediately fry the motor driver and render it unusable. \
\
> ❗ The CAN adapter is a little buggy. When you first plug it in, the LED will be red. You must unplug it and plug it in a second time so that the green light flashes. Then you can run the CAN udev scripts.
2. **Host setup (once)**: `./src/interfacing/can/scripts/can_udev.sh install` → `/dev/canable`.
   `watod-config.local.sh`: `ACTIVE_MODULES="interfacing"`, `MODE_OF_OPERATION="develop"`.
3. **Bring up**: `can.launch.py` starts `can_node` + SLCAN (`/dev/canable` → `can0` @ 1 Mbps).
   ```bash
   ./watod build && ./watod up -d
   ./watod -t interfacing
   source /opt/watonomous/setup.bash
   ```
4. **Verify**:
   ```bash
   ros2 node list                  # /can_node
   candump can0                    # e.g. 0x290A–0x290E
   ros2 topic echo /interfacing/motorFeedback common_msgs/msg/MotorFeedback --once
   ```

## 1. Calibrate
Per-joint zero + limits, from live motor feedback, with the arm **hanging** (the URDF zero).
**Re-run after every power-on** for elbow.pitch, elbow.roll, and shoulder.yaw (AK80-9 motors) —
their single-turn absolute encoders don't reliably survive a power cycle; shoulder.pitch/roll
(AK10-9) have so far.

Per joint: confirm motor id → home zero → one end Enter → other end Enter → writes `zero_offset`/limits/`can_id`.
```bash
source /opt/watonomous/setup.bash
ros2 run can calibrate_arm.py \
  --arm-side left --write-mapping
```
Prompt: **Enter**=yes · id=correct id · **s**=skip · **q**=quit.

> You will hear buzzing as the motors are activated. Don't worry, that is normal.


Gripper is unique: 
1. Set home position just small of fully open (5deg smaller). 
2. First pose: Fully open. Second pose: fully closed
3. If gripper lower limit in `arm_calibration` >0, swap the value with 0

## 2. Visualize
Read-only mirror of live motor feedback in the browser. Confirms calibration looks right
before commanding anything.

```bash
./watod -t simulation_mj
python3 /workspace/humanoid/src/interfacing/can/scripts/live_arm_mjviser.py --arm-side left
# open http://localhost:8080
```

> **The command frame is the URDF frame** (`pioneer_bimanual_arm.urdf`): calibrate hanging, with
> each joint's `direction` matching the URDF. Before trusting the view: move each of the six
> joints by hand, one at a time, and confirm the on-screen joint turns the same way and stops
> at the same angle. Try fixes with `--flip JOINT` / `--offset JOINT=DEG` (viewer-only), then
> make them real: a flip is the joint's `direction` in
> [joint_command/config/arm_calibration.yaml](joint_command/config/arm_calibration.yaml)
> (then re-run `calibrate_arm.py`); an offset means it was not zeroed hanging (re-run it).

→ script docstring in `live_arm_mjviser.py` for the angle math and full flag reference.

## 3. Setup joint_command (real motor control)
Only after 1–2 look right. Clear workspace, hardware E-stop on the 48 V supply in reach.
Everything goes through `joint_command` (`ArmPose` on
`/arm/<side>/joint_targets`): it clamps, rate-limits, seeds from live feedback (no startup jump) and
runs the MIT watchdog. One node drives every arm in `joint_command.yaml`'s `arms`; each joint's
`active` in `arm_actuators.yaml` picks what moves.

``` bash
./watod -t joint_command
source /opt/watonomous/setup.bash
ros2 launch joint_command joint_command.launch.py

```

A fault on any arm stops every arm: AK joints sink damped, the GL40s go limp. Support the arms
and restart the node.

## 4. Move the arm

### Teleop with leader arm
1. Inside `simulation_mj` shell:
```bash
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
python3 calibrate_leader.py --port /dev/ttyACM1 --side left # Set the leader arm to joints hanging straight down and rotated to face forward, gripper fully open
```
> ❗If the command fails, you might need to do `ls /dev` and search for which `ttyACM*` ports are running 
2. Configure which real-arm joints you want to move. In `arm_actuators.yaml`, configure the `active` attribute for each joint.

3. vRun the teleop script through a dry run, verifying that leader arm direction is aligned with real-arm:
```bash
python3 pioneer_leader_arm_teleop.py --target real 
```

4. Run live script. Ensure the `joint_command` node is running
```bash
python3 pioneer_leader_arm_teleop.py --target real --live
```

>⚠️ Currently to shutdown the teleop, you must Ctrl-C the teleop script before Ctrl-C the `joint_command` node. Otherwise the teleop script will continue publishing `ArmPose` messages, conflicting with the `joint_command` node's shutdown process.

 ### Other


- Scripted moves and benchmarks: `tools/arm_roundtrip.sh` → [can/README.md](can/README.md#angle-benchmarks-arm_roundtrippy)
- Node, configs and tuning → [joint_command/README.md](joint_command/README.md)
