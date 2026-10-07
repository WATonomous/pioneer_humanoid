# Real arm bring-up: connect → calibrate → visualize → move

Follow top to bottom. Package details: [can/README.md](can/README.md) (CAN bridge, MIT mode,
benchmarks), [joint_command/README.md](joint_command/README.md) (command node, configs).

## 0. Connect
1. **Hardware**: battery + E-stop closed (motor power ≠ CAN power). CANable USB → host; CAN_H/CAN_L → arm (120 Ω term).
2. **Host setup (once)**: `./src/interfacing/can/scripts/can_udev.sh install` → `/dev/canable`.
   `watod-config.local.sh`: `ACTIVE_MODULES="interfacing"`, `MODE_OF_OPERATION="develop"`.
3. **Bring up**: `can.launch.py` starts `can_node` + SLCAN (`/dev/canable` → `can0` @ 1 Mbps).
   ```bash
   ./watod build && ./watod up -d
   ./watod -t interfacing
   source /opt/watonomous/setup.bash
   ```
   A gs_usb / candleLight adapter instead: [can/README.md](can/README.md#gs_usb--candlelight-adapter).
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
python3 /root/ament_ws/src/interfacing/can/scripts/calibrate_arm.py \
  --arm-side left --write-mapping --calibration /calibration/arm_calibration.yaml
```
Prompt: **Enter**=yes · id=correct id · **s**=skip · **q**=quit.

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

## 3. Move (real motor control)
Only after 1–2 look right. Everything goes through `joint_command` (`ArmPose` on
`/arm/joint_targets`): it clamps, rate-limits, seeds from live feedback (no startup jump) and
runs the MIT watchdog. Start it with `ros2 launch joint_command joint_command.launch.py`.

First powered moves, in this order:
1. Clear workspace, hardware E-stop on the 48 V supply in reach, a hand under the arm.
2. One joint at a time: every other joint `active: false` in `arm_actuators.yaml` (restart the node).
   Small, slow moves from the current pose.
3. Then the whole arm, slowly; then the gripper.

A fault stops the whole arm: AK joints sink damped, the GL40s go limp. Support the arm and
restart the node.

- Leader-arm teleop: `pioneer_leader_arm_teleop.py --target real` (dry run), then `--live`
  → [teleop/pioneer_leader_arm_teleop/README.md](../teleop/pioneer_leader_arm_teleop/README.md#real-arm-dry-run-then---live)
- Scripted moves and benchmarks: `tools/arm_roundtrip.sh` → [can/README.md](can/README.md#angle-benchmarks-arm_roundtrippy)
- Node, configs and tuning → [joint_command/README.md](joint_command/README.md)
