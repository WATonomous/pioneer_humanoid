# Real arm session checklist: power on → calibrate → teleop → shut down

The commands for one session driving the real right arm with the leader arm, in order.
Background and first-time setup (udev rule, wiring, what each step does):
[ARM_BRINGUP.md](ARM_BRINGUP.md).

> **Hardware E-stop within reach for every step where the motors are powered.**

## 1. Plug in and power on

1. Plug the CAN adapter into the laptop. If the LED is red, unplug and replug it until it flashes green.
2. Plug in the leader arm (USB).
3. Check both devices on the laptop:

   ```bash
   ls -l /dev/canable     # CAN adapter -> ttyACMx
   ls /dev/ttyACM*        # leader arm is the ttyACM that /dev/canable does NOT point to
   ```

4. Connect the battery, then **release the E-stop** to power the motors.

## 2. Start the containers (laptop terminal, `~/humanoid`)

The `interfacing` container starts the CAN connection as soon as it starts, so the adapter has to be
plugged in first.

```bash
./watod build interfacing joint_command     # first time, or after code changes
./watod up -d interfacing joint_command simulation_mj
```

## 3. Check CAN

```bash
./watod -t interfacing
source /opt/watonomous/setup.bash
ros2 node list               # expect /can_node
candump can0                 # expect frames like 0000290B–0000290E, 00002968 (Ctrl+C to stop)
ros2 topic echo /interfacing/motorFeedback common_msgs/msg/MotorFeedback --once
```

- **No frames?** Check that the E-stop is released, the battery is connected, and the adapter LED is green.
- **`can0` does not exist, or `can_node` logs `No such device`?** The adapter dropped off USB.
  `can_node` does not reconnect: replug the adapter, then `./watod down && ./watod up -d ...` (step 2).
- **`ros2 node list` fails with `!rclpy.ok()`?** The ROS 2 daemon is stuck:
  `ros2 daemon stop; ros2 daemon start`.

## 4. Calibrate the arm (every power-on)

Arm **hanging straight down**. The motors buzz during this step, which is normal.

```bash
# still in the interfacing container
ros2 run can calibrate_arm.py --arm-side right --write-mapping
```

- Per joint: confirm the motor id → set home → move to one end, Enter → move to the other end, Enter
- Gripper: home about 5° short of fully open. First end: fully open. Second end: fully closed.
- Afterwards: if the gripper's `lower_limit` in `arm_calibration.yaml` is above 0, swap it with 0.

Optional: keep the viewer from step 5 open while calibrating.

## 5. Check calibration in the viewer (nothing moves)

New laptop terminal:

```bash
./watod -t simulation_mj
python3 /workspace/humanoid/src/interfacing/can/scripts/live_arm_mjviser.py --arm-side right
```

Open <http://localhost:8080>. Move **each joint by hand, one at a time**:

- [ ] Turns the same way on screen
- [ ] Stops at the same angle

Wrong direction: change that joint's `direction` in `arm_calibration.yaml`, then redo step 4.
Wrong angle: it wasn't zeroed while hanging. Redo step 4.

## 6. Choose the active joints

In [joint_command/config/arm_actuators.yaml](joint_command/config/arm_actuators.yaml), set
`active: true` on **one joint only** for the first run, and `active: false` on all others.

- [ ] First joint to test: ________

## 7. Start joint_command (motor control)

Arm supported, workspace clear, hand near the E-stop. New laptop terminal:

```bash
./watod -t joint_command
source /opt/watonomous/setup.bash
ros2 launch joint_command joint_command.launch.py
```

If it reports a fault, the arm goes limp or sinks. Support the arm, then restart the node.

## 8. Teleop with the leader arm

New laptop terminal:

```bash
./watod -t simulation_mj
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
```

1. Calibrate the leader: hold it hanging down, facing forward, gripper fully open.

   ```bash
   # /dev/ttyACM0 or /dev/ttyACM1: whichever is the leader (step 1)
   python3 calibrate_leader.py --port /dev/ttyACM0 --side right
   ```

2. **Dry run** (sends nothing to the motors). Check that the directions match the real arm:

   ```bash
   python3 pioneer_leader_arm_teleop.py --target real --arms right
   ```

3. **Live**, with joint_command running:

   ```bash
   python3 pioneer_leader_arm_teleop.py --target real --live --arms right
   ```

4. Small, slow moves first. Once the joint behaves, stop (step 9), activate the next joint (step 6)
   and repeat steps 7–8.

## 9. Shut down (in this order)

1. Ctrl+C the **teleop** script
2. Ctrl+C **joint_command**
3. Support the arm, press the **E-stop**
4. `./watod down` (laptop terminal)
