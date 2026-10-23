# Five-servo leader → Pioneer arm

This teleop maps the five-servo leader onto the physical-left arm in Isaac Sim:

| Leader | Servo ID | Sim joint | Motion |
|---|---:|---|---|
| A | 2 | `joint1L` | shoulder flexion |
| B | 3 | `joint2l` | shoulder abduction |
| C | 1 | `joint3l` | shoulder rotation |
| D | 5 | `joint4l` | elbow flexion |
| E | 4 | `joint5l` | forearm rotation |

The leader's fully straight startup pose becomes zero: A through E start at
0 degrees, including the elbow. Approximate human-style left-arm limits apply
in both the controller and physics: A -60..170, B -30..170, C -90..90,
D -150..0, E -90..90 degrees. These are independent joint approximations,
not a full anatomical model; see `arm_limits.py` for the reference and choices.
The source robot assets are unchanged. Only the absolute target is clamped:
the encoder baseline and accumulated angle are never clamped or shifted.
Thus 175 degrees maps to 170, and returning to zero maps to zero.
Encoder positions are continuously unwrapped, so crossing the
physical encoder's 0/4095 boundary never reverses the simulated joint. All
leader servos remain torque-off. Click **ZERO LEADER**
in the **Leader Arm Controls** window to re-zero and reset the simulated arm.
The **R** key is also available when the 3D viewport has keyboard focus.
Return along the same path after an overshoot; the single-turn encoders cannot
resolve motion exceeding half a revolution between successful samples.

## Quick start

From the repository root on the host:

```bash
./src/teleop/leader_arm_teleop/start_teleop.sh
```

This single command detects the leader's USB serial device, starts the
`simulation_isaac` container if needed, and launches Isaac Sim. Press
**Ctrl+C** to stop cleanly and release the serial port. Extra controller
options are forwarded directly, for example:

```bash
./src/teleop/leader_arm_teleop/start_teleop.sh --filter-alpha 0.5
```

If more than one USB serial device is connected, select the leader explicitly:

```bash
./src/teleop/leader_arm_teleop/start_teleop.sh --port /dev/ttyACM0
```

The stable `/dev/serial/by-id/...` path is preferred automatically when it is
available. `TELEOP_PORT=/dev/...` can also set the port without a command-line
flag.

## Manual container launch

To launch directly inside the `simulation_isaac` container:

```bash
cd /workspace/humanoid/src/teleop/leader_arm_teleop
/workspace/isaaclab/isaaclab.sh -p leader_arm_teleop.py --port /dev/ttyACM0
```

Default directions for A,B,C,D,E are `1,-1,1,-1,1`: B and D are inverted.
If another axis moves backward, override the directions without changing code:

```bash
/workspace/isaaclab/isaaclab.sh -p leader_arm_teleop.py --signs 1,-1,1,-1,1
```
