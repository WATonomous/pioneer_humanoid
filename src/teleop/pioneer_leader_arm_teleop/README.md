# pioneer_leader_arm_teleop

The 7-servo leader arm (STS3215, torque always off) drives the Pioneer left arm joint to joint,
with no IK, in any registered scene: Isaac Sim (default) or plain MuJoCo (`--target mujoco`, CPU).
`--target real` compares the leader with the real arm (dry run, publishes nothing; #327).
Optional recording in the shared `dataset_schema_pioneer_v1.yaml` format, from either simulator.

Angles map **one-to-one** from a shared zero: arm hanging straight down under gravity, gripper open.
That is the URDF zero, the real arm's calibrated zero, and the leader's (`calibrate_leader.py`).

| Servo | Bus ID | Sim joint | Default sign |
|-------|--------|-----------|--------------|
| A | 2 | `joint1L` shoulder flexion | +1 |
| B | 3 | `joint2l` shoulder abduction | +1 |
| C | 1 | `joint3l` shoulder rotation | −1 |
| D | 5 | `joint4l` elbow flexion | −1 |
| E | 4 | `joint5l` forearm rotation | +1 |
| F | 7 | `joint6l` wrist | −1 |
| G | 6 | gripper (41.5° open → 0° closed) | +1 |

## Calibrate (once per leader)

```bash
python calibrate_leader.py     # hang the leader straight down, gripper open, rotation joints on their marks; Enter
```

Writes `leader_calibration.json` here (zero encoder counts per servo); teleop refuses to start without it.
Redo it after re-mounting a servo.

## Run (Isaac)

The leader plugs in over USB (`/dev/ttyACM0`); the `simulation_isaac` container is privileged and mounts `/dev`. On the host you need the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and in).

```bash
# on the host, once: build and start the Isaac container
./watod build simulation_isaac        # rebuild after pulling Dockerfile or package changes
./watod up -d
./watod -t simulation_isaac           # shell inside the container

# inside the container
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
$PYTHON encoder_test.py                   # first time: all 7 servos respond? (--ids 2,3 --raw for a subset)
$PYTHON calibrate_leader.py                # first time: store the hanging-pose zero

/workspace/isaaclab/isaaclab.sh -p pioneer_leader_arm_teleop.py --scene push
/workspace/isaaclab/isaaclab.sh -p pioneer_leader_arm_teleop.py --scene vial_rack --record
```

Scenes: `bare` (default), `push`, `vial_rack`, or any scene registered in `humanoid_isaac_scenes` (an unknown name lists them).

## Run (MuJoCo)

CPU only. Either in the `simulation_mj` container (needs the host's X display; it reserves a GPU for mjlab):

```bash
xhost +local:                              # on the host, once per login: let the container use the display
./watod build simulation_mj && ./watod up -d simulation_mj
./watod -t simulation_mj
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
python3 pioneer_leader_arm_teleop.py --target mujoco --scene peg_insert [--record]
```

or on any machine with a display, no Docker (`dialout` group as above):

```bash
pip install mujoco feetech-servo-sdk                  # teleop
pip install torch "lerobot @ git+https://github.com/huggingface/lerobot.git@e670ac5daf9b76" pynput   # + --record (and ffmpeg with libsvtav1)
python pioneer_leader_arm_teleop.py --target mujoco --scene peg_insert   # macOS: mjpython
```

Scenes: `bare`, `peg_insert`, `zip_tie`, `drawer_stow`, `matcha`, `duplo`, or any scene in `humanoid_mujoco_scenes` (see `src/simulation/mujoco_scenes/`).

- **Launch:** with the physical leader in its calibrated straight/resting pose, the simulated left arm starts in that same pose and follows every joint immediately. There is no home-position gate or re-zero.
- **Live controls:** the **Leader Arm Controls** window shows physical and target angles. Toggle any
  **Inverted** box and click **Apply directions**; the current sim pose is preserved, so it does not
  jump while the direction changes. Enter desired angles and click **Held pose → defaults** to map
  the physical pose you are holding to those values. **Held pose → all zero** is the quick live-zero;
  **Use saved calibration** removes that live offset. No restart is required.
- **Directions:** the startup direction order A..G is `1,1,-1,-1,1,-1,1`; `--signs` can still
  override it from the command line.
- **Wrist preview:** recording with `--cameras ego,wrist_left` opens a small **Left Wrist Camera**
  window using the exact wrist frames that are written into the dataset.
- **Recording status:** a separate color-coded window shows **READY**, **● RECORDING** (with live
  frame count), or **SAVING**, plus the number of demos saved. Reaching `--num_episodes` does not
  close Isaac; you can continue recording additional demos until you close it yourself.
- **R:** reset every task object while snapping the simulated left arm to the leader's current physical pose. Move the physical leader to the bent-90° start pose first, then press R. During a take, R also discards that take.
- **`--record`:** `S` start · `N` save (then auto-reset) · `D` while recording discards the
  current take; `D` while READY removes the most recently saved demo and decrements the counter.
   Saved-demo removal can take several seconds because LeRobot rebuilds its shared video/parquet
   chunks. Output goes to `<repo>/datasets/pioneer_v1_left_arm/sim/`; use
   `--cameras ego,wrist_left` / `none`.
- **Multi-step scenes** (`drawer_stow`, `matcha`, `duplo`): the terminal prints `[TASK] step k/n: <instruction>` as each step is done and
  `all steps done -- N to save` at the end. Recorded frames carry the current step's instruction as their `task` and a
  `subtask_index`, so a take is labelled step by step with no hand annotation.
  In the MuJoCo viewer these keys (and R) also toggle display flags (shadows, reflections, …); harmless.
- Other flags: `--port`, `--baud`, `--calibration` (default `leader_calibration.json`), `--filter-alpha` (target smoothing, default 0.35).

Targets are clamped to the arm's URDF limits. Wrist damping is lowered to 2.5 in this teleop only, so the sim wrist keeps up with the leader.

## Real arm (dry run)

Read-only: subscribes to `/interfacing/motorFeedback`, creates no publisher, so it cannot move the arm.
It prints leader vs real joint angles in the URDF frame (also the command frame `ArmPose` / `joint_command` use),
their difference, and how far the leader is from home. Use it to check the calibrations before any
live teleop (`--live` is a later step of #327).

```bash
# interfacing up, arm powered (motors not commanded); then in the simulation_mj container:
./watod -t simulation_mj
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
python3 pioneer_leader_arm_teleop.py --target real --port /dev/ttyACM1   # leader port: not the CANable's
python3 pioneer_leader_arm_teleop.py --target real --self-test           # angle math only, no ROS / leader
```

- Leader: `sign × leader` (clamped to the URDF limits, as in sim).
  Real: `zero_offset + motor / direction` from `hardware_mapping.yaml`. Same frame, compared directly.
- Check: pose both arms the same, motors off; every joint should agree within 5° across its range.
  Fix `--signs` (leader) or the real arm's `direction` / calibration (`calibrate_arm.py`) until it does.
- Warnings (end of each row): `>5` (disagreement), `NO FEEDBACK` (none for 0.5 s), `urdf-clamp`, `outside hw [lo,hi]`
  (the leader's target is outside `hardware_mapping.yaml`'s limits; `joint_command` would clamp it).
- The gripper is shown, not compared: the GL40 (id 21) has no command path yet.

## Files

| file | role |
|------|------|
| `pioneer_leader_arm_teleop.py` | entry; `--target isaac\|mujoco\|real` picks the backend before any simulator import |
| `isaac_sim.py` / `mujoco_sim.py` | sim backends |
| `real_arm.py` | real-arm backend: dry run, leader vs `/interfacing/motorFeedback`, publishes nothing |
| `leader_mapping.py` | shared: 1:1 mapping, home check, filter, gripper closure, leader read, wall-clock pacing |
| `servo_leader.py` / `arm_limits.py` | servo bus reader / clamp + gripper fraction |
| `calibrate_leader.py` | stores the hanging-pose zero |
| `encoder_test.py` | servo bring-up, no simulator |
