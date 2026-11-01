# pioneer_leader_arm_teleop

The 7-servo leader arm (STS3215, torque always off) drives the Pioneer left arm joint to joint,
with no IK, in any registered scene: Isaac Sim (default) or plain MuJoCo (`--target mujoco`, CPU).
`--target real` compares the leader with the real arm (dry run, publishes nothing); `--live` then drives it (#327).
Optional recording in the shared `pioneer_humanoid_dataset_schema.yaml` format, from either simulator.

Angles map **one-to-one** from a shared zero: arm hanging straight down under gravity, gripper open.
That is the URDF zero, the real arm's calibrated zero, and the leader's (`calibrate_leader.py`).

| Servo | Bus ID | Sim joint | Default sign |
|-------|--------|-----------|--------------|
| A | 2 | `joint1L` shoulder flexion | +1 |
| B | 3 | `joint2l` shoulder abduction | +1 |
| C | 1 | `joint3l` shoulder rotation | −1 |
| D | 5 | `joint4l` elbow flexion | +1 |
| E | 4 | `joint5l` forearm rotation | +1 |
| F | 7 | `joint6l` wrist | −1 |
| G | 6 | gripper (41.5° open → 0° closed) | +1 |

## Calibrate (once per board)

The leader unit has one board per arm, each on its own USB serial port (not the CANable's).

```bash
python calibrate_leader.py --side left --port /dev/ttyACM0     # hang the leader straight down, gripper open, rotation joints on their marks; Enter
python calibrate_leader.py --side right --port /dev/ttyACM1 --signs=<A..G>   # only for two arms; right has no default signs (keep the =: they may start with -)
python calibrate_leader.py --side left --port /dev/ttyACM0 --from leader_calibration.json   # convert the old file, no re-hanging
```

Writes that board into `leader_arm.json` here and enables it: zero encoder counts per servo, the signs (left
default `1,1,-1,1,1,-1,1`) and the board's USB serial number. The teleop finds each board by that serial, so
plugging the boards in the other order can't swap the arms; two boards with the same serial (or none) are refused.
The right board stays disabled until it is calibrated. Teleop refuses to start without the file.
**Commit `leader_arm.json`:** it describes the team's one leader unit, so nobody else has to calibrate; a different
unit is refused by its USB serial. Recalibrate and commit again after re-mounting a servo.

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
$PYTHON calibrate_leader.py --side left --port /dev/ttyACM0   # first time: store the hanging-pose zero

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

- **Home:** the arm starts at home (elbow bent 90°, forearm forward, gripper open) and follows the leader only once every leader joint is within 3° of home and the gripper is open. The status line lists the joints still off.
- **Directions:** move one leader joint at a time. If a sim joint goes the wrong way, re-run `calibrate_leader.py` for that side with that entry flipped in `--signs` (order A..G, left default `1,1,-1,1,1,-1,1`).
- **6:** reset the arm and every object in the scene; the arm waits at home until the leader is back at home. During a take, it also discards the take.
- **`--record`:** `7` start · `8` save (then auto-reset) · `9` discard → `<repo>/datasets/pioneer_v1_left_arm/sim/` · `--cameras ego,wrist_left` / `none`.
  Digits because every letter, 0-5 and Esc are MuJoCo viewer shortcuts (Esc stays the viewer's free camera here). The status line starts `REC <frames>` while recording.
- **Multi-step scenes** (`drawer_stow`, `matcha`, `duplo`): the terminal prints `[TASK] step k/n: <instruction>` as each step is done and
  `all steps done -- 8 to save` at the end. Recorded frames carry the current step's instruction as their `task` and a
  `subtask_index`, so a take is labelled step by step with no hand annotation.
- **Both arms (MuJoCo):** `--arms bimanual` (or `right` for the right arm alone), once the right board is calibrated (`calibrate_leader.py --side right`).
  Recording waits for every leader at home; frames are the left arm's 7 values then the right's, into
  `<repo>/datasets/pioneer_v1_bimanual/sim/` (`pioneer_v1_right_arm` for the right arm alone).
  The right wrist camera is off in the schema: add it with `--cameras ego,wrist_left,wrist_right`.
- Other flags: `--baud`, `--leader-arm` (default `leader_arm.json` here), `--filter-alpha` (target smoothing, default 0.35).

Targets are clamped to the arm's URDF limits. Wrist damping is lowered to 2.5 in this teleop only, so the sim wrist keeps up with the leader.

## Synthetic demos (MuJoCo, no leader)

A simulated person "uses" the leader arm, so you get teleop demos without a human. Demos go through the same
`LeaderMapping` and recorder as `--target mujoco --record`, so the dataset format is identical (same features,
per-step `task` and `subtask_index`). Output: `<repo>/datasets/pioneer_v1_left_arm/sim_synthetic/<scene>/`, kept apart
from human takes (`--dataset_root` to override).

```bash
python synthetic/synthetic_teleop.py --scene peg_insert --num_episodes 3 --preview peg.mp4   # watch, no dataset
MUJOCO_GL=egl python synthetic/synthetic_teleop.py --scene duplo --record --num_episodes 100  # CPU only: MUJOCO_GL=osmesa
```

Scenes: `drawer_stow`, `peg_insert`, `zip_tie`, `duplo`, `matcha`.

### How it works

```
for seed in seed, seed+1, ...:                       # 1. try takes without cameras (fast, --workers in parallel)
    reset the scene with seed                        #    random layout
    operator = random person (speed, aim, tremor)    #    random style
    run the scene's plan; keep seed if success       #    stop at --num_episodes successes
for each kept seed:                                  # 2. replay it with cameras and save (same seed = same take)
    record the episode; log seed + style + notes to meta/synthetic_takes.jsonl
```

A rerun with the same `--dataset_root` skips seeds already saved, so an interrupted run resumes.

### Files

| File | What it is |
|---|---|
| `synthetic/synthetic_teleop.py` | entry point: the loop above, recording, preview |
| `synthetic/human_operator.py` | how the person moves: `op.see`, `op.move`, `op.arc`, `op.set_grip` (motion model in its docstring) |
| `synthetic/synthetic_tasks.py` | what the person does in each scene: one `plan` per scene (success comes from the scene's `success`) |

### Adding a scene

```python
def my_plan(op, model, data, notes):
    p = op.see("cube")                          # where it is (a few mm off, like an eye)
    op.arc(p + (0, 0, 0.05), clear_z=0.15)      # lift, across, down
    op.move(p, tol=0.003); op.set_grip(1.0)     # grasp
    if not lifted(model, data):                 # check, retry or give up (the take is discarded)
        raise TaskFailed("grasp")
    ...

TASKS["my_scene"] = Task(my_plan, "pick up the cube")
```

The scene itself registers `success` (or a multi-step `progress`): see [mujoco_scenes](../../simulation/mujoco_scenes/README.md).

### Notes

- Recording is slow on CPU (rendering the cameras); use `MUJOCO_GL=egl` with a free GPU, and `--image_size 320x240`
  if the policy takes that size.
- `--pace`: 0.5 slow beginner, 1 practised operator (default), 1.5 brisk.
- Exits with an error if fewer than `--num_episodes` takes succeed within `--max_tries` (the ones found are kept).

## Real arm (dry run, then `--live`)

Without `--live` it is read-only: subscribes to `/interfacing/motorFeedback`, creates no publisher, so it
cannot move the arm. It prints leader vs real joint angles in the URDF frame (also the command frame
`ArmPose` / `joint_command` use), their difference, and how far the leader is from home. Use it to check
the calibrations before any live teleop.

```bash
# interfacing up, arm powered (motors not commanded); then in the simulation_mj container:
./watod -t simulation_mj
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
python3 pioneer_leader_arm_teleop.py --target real          # left board from leader_arm.json
python3 pioneer_leader_arm_teleop.py --target real --live   # needs joint_command running
python3 pioneer_leader_arm_teleop.py --target real --self-test           # angle math only, no ROS / leader
```

- Leader: `sign × leader`, clamped to the `arm_calibration.yaml` limits `joint_command` enforces (the URDF
  limits for a joint without `limit_range`; the sim backends keep the URDF limits).
  Real: `zero_offset + motor / direction` from `arm_calibration.yaml`. Same frame, compared directly.
- Check: pose both arms the same, motors off; every joint should agree within 10° across its range.
  Fix the leader's signs (`calibrate_leader.py --signs`) or the real arm's `direction` / calibration (`calibrate_arm.py`) until it does.
- Warnings (end of each row): `>10` (disagreement), `NO FEEDBACK` (none for 0.5 s), `limit-clamp` (the leader was cut to those
  limits), `real outside hw [lo,hi]` (the real arm is outside `arm_calibration.yaml`'s limits: stale calibration).
- Gripper: compared as position (0 open .. 1 closed), leader vs the real GL40 (id 21); `>0.1` and
  `real outside hw` warn.

`--live`:
0. Optionally homes the real elbow.pitch first. Once every active joint has feedback, it asks `[y/N]`.
   On `y` it ramps elbow.pitch from its measured angle to home (−90°) at `--home-rate` deg/s (default 5,
   at most its `velocity_max`), holding the other joints where they are. It then keeps publishing that
   pose until the leader takes over. It refuses if elbow.pitch is not `active` or home is outside its
   `arm_calibration.yaml` limits (re-run `calibrate_arm.py` sweeping past home).
1. Shows the dry run until the gate passes: on every joint `active` in `arm_actuators.yaml`, the real arm is
   inside its `arm_calibration.yaml` limits and the leader is within 3° of home and within 10° of the real
   arm; the gripper (if active) is inside its limits and within 0.1. The screen lists what fails.
2. Asks you to press Enter (anything else quits), re-checks the gate, then publishes `ArmPose` on
   `/arm/left/joint_targets` at 50 Hz, gripper included. `joint_command` clamps, rate-limits and runs the watchdog.
3. Ctrl-C stops publishing. `joint_command` holds the last pose for `command_timeout_sec` (10 s), then
   sets kp = 0 and the arm sinks: support it.

## Files

| file | role |
|------|------|
| `pioneer_leader_arm_teleop.py` | entry; `--target isaac\|mujoco\|real` picks the backend before any simulator import |
| `isaac_sim.py` / `mujoco_sim.py` | sim backends |
| `real_arm.py` | real-arm backend: dry run (leader vs `/interfacing/motorFeedback`), `--live` publishes `/arm/left/joint_targets` |
| `leader_mapping.py` | shared: 1:1 mapping, home check, filter, gripper closure, leader read, wall-clock pacing |
| `servo_leader.py` / `arm_limits.py` | servo bus reader / clamp + gripper fraction |
| `calibrate_leader.py` | stores a board's hanging-pose zero, signs and USB serial in `leader_arm.json` |
| `encoder_test.py` | servo bring-up, no simulator |
| `synthetic/synthetic_teleop.py` | synthetic demos: simulated operator → same mapping and recorder as `mujoco_sim.py` |
| `synthetic/human_operator.py` / `synthetic/synthetic_tasks.py` | the operator's motion model / what it does in each scene |
