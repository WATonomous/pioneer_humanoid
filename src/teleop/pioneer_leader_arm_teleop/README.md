# pioneer_leader_arm_teleop

The leader arm (STS3215 servos, torque always off: 8 for the default 8dof arms, 7 for the old arms) drives the Pioneer
arms joint to joint, with no IK, in any registered scene: plain MuJoCo (default, CPU) or Isaac Sim (`--target isaac`),
both the 8dof arms by default.
`--target real` compares the leader with the real arm (dry run, publishes nothing); `--live` then drives it (#327).
Optional recording (`--record`) in the robot's schema: `pioneer_8dof_dataset_schema.yaml` by default.

Angles map **one-to-one** from a shared zero: arm hanging straight down under gravity, gripper open.
That is the URDF zero, the real arm's calibrated zero, and the leader's (`calibrate_leader.py`).

Old arms (`--robot armv2`, 7 servos):

| Servo | Bus ID | Sim joint | Default sign |
|-------|--------|-----------|--------------|
| A | 2 | `joint1L` shoulder flexion | +1 |
| B | 3 | `joint2l` shoulder abduction | +1 |
| C | 1 | `joint3l` shoulder rotation | −1 |
| D | 5 | `joint4l` elbow flexion | +1 |
| E | 4 | `joint5l` forearm rotation | +1 |
| F | 7 | `joint6l` wrist | −1 |
| G | 6 | gripper (41.5° open → 0° closed) | +1 |

### The 8dof arms (`--robot 8dof`, MuJoCo and Isaac)

`--target mujoco` and `--target isaac` drive `pioneer_bimanual_arm_8dof` (7-DoF + gripper per arm) by default; MuJoCo
keeps the old arms with `--robot armv2`. Isaac has only the 8dof arms, the real arm only the old arms. The 8dof arms add a 7th joint, wrist yaw, after wrist pitch:
the leader drives it with an **eighth servo on bus ID 8**. Labels follow joint order: **G = wrist yaw (new), H = gripper**
(the old leader's G, same bus ID 6).

| Servo | Bus ID | 8dof joint (left) | Default sign left / right |
|-------|--------|-------------------|---------------------------|
| A | 2 | `left_shoulder_pitch` | +1 / −1 |
| B | 3 | `left_shoulder_roll` | +1 / −1 |
| C | 1 | `left_shoulder_yaw` | +1 / −1 |
| D | 5 | `left_elbow` (0 = straight) | +1 / +1 |
| E | 4 | `left_forearm_roll` | −1 / +1 |
| F | 7 | `left_wrist_pitch` | +1 / +1 |
| **G** | **8** | `left_wrist_yaw` (**new servo**) | +1 / +1 |
| H | 6 | gripper | +1 / +1 |

- **Default signs** (`SIDE_SIGNS_8DOF`): the old signs times each 8dof joint's direction relative to the old arm's, from
  the two models. Not checked on the leader: move each joint alone and compare with the sim, then calibrate with `--signs`.
- **Calibration:** its own file, `leader_arm_8dof.json`, so `leader_arm.json` stays valid for the old arms:
  `python calibrate_leader.py --side right --robot 8dof` (the new servo G on ID 8 first: `encoder_test.py --robot 8dof`).
- **Recording:** `pioneer_8dof_dataset_schema.yaml`, 8 values per arm (7 joints + gripper), dataset `datasets/pioneer_8dof_*`.
- **`--lock-wrist`** freezes both wrist joints (F and G).

## Calibrate (once per board)

The leader unit has one board per arm, each on its own USB serial port (not the CANable's).

```bash
python calibrate_leader.py --side left     # hang the leader straight down, gripper open, rotation joints on their marks; Enter
python calibrate_leader.py --side right --port <your-port>   # --port only when more than one board is plugged in
python calibrate_leader.py --side right --signs=<A..G>   # other directions than SIDE_SIGNS (keep the =: they may start with -)
python calibrate_leader.py --side left --from leader_calibration.json   # convert the old file, no re-hanging
```

Writes that board into `leader_arm.json` here and enables it: zero encoder counts per servo, the signs (`--signs`,
default the side's `SIDE_SIGNS` in `leader_mapping.py`) and the board's USB serial number. The teleop finds each board
by that serial on whatever port it gets on your machine, so no port is stored and plugging the boards in the other order
can't swap the arms; a board without a USB serial, or two with the same one, are refused.
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
$PYTHON encoder_test.py                    # first time: all 8 servos respond? (--ids 2,3 --raw for a subset)
$PYTHON calibrate_leader.py --side right   # first time: store the hanging-pose zero (leader_arm_8dof.json)

/workspace/isaaclab/isaaclab.sh -p pioneer_leader_arm_teleop.py --target isaac --scene push
/workspace/isaaclab/isaaclab.sh -p pioneer_leader_arm_teleop.py --target isaac --scene vial_rack --arms bimanual --record
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

- **Home:** the arm starts at home (elbow bent 90°, forearm forward, gripper open) and follows the leader only once every leader joint is within 15° of home and the gripper is open. The status line lists the joints still off.
- **Directions:** move one leader joint at a time. If a sim joint goes the wrong way, re-run `calibrate_leader.py` for that side with that entry flipped in `--signs` (order A..G; the defaults are `SIDE_SIGNS` in `leader_mapping.py`).
- **Keys** (all teleops side by side: [teleop README → Keyboard controls](../README.md#keyboard-controls)):

  | Key | Isaac | MuJoCo |
  |-----|-------|--------|
  | reset arm + scene (the arm waits at home until the leader is back at home; discards a take in progress) | `R` | `6` |
  | `--record`: start | `S` | `7` |
  | `--record`: save (then auto-reset) | `N` | `8` |
  | `--record`: discard | `D` | `9` |
  | stop and finalize the dataset | `Esc` | close the window |

  MuJoCo uses digits because every letter, 0-5 and Esc are viewer shortcuts (Esc stays the viewer's free camera).
  Its window's top-left corner shows whether you are recording (and for how long), the key to press next, and
  `SAVED` / `take discarded` for 3 s after 8 / 9. Until the arm follows (with or without `--record`) it also lists each
  joint's offset from home (the gripper as % open) under `bring to position`, `<` marking the ones still out of range.
- **Arm:** the right arm by default (`--arms right`): the leader's left arm doesn't register its servos. `--arms left` once it does.
- **`--record`** → `<repo>/datasets/pioneer_v1_right_arm/sim/` · cameras `ego` + the driven arm's wrist (`--cameras` to change, `none` for none).
- **Multi-step scenes** (`drawer_stow`, `matcha`, `duplo`): the terminal prints `[TASK] step k/n: <instruction>` as each step is done and
  `all steps done -- 8 to save` at the end. Recorded frames carry the current step's instruction as their `task` and a
  `subtask_index`, so a take is labelled step by step with no hand annotation.
- **Both arms:** `--arms bimanual`, once both boards are calibrated (`calibrate_leader.py --side left` / `right`).
  Recording waits for every leader at home; frames are the left arm's 7 values then the right's, into
  `<repo>/datasets/pioneer_v1_bimanual/sim/` (`pioneer_v1_left_arm` for the left arm alone), with both wrist cameras.
- Other flags: `--baud`, `--leader-arm` (default `leader_arm.json` here), `--filter-alpha` (target smoothing, default 0.35).
  `"lock_wrist": true` under an arm in `leader_arm.json` freezes that arm's wrist (F) rigidly at home (MuJoCo: motor off, joint pinned every physics step; Isaac: its joint limits closed onto home) and ignores the leader's wrist; `false` lets it follow the leader. `--lock-wrist` / `--no-lock-wrist` override it for one run. Recalibrating keeps the setting.

Targets are clamped to the arm's URDF limits. The old arm's wrist damping is lowered to 2.5 in this teleop only (MuJoCo `--robot armv2`), so the sim wrist keeps up with the leader.

## Synthetic demos (MuJoCo, no leader)

A simulated person "uses" the leader arm, so you get teleop demos without a human. Demos go through the same
`LeaderMapping` and recorder as `--target mujoco --record`, so the dataset format is identical (same features,
per-step `task` and `subtask_index`). Always the 8dof arm (`pioneer_bimanual_arm_8dof`), left arm. Output:
`<repo>/datasets/pioneer_8dof_left_arm/sim_synthetic/<scene>/`, kept apart from human takes (`--dataset_root` to override).

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
python3 pioneer_leader_arm_teleop.py --target real --arms right            # right board -> right arm
python3 pioneer_leader_arm_teleop.py --target real --arms bimanual         # both boards, both arms
python3 pioneer_leader_arm_teleop.py --target real --live   # needs joint_command running
python3 pioneer_leader_arm_teleop.py --target real --self-test           # angle math only, no ROS / leader
```

- `--arms left` (default), `right` or `bimanual`: each arm is driven by its own leader board, so the right board
  must be calibrated first (`calibrate_leader.py --side right`). Check its signs (`SIDE_SIGNS`) with this
  dry run. A flipped sign on a joint whose home is 0° passes the gate, so sweep every joint.
  With `bimanual` the screen shows a table per arm.

- Leader: `sign × leader`, clamped to the `arm_calibration.yaml` limits `joint_command` enforces (the URDF
  limits for a joint without `limit_range`; the sim backends keep the URDF limits).
  Real: `zero_offset + motor / direction` from `arm_calibration.yaml`. Same frame, compared directly.
- Check: pose both arms the same, motors off; every joint should agree within 10° across its range.
  Fix the leader's signs (`calibrate_leader.py --signs`) or the real arm's `direction` / calibration (`calibrate_arm.py`) until it does.
- Warnings (end of each row): `>10` (disagreement), `NO FEEDBACK` (none for 0.5 s), `limit-clamp` (the leader was cut to those
  limits), `real outside hw [lo,hi]` (the real arm is outside `arm_calibration.yaml`'s limits: stale calibration).
- Gripper: compared as position (0 open .. 1 closed), leader vs the real GL40 (`gripper.open_close` in
  `arm_calibration.yaml`: id 21 left, 36 right); `>0.1` and `real outside hw` warn.

`--live`:
0. Asks **start at home? `[y/N]`** (one question for both arms with `bimanual`).
   - `y`: every `active` joint ramps from its measured angle to home (shoulders 0°, elbow.pitch −90° left /
     +90° right) at `--home-rate` deg/s (default 5, at most every active joint's `velocity_max`), then holds
     there. Put the leader at home by hand. Refused, nothing moves, if an active joint's home is outside its
     `arm_calibration.yaml` limits (re-run `calibrate_arm.py` sweeping past home); it then carries on as `n`.
   - `n`: no homing; start from wherever the arm is.
1. Shows the dry run until the gate passes: on every joint `active` in `arm_actuators.yaml`, the real arm is
   inside its `arm_calibration.yaml` limits and the leader is within 15° of the real arm (and within 15° of
   home after `y`); the gripper (if active) is inside its limits and within 0.1. The screen lists what fails.
   With `bimanual` both arms must pass.
2. Asks you to press Enter (anything else quits), re-checks the gate, then publishes `ArmPose` on
   `/arm/<side>/joint_targets` at 50 Hz, gripper included; with `bimanual` one Enter starts both arms.
   `joint_command` clamps, rate-limits and runs the watchdog.
3. Ctrl-C stops publishing. `joint_command` holds the last pose for `command_timeout_sec` (10 s), then
   sets kp = 0 and the arm sinks: support it.

## Real arm, phased start: `real_arm_teleop.py`

The leader can start anywhere; the real arm comes to it, slowly. Enter advances, `q` + Enter quits.
At start it asks **start at home? `[y/N]`**: `y` adds the HOME step below, `n` skips it.

```bash
python3 real_arm_teleop.py --arms right [--slow-rate 15] [--home-rate 5] [--skip-zero]   # needs interfacing + joint_command up
python3 real_arm_teleop.py --self-test                      # logic only, no ROS / leader
```

0. **ZERO** (read-only): hang the leader and the real arm straight down (leader gripper open), Enter: the
   leader's reading becomes its zero for this run. Refuses if an active real joint reads more than 5° from 0
   (recalibrate with `calibrate_arm.py`); the real arm's zero is never changed here. `--skip-zero` skips it.
1. **CHECK** (read-only, publishes nothing): leader / real per joint. Enter starts HOME (`y`) or SLOW
   (`n`) once every `active` joint has feedback and is inside its `arm_calibration.yaml` limits.
   - **HOME** (`y` only): every `active` joint ramps from its measured angle to home at `--home-rate` deg/s
     (default 5) and holds; put the leader at home by hand. Enter starts SLOW once the ramp is done and every
     active leader joint is within 15° of home. Refused at the question if an active joint's home is outside
     its calibrated limits (it then runs as `n`).
2. **SLOW**: starts from the real arm's measured pose and moves each joint toward the leader at most
   `--slow-rate` deg/s (default 15, at most each joint's `velocity_max`), then keeps following at that
   speed: test the joints. Enter starts LIVE once every target is within 2° of the leader.
3. **LIVE**: full teleop; `joint_command`'s `velocity_max` still applies. Enter drops back to SLOW.

Lost feedback on an active joint holds the last target (still published) and falls back to SLOW.
Ctrl-C stops publishing: `joint_command` holds for `command_timeout_sec`, then kp = 0 and the arm sinks.

## Files

| file | role |
|------|------|
| `pioneer_leader_arm_teleop.py` | entry; `--target isaac\|mujoco\|real` picks the backend before any simulator import |
| `isaac_sim.py` / `mujoco_sim.py` | sim backends |
| `real_arm.py` | real-arm backend (`--arms left\|right\|bimanual`): dry run (leader vs `/interfacing/motorFeedback`), `--live` publishes `/arm/<side>/joint_targets` |
| `real_arm_teleop.py` | real arm, phased start: ZERO (leader) -> CHECK -> [HOME if `y`] -> SLOW (arm moves to the leader at `--slow-rate`) -> LIVE, Enter to advance |
| `leader_mapping.py` | shared: 1:1 mapping, home check, filter, gripper closure, leader read, wall-clock pacing |
| `servo_leader.py` / `arm_limits.py` | servo bus reader / clamp + gripper fraction |
| `calibrate_leader.py` | stores a board's hanging-pose zero, signs and USB serial in `leader_arm.json` |
| `encoder_test.py` | servo bring-up, no simulator |
| `synthetic/synthetic_teleop.py` | synthetic demos: simulated operator → same mapping and recorder as `mujoco_sim.py` |
| `synthetic/human_operator.py` / `synthetic/synthetic_tasks.py` | the operator's motion model / what it does in each scene |
