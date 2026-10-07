# pioneer_leader_arm_teleop

The 7-servo leader arm (STS3215, torque always off) drives the Pioneer left arm joint to joint,
with no IK, in any registered scene: Isaac Sim (default) or plain MuJoCo (`--target mujoco`, CPU).
`--target real` compares the leader with the real arm (dry run, publishes nothing); `--live` then drives it (#327).
Optional recording in the shared `dataset_schema_pioneer_v1.yaml` format, from either simulator.

Angles map **one-to-one** from a shared zero: arm hanging straight down under gravity, gripper open.
That is the URDF zero, the real arm's calibrated zero, and the leader's (`calibrate_leader.py`).

| Servo | Bus ID | Sim joint | Default sign |
|-------|--------|-----------|--------------|
| A | 2 | `joint1L` shoulder flexion | +1 |
| B | 3 | `joint2l` shoulder abduction | −1 |
| C | 1 | `joint3l` shoulder rotation | −1 |
| D | 5 | `joint4l` elbow flexion | +1 |
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

- **Home:** the arm starts at home (elbow bent 90°, forearm forward, gripper open) and follows the leader only once every leader joint is within 3° of home and the gripper is open. The status line lists the joints still off.
- **Directions:** move one leader joint at a time. If a sim joint goes the wrong way, restart with that entry flipped in `--signs` (order A..G, default `1,-1,-1,1,1,-1,1`).
- **R:** reset the arm and every object in the scene; the arm waits at home until the leader is back at home. During a take, it also discards the take.
- **`--record`:** `S` start · `N` save (then auto-reset) · `D` discard → `<repo>/datasets/pioneer_v1_left_arm/sim/` · `--cameras ego,wrist_left` / `none`.
- **Multi-step scenes** (`drawer_stow`, `matcha`, `duplo`): the terminal prints `[TASK] step k/n: <instruction>` as each step is done and
  `all steps done -- N to save` at the end. Recorded frames carry the current step's instruction as their `task` and a
  `subtask_index`, so a take is labelled step by step with no hand annotation.
  In the MuJoCo viewer these keys (and R) also toggle display flags (shadows, reflections, …); harmless.
- Other flags: `--port`, `--baud`, `--calibration` (default `leader_calibration.json`), `--filter-alpha` (target smoothing, default 0.35).

Targets are clamped to the arm's URDF limits. Wrist damping is lowered to 2.5 in this teleop only, so the sim wrist keeps up with the leader.

## Synthetic demos (MuJoCo, no leader)

`synthetic_teleop.py` records demos without a person: a simulated operator moves a virtual leader through the
same `LeaderMapping` (home engage, filter, clamp) and recorder as `--target mujoco --record`, so the dataset has
the same features, per-step `task` and `subtask_index`. Headless, CPU is fine.

```bash
MUJOCO_GL=egl python synthetic_teleop.py --scene drawer_stow --record --num_episodes 10   # CPU only: MUJOCO_GL=osmesa
python synthetic_teleop.py --scene peg_insert --num_episodes 3 --preview peg.mp4          # just look, no dataset
```

- **Human-like motion** (`human_operator.py`): the operator aims the follower's gripper and moves the leader to match
  (IK of the leader, a 1:1 copy of the arm). Reaches are overlapping minimum-jerk strokes (bell-shaped speed, slightly
  curved paths) timed by Fitts' law; the first stroke lands a few % off, and after a reaction time the operator
  corrects what they see, stopping when the arm is blocked rather than pushing harder. Tremor (8–12 Hz, sub-mm), slow
  drift, wrist wobble, noisy judgement of where objects are, hesitation before grasps. Speed, accuracy, tremor and so on
  are drawn per episode, like different people.
- **Plans** (`synthetic_tasks.py`): `drawer_stow`, `peg_insert`, `zip_tie`, written like instructions to a person:
  look, reach, grasp, check it worked and recover if not (re-grasp, pull again, stand a crooked peg back up), place.
- **Seeds:** episode *i* uses `--seed + i` for the layout and the operator. Takes are first run without cameras on
  `--workers` processes; only successful ones are replayed (deterministically) with cameras and saved, like discarding
  a botched take. Measured: drawer_stow 29/30, peg_insert 18/20, zip_tie 20/20 seeds succeed.
- **Speed / memory:** the physics and the operator run ~3–10× faster than real time; the cameras are the slow part
  on CPU (osmesa: ~0.5 s per 640×480 frame), so recording renders them on up to 3 spawned processes (~1.2 GB each)
  from each frame's poses. The recorder keeps a take's frames in RAM (~5.5 GB for two cameras): budget ~14 GB.
  With a GPU (`MUJOCO_GL=egl`) rendering is not the bottleneck.
- Adding a scene: write `plan(op, model, data, notes)` and a success check in `synthetic_tasks.py`.

## Real arm (dry run, then `--live`)

Without `--live` it is read-only: subscribes to `/interfacing/motorFeedback`, creates no publisher, so it
cannot move the arm. It prints leader vs real joint angles in the URDF frame (also the command frame
`ArmPose` / `joint_command` use), their difference, and how far the leader is from home. Use it to check
the calibrations before any live teleop.

```bash
# interfacing up, arm powered (motors not commanded); then in the simulation_mj container:
./watod -t simulation_mj
cd /workspace/humanoid/src/teleop/pioneer_leader_arm_teleop
python3 pioneer_leader_arm_teleop.py --target real --port /dev/ttyACM1   # leader port: not the CANable's
python3 pioneer_leader_arm_teleop.py --target real --port /dev/ttyACM1 --live   # needs joint_command running
python3 pioneer_leader_arm_teleop.py --target real --self-test           # angle math only, no ROS / leader
```

- Leader: `sign × leader` (clamped to the URDF limits, as in sim).
  Real: `zero_offset + motor / direction` from `arm_calibration.yaml`. Same frame, compared directly.
- Check: pose both arms the same, motors off; every joint should agree within 5° across its range.
  Fix `--signs` (leader) or the real arm's `direction` / calibration (`calibrate_arm.py`) until it does.
- Warnings (end of each row): `>5` (disagreement), `NO FEEDBACK` (none for 0.5 s), `urdf-clamp`, `outside hw [lo,hi]`
  (the leader's target is outside `arm_calibration.yaml`'s limits; `joint_command` would clamp it).
- Gripper: compared as position (0 open .. 1 closed), leader vs the real GL40 (id 21); `>0.1` warns.

`--live`:
1. Shows the dry run until the gate passes: on every joint `active` in `arm_actuators.yaml`, the leader is
   within 3° of home and within 5° of the real arm, and the gripper within 0.1. The screen lists what fails.
2. Asks you to type `live` (anything else quits), re-checks the gate, then publishes `ArmPose` on
   `/arm/joint_targets` at 50 Hz, gripper included. `joint_command` clamps, rate-limits and runs the watchdog.
3. Ctrl-C stops publishing. `joint_command` holds the last pose for `command_timeout_sec` (10 s), then
   sets kp = 0 and the arm sinks: support it.

## Files

| file | role |
|------|------|
| `pioneer_leader_arm_teleop.py` | entry; `--target isaac\|mujoco\|real` picks the backend before any simulator import |
| `isaac_sim.py` / `mujoco_sim.py` | sim backends |
| `real_arm.py` | real-arm backend: dry run (leader vs `/interfacing/motorFeedback`), `--live` publishes `/arm/joint_targets` |
| `leader_mapping.py` | shared: 1:1 mapping, home check, filter, gripper closure, leader read, wall-clock pacing |
| `servo_leader.py` / `arm_limits.py` | servo bus reader / clamp + gripper fraction |
| `calibrate_leader.py` | stores the hanging-pose zero |
| `encoder_test.py` | servo bring-up, no simulator |
| `synthetic_teleop.py` | synthetic demos: simulated operator → same mapping and recorder as `mujoco_sim.py` |
| `human_operator.py` / `synthetic_tasks.py` | the operator's motion model / what it does in each scene |
