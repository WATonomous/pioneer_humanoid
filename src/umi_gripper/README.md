# umi_gripper: handheld gripper demos for the Pioneer arms

A handheld copy of the arm's gripper that you carry through a task, in the style of UMI
([universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface)). Three things
ride on it:

- **a Quest controller** (Quest 2 or 3) that tracks the gripper's pose,
- **the robot's wrist camera model**, in the robot's mount position,
- **an STS3215 servo used as an encoder** on the jaw linkage (torque off, read exactly as the leader arm reads its
  servos), which measures how far the jaws are closed.

Each demo is recorded raw, then converted into a LeRobot dataset in the gripper's **tool frame** (TCP pose + jaw
closure + wrist image, 25 fps). Because the tool frame has the robot's tool axes, a policy trained on this data drives
either 8dof arm through IK.

```
headset page (Meta Browser, existing quest teleop index.html) ──poses──┐
wrist camera (wrist_cam.rectify, 640x480 pinhole)            ──frames──┼─> collect.py ─> datasets/umi_raw/<session>/take_N
STS3215 jaw encoder (servo_leader.ServoLeader, torque off)   ──counts──┘                         │
                                                                                                   v
                         postprocess.to_lerobot: clock sync, latency, 25 fps, quality checks ─> datasets/pioneer_umi/real
                         postprocess.reachability / tools.replay_mujoco: can the 8dof arm follow it?
```

## Hardware

| Part | Notes |
|---|---|
| Quest 2 (default) or Quest 3 | Developer mode, `adb`, Meta Browser. Strap the headset on your forehead and look under it. Its screen shows a blank scene; it is only there to track the controller. |
| Controller mount | Your CAD. Keep the tracking ring (Quest 2) or the front face (Quest 3) clear and facing the headset. |
| Jaws and fingers | The same geometry as the 8dof arm's gripper (`assets/pioneer_bimanual_arm_8dof`), or the policy sees a different gripper. |
| Wrist camera | The robot's model, in the robot's position relative to the jaws (`arm_8dof_params._WRIST_POS`), calibrated with `perception/camera_calibration`. |
| STS3215 + bus board | The same board as the leader arm. Give the servo its own ID (default 1) and connect it to the jaw linkage. |
| Laptop | USB to the Quest (`adb reverse`), the camera and the encoder board. Linux or macOS. |

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e src/umi_gripper[test,replay] -e src/robot_learning[sim]
```

The package imports what it reuses straight from the repo: `quest_controller_teleop/transport.py` + `index.html`,
`pioneer_leader_arm_teleop/servo_leader.py`, `perception/camera_calibration/wrist_cam.py`, `pioneer_humanoid`, and
`robot_learning`'s LeRobot sink. Run commands from `src/umi_gripper` (or install it with `-e`).

## Configure

`config/umi_gripper.yaml` holds the shared defaults: which headset (`quest2` | `quest3`), which controller is
mounted, the CAD mount per headset, the width source (`encoder` | `trigger`), the camera device, and the
quality limits. The calibration tools write the measured values for this rig into `config/umi_calibration.json`,
which overrides the yaml. Commit that file, as the leader arm's `leader_arm.json`: there is one rig, so nobody
else has to re-measure it.

Frames (details in the yaml):
- **world**: the headset's floor frame. It never enters the data: everything is stored relative to the episode's first pose.
- **grip**: the controller's WebXR grip pose.
- **tcp**: the robot's tool frame. The fingers point along -Z, and the origin is between the jaws, 25 mm up from the tips.
- **camera**: the camera's OpenCV frame.

Set `mounts.<headset>.grip_to_tcp.rpy_deg` from your CAD. Calibration can't measure it.

## Calibrate (once per rig, and after changing the mount)

```bash
cd src/umi_gripper
# 0. bring-up: does the encoder answer, and is the controller streaming?
python ../teleop/pioneer_leader_arm_teleop/encoder_test.py --ids 1 --raw
adb reverse tcp:8444 tcp:8444 && python -m umi_gripper.quest_tracker      # then open http://localhost:8444/ on the Quest
# 1. jaw range: fully open, fully closed
python -m umi_gripper.calibration.calibrate_width
# 2. TCP offset: TCP point in a divot, click the thumbstick, tilt/twist about it for 10 s (residual must be < 3 mm)
python -m umi_gripper.calibration.calibrate_tcp
# 3. camera mount + latency: record 20-30 s of waving the gripper at a fixed printed ArUco marker, then
python -m umi_gripper.collect --task calibration
python -m umi_gripper.calibration.calibrate_camera ../../datasets/umi_raw/<session>/take_0000 --marker-size 0.10
```

Step 3 prints how far the handheld camera sits from the robot's camera position. If that distance is large, fix
the mount: the whole idea is that the camera sees the jaws exactly as the robot's camera does.

## Collect

```bash
src/umi_gripper/start_umi.sh --task "put the cup on the plate"      # adb reverse, opens the page, taps into VR
# or by hand: adb reverse tcp:8444 tcp:8444; python -m umi_gripper.collect --task "..."; open http://localhost:8444/
```

| On the mounted controller | |
|---|---|
| primary (B on the right controller, Y on the left) | start a take / save it |
| secondary (A / X) | discard the take |
| Ctrl+C | end the session (an unsaved take is discarded) |

The headset page and the terminal show `Ready` / `REC 3.2 s`, whether the controller is tracked, and the jaw
closure. Flags: `--headset quest3`, `--controller left`, `--width trigger` (no encoder: the trigger, mechanically
linked to the jaws), `--no-camera` (tracking tests), `--camera /dev/v4l/by-id/...`.

Takes land in `datasets/umi_raw/<date_time>/take_NNNN/` (git-ignored). Every stream is stored as it arrived, with
host timestamps: the raw grip poses, encoder counts and JPEG frames. Nothing is resampled at this stage, so
recalibrating later re-converts old takes.

**Tracking tips:** keep the controller where the headset's cameras can see it. Turn your head toward your work;
don't reach behind objects or your back. Short tracking gaps (< 100 ms) are interpolated over. Longer gaps inside a
take reject it; gaps at the start or end are trimmed off.

## Convert, check, replay

```bash
python -m umi_gripper.postprocess.to_lerobot ../../datasets/umi_raw --dry-run          # which takes pass, and why not
python -m umi_gripper.postprocess.to_lerobot ../../datasets/umi_raw                    # -> datasets/pioneer_umi/real
python -m umi_gripper.postprocess.reachability ../../datasets/umi_raw --arm left       # can the 8dof arm follow?
python -m umi_gripper.tools.replay_mujoco <take> [--video out.mp4]                     # watch it on the arm
```

Conversion moves every stream onto the host clock, removes each stream's latency, and resamples at 25 fps: the TCP
pose is interpolated, the jaw closure linearly interpolated, and each step takes the nearest camera frame. A take
is then rejected for any of: an interior tracking or camera gap, a TCP jump (> 2 m/s or > 15 rad/s), a headset
re-centre partway through, or a length under 1 s. The script writes a report next to the dataset listing each
take and why it was kept or dropped.

The reachability check places the demo's first pose at the arm's home TCP pose, then tracks every step with mink IK
at a joint speed cap. That catches demos the arm can't reach, or can't follow at that speed.

## Data contract (`src/robot_learning/config/pioneer_umi_dataset_schema.yaml`)

| Field | Content |
|---|---|
| `observation.state` (10) | TCP position (m) + 6D rotation in the episode's first TCP frame, then `gripper` (0 open … 1 closed) |
| `action` (10) | the next step's state: where the hand went |
| `observation.images.wrist` | 640x480 RGB, the rectified wrist camera |
| `task` | `--task` |

Storing poses relative to the episode's first frame keeps the headset's world frame out of the data. The pose of
any step relative to any other step is still exact, so a training pipeline can build UMI's relative-to-current
actions directly from these columns.

## On the real arm

Not built yet, on purpose. A tool-frame policy runs through `postprocess.reachability.TcpIK`, which turns TCP
targets into velocity-capped joint targets. Today that IK only drives the sim (`replay_mujoco`). The real arm is
still the old 6-DoF arm, and anything that moves it must go through `joint_command` and follow the
`real-hardware-safety` process: dry run first, one joint at a time, the hardware E-stop within reach.

## Files

```
src/umi_gripper/
├── start_umi.sh                     adb + headset page + collect, one command
├── config/umi_gripper.yaml          shared defaults; umi_calibration.json = this rig's measurements
└── umi_gripper/
    ├── quest_tracker.py             every pose of the mounted controller (reuses the quest teleop page + validation)
    ├── width.py                     STS3215 jaw encoder (servo_leader, torque off) or the trigger -> closure 0..1
    ├── camera.py                    wrist camera thread, wrist_cam.rectify
    ├── clock.py                     headset clock -> host clock (sliding minimum offset)
    ├── take.py                      raw take format
    ├── collect.py                   the collection session
    ├── calibration/                 calibrate_width, calibrate_tcp (pivot), calibrate_camera (hand-eye + latency)
    ├── postprocess/                 sync (25 fps), quality, reachability (TCP IK), to_lerobot
    └── tools/replay_mujoco.py       a take on the 8dof arm in MuJoCo
```

Tests: `python -m pytest src/umi_gripper/tests`. They build synthetic takes with known motion and check the clock
mapping, resampling, quality checks, pivot / hand-eye / latency calibration, ArUco pose recovery, the pose server,
the session logic and IK reachability. None of them need hardware.
