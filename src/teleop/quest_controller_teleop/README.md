# Quest 2 controller teleoperation

Control the Pioneer bimanual arms in Isaac Sim or MuJoCo with Meta Quest 2 Touch
controllers. Arm movement is always on while tracking is valid; no grip button
is needed. The IK follows Aurel Arnold's
[VR teleoperation methodology](https://aurelarnold.xyz/blog/vr-teleoperation-stack/#control-that-feels-natural),
adapted to Pioneer's joint geometry.

## MuJoCo (no GPU, runs on a laptop)

Any scene in `humanoid_mujoco_scenes` (default `bimanual_sorting`), same page, IK and
controls as Isaac. Needs `mujoco`, `aiohttp` and `Pillow`. Plug the Quest into the same machine:

```bash
mjpython src/teleop/quest_controller_teleop/mujoco/mujoco_sim.py --scene bimanual_sorting
adb reverse tcp:8444 tcp:8444
```

Then open `http://localhost:8444/` in Meta Browser and select **Enter VR**. The headset
shows the ego camera (mono) and both wrist cameras; the MuJoCo window shows the scene on
the laptop; **6** with that window focused resets, like left Y. The window marks each IK target with a sphere
(green tracking, orange near the lead limit, red pinned). `--headless` drops the window and runs under
plain `python` (`mjpython` is only needed for the window on macOS). Recording and the cloth scene are Isaac-only.

**Tuning flags** (both backends): `--gain`, `--rotation-gain`, `--joint-speed` (shoulder),
`--elbow-speed` (above the shoulder's, the elbow leads), `--wrist-speed`, `--position-lead`,
`--angular-lead`, `--position-damping` and `--orientation-damping` (baseline DLS lambda; higher is
smoother but slower to close an error). `--help` lists the defaults.

**Teleop joint limits** live in [`joint_limits.yaml`](joint_limits.yaml): `jN: [min, max]` in degrees,
left-arm convention (the right arm is the left mirrored, every angle negated), with the reason for each.
They narrow only what the IK may command, inside the hardware limits: shoulder flexion -40..90, abduction
-20..20 and upper-arm rotation -10..45 (together they keep the elbow from swinging outward when the hand
twists), elbow -110..-35 (no deep fold; always visibly bent), forearm roll +-75, wrist up/down -45..120.
Edit the file, or keep another set and pass `--joint-limits my_limits.yaml`; a joint left out keeps its
hardware range.

MuJoCo defaults to `--controller-frame base` (hand up = wrist up, forward = forward): in the ego
camera's frame, tilted 50 deg down, raising a hand moved the wrist forward as well. Headset cameras
render on their own thread (`mujoco/mujoco_cameras.py`), so the 10 ms control loop never waits on them.

## Isaac: start from the host

Enter the existing container:

```bash
docker start watod_rwahib-simulation_isaac-1
docker exec -it watod_rwahib-simulation_isaac-1 bash
```

Inside the container, start the cloth-folding scene:

```bash
cd /workspace/humanoid
bash src/teleop/quest_controller_teleop/isaac/run_quest_controller_teleop_always_on.sh \
  --cloth-folding --single-task-thread
```

For the default pick-and-place scene, omit `--cloth-folding`. There is one
launcher, with the same always-on control in both scenes. `--scene cloth_folding`
is equivalent to `--cloth-folding`; `--garment Top_Long_Seen_1` selects a garment.
Append `--headless` for headset-only viewing.

On the host, in a second terminal:

```bash
adb devices
adb reverse tcp:8444 tcp:8444
```

Authorize USB debugging in the headset if needed. Open `http://localhost:8444/`
in Meta Browser, refresh it after code changes, and select **Enter VR**. The
simulator serves controller input, status, and camera images; no separate web
server or ROS node is needed. Stop the simulator with **Ctrl+C** before restarting
or changing scenes. Ctrl+Z only suspends it and retains GPU memory.

If Python dependencies are missing, install these inside the container:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install \
  -r src/teleop/quest_controller_teleop/requirements-isaac.txt
```

The IK uses NumPy and the existing robot URDF. No external IK package or helper
process is required.

## Controls

| Input | Action |
| --- | --- |
| Controller motion | Move that arm automatically while tracked |
| Index trigger | Proportional gripper closure |
| A / X | Hold for 25% translation and rotation increments |
| Left Y | Reset the scene, arms, and grippers once per press |
| Thumbstick click | Hold to pause that arm and collect wrist-pivot calibration; release to fit and resume |

Tracking loss or input older than 200 ms pauses movement. Tracking return
reanchors the target at the measured arm pose, avoiding a jump from controller
movement during the interruption. Reset and calibration also reanchor before
resuming. Releasing the grip button does not pause movement.

For pivot calibration, hold the thumbstick click and rotate the controller about
several axes while keeping your anatomical wrist pivot stationary. Release after
3–5 seconds. Offsets are saved in the ignored `wrist_calibration.json`; failures
appear in the status message. Calibration helps prevent controller grip-origin
motion during wrist rotation from becoming unintended arm translation.

## Motion mapping and IK

The control path is `index.html` → `transport.py` → `scene_driver.py` →
`article_control.py` → Isaac joint targets. `control.py` supplies URDF geometry,
rotation utilities, and pivot fitting. WebXR sends the grip-space quaternion in
XYZW order. Translation and rotation use the same coordinate mapping, including
the actual stereo-camera orientation and headset heading captured on engagement.
Use `--controller-frame base` for upright robot-base axes.

The blog-based sequential solver warm-starts from the previous command, solves
position, recomputes FK, then solves orientation using a base-frame SO(3) error
and matching rotational Jacobian. Position uses joints 1/2/4; orientation uses
3/5/6. This grouping preserves three rotation directions at Pioneer's home pose.
The wrist anchor is joint 6's origin (the wrist pivot), identical on both arms.
Joint 3 also moves the wrist anchor, so position and orientation are coupled.
Pioneer's two distal joints cannot reproduce every 3D rotation alone; some
proximal movement is required.

The wrist improvements retained here include pivot calibration, the robot-specific
orientation grouping, separate adaptive damping, faster distal-joint speed caps,
and no orientation spring toward home. Physical joint limits remain enforced.
See [the article comparison](IK_ARTICLE_AUDIT.md) for equations and adaptations.

Targets are anchored to the hand (`AnchoredTarget`): the request is the engagement
target plus the gain times the hand's motion since engagement. Translation lead is
capped at 8 cm and rotation lead at 0.6 rad from measured FK, but the cap only limits
what the IK is sent each frame, so once the arm catches up it lines up with the hand
again. The article's policy instead absorbs the excess for good, which in always-on
control only accumulates: after 15 s of fast scripted wandering and a return to the
start, the arm was 25 cm / 43 deg off with it and 0.1 cm / 0 deg anchored. Pressing or
releasing precision (A/X) re-bases the reference, without a jump. Pausing and resuming
tracking or calibration starts a new reference.

Default translation gain is 1.5, rotation gain 1, joint-speed cap 0.9 rad/s on
joints 1–4, and 3.6 rad/s on joints 5–6. To set them explicitly:

```bash
bash src/teleop/quest_controller_teleop/isaac/run_quest_controller_teleop_always_on.sh \
  --cloth-folding --single-task-thread \
  --gain 1.5 --joint-speed 0.9 --wrist-speed 3.6 \
  --rotation-gain 1.0 --angular-lead 0.6
```

## Scenes, recording, and runtime

Cloth mode requires `src/simulation/garment_fold_task` and its assets. It uses
CUDA PhysX particle cloth with Fabric disabled for mesh rendering. Cloth
grasping still needs tuning; scene loading does not establish reliable folding.
Reset restores initial cloth particle positions and pose with zero velocity.

Append `--record` to record the left arm using the shared robot-learning recorder.
With the Isaac window focused, **S** saves an episode and **D** discards it.
Recording begins when a controller target becomes active. Exiting does not save
the current episode. Cloth mode has its own dataset schema and output directory.

`--single-task-thread` retains the startup workaround for the native Carbonite
asset/tasking mutex crash. This is not a verified root-cause fix. HTTP/WebSocket
handling uses Kit's main-thread async loop, serviced before input consumption on
physics ticks. Scene operations and camera capture also stay on the main thread.
