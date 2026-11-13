> For the current NASA cargo policy, use [the HG-DAgger runner](../../../assets/nasa_cargo/MOVING_TRAY.md).
> It includes Wilson's latest compatible smoothing, anchored mapping, packet prediction and tab-ownership updates.
> [Import details](IMPORTED_FROM.md) explain the six-joint Isaac / seven-joint MuJoCo compatibility boundary.
> The standalone Isaac commands below retain their original defaults.

# Quest 2 controller teleoperation

For the trained **NASA cargo policy with HG-DAgger interventions**, use the
[cargo VR workflow](../../../assets/nasa_cargo/VR_HG_DAGGER.md). It uses this
controller stack through `hg_dagger.py --teleop vr`, with the existing cargo
policy observation/action contract. The standalone scenes described below use
their own recording configuration.

Control the Pioneer bimanual arms in Isaac Sim with Meta Quest 2 Touch
controllers. Arm movement is always on while tracking is valid; no grip button
is needed. The IK follows Aurel Arnold's
[VR teleoperation methodology](https://aurelarnold.xyz/blog/vr-teleoperation-stack/#control-that-feels-natural),
adapted to Pioneer's joint geometry.

## Start from the host

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

The control path is `index.html` → `transport.py` → `isaac_control.py` →
`article_control.py` → Isaac joint targets. `control.py` supplies URDF geometry,
rotation utilities, and pivot fitting. WebXR sends the grip-space quaternion in
XYZW order. Translation and rotation use the same coordinate mapping, including
the actual stereo-camera orientation and headset heading captured on engagement.
Use `--controller-frame base` for upright robot-base axes.

The blog-based sequential solver warm-starts from the previous command, solves
position, recomputes FK, then solves orientation using a base-frame SO(3) error
and matching rotational Jacobian. Position uses joints 1/2/4; orientation uses
3/5/6. This grouping preserves three rotation directions at Pioneer's home pose.
Joint 3 also moves the wrist anchor, so position and orientation are coupled.
Pioneer's two distal joints cannot reproduce every 3D rotation alone; some
proximal movement is required.

The wrist improvements retained here include pivot calibration, the robot-specific
orientation grouping, separate adaptive damping, faster distal-joint speed caps,
and no orientation spring toward home. Physical joint limits remain enforced.
See [the article comparison](IK_ARTICLE_AUDIT.md) for equations and adaptations.

Targets follow the article's incremental slipping-clutch policy. Translation
lead is capped at 8 cm and rotation lead at 0.6 rad from measured FK; excess
movement is absorbed. At rotation gain 1, mapped controller increments match
until a lead or physical limit is reached. Absorption can change alignment.
Pausing and resuming tracking or calibration starts a new reference.

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
