# Quest 2 controller teleoperation

Control both Pioneer arms in Isaac Sim using Meta Quest 2 Touch controllers.
The control method follows Aurel Arnold's
[VR teleoperation methodology](https://aurelarnold.xyz/blog/vr-teleoperation-stack/#control-that-feels-natural),
adapted to Pioneer's joint geometry. This entry point controls simulation only.

```text
Quest browser: index.html / WebXR poses and buttons
    → transport.py: WebSocket /ws on port 8444
    → isaac_control.py: input freshness, clutch targets, calibration, reset
    → control.py: URDF kinematics and position-priority IK
    → isaac/run_quest_controller_teleop.py: joint targets and Isaac physics
    ← stereo and wrist-camera JPEGs over HTTP
    ← status and IK/limit haptics over WebSocket
```

## Start

Run inside the existing `simulation_isaac` container, from `/workspace/humanoid`:

```bash
"${PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}" -m pip install \
  -r src/teleop/quest_controller_teleop/requirements-isaac.txt
bash src/teleop/quest_controller_teleop/isaac/run_quest_controller_teleop.sh
```

On the host, connect the Quest over USB:

```bash
adb devices
adb reverse tcp:8444 tcp:8444
```

Open `http://localhost:8444/` in Meta Browser and enter VR. Release both grips
before squeezing to move. The server handles the page, controller input, and
camera images directly; ROS and a separate web server are not required.
Append `--headless` for headset-only viewing.

## Scenes and recording

The default is pick-and-place with the table, box, and container. To load cloth:

```bash
bash src/teleop/quest_controller_teleop/isaac/run_quest_controller_teleop.sh \
  --cloth-folding --garment Top_Long_Seen_1
```

`--scene cloth_folding` is equivalent to `--cloth-folding`. Stop the process
before changing scenes or starting another server on the same port.

Cloth mode requires the `src/simulation/garment_fold_task` workspace and its
assets. It uses CUDA PhysX particle cloth with Fabric disabled to synchronize
mesh rendering. Cloth grasping remains untuned; loading the scene does not
establish reliable folding.

Append `--record` to record the left arm using the shared robot-learning recorder.
With the Isaac window focused, **S** saves an episode and **D** discards it.
Recording begins when a controller target becomes active. Exiting does not save
the current episode. Cloth mode uses its own dataset schema and output directory.

## Controls

| Input | Action |
| --- | --- |
| Grip | Hold to move that arm; release to hold its command |
| Trigger while gripping | Proportional gripper closure |
| A / X | Hold for 25% translation and rotation increments |
| Left Y | Reset the selected scene, arms, and grippers once per press |
| Thumbstick click with grip released | Collect wrist-pivot calibration; release to fit and save |

For pivot calibration, rotate the controller while keeping your wrist pivot
stationary. The fitted offsets are stored in the ignored `wrist_calibration.json`.
Calibration failure is reported in the status message.

After a reset or tracking loss, release the grips before reengaging. Input older
than 200 ms disengages control. Reset preserves saved wrist-pivot offsets; cloth
reset restores the initial particle positions and garment pose with zero velocity.

## Motion mapping and IK

Engaging grip anchors the target at the measured arm pose. Subsequent controller
increments move the target; reengaging does not jump to the controller's absolute
position. Translation gain defaults to 1.5 and rotation gain to 1.

The default coordinate mapping uses the actual stereo camera orientation and
headset heading captured at clutch engagement. Use `--controller-frame base`
for upright robot-base axes. Translation and rotation share the same mapping.

The solver uses URDF forward kinematics and Jacobians. Joints 1–4 position the
joint5 anchor; the remaining motion that preserves this position, including
joints 5–6, adjusts orientation. Joint limits, command speed limits, damping,
and target lead limits constrain the result. Some orientations are unreachable
with the anchor held fixed.

```bash
bash src/teleop/quest_controller_teleop/isaac/run_quest_controller_teleop.sh \
  --gain 1.5 --joint-speed 1.0 --wrist-speed 4.0 \
  --rotation-gain 1.0 --angular-lead 0.6
```

## Diagnostics

The headset HUD and `http://localhost:8444/status` report tracking, orientation
errors, joint stops, and absorbed motion. Add `--controller-diagnostics` for
actual/commanded joint angles and target/model comparisons once per wall second
while clutched. Small IK error with large motor error suggests actuator tracking
or contacts; large IK error points toward solver or pose constraints.

For Carbonite asset/tasking mutex crashes, `--single-task-thread` applies the
repository's startup workaround. This is not a verified root-cause fix. The runner
also enables Python fault-handler thread dumps for native aborts. HTTP/WebSocket
handling runs on Kit's main application loop; scene changes and camera capture
remain on the main thread.
