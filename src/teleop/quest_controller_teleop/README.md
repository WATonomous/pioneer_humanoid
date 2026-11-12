# Quest 2 controller teleoperation

Drive the Pioneer bimanual rig's **7-DoF + 1-DoF-gripper arms**
([`pioneer_bimanual_arm_8dof`](../../../assets/pioneer_bimanual_arm_8dof/README.md)) in MuJoCo with Meta Quest 2 Touch
controllers. MuJoCo only (no Isaac: the arm has no USD yet); no recording yet.

## Run (no GPU, runs on a laptop)

1. `pip install -r src/teleop/quest_controller_teleop/requirements.txt` (plus `numpy`, `Pillow`, `pyyaml`).
2. Plug the Quest in (USB debugging allowed) and run, from the repo root:
   ```bash
   src/teleop/quest_controller_teleop/start_quest.sh --scene bimanual_sorting
   ```
   It forwards the port, leaves one teleop tab in Meta Browser, starts the sim and clicks into VR. If the headset
   shows the page instead, pull the trigger anywhere on it (any click enters VR). It runs `.venv/bin/mjpython`;
   set `PY=python` off macOS.
3. The headset shows the ego and both wrist cameras. To watch only the MuJoCo window (headset just tracks the
   controllers), add `--no-headset-video`: the GPU then goes to the window (~60 fps instead of ~20).

By hand: `mjpython src/teleop/quest_controller_teleop/mujoco/mujoco_sim.py --scene bimanual_sorting`,
`adb reverse tcp:8444 tcp:8444`, then open `http://localhost:8444/` in Meta Browser. The window opens on the ego
camera (Esc frees it). Only Meta Browser can take control, and of several open tabs the one in VR has it.

Any scene in `humanoid_mujoco_scenes` works. `--help` lists the tuning flags (gains, speed caps, lead limits, IK, input filter) and their defaults.

## Controls

| Input | Action |
| --- | --- |
| Controller motion | Moves that arm (always on while tracked; no grip button) |
| Index trigger | Proportional gripper closure |
| A / X | Hold for 25% translation and rotation (precision) |
| Left Y, or **6** in the MuJoCo window | Reset the scene, arms and grippers |
| Thumbstick click | Hold, rotate the controller about your still wrist for 3–5 s, release: wrist-pivot calibration (saved in git-ignored `wrist_calibration.json`) |

Input older than 200 ms pauses the arm; when tracking returns (or after a reset / calibration) it re-anchors at the measured pose, so it never jumps.

## How it works

`index.html` → `transport.py` → `scene_driver.py` → `mink_control.py` → joint targets.

- **Input filter** (`input_filter.py`): a 1-euro filter on each controller pose (OpenArm's scheme): smooth when the hand is still, little lag when it moves; speed caps spread tracking jumps.
- **Targets** (`article_control.py`): the arm pose at engagement plus 1.5× the hand's motion, at most 8 cm / 0.6 rad ahead of the arm.
- **IK** (`mink_control.py`): one mink QP per tick, position at the wrist pivot first, gripper orientation second (weight 0.2), a light pull toward home; each joint's URDF limit and speed cap (0.9 rad/s joints 1–4, 3.6 rad/s 5–7) are hard constraints.
- **Joint limits:** the URDF's, plus one rule: the elbow stays at least 10° bent (away from the straight-arm singularity). `--body-keepout` adds an approximate torso keep-out (off: the stand has no torso).

Benchmarks behind these choices: PRs #535 and #540.
