# Quest 2 controller teleoperation

Drive the Pioneer bimanual rig's **7-DoF + 1-DoF-gripper arms**
([`pioneer_bimanual_arm_8dof`](../../../assets/pioneer_bimanual_arm_8dof/README.md)) in MuJoCo with Meta Quest 2 Touch
controllers. MuJoCo only (no Isaac: the arm has no USD yet). The headset only tracks; you watch the Mac's window.

## Run (no GPU, runs on a laptop)

1. Once: `adb` (`brew install android-platform-tools`), the Quest in developer mode, and a repo-root `.venv` with
   `pip install -r src/teleop/quest_controller_teleop/requirements.txt` (plus `numpy`, `pyyaml`). The launcher runs
   `.venv/bin/python` (`PY=` overrides).
2. Plug the Quest in (accept USB debugging, "always allow") and run, from the repo root:
   ```bash
   src/teleop/quest_controller_teleop/start_quest.sh --scene bimanual_sorting
   ```
   It forwards the port, leaves one teleop tab in Meta Browser, starts the sim and clicks into VR for you (retrying
   until poses arrive), so the VR controllers never touch the page. If it gives up (headset asleep), wake the
   headset and rerun, or pull the trigger anywhere on the page.
3. The Mac shows one camera window: ego on top, both wrist cameras below (~30 fps). **6** resets; closing it stops
   the sim. The headset shows a plain dark scene: it only streams poses.

By hand: `python src/teleop/quest_controller_teleop/mujoco/mujoco_sim.py --scene bimanual_sorting`,
`adb reverse tcp:8444 tcp:8444`, then open `http://localhost:8444/` in Meta Browser. Only Meta Browser can take
control, and of several open tabs the one in VR has it.

Any scene in `humanoid_mujoco_scenes` works. `--help` lists the tuning flags (gains, speed caps, lead limits, IK, input filter) and their defaults.

## Controls

| Input | Action |
| --- | --- |
| Controller motion | Moves that arm (always on while tracked; no grip button) |
| Index trigger | Proportional gripper closure |
| A / X | Hold for 25% translation and rotation (precision) |
| Left Y, or **6** in the window | Reset the scene, arms and grippers (discards an unsaved take) |
| Right B, or **7** / **8** in the window | With `--record`: start a take / save it (the scene then resets); **9** discards |
| Thumbstick click | Hold, rotate the controller about your still wrist for 3–5 s, release: wrist-pivot calibration (saved in git-ignored `wrist_calibration.json`) |

Input older than 200 ms pauses the arm; when tracking returns (or after a reset / calibration) it re-anchors at the measured pose, so it never jumps.

## Recording

Add `--record` to record LeRobot datasets for training (`pip install -e src/robot_learning[sim]` first):

```bash
src/teleop/quest_controller_teleop/start_quest.sh --scene bowling --record \
  --task_description "stand the bottles up"
```

Both arms, 25 fps of sim time, schema `pioneer_8dof_dataset_schema.yaml` (8 values per arm), cameras ego and both
wrists at 640x480, into `datasets/pioneer_8dof_bimanual/sim` (later sessions append). The window shows `REC <s>` while
recording. Train on it as in `src/robot_learning/README.md`.

Multi-step scenes (those with a `progress` function, e.g. `drawer_stow`) record the current step's instruction as each
frame's task ("open the drawer", "put the red block in the drawer", ...) plus a `subtask_index` feature, as the leader
teleop does: sim evaluation prompts the policy with the same per-step instruction. The window and the terminal show the
current step; save the take once it says all steps are done.

```bash
src/teleop/quest_controller_teleop/start_quest.sh --scene drawer_stow --record
```

`drawer_stow` is a left-arm task: train on the left arm with `humanoid_robot_learning.select_arm`
([robot_learning README](../../robot_learning/README.md)).

## How it works

`index.html` → `transport.py` → `scene_driver.py` → `mink_control.py` → joint targets.

- **Input filter** (`input_filter.py`): a 1-euro filter on each controller pose (OpenArm's scheme): smooth when the hand is still, little lag when it moves; speed caps spread tracking jumps.
- **Targets** (`article_control.py`): the arm pose at engagement plus 1.5× the hand's motion, at most 8 cm / 0.6 rad ahead of the arm.
- **Between packets** (`scene_driver.py`): the 90 Hz headset vs 100 Hz ticks; empty ticks move on along the hand's
  last velocity, so motion never goes stop-go.
- **IK** (`mink_control.py`): one mink QP per tick, position at the wrist pivot first, gripper orientation second (weight 0.2), a light pull toward home; each joint's URDF limit and speed cap (0.9 rad/s joints 1–4, 3.6 rad/s 5–7) are hard constraints.
- **Motors** (`mujoco/mujoco_sim.py`): position PD plus velocity and gravity feedforward (the motors' MIT mode), so
  the sim arm doesn't trail its command.
- **Window and recording** (`mujoco/mujoco_cameras.py`): one render thread draws the three cameras for the window
  and for the recorder, so they never compete for the GPU.
- **Joint limits:** the URDF's, plus one rule: the elbow stays at least 10° bent (away from the straight-arm singularity). `--body-keepout` adds an approximate torso keep-out (off: the stand has no torso).

Benchmarks behind these choices: PRs #535 and #540.
