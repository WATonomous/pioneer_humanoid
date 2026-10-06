# Teleop

All teleoperation entry points, one home. **Everything runs inside the
`simulation_isaac` watod container** (Isaac Lab 2.3.2) — do not use host Isaac Lab.

```bash
# host: ACTIVE_MODULES="simulation_isaac" in watod-config.local.sh
./watod up -d
./watod -t simulation_isaac          # shell into the container
```

The repo is bind-mounted at `/workspace/humanoid`. Shared arm config + IK helpers:
`src/pioneer_humanoid`. Shared recorder: `src/robot_learning`.

| Folder | Input | Sim robot | Notes |
|--------|-------|-----------|-------|
| [`quest_teleop/`](quest_teleop/) | Quest 2 hand tracking | pioneer bimanual, both arms | `bridge/` (WebXR → `/quest_teleop`) + `sim/` (weighted-DLS fingertip IK); see its README |
| **`keyboard_teleop/`** | keyboard + IK | pioneer bimanual, left arm | Isaac (`keyboard_teleop.py`) or plain MuJoCo, no container (`mujoco_keyboard_teleop.py`); see below |
| [`pioneer_leader_arm_teleop/`](pioneer_leader_arm_teleop/) | 7-servo leader arm | pioneer bimanual, left arm | any `--scene`, `--record`; see its README |
| [`task_space_controller/`](task_space_controller/) | viewport pose gizmo + IK | pioneer bimanual, left arm | `--publish-real-left-arm` drives the real arm — **its README covers the CAN pipeline + e-stop** |
| `humanoid-record` (CLI) | ROS topics | real pioneer arm | `src/robot_learning` |

## keyboard_teleop

Left arm follows the keyboard via differential IK (absolute-pose target, DLS).

```bash
# inside the simulation_isaac container:
cd /workspace/humanoid/src/teleop/keyboard_teleop
PYTHONPATH=$(pwd) /workspace/isaaclab/isaaclab.sh -p keyboard_teleop.py [--scene bare|push] [--record]
```

- **Move:** `W/S` x · `A/D` y · `Q/E` z · `Z/X` `T/G` `C/V` rotate · **hold `Shift`** = fine
- **`K`** toggle gripper · **`R`** reset arm
- **`--scene`:** `bare` (default) or `push` (table + ramp-box + block + lightbox)
- **`--record`** (image already has `humanoid-robot-learning`): `S` start · `N` save · `D` discard · `Esc` stop → `<repo>/datasets/pioneer_v1_left_arm/sim/` · `--cameras ego,wrist_left` / `none`

### MuJoCo backend (`mujoco_keyboard_teleop.py`)

CPU only, no container: any `humanoid_mujoco_scenes` scene, recording in the leader teleop's format (the two
can share a dataset; `leader_angles` / `leader_counts` are NaN for keyboard takes).

```bash
pip install mujoco pynput && pip install -e src/pioneer_humanoid -e src/simulation/mujoco_scenes -e "src/robot_learning[sim]"
python src/teleop/keyboard_teleop/mujoco_keyboard_teleop.py --scene tidy_table --record --cameras top,wrist_left
```

- **Keep the terminal focused**, not the viewer: keys are read globally (pynput), and the MuJoCo viewer maps
  most letters to display toggles. macOS: run with `mjpython` and allow the terminal under Privacy >
  Accessibility.
- The gripper always points down; keys move the grasp point (between the finger pads). The first move key
  glides the arm from home to gripper-down over the table, then you have it.
- **Move:** `W/S` x · `A/D` y · `Q/E` up/down · `C/V` turn · **hold `Shift`** = fine (quarter speed)
- **`K`** open / close (eases over 0.4 s) · **`J`** narrow (default, ~66 mm) / wide opening · **`H`** glide home
  (end each take with it) · **`R`** reset arm + scene · `Esc` quit
- **`--record`:** `P` start a take · `N` save (then a new layout) · `B` discard. Not `S`/`D`, which move the arm.
- Contacts stay calm: speeds ramp, joint targets are rate-limited, the finger tips can't be commanded into the
  table, and pushing into something holds the arm where it stopped (~4 N held, not the ~110 N the stiff
  joints would otherwise press with). Hitting something at full speed still taps it (~140 N for an instant);
  hold Shift near things (~75 N).
- Tests: `pytest src/teleop/keyboard_teleop/test_mujoco_keyboard_teleop.py` (headless).

## Notes

- Upper body = 6-DOF arm + 15-DOF hand. Lower body (unused): 4D `[x, y, yaw, height]`.
- Retargeter reference: `IsaacLab/.../devices/openxr/retargeters/humanoid/unitree`
