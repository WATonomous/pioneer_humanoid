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
| **`keyboard_teleop/`** | keyboard + IK | pioneer bimanual, left arm | see below |
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
- **`--scene`:** any `humanoid_isaac_scenes` scene — `bare` (default, empty lightbox workcell), `push` (ramp-box + block), `vial_rack`, …
- **`--record`** (image already has `humanoid-robot-learning`): `S` start · `N` save · `D` discard · `Esc` stop → `<repo>/datasets/pioneer_v1_left_arm/sim/` · `--cameras ego,wrist_left` / `none`

## Notes

- Upper body = 6-DOF arm + 15-DOF hand. Lower body (unused): 4D `[x, y, yaw, height]`.
- Retargeter reference: `IsaacLab/.../devices/openxr/retargeters/humanoid/unitree`
