# Teleop

All teleoperation entry points, one home. The MuJoCo ones (Quest, leader arm by default) run natively, even on a
laptop; **the Isaac ones run inside the `simulation_isaac` watod container** (Isaac Lab 2.3.2), not host Isaac Lab.

```bash
# host: ACTIVE_MODULES="simulation_isaac" in watod-config.local.sh
./watod up -d
./watod -t simulation_isaac          # shell into the container
```

The repo is bind-mounted at `/workspace/humanoid`. Shared arm config + IK helpers:
`src/pioneer_humanoid`. Shared recorder: `src/robot_learning`.

| Folder | Input | Sim robot | Notes |
|--------|-------|-----------|-------|
| [`quest_controller_teleop/`](quest_controller_teleop/) | Quest 2 Touch controllers | pioneer_bimanual_arm_8dof (7-DoF + gripper), both arms, MuJoCo | `start_quest.sh` (one command), `--record`; see its README |
| **`keyboard_teleop/`** | keyboard + IK | pioneer bimanual, left arm | see below |
| [`pioneer_leader_arm_teleop/`](pioneer_leader_arm_teleop/) | leader arm (8 servos; 7 for the old arms) | pioneer_bimanual_arm_8dof in MuJoCo (default); old arms in Isaac / real | any `--scene`, `--record`; see its README |
| [`task_space_controller/`](task_space_controller/) | viewport pose gizmo + IK | pioneer bimanual, left arm | `--publish-real-left-arm` drives the real arm — **its README covers the CAN pipeline + e-stop** |
| `humanoid-record` (CLI) | ROS topics | real pioneer arm | `src/robot_learning` |

## keyboard_teleop

Left arm follows the keyboard via differential IK (absolute-pose target, DLS).

```bash
# inside the simulation_isaac container:
cd /workspace/humanoid/src/teleop/keyboard_teleop
PYTHONPATH=$(pwd) /workspace/isaaclab/isaaclab.sh -p keyboard_teleop.py [--scene bare|push] [--record]
```

Keys: see [Keyboard controls](#keyboard-controls) below.

- **`--scene`:** any `humanoid_isaac_scenes` scene — `bare` (default, empty lightbox workcell), `push` (ramp-box + block), `vial_rack`, …
- **`--record`** (image already has `humanoid-robot-learning`) → `<repo>/datasets/pioneer_v1_left_arm/sim/` · `--cameras ego,wrist_left` / `none`

## Keyboard controls

Every key of every teleop in one place. Isaac Sim keys need the Isaac Sim window focused; MuJoCo keys need the
MuJoCo viewer focused. Recording keys only do something with `--record`. **If you change a binding, update this table.**

| Action | `keyboard_teleop` (Isaac) | leader arm, Isaac | leader arm, MuJoCo | Quest (Isaac) |
|--------|---------------------------|-------------------|--------------------|---------------|
| Move end effector x / y / z | `W`/`S` · `A`/`D` · `Q`/`E` | leader arm | leader arm | wrists |
| Rotate roll / pitch / yaw | `Z`/`X` · `T`/`G` · `C`/`V` | leader arm | leader arm | wrists |
| Fine motion (hold) | `Shift` | — | — | — |
| Gripper | `K` toggle | leader gripper | leader gripper | thumb–index pinch |
| Reset | `R` (arm only) | `R` (arm + scene; discards a take in progress) | `6` (arm + scene; discards a take in progress) | `T` (objects + arms to rest) |
| Re-home to current wrist pose | — | — | — | `R` |
| Start recording | `S` ¹ | `S` | `7` | automatic when the headset connects |
| Save episode | `N` | `N` | `8` (then auto-reset) | `S` (keeps recording) |
| Discard episode | `D` ¹ | `D` | `9` | `D` (keeps recording) |
| Stop session, finalize dataset | `Esc` | `Esc` | close the window (`Esc` is the viewer's free camera) | close Isaac Sim |

¹ `S` and `D` are also move keys in `keyboard_teleop`: pressing them to start / discard also nudges the arm.

MuJoCo uses digits because every letter, `0`-`5` and `Esc` are MuJoCo viewer shortcuts. Where the keys are defined:
recording defaults `S`/`N`/`D`/`Esc` in `src/robot_learning/humanoid_robot_learning/recording/episode_keys.py`;
MuJoCo overrides in `pioneer_leader_arm_teleop/mujoco_sim.py` (`_KEY_RESET`, `RECORD_KEYS`);
`keyboard_teleop` motion keys are Isaac Lab's `Se3Keyboard` defaults.

## Notes

- Upper body = 6-DOF arm + 15-DOF hand. Lower body (unused): 4D `[x, y, yaw, height]`.
- Retargeter reference: `IsaacLab/.../devices/openxr/retargeters/humanoid/unitree`
