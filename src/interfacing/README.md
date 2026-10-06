# Real arm bring-up: calibrate → visualize → move

Three separate steps, in three different packages. This page is just the entry point —
each step's real detail lives in its own doc.

## 1. Calibrate
Per-joint zero + limits, from live motor feedback. **Re-run after every power-on** for
elbow.pitch, elbow.roll, and shoulder.yaw (AK80-9 motors) — their single-turn absolute
encoders don't reliably survive a power cycle; shoulder.pitch/roll (AK10-9) have so far.

→ [can/README.md](can/README.md) — hardware
bring-up, `can_node`, `calibrate_arm.py` usage.

## 2. Visualize
Read-only mirror of live motor feedback in the browser. Confirms calibration looks right
before commanding anything.

```bash
./watod -t simulation_mj
python3 /workspace/humanoid/src/interfacing/can/scripts/live_arm_mjviser.py --arm-side left
# open http://localhost:8080
```

> **The command frame is the URDF frame** (`pioneer_bimanual_arm.urdf`): calibrate hanging, with
> each joint's `direction` matching the URDF. Before trusting the view: move each of the six
> joints by hand, one at a time, and confirm the on-screen joint turns the same way and stops
> at the same angle. Try fixes with `--flip JOINT` / `--offset JOINT=DEG` (viewer-only), then
> make them real: a flip is the joint's `direction` in
> [joint_command/config/arm_calibration.yaml](joint_command/config/arm_calibration.yaml)
> (then re-run `calibrate_arm.py`); an offset means it was not zeroed hanging (re-run it).

→ script docstring in `live_arm_mjviser.py` for the angle math and full flag reference.

## 3. Move (optional, real motor control)
Only after 1–2 look right. Rate-limited, seeds from live feedback (no startup slam).

→ [joint_command/MOVE_ARM_RUNBOOK.md](joint_command/MOVE_ARM_RUNBOOK.md)
