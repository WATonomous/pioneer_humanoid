---
name: real-hardware-safety
description: >
  Safety process for the WATonomous humanoid arm. Use it ANY time code that runs on, or could
  run on, the real arm is written, reviewed or changed: motor control, CAN, joint_command and its
  configs, teleop, calibration, sim-to-real, or any script that commands the CubeMars motors.
  Also when the user says "real arm", "on the robot", "test on hardware", "deploy", "calibrate",
  "zero", "motor test", or just asks for "a quick script to move the arm". AI-written arm code has
  damaged this arm before (stripped mounting screws from a sim/real zero mismatch at full speed).
---

# Real hardware safety

The arm has been damaged by code that moved it before anyone checked where it was. Follow this
process for anything that can command a motor. The **numbers** (torque caps, speeds, gains,
limits) live in config, not here:

- `src/interfacing/joint_command/config/arm_actuators.yaml`: per-joint caps, speeds, gains,
  watchdog, `active`. Each value carries its reason in a comment; change them only with bench data.
- `src/interfacing/joint_command/config/arm_calibration.yaml`: zero, direction and limits,
  written by `calibrate_arm.py`.

## Rules

1. **Everything goes through `joint_command`.** Publish `ArmPose` on `/arm/joint_targets`; never
   send `MotorCmd` straight to `/interfacing/motorCMD` to move the arm. `joint_command` clamps to
   the limits, rate-limits, seeds from live feedback (no startup jump) and runs the MIT watchdog.
   New code must not bypass or loosen those layers.
2. **Never assume where the arm is.** Read feedback first; the first command must start from the
   measured pose. Calibration goes stale (the AK80-9s lose their zero on power-off): recalibrate
   hanging, then confirm in `live_arm_mjviser.py` that every joint turns the same way as on screen.
3. **Dry run before live.** Read-only first (`--target real` dry run, the viewer, a dry-run
   `motor_cmd_topic`); command motors only once the numbers agree.
4. **One joint at a time first.** Use `active: false` on the rest, arm supported, small slow moves
   relative to the current pose; then the whole arm, slowly.
5. **Hardware E-stop on the 48 V supply, in reach, for every powered test.** A software stop is
   not a substitute: if the PC hangs, only the hardware E-stop stops the arm.
6. **Don't raise a cap, speed or gain without a bench test** that justifies it, and say why in the
   config comment. Never turn off the position clamp or a watchdog check to "make it work".
7. **Check the arm itself:** mounting screws tight, brackets not cracked, cables with slack, arm
   clamped down, nothing in its reach.

## When reviewing or writing such code

- Say plainly what was and wasn't run on hardware.
- Flag anything that can move a motor without passing rules 1-3, and fix it rather than only
  noting it.
- End with the E-stop reminder (rule 5) and what still has to be checked on the bench.
