# AKH70-48 bench sysid (preliminary)

Bench measurements on one CubeMars AKH70-48 (hip pitch and knee actuator), taken in the
CubeMars tool before the CAN adapter arrived. They confirm the friction value already in the
sim config and add a viscous term. Everything here is preliminary: currents were read off the
tool's waveform graph, on a 30 V supply, on a 3D-printed rig.

Measured 2026-11-01 to 2026-11-09 by Aydan Ling.

## Summary

| Parameter | Value | Confidence | Used in config? |
|---|---|---|---|
| Coulomb friction (output) | 1.8–2.5 N·m | ±20% | Yes: `friction=2.2` kept |
| Viscous friction (output) | 0.9–1.2 N·m·s/rad | ±30% | Commented out (needs an Isaac Lab version with `viscous_friction`) |
| Torque per commanded amp (output) | 12.1 N·m/A | ±10%, see open question 1 | No |
| Friction holding at 0 A after loading | 1.3 N·m | one run | No |
| Phase resistance / inductance | 223.16 mΩ / 67.82 µH | driver calibration | No |
| Pole pairs / gear ratio | 21 / 48 | driver calibration / config | No |

Not measured yet: armature, command delay, torque–speed curve at 48 V, thermal rise.

## Setup

- Supply: 30 V, 10 A limit (the robot runs at 48 V).
- Motor on a printed plate and stand, clamped to a board. Printed arm on the output, holes at
  100/125/150/175 mm.
- CubeMars tool, servo page. Speed test: S row (speed loop). Torque test: I row (current loop).
- Speed conversion: 1 output RPM = 1,008 ERPM (21 pole pairs × 48).

## Test 1: friction vs speed (free output)

Constant speed with nothing on the output; the current needed is friction. Warmed up 5 min at
10 RPM first. Motor 27.2 °C at session start, 30.7 °C at test start. Iq read by eye off the
waveform graph (about ±0.02–0.03 A), so forward and reverse came out identical.

| Output RPM | Command (ERPM) | Iq forward (A) | Iq reverse (A) |
|---|---|---|---|
| 1 | 1,008 | 0.200 | −0.200 |
| 2 | 2,016 | 0.225 | −0.225 |
| 5 | 5,040 | 0.270 | −0.270 |
| 10 | 10,080 | 0.340 | −0.340 |
| 15 | 15,120 | 0.375 | −0.375 |
| 20 | 20,160 | 0.400 | −0.400 |

Straight-line fit, Iq = c + b·ω (ω in rad/s at the output): **c = 0.208 A, b = 0.101 A·s/rad**,
largest residual 0.026 A. The slope flattens with speed (0.145 A·s/rad below 10 RPM, 0.08 above
5 RPM). The data stops at 2.1 rad/s; the motor's limit is 3.67 rad/s.

In torque units this depends on which torque constant is right (open question 1):

| | × 8.81 N·m/A (CubeMars) | × 12.1 N·m/A (test 2) |
|---|---|---|
| Coulomb | 1.83 N·m | 2.52 N·m |
| Viscous | 0.89 N·m·s/rad | 1.22 N·m·s/rad |

## Test 2: torque per amp (lever on a scale)

Arm level, a standoff in the 175 mm hole pressing down on a kitchen scale (5 kg range). Scale
tared once with the motor enabled at 0 A and the arm resting on it. Current stepped up and back
down in the I row without lifting the arm. Torque = scale (kg) × 9.81 × 0.175 m.

Run of 2026-11-09 (the clean run):

| Commanded (A) | Scale, going up (kg) | Scale, coming down (kg) |
|---|---|---|
| 0.30 | 1.10 | 0.755 |
| 0.45 | 2.00 | 2.14 |
| 0.60 | 3.21 | (top of run) |
| 0.15 | not recorded | 0.754 |
| 0 | 0 (tare) | 0.755 |

- Up pass, line fit: **7.03 kg/A → 12.1 N·m per commanded amp**. The line reaches zero load at
  0.15 A, so about **1.8 N·m** is lost to gearbox friction before torque reaches the output.
- Down pass: the gearbox locked at 0.755 kg from 0.30 A to 0 A, so it holds about **1.3 N·m**
  with no current. The down pass therefore shows friction, not the torque constant, and the two
  passes were not averaged.
- An earlier run (2026-11-05, old stand, arm swung into the scale instead of starting at rest)
  gave 2.00, 3.00, 4.01 kg at 0.30, 0.45, 0.60 A: slope 6.7 kg/A, about 11.5 N·m/A.

## Open questions

1. **Which "amp"?** 12.1 N·m/A is 37% above CubeMars' 8.81 N·m/A and above the 11.2 N·m/A
   expected from KV41 × 48 before gearbox losses. The tool's status panel showed 0.12 A phase
   current for a 0.15 A command (one reading; ratio 0.8). The likely explanation is a different
   current convention (peak vs RMS, or a √(3/2) transform factor), not a stronger motor. Until
   this is settled, do not rescale torques in sim. The check: command torque in MIT mode over
   CAN against the scale or a known weight (`torque_accuracy` and `gravity_hold` in the sysid
   package).
2. **The tool's T row is not a plain torque command.** At T = 2 and T = 8 the scale read 0 kg
   and the supply current did not change. Avoid it.
3. **Friction asymmetry and temperature.** Not resolvable from graph-read currents. End
   temperature of test 1 was not recorded.
4. **Rig slack.** The stand showed 2–3 mm of play under load. It does not affect these static
   tests but must be removed before tests that reverse direction.

## What this PR changes

- `whole_body.py`: hip pitch and knee keep `friction=2.2` (now measured). Gains lowered to what
  the drivers can run: CubeMars MIT mode caps Kd at 5, so Kd 17 → 5 and 12 → 5, with Kp lowered
  to keep damping reasonable (267 → 150 for hip pitch and knee, 146 → 100 for hip roll and yaw).
- `rough_env_cfg.py`: `max_delay` 4 → 1. The delay term counts policy steps (20 ms), so 4 meant
  up to 80 ms.

The gain and delay changes are not measurements; they affect every locomotion training run and
need a retrain.

## Next

CAN-based sysid on the same motor (latency, chirp, step, free swing, gravity hold, torque–speed
at 48 V), then the AK80-64 and AK60-39.
