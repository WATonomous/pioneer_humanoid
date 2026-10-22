# Pioneer humanoid: physics model and motor test

The final Pioneer humanoid design (`assets/pioneer_humanoid/`) as a MuJoCo physics model, with a tool to move
each simulated motor and see what it does. Every motor moves to its target **without overshooting**, within its
own **joint limits, torque cap and speed limit**, and the legs carry their real masses.

```bash
cd src/simulation/pioneer_humanoid_model
uv sync                                                  # once: mujoco, viser, numpy, pyyaml
uv run python motor_test.py                              # pelvis held in mid-air: every limb moves freely
uv run python motor_test.py --base stand                 # free on the floor
uv run python motor_test.py --sweep --report motor_report.md   # test every motor, no viewer
```

Open **http://localhost:8080** (on a remote machine: `ssh -N -L 8080:localhost:8080 <you>@<host>`).

## The motor test app

- **One slider per motor, grouped by body part:** torso & head (3), each arm (7, including the wrist), each
  hand (16), each leg (6). That makes 61 motors. The 8 finger DIP joints follow their PIP joint (the URDF's
  `<mimic>`) and have no motor of their own.
- **Moving a slider:** the motor moves to the new angle at no more than its speed limit and stops on it.
- **Under each slider:**
  - the actual angle;
  - the speed against the speed limit;
  - the torque against the torque cap;
  - the mass the motor moves;
  - whether its specs are *real*, *assumed* or *placeholder*.

  Flags: "⚠️ can't reach" means it has stopped more than 3° from its target; "🔴 at torque cap" means it's at
  its torque limit.
- **Sweep this motor / every motor in turn:** jumps the target to each limit and back.

## Per-joint specs (`joint_specs.yaml`, table in `JOINT_LIMITS.md`)

Every motor's **limits, torque cap, speed limit and rotor inertia**, and the **link masses**, live in one file,
with each value labelled with its source. Edit it as real specs arrive; nothing else needs changing.
**[`JOINT_LIMITS.md`](JOINT_LIMITS.md)** is the same as a readable table, with the mass each joint moves; after
editing the YAML, regenerate it with `uv run python motor_test.py --limits JOINT_LIMITS.md`.

| Joints | Source |
|---|---|
| Legs (12): torque, speed, link masses | **real**: the team's leg model (`assets/whole_body_humanoid`, the same "wato legs"). For example, hip pitch and knee 222 Nm at 3.67 rad/s; the legs weigh 34 kg |
| Arms (14): torque, speed | **assumed**: the previous Wato arm's motors (shoulders AK10-9 53 Nm, upper arm / elbow / forearm AK80-9 18 Nm, wrists GL40 0.73 Nm, 6 rad/s). Confirm for armV2 |
| Fingers, waist, neck (35) | **placeholder**: the URDF's 10 Nm and 6 rad/s; no real spec exists yet (the waist has no motor in the CAD) |
| Joint limits | the URDF, with the knee and elbow fixes (see `assets/pioneer_humanoid/README.md`) |

The robot weighs 38.2 kg: 34 kg of real legs, plus the URDF's placeholder upper body.

## How the motors are controlled (`Controller` in `motor_test.py`)

Each joint has a torque motor. Every 1 ms step, a controller sets its torque:

```
torque = clip( kp·(ref − q) + kd·(ref_vel − q̇) + [M·ref_acc + bias]_joint ,  ±torque cap )
```

- **The reference (`ref`)** moves towards your slider value on a **trapezoidal profile**:
  - its speed is at most the joint's speed limit;
  - its acceleration is limited both by half the motor's torque and by how fast the motor's loop can track;
  - it brakes at half that rate, so it lands on the target instead of arriving with momentum.
- **Feedforward:** `M·ref_acc + bias` is the torque that the planned accelerations of *all* joints need, plus
  gravity. It comes from MuJoCo's inverse dynamics, so a swinging shoulder doesn't shove the elbow, and gravity
  doesn't sag the arm.
- **Damping (`kd`)** is critical, recomputed 100 times a second from the inertia the joint actually moves in its
  current pose. Stiffness and damping stay within what the time step can integrate stably. A real motor driver's
  loop is about 1000× faster than the simulation and doesn't need that limit.
- **The damping is part of the capped torque,** as on a real driver. MuJoCo's built-in position motor applies its
  damping outside the torque cap, which let the hip's damping cancel most of its 222 Nm.

What it fixed: dragging the left shoulder pitch from 0° to −90° used to overshoot by **24°** and wobble for
**0.8 s**. It now overshoots by **0.28°** and settles in **0.3 s**.

## Results (`--sweep` writes the full per-motor table)

**61 of 61 motors pass** in `hang` mode. Each motor's target jumps to both limits and back, plus a step to half
of each side of its range, and every motor:
- reaches both limits;
- never goes more than 0.5° past a target (worst 0.35°, median 0.00°);
- stays within 5% of its speed limit (worst +2.7%);
- settles in under 0.4 s;
- spends under 5% of the time at its torque cap.

## Caveats

- **The upper body's masses and the finger, waist and neck motors are still placeholders.** The arm motors are
  assumed. Re-run the sweep when real specs go into `joint_specs.yaml`.
- **Self-collision isn't simulated** (the URDF's collision shapes are bounding boxes). Use the viewer to see
  whether a pose makes parts pass through each other.
- **A URDF import pitfall, handled in `model.py`:** MuJoCo turns each joint's URDF `effort` into a joint-level
  force clamp. With the URDF's 10 Nm placeholders, that silently capped every motor at 10 Nm whatever its
  actuator said.
