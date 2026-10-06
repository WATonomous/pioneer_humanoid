# Joint command: `joint_command_node` + `joint_command_core`

We convert high-level arm joint targets (`ArmPose`) into per-motor CAN commands (`MotorCmd`), with YAML-driven calibration and runtime safety moderation (clamp, rate limit, smoothing). Intended for policy / teleop outputs before hardware.

## Pipeline

**Input:** `common_msgs/ArmPose` on `/arm/joint_targets` (6 angles: 3 shoulder, 2 elbow, 1 wrist).

**Output:** six `common_msgs/MotorCmd` messages on `/interfacing/motorCMD` (`POSITION_LOOP` by default).

**Node behavior:**
1. Each `ArmPose` is cached.
2. A timer at `control_rate_hz` runs moderation one step and publishes, so `velocity_max` is a
   true deg/s bound however fast `ArmPose` arrives.
3. Nothing is published until the rate-limiter has been seeded from real feedback.
4. If no `ArmPose` arrives within `command_timeout_sec`, commands stop and the next one re-seeds.

## Per-joint processing (`armPoseToMotorCmds`)

For each joint $i$, let $q^{\mathrm{in}}_i$ be the incoming angle (degrees, same units as `arm_calibration.yaml`).

Repeat in order, once per control tick:

1. **Position clamp** — if enabled, clip to hardware limits:
   $$
   q \leftarrow \mathrm{clip}(q,\ q_{\min},\ q_{\max}).
   $$
2. **Low-pass** — exponential smoothing with $\alpha =$ `low_pass_alpha`:
   $$
   q \leftarrow \alpha\, q^{\mathrm{prev}} + (1-\alpha)\, q.
   $$
3. **Velocity limit** — cap change per control tick using previous moderated target $q^{\mathrm{prev}}_i$:
   $$
   \Delta q_{\max} = \frac{\texttt{velocity\_max}}{\texttt{control\_rate\_hz}}.
   $$
4. **Delta limit** — additional per-step cap `delta_max` (degrees/tick).
5. **Position clamp again** — limits still hold after smoothing. If it bites (joint outside its
   limits), the move back into range is rate-limited too, not snapped.
6. **Calibration** — map to motor frame before publish:
   $$
   q_{\mathrm{motor}} = \texttt{direction} \cdot (q - \texttt{zero\_offset}).
   $$

Store $q$ as $q^{\mathrm{prev}}$ for the next tick.

## MIT joints

`control_type` is set per joint in `arm_actuators.yaml` (`-1` = node default). All six joints
currently run `MIT_CONTROL` (0): the GL40 wrist (`mit_family: gl2`) and the AKs (`ak`). See "MIT
mode" in [can/README.md](../can/README.md).

A MIT joint is a PD drive with no internal limit checking, so `joint_command` adds:

| Field | Role |
|---|---|
| `mit_kp` / `mit_kd` | stiffness / damping (physical units) |
| `mit_max_torque` | fault above this torque |
| `mit_max_track_err` | fault if the joint lags its setpoint by more (deg) |
| `mit_feedback_timeout` | fault after this long without feedback |
| `mit_family` | `gl2` or `ak`; they disagree on the status byte (AK 1 = over-temperature) |
| `mit_fault_action` | `limp` (kp = kd = 0, `MIT_EXIT`) or `damp` (kp = 0, kd = `mit_fault_kd`); default `damp` for `ak` |
| `mit_fault_kd` | damping for `damp`, in (0, 5] |

**Startup rule** (the node refuses to launch otherwise): quantised `mit_kp` × `mit_max_track_err`
(rad) + `gravity_ff_max_torque` ≤ `mit_max_torque`.

Lifecycle: `MIT_ENTER` at startup, zero-stiffness frames until seeded, then gains, then
zero-stiffness frames again when the stream goes stale. A fault latches: `limp` joints get
`MIT_EXIT`, `damp` joints keep getting damping frames (a silent AK trips its CAN timeout and drops
the arm). On Ctrl-C, `damp` joints are damped for `mit_shutdown_damp_sec` (default 2 s), then every
MIT joint is exited. Stop `joint_command` **before** `can_node`, with the arm supported.

## Gravity feed-forward

MIT joints can be sent the torque that holds the arm's weight, in `MotorCmd.torque`:

$$\tau_{\mathrm{ff}} = \mathrm{clip}\big(\texttt{gravity\_ff\_scale} \cdot r \cdot \tau_{\mathrm{model}},\ \pm\texttt{gravity\_ff\_max\_torque}\big)$$

$\tau_{\mathrm{model}}$ ([gravity_model.cpp](src/gravity_model.cpp)) is the left arm's static load
from the URDF masses at the commanded pose; $r$ ramps 0 → 1 over 1 s after each seed. It is zeroed
while any joint's angle is unknown, unless an unpowered joint sets `gravity_assume_deg` (valid
only while that joint is strapped at that angle).

The model reads command-frame angles as URDF angles: **the command frame is the URDF frame.**
`arm_calibration.yaml`'s zero is the URDF zero (arm hanging straight down, elbow straight) and
each joint's `direction` makes positive turn the URDF's positive way: shoulder pitch swings the
arm forward, shoulder roll out to the side, elbow pitch backward. (The old `urdf_direction` /
`urdf_offset_deg` keys are refused: fold any correction into `direction` / `zero_offset`.)

Bring-up, one joint at a time, arm supported, `gravity_ff_scale: 0`:
1. Calibrate with the arm hanging (`calibrate_arm.py`); the seed log should then read ~0 there.
2. Jog each joint positive and check its direction against the list above. A wrong one: flip its
   `direction` in `arm_calibration.yaml` and re-run `calibrate_arm.py`.
3. At a few poses, the `Gravity model ... pred X meas Y` log (every 5 s) must agree in sign and
   roughly in size.
4. Set `gravity_ff_scale: 0.5`, confirm the sag shrinks, then go to 1.0.

A wrong sign doubles the sag; the startup rule and the tracking watchdog bound it.

## Gripper

The GL40 gripper (id 21, `arm_calibration.yaml` `gripper.open_close`) is a 7th joint, present only
when `arm_actuators.yaml` has a `gripper.open_close` block. Without one, the node is the 6-joint arm.
Leave it out of a run with its `active: false` (see Launch).

- **Command:** `ArmPose.gripper_closure`, 0 = open .. 1 = closed, used while
  `include_gripper` is true. Closure 0 → command 0° (its calibrated zero), 1 → `upper_limit`, then
  the same clamp / low-pass / rate limit as any joint. Without `include_gripper` (or with a
  non-finite closure) it **holds where it is**, so a publisher that omits it never drops an object.
- **Calibrate it open** with `calibrate_arm.py`, `direction` so that closing is positive, limits
  open (≈ 0) .. closed. The node refuses limits that are not `[<= 0, > 0]`.
- **Grasping never faults:** closing on an object leaves it short of its target, so
  `mit_max_track_err` must cover the full travel (enforced). With the startup rule, that also caps
  the squeeze: `kp × travel <= mit_max_torque`.
- It must be MIT (`POSITION_LOOP` would be full stiffness), with no gravity feed-forward. Its
  watchdog faults halt the whole arm like any MIT joint; an unpowered gripper is just excluded.

## Excluded joints

At seeding, a joint with no feedback in the last 0.5 s, or physically outside its limits (stale
calibration: re-run `calibrate_arm.py`), is **excluded** until the next seed. Servo joints get no
command and MIT joints get their fault action. It is never ramped from an assumed 0 or clamped to
a limit, and the rest of the arm keeps working.

## Config files

| File | Role |
|------|------|
| `config/joint_command.yaml` | ROS params: arm side, topics, control rate, control type |
| `config/arm_calibration.yaml` | Per-joint `can_id`, limits, `direction`, `zero_offset` |
| `config/arm_actuators.yaml` | `active` per joint, moderation toggles, per-joint `velocity_max`, `delta_max`, `low_pass_alpha`, `control_type`, MIT gains and limits |

`arm_actuators.yaml` uses a top-level `safety:` key with `global` defaults and optional `joints` overrides (shoulder/elbow/wrist paths match `arm_calibration.yaml`).

## Tuning `arm_actuators.yaml`

Units are **degrees** and **deg/s**. At 50 Hz, `velocity_max: 100` implies up to **2.0°/tick** from the velocity limiter.

Start conservative on hardware, then increase until motion is responsive without jitter or limit hitting. Current values are bench defaults, not policy-tuned.

| Parameter | Effect |
|-----------|--------|
| `velocity_max` | Max joint speed (converted to °/tick) |
| `delta_max` | Hard cap on ° change per tick |
| `low_pass_alpha` | Higher → smoother/slower (e.g. `0.85`) |
| `enable_*` | Toggle each stage without recompiling |
| `control_type` | Per joint; `0` = MIT_CONTROL, `4` = POSITION_LOOP, `-1` = node default |
| `mit_*`, `gravity_*` | MIT gains, fault thresholds and feed-forward (see above) |

## Tests

`colcon test --packages-select joint_command` runs gtests against the **shipped** config, so an
unsafe edit (clamp off, velocity past the 2 rad/s testing ceiling, broken gain rule) fails the build.

## Launch

```bash
ros2 launch joint_command joint_command.launch.py
```

**Which actuators a run uses:** each joint's `active: true` in `arm_actuators.yaml`. Set it to
`false` (e.g. every joint but one for a single-joint test) and restart the node. An inactive
joint gets no command at all (no `MIT_ENTER`, no watchdog), and to the gravity model its angle is
unknown, as if unpowered (`gravity_assume_deg` applies). The node logs which joints are active,
and refuses to start with none.

**Defaults:** `arm_side=left`, `control_rate_hz=50`, `control_type=POSITION_LOOP` (4), overridden per joint in `arm_actuators.yaml`.
