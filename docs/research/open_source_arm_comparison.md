# Pioneer vs open-source arm projects: data collection, training, eval, deployment

*Written 2026-11-12.*

This compares our manipulation stack with about 12 open-source arm and robot-learning projects. Our side comes from reading the repo; theirs comes from their docs, papers and source code (OpenArm's `openarm_teleop` source was read directly). Items marked **[unverified]** came from memory or third-party pages; check them before relying on them.

**Projects covered:**
- OpenArm (Enactic), the closest analogue to our arm
- ALOHA / ALOHA 2 / Mobile ALOHA
- GELLO
- LeRobot (SO-101, Koch, HopeJR, XLeRobot)
- AgileX PiPER
- Trossen (trossen_arm, WidowX AI)
- UMI
- DROID and RoboArena
- VR teleop systems: Open-TeleVision, Bunny-VisionPro, BEAVR
- Pollen Reachy 2
- ToddlerBot and Berkeley Humanoid Lite
- openpi (pi0 / pi0.5) and RTC
- Unitree G1 stack (`xr_teleoperate`, `unitree_IL_lerobot`)
- AhaRobot

> **"Pixel":** I couldn't find an open-source robot *arm* called Pixel. The only "Pixel" turned up is Naz Louis's walking desk robot, whose code is promised but not confirmed released. Possibly meant: AgileX **PiPER** (covered below), the AgileX Pika gripper, or the Seeed reBot B601-DM. A link would settle it.

---

## 1. Where we already match or beat them

Keep these and don't rewrite them:

| Area | Ours | Comparison |
|---|---|---|
| Quest teleop engineering | WebXR + versioned WebSocket packets, single tab in control, 1-euro + SLERP filter, mink QP IK with speed boxes, elbow-singularity rule, 200 ms stale-input pause with re-anchoring, 6 test files | On par with or ahead of OpenArm's VR (prototype, ~20 FPS), Unitree `xr_teleoperate` (30 Hz, weighted moving filter) and Open-TeleVision (60 Hz CLIK). Few of them have tests. |
| Synthetic demos | "Simulated human" on the leader mapping: minimum-jerk + Fitts' law, tremor, OU drift, corrective strokes; keeps only successes, then re-renders with cameras | None of the surveyed projects do this. ALOHA uses scripted sim policies, which are much less human-like. |
| Real-arm safety layer | `joint_command`: clamp → LPF → velocity/delta limits, seeded from feedback, MIT watchdog (torque, tracking error, feedback timeout, temperature), any fault halts all arms, gtests against the *shipped* config in CI | Matches OpenArm (limits in code), trossen_arm (fault → idle all joints) and Reachy (`max_relative_target`). Testing the shipped config in CI is better than any of them. |
| Leader-arm start gate | Phased start (ZERO → CHECK → HOME → SLOW → LIVE), dry run by default, 15° engage tolerance | Same idea as Unitree's "align with the initial pose before start" and OpenArm's low-gain ramp to the start pose. Ours is more thorough. |
| Camera sim-to-real | ChArUco intrinsics, real frames rectified to the same ideal pinhole the sim renders | DROID ships calibrations, but nobody surveyed rectifies real frames to match sim. |
| Eval statistics | 40 held-out seeds, success has to hold for 0.4 s, Wilson 95% CI | Most projects report bare success rates over 10–20 trials. |
| VLA know-how | Have run RTC; found temporal ensembling worse (SO-101 study); pi0.5 uses delta joint actions with an absolute gripper | Matches openpi's recommendations and the RTC paper's finding that temporal ensembling oscillates. |

---

## 2. The biggest differences

These are the main reasons our pipeline can't produce a real-robot policy yet.

1. **Sim and real are different robots.** The pipeline defaults to the 8dof arm, which exists only in MuJoCo. The real arm is the old 6-DoF arm with GL40 wrist and gripper. Their datasets can't be combined, and the right arm has only 3 shoulder joints active. Every project surveyed collects on the same embodiment it deploys on.
2. **No policy deployment and no real-robot eval.** Nothing in `robot_learning` publishes `/arm/*/joint_targets`, and the RTC driver has a `TODO(VLA)`. Every reference project runs policies on hardware: LeRobot `lerobot-rollout`, the openpi websocket server, Unitree's `--send_real_robot`.
3. **Real recordings fake the state.** `pioneer_humanoid_dataset_schema.yaml:51` has `state_topic: null  # state mirrors action`. On top of that:
   - only the left arm is recorded;
   - there is no ego camera;
   - frames are a 25 Hz snapshot of whatever message arrived last, with no timestamp alignment;
   - velocity, current and torque are not recorded.

   OpenArm records qpos, qvel, qtorque and EE pose at about 250 Hz with nanosecond timestamps. ALOHA records qpos, qvel and effort. trossen_arm adds microsecond timestamps, external effort and temperatures.
4. **No dataset QA gate.** OpenArm's `openarm-dataset-validate` and LeRobot's cadence report catch bad episodes before training. We have nothing equivalent.
5. **No sim system identification.** Sim gains are kp 600–2270, while real MIT gains are kp 0.2–200. Link masses are estimates, and the left gravity model is turned off ("sign unchecked"). ALOHA 2 fit its MuJoCo model to 11 real trajectories. ToddlerBot fits per motor model and gets 1.3° tracking error.
6. **Bimanual policies can't be scored.** `sim/evaluate.py` evaluates the left arm only, so the 16-dim bimanual Quest datasets have no eval.

Also worth knowing: the commit "Add trajectory evaluation tools" (#429) is **SLAM** benchmarking (RPE, Sim(3)), not manipulation eval.

---

## 3. Project by project: what's different and what to borrow

### OpenArm (Enactic)

This is the closest analogue: 7-DoF bimanual, QDD motors in MIT mode, CAN.

- **Hardware and control.** Damiao motors on CAN-FD with a **1 kHz** loop. Each arm gets MIT commands `{kp, kd, q, dq, τ_ff = g(q) + friction(dq)}`. Gravity comes from KDL on the URDF. Friction uses the per-joint model `Fo + Fv·ω + Fc·tanh(k·ω)` with YAML parameters.
- **Teleop.**
  - Bilateral position-position: the leader tracks the follower and the follower tracks the leader. This gives force feedback without a force/torque sensor. It needs ≥500 Hz.
  - Unilateral mode: the leader runs pure torque mode, compensating gravity plus 0.3× friction and 0.1× Coriolis.
  - KER is a motorless 1:1 leader, comparable to our Feetech leader.
- **Safety constants.** 8 rad/s and 20 Nm limits on every joint, a low-gain ramp to the start pose, and a 90 Hz velocity filter.
- **Data pipeline.** The pipeline is a **dora-rs dataflow** of small nodes (leader, robot, recorder, replayer, policy server, success classifier, eval UI). The leader and robot are swapped with environment variables (`LEADER=webxr ROBOT=mujoco`). **Every hardware node has a dummy twin**, so the whole pipeline runs without hardware.
- **Native dataset format.**
  - Raw data is kept at the control rate. `Dataset.sample(hz)` resamples it, with optional zero-phase Butterworth smoothing, and it exports to LeRobot v2.1/v3, GR00T and Rerun.
  - `validate` checks for NaNs, bad quaternions, joint jumps over 1 rad, limit violations and episodes under 2 s, and writes a `valid` flag. `repair` fills single-frame gaps.
  - `merge` refuses datasets whose equipment, version or rate don't match.
  - Each episode carries `success`, the operator and the leader device.
- **Eval.** The OpenArm Cell standardises the setup: lighting, cameras, light curtains, e-stop and a **calibration jig**. OpenArm Online scores Docker-submitted policies in sim.
- **LeRobot classes.** `openarm_follower`, `bi_openarm_follower` and `openarm_leader` exist, with `side=left/right` per-side limits and one CAN interface per arm.

**Borrow:**
- τ_ff = gravity + identified friction on the follower.
- A dummy twin for every node.
- A native raw format with export to LeRobot.
- The validator, repair and merge-guard trio.
- A calibration jig.
- The LeRobot robot-class pattern.

### ALOHA / ALOHA 2 / Mobile ALOHA

- **Setup.** 50 Hz joint-space leader-follower with no force feedback. Four cameras; ALOHA 2 uses D405s at 848×480. One HDF5 file per episode holding qpos, qvel, effort and action.
- **ALOHA 2 leader changes.**
  - Gravity compensation is **passive**, using retractable hangers. Operators preferred it to active software compensation (1.38 vs 0.97 insertions/min).
  - The low-friction gripper rail cuts opening force from 14.7 N to 0.84 N.
- **Tools.** `replay_episodes.py` replays a recorded episode open-loop on the real robot to check calibration. `visualize_episodes.py` and a `sleep.py` park pose round it out.
- **Operations.** Foot pedals save or discard episodes. Operator, time and robot ID are logged. **The rig auto-shuts down on missing data.** Operators take ≥2-minute breaks every 10 minutes, and the rig produces hundreds of demos per day.
- **ACT recipe.** Chunk 100, `kl_weight` 10, lr 1e-5, trained long past the loss plateau. LeRobot's guidance is the opposite (5–10 epochs), so **test both on our data**. Mobile ALOHA added co-training with related data.
- **Sim model.** The MuJoCo Menagerie ALOHA 2 was sysID'd from 11 real trajectories. Fitted: gains, damping, armature, friction and torque limits.

**Borrow:**
- Open-loop replay on hardware.
- Foot pedals.
- Auto-stop on missing sensor data.
- Passive gravity offload for the leader.
- Sysid from about 10 trajectories.
- Co-training (synthetic + real).

### GELLO

- Our leader arm is already GELLO-style: torque-off servos, a 1:1 kinematic replica.
- **Calibration.** `gello_get_offset.py` puts the leader in a known pose, takes the joint signs, and **solves** for per-servo offsets. Our 8dof leader signs are "derived from the models, not checked on the leader".
- **Force feedback.** Newer GELLO adds gravity comp and force feedback through FACTR, using the servos' own motors.

**Borrow:**
- An automated sign and offset solver for the 8dof leader.
- Optional low-torque gravity assist on the Feetech servos instead of torque-off.

### LeRobot (SO-101, Koch, HopeJR, XLeRobot)

- **Calibration.** One calibration file per `--robot.id`, reused for teleop, record, replay and rollout. A missing calibration is detected automatically.
- **Cadence report.** After each episode it prints the achieved Hz, the ticks over budget, and a per-step timing split (observe, teleop, send, record). LeRobot timestamps are derived from frame index ÷ fps, so a session that actually ran at 21 Hz is silently labelled 25 Hz. We have exactly this risk.
- **Recording.** `lerobot-record` supports re-record and next-episode keys, `--resume`, and `finalize()` before push.
- **Inspection.** `lerobot-replay`, the Dataset Visualizer and Rerun.
- **Dataset tools.** Edit, merge, delete and `recompute_stats`.
- **`lerobot-rollout` modes.** `base`, `sentry` (always recording, auto-upload to `rollout_*` datasets), `highlight`, `dagger` (human corrections) and `episodic`.
- **Inference.** Async inference splits a PolicyServer from a RobotClient over gRPC (`actions_per_chunk` about 50, `chunk_size_threshold` 0.5–0.6). RTC is enabled with `--inference.type=rtc` (execution horizon 8–12, guidance 10), plus a trained-RTC mode for pi0.5.
- **Data tips.** At least 50 episodes, about 10 per object position, fixed cameras, add variation gradually. The task should be doable from the camera images alone.

**Borrow:**
- The cadence report.
- Rollout datasets as the eval record.
- DAgger mode.
- Async inference and RTC for real deployment.
- A single LeRobot version pin. Ours are scattered: `>=0.3`, `<0.5`, git `e670ac5`, 0.4.3, `>=0.7`.

### AgileX PiPER

- **Hardware and control.** CAN at 1 Mbps with roughly 200 Hz feedback, and both position mode and MIT mode. The SDK explicitly warns that misusing MIT mode can damage the arm.
- **Multi-arm CAN.** CAN adapters are bound to USB ports so interface names stay predictable.
- **Gravity fitting.** The community `piper_control` package **fits a MuJoCo gravity model from logged (q, effort) samples**. That is a direct recipe for fixing our left-arm gravity table.

### Trossen (trossen_arm, WidowX AI)

- **Control modes.** Per-joint modes: `idle`, `position`, `velocity`, `external_effort` and `effort`. In `external_effort` the controller compensates gravity and friction itself, so zero effort means the arm floats. That mode is what their leader arms and gravity-comp demos use.
- **Interpolation by goal time.** Quintic above 0.2 s, linear between 1 ms and 0.2 s, immediate below that.
- **Faults and feedback.** Any error idles all joints and logs which joint and value caused it. Feedback carries a µs timestamp plus position, velocity, acceleration, effort, external effort and temperatures.
- **SIMPLER.** Validates sim as an eval proxy for real by measuring rank agreement: Pearson r and MMRV over about 1,500 real and 1,500 sim episodes.

**Borrow:**
- An "external effort" style mode, i.e. a gravity-floating arm for homing and hand-guiding.
- Quintic moves to the first target.
- A SIMPLER-style check that sim success predicts real success before trusting MuJoCo numbers.

### UMI

- **Collection.** Demos are collected with a handheld gripper and no robot.
- **Feasibility check.** Every demo is checked automatically against the target robot's kinematics.
- **Latency matching.** Camera, proprioception and action latencies are measured per sensor (by filming a clock) and compensated at inference. Without it, motion was jerky.
- **Policy design.** Relative, camera-centric actions. A CLIP ViT beat ResNet-34. Inter-gripper proprioception helped bimanual tasks.
- **Throughput.** 111 demos/hour, against 35/hour with a SpaceMouse.

**Borrow:**
- Measure our end-to-end latency per sensor, then compensate.
- A kinematic and self-collision feasibility check on every demo. The shoulder-yaw collision with the stand post at −78…−100° is a known case our IK doesn't avoid.

### DROID and RoboArena

- **One standard rig.** The same rig (Franka, two ZED 2 cameras plus a wrist ZED Mini, Quest 2 teleop) was used at 13 institutions by about 50 collectors.
- **Shipped metadata.** Calibrations, success flags and language annotations ship with the data.
- **RoboArena eval.** Blinded pairwise A/B evaluation; rankings converge within about 100 pairs.

**Borrow:**
- Standardise the rig and camera extrinsics before more people collect.
- Use blind A/B comparison when comparing policies on the real arm.

### VR teleop: Open-TeleVision, Bunny-VisionPro, BEAVR

- **Open-TeleVision.**
  - Stereo video from an actuated head camera, streamed to the headset, **clearly beat mono** in their user study.
  - Wrist position is taken relative to the head; orientation is absolute.
  - CLIK IK with a nullspace offset.
  - Policy: ACT with a DINOv2 backbone, chunk 60, about 20 demos per task.
- **Bunny-VisionPro.** Real-time singularity and collision avoidance, and about $1.20 vibration motors for haptics.
- **BEAVR.** Claims ≤35 ms latency with zero-copy streaming and records straight into LeRobot format.

**What this means for us:**
- We just *removed* headset video (#554). That's reasonable for latency, but it means operators rely on direct sight. Revisit it once the real arm is in use, ideally with stereo.
- Quest controllers have **haptic rumble**. Driving it from follower tracking error, joint-limit proximity or IK-infeasible state is a cheap form of force feedback.
- Measure and publish a motion-to-motion latency number.

### Pollen Reachy 2

- **Latency.** They publish measured figures: 74 ms hand-to-gripper and 135 ms glass-to-glass video.
- **Soft e-stop.** A 3-button VR combo makes the arms compliant. They state it does not replace the hardware e-stop.
- **Warnings vs errors.** The UI separates warnings from errors.
- **Recording.**
  - VR drives the robot while LeRobot only records (`use_external_commands=false`).
  - They recommend a separate recording PC with wired Ethernet.
  - Recording runs at 15 fps.
- **Idle joints.** **"Recording all joints when only some are moving can reduce model quality."** They use per-part `with_<part>` flags. This applies to us because the right arm and the wrist are often inactive.
- **Warning sign.** Their VR recording docs broke when a server was dropped. Pin teleop and recorder versions together.

### ToddlerBot (Stanford) and Berkeley Humanoid Lite

- **ToddlerBot calibration.** 3D-printed fixtures click joints into zero in about a minute. Relevant to us because our AK80-9s lose their zero on every power cycle.
- **ToddlerBot sysID, once per motor model.**
  - Backdrive at constant RPM: the intercept gives frictionloss, the slope gives damping.
  - Spin-down gives armature (I = 2E/ω²).
  - Optuna then jointly fits 9 parameters (PD gains, velocity-dependent torque limit, braking torque, and others) against chirp tracking, reaching 1.3° error.
  - Validated on a second robot.
- **ToddlerBot policy.** Diffusion with a 16-step chunk. It drops the first 3 actions to cover latency and executes the next 5. Tested with 20 trials per task.
- **Berkeley Humanoid Lite.**
  - Separate rates: actuator 250 Hz, policy 25 Hz.
  - Measured transmission stiffness and backlash.
  - End-effector repeatability of 3.4 mm std over 100 repetitions, which is a good acceptance test for our arm.

### openpi and RTC

- **Actions and normalisation.**
  - Delta joint actions with an absolute gripper (we already do this).
  - Quantile q01/q99 normalisation for pi0.5. **Recompute it after every merge**: merged LeRobot datasets hold a conservative envelope, not true quantiles.
  - Rarely used dimensions get tiny ranges and can blow up training, so check `norm_stats.json`.
- **Data and defaults.**
  - 1–20 h of data for fine-tuning.
  - Defaults: 30k steps at batch 32.
  - To benefit from pretraining, joint sign conventions and gripper units should roughly match the base model (`adapt_to_pi`).
- **LeRobot gotcha.** `--policy.pretrained_path` silently resets `n_action_steps` and similar fields; `--policy.path` doesn't.
- **RTC.** On real robots it holds up with up to +200 ms of injected delay. Temporal ensembling caused oscillations that tripped a protective stop.
- **Our pi0.5 dataset is inconsistent.** `notmaxxx/pioneer_vla_pick_place` has 8 dims (6 joints plus 2 finger positions in metres), while our schema is 7 dims with a closure fraction. Fix this before the next fine-tune.

### Unitree G1 (`xr_teleoperate`, `unitree_IL_lerobot`)

- **Teleop and safety.**
  - Separate keys for start, record and quit.
  - Arms are referenced to head yaw.
  - Pressing both joysticks gives a soft e-stop in damping mode.
  - **On exit, the arms return to the initial pose within 5 s.**
  - `--sim` runs the identical code in Isaac.
- **Data tools.** A PyQt editor trims episodes and deletes failed ones.
- **Eval.** Three modes from one script: **real, dataset-replay with the robot off, and sim**. `replay_robot.py` sanity-checks episodes on hardware.

### AhaRobot

- **Foot pedals** for clutch and episode start/stop, so both hands stay on the task.
- Our Quest mapping is "always on, no clutch". A clutch (grip button or pedal) lets the operator re-centre without moving the arm.

---

## 4. Improvement list

Ordered roughly by impact for getting a real-arm policy working. Effort: S is under a day, M is a few days, L is a week or more.

### A. Safety and calibration (real arm)

1. **(M) Repeatable zero.**
   - Add a 3D-printed click-in calibration jig (ToddlerBot, OpenArm Cell) so re-zeroing after each power cycle takes about 1 minute instead of a full sweep.
   - Refuse to enable torque if |q − q_home_expected| exceeds a threshold. This directly targets the stripped-screws incident.
2. **(S) Calibration id as the single source of truth.** Store a calibration file per physical arm id (LeRobot style). Write its hash, plus the kp/kd values and gravity scale in use, into every dataset's metadata.
3. **(S) Soft e-stop to damping, not limp, from the teleop UI.** Use a Quest button combo or both sticks (Reachy, Unitree). Keep the 48 V hardware e-stop as the real one.
4. **(S) Return-to-home within a fixed time on exit, using a quintic profile** (Unitree, trossen goal-time rule).
5. **(M) Fix gravity compensation.** Fit the left-arm gravity model from logged (q, current) samples (`piper_control` approach) and enable it. Then lower kp, since compliant, overdamped gains are better for BC (Bronars et al. 2026).
6. **(S) Auto-stop recording and teleop on missing sensor data** (ALOHA 2): camera drop, stale feedback, or CAN errors.

### B. Data collection

7. **(M) Record real measured state.** Set `state_topic` to motor feedback for both arms and drop "state mirrors action".
8. **(M) Record raw at the control rate with monotonic timestamps, and resample when exporting to LeRobot.**
   - Signals: q, dq, current/torque, τ_ff, temperature, kp/kd, raw leader q or VR poses, and each camera frame's capture timestamp.
   - Follows OpenArm and trossen. It also stops tying the raw data to LeRobot version churn.
9. **(S) Per-episode cadence report** (LeRobot): achieved Hz, worst gap, per-stage timing. Flag or reject drifting episodes.
10. **(M) Dataset validator as a training gate** (OpenArm): NaNs, joint jumps, limit hits, minimum duration, timestamp gaps, frozen camera frames. Add `valid` and `success` flags per episode, and a merge guard that refuses mismatched schema, robot or fps.
11. **(S) Open-loop replay on hardware** (`humanoid-replay`, as in ALOHA and LeRobot) through `joint_command` in slow mode. This is the cheapest calibration and zero check.
12. **(S) Record more episode metadata:** operator, teleop device (Quest or leader), calibration hash, git SHA, scene seed.
13. **(S) Drop idle joints from state and action**, e.g. the inactive right arm or wrist (Pollen finding).
14. **(S) Operator ergonomics.**
    - Foot pedal (USB HID, a few dollars) for save, discard and clutch.
    - A clutch button on the Quest.
    - Breaks every 10 minutes.
15. **(M) Real ego camera,** with its intrinsics measured rather than "ESTIMATE", and sim camera poses matched to real extrinsics.
16. **(M) Visual domain randomisation in MuJoCo** (lighting, textures, camera jitter). Today only object placement is randomised.
17. **(S) Kinematic and collision feasibility check on synthetic and Quest demos** (UMI), including the shoulder-yaw / stand-post collision.

### C. Teleop

18. **(M) Quest teleop on the real arm,** publishing through `joint_command`. Reuse the leader arm's ZERO → CHECK → SLOW → LIVE gate.
19. **(S) Haptic rumble on the Quest controllers** for tracking error, joint-limit proximity and IK infeasibility. This is cheap force feedback.
20. **(S) Measure and publish end-to-end teleop latency** (Reachy publishes 74 ms; BEAVR targets ≤35 ms).
21. **(S) Automated leader sign and offset solver for the 8dof leader** (GELLO `get_offset`). The signs are currently unverified.
22. **(L, later) Powered leader or bilateral control** (OpenArm, GELLO/FACTR) once the follower runs a faster loop. Bilateral needs ≥500 Hz; we're at 50 Hz command and 100 Hz CAN polling on one shared bus. **One CAN bus per arm** is a prerequisite.

### D. Sim-to-real

23. **(L) Per-motor-model sysID rig**, one each for AK10-9, AK80-9, AK45-10 and AK40-10.
    - Load-arm pendulum.
    - Constant-velocity backdrive to get friction and damping.
    - Spin-down to get armature.
    - Chirp excitation, then fit in MuJoCo with Optuna or CMA-ES (ToddlerBot, ALOHA 2).
    - Push the results into the MJCF. **Use the same gains in sim and real**, picking kp/kd from kp = I·ωn², kd = 2Iζωn (BeyondMimic).
24. **(S) Weigh the real parts** and replace the estimated link masses.
25. **(S) Resolve the `WRIST_FOCAL_MM` inconsistency.** The calibrated pinhole YAML and the 110° FOV constant in `arm_params.py` disagree; check which one the sim renders.
26. **(M) Check sim-versus-real ranking agreement** (SIMPLER) before trusting MuJoCo success rates to pick checkpoints.

### E. Training

27. **(S) Fix the pi0.5 dataset and schema mismatch** (8 vs 7 dims, metres vs closure fraction).
28. **(S) Recompute quantile stats after every merge** and check `norm_stats.json` for degenerate dimensions.
29. **(S) Use `--policy.path`, not `pretrained_path`,** when the chunk settings matter.
30. **(M) Co-train synthetic and real demos** (Mobile ALOHA). Our synthetic generator is a real asset here once the embodiments match.
31. **(S) ACT epoch sweep.** ALOHA says train long, LeRobot says 5–10 epochs; settle it on our data.
32. **(S) Turn on LeRobot image augmentation** for real data, if it isn't already.

### F. Eval and deployment

33. **(M) Bimanual eval in `sim/evaluate.py`** (16-dim), so Quest datasets can be scored.
34. **(L) Real-arm policy runner.**
    - Run the policy server as a separate process (openpi websocket or LeRobot async gRPC).
    - The robot-side client publishes only through `joint_command`.
    - Use RTC for flow-matching policies.
    - Hook up the `TODO(VLA)` path.
35. **(M) One eval script with three modes** (Unitree): sim, offline dataset replay (already done for pi0.5), and real.
36. **(M) Record every real rollout as a `rollout_*` dataset** (LeRobot sentry/episodic). Add DAgger corrections from the leader arm.
37. **(S) Real eval protocol.**
    - Fixed trial count (≥20 per condition).
    - Fixed lighting and camera poses.
    - Saved videos.
    - The Wilson CI we already compute.
    - Blind A/B when comparing two policies (RoboArena).
    - Re-evaluate after re-zeroing to catch calibration sensitivity.
38. **(S) Accept-test the arm itself:** end-effector repeatability over 100 repeats (Berkeley Humanoid Lite reports 3.4 mm std).

### G. Repo structure and dev practices

39. **(M) Dummy twin for every hardware node** (OpenArm): fake CAN or motor feedback, fake cameras, fake leader, fake policy. CI can then run record → validate → train-a-few-steps → eval end to end.
40. **(S) Run the Python tests in CI.** Today `build_and_unitest.yml` runs `colcon test` and nothing calls pytest. `robot_learning` (recorder, schema, evaluate, DiT) has no tests at all.
41. **(S) One LeRobot version pin** across `robot_learning`, the leader install and SO-101.
42. **(M) Consider LeRobot robot classes** (`pioneer_follower`, `bi_pioneer_follower`, `pioneer_leader`) that wrap `joint_command` over ROS. This gets record, replay, rollout, async inference and RTC for free while keeping our safety layer. It is the OpenArm and Reachy pattern.
43. **(S) Fix doc drift:**
    - The 8dof asset README says the leader and eval are "not yet ported", but they are.
    - The push-eval script says 10 Hz, but the schema is 25 fps.
    - The bowling scene is missing from the scenes table.
44. **(S) Predictable CAN interface names per arm** (udev or USB-port binding, as PiPER and OpenArm do) once the buses are split.

---

## 5. Suggested order

1. **Unblock real data:** items 7, 8, 9, 11, 1 and 2. Record real state with timestamps, check cadence, replay on hardware, and add the calibration jig and metadata.
2. **Close the loop on hardware:** items 34, 18 and 37. Build the policy runner through `joint_command`, add real Quest teleop, and set a real eval protocol.
3. **Make sim trustworthy:** items 23, 24, 5 and 26. SysID, real masses, gravity compensation, and the sim/real ranking check.
4. **Scale:** items 10, 39, 40, 42, 30 and 36. Validator, dummy-twin CI, LeRobot classes, co-training, and rollout datasets with DAgger.

---

## Sources

- **OpenArm:** https://docs.openarm.dev/ · https://github.com/enactic/openarm_teleop · https://github.com/enactic/dora-openarm-data-collection · https://docs.openarm.dev/dataset/api · https://huggingface.co/docs/lerobot/openarm
- **ALOHA 2:** https://arxiv.org/abs/2405.02292 · ACT: https://github.com/tonyzhaozh/act
- **GELLO:** https://github.com/wuphilipp/gello_software
- **LeRobot:** https://huggingface.co/docs/lerobot/lerobot-dataset-v3 · https://huggingface.co/docs/lerobot/main/en/il_robots · https://huggingface.co/docs/lerobot/main/en/async · https://huggingface.co/docs/lerobot/en/rtc · https://huggingface.co/docs/lerobot/pi05
- **PiPER:** https://github.com/agilexrobotics/piper_sdk · https://pypi.org/project/piper-control/
- **Trossen:** https://docs.trossenrobotics.com/trossen_arm/main/programming_guide/concepts.html · SIMPLER: https://simpler-env.github.io
- **UMI:** https://umi-gripper.github.io/ · DROID: https://droid-dataset.github.io/ · RoboArena: https://arxiv.org/abs/2506.18123
- **VR teleop:** Open-TeleVision: https://arxiv.org/abs/2407.01512 · Bunny-VisionPro: https://dingry.github.io/projects/bunny_visionpro · BEAVR: https://arxiv.org/abs/2508.09606
- **Reachy 2:** https://docs.pollen-robotics.com/ · https://huggingface.co/docs/lerobot/reachy2
- **ToddlerBot:** https://arxiv.org/abs/2502.00893 · https://hshi74.github.io/toddlerbot/tools/02_sysID.html · Berkeley Humanoid Lite: https://arxiv.org/abs/2504.17249
- **openpi:** https://github.com/Physical-Intelligence/openpi · RTC: https://arxiv.org/abs/2506.07339
- **Unitree:** https://github.com/unitreerobotics/xr_teleoperate · https://github.com/unitreerobotics/unitree_IL_lerobot
- **AhaRobot:** https://arxiv.org/abs/2503.10070
- **Gains and friction:** Bronars et al.: https://arxiv.org/abs/2604.02523 · BeyondMimic: https://arxiv.org/abs/2508.08241 · BAM friction: https://arxiv.org/abs/2410.08650
