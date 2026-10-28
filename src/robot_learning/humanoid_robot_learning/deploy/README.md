# Deploy: run a trained policy

| Module | Runs on | What it does |
|--------|---------|--------------|
| `real_arm.py` | real left arm | `PolicyDrivenRealArm`: policy actions become `ArmPose` on `/arm/joint_targets`, and motor feedback becomes `observation.state` |
| `pioneer_interface.py`, `rtc_driver.py` | Isaac Sim | Real-Time Chunking rollout of a flow-matching policy (`scripts/pioneer_push_eval_rtc.py`) |

Sim success rate on MuJoCo scenes: `humanoid_robot_learning.sim.evaluate` (see the [robot_learning README](../../README.md)).

## Real arm: `PolicyDrivenRealArm`

It uses the same data contract the teleops record (`config/pioneer_humanoid_dataset_schema.yaml`), and
works with any LeRobot checkpoint trained on it: ACT, `humanoid_dit`, SmolVLA, ...
It does a **dry run by default**: it reads feedback and cameras, runs the policy and prints targets
next to the measured pose. It never creates a publisher. `--live` commands the arm.

### Before any `--live` run

Follow `.claude/skills/real-hardware-safety` and [ARM_BRINGUP.md](../../../interfacing/ARM_BRINGUP.md):

1. **Hardware E-stop on the 48 V supply, within reach.** Arm clamped down, screws tight, nothing in its reach.
2. **Calibrate after every power-on** (`calibrate_arm.py`; the AK80-9s lose their zero). Then confirm in
   `live_arm_mjviser.py` that every joint turns the same way as in the viewer. The policy was trained in the
   URDF frame, and calibration is what makes the real command frame equal to it.
3. Run the **leader-arm teleop dry run** (`pioneer_leader_arm_teleop.py --target real`) to check the frames agree.
4. **Start with one joint**: set `active: false` on the rest in `arm_actuators.yaml` and restart joint_command.
   Keep the policy's moves small and slow. The adapter needs feedback from every joint (see *Limits* below).

### Run

1. **Bring up the arm**: follow [ARM_BRINGUP.md](../../../interfacing/ARM_BRINGUP.md) §0–3 (can_node, calibrate, joint_command).
2. **Open the policy container:**
   ```bash
   ./watod -t simulation_mj
   cd /workspace/humanoid/src/robot_learning
   ```
3. **Dry run** (nothing moves). Check the printed target and measured angles agree:
   ```bash
   python3 -m humanoid_robot_learning.deploy.real_arm --policy <run>/checkpoints/last/pretrained_model
   ```
4. **Live:** add `--live`, then type `live` at the prompt:
   ```bash
   python3 -m humanoid_robot_learning.deploy.real_arm --policy <run>/checkpoints/last/pretrained_model --live
   ```
5. **Stop:** press Ctrl-C and the arm holds where it is. Support the arm, then press Ctrl-C again.
   10 s later joint_command sets kp = 0 and the arm goes limp.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--policy` | required | `pretrained_model` directory of a LeRobot checkpoint |
| `--live` | off | publish `/arm/joint_targets` after the preflight passes and you type `live` |
| `--task` | the training dataset's only task | task text for language-conditioned policies |
| `--device` | `cpu` | torch device |
| `--fps` | schema `fps` (25) | policy rate. Use the rate the policy was trained at |
| `--max_s` | 60 | stop and hold after this long |
| `--start-tol` | 15° | the first target must be this close to the measured pose on every joint |
| `--max-gap` | 30° | stop and hold if a target gets this far from the measured pose |
| `--arm-calibration` | auto, the same file joint_command loads | `arm_calibration.yaml` |
| `--schema` | `config/pioneer_humanoid_dataset_schema.yaml` | fps and camera topics |

`pip install -e src/robot_learning` also installs the same CLI as `humanoid-deploy-real`.

### What a `--live` run checks

1. **Preflight.** It waits up to 5 s for data, then refuses to start if any of these hold:
   - a joint or the gripper has no feedback;
   - a camera the policy reads has no image (or no `topic` in the schema);
   - a joint is outside its calibrated limits (stale calibration: joint_command would exclude it);
   - something else already publishes `/arm/joint_targets` (teleop);
   - joint_command is not subscribed.
2. **`live` prompt.** Have the arm supported and a hand on the E-stop before you type it.
3. **First action.** If it is more than `--start-tol` from the arm, the run is refused and nothing is published.
4. **Every tick** (25 Hz): state from feedback, then the policy action, then the checks, then `ArmPose`.
   One status line per second shows target and measured angle per joint, the gripper command and the
   step time. `LATE` means inference was slower than `--fps`. joint_command keeps the last target
   meanwhile, but the policy then runs slower than it was trained to.
5. **Stop.** Ctrl-C, `--max_s`, lost feedback or camera, or a target past `--max-gap` all stop the run.
   The adapter then **holds**: it keeps publishing the measured pose, with the gripper left where it is,
   until you press Ctrl-C. After that, joint_command keeps the last pose for `command_timeout_sec` (10 s),
   then sets kp = 0 and the arm sinks. **Support it.**

### The path from `publish()` to the motors

| Step | Where | What happens to the action |
|------|-------|----------------------------|
| Policy | `make_policy_fn` | Same `load_policy` + `predict_action` as the sim eval. Gives `(7,)`: 6 joints in rad (sim `joint1L..joint6l`) + gripper 0 open .. 1 closed |
| Checks | `action_failures` | 7 finite values; first target within `--start-tol` of the arm, later ones within `--max-gap` |
| To ArmPose | `arm_pose_io.vector_to_arm_pose` | rad → deg. Slot *i* = dataset index *i* = joint_command's `jointPaths()[i]`: shoulder pitch, **yaw**, **roll**, elbow pitch, roll, wrist pitch. `include_gripper` and `gripper_position` are set; a 6-value hold pose leaves the gripper where it is |
| Topic | `/arm/joint_targets` | `header.stamp` is set. joint_command ignores `name` / `is_left`; the arm is its `arm_side` parameter (`left`) |
| Safety | `joint_command_node` | Drops non-finite poses. Waits for fresh feedback and seeds the rate limiter from it, so there is no start jump. Per 50 Hz tick: position clamp → low-pass → velocity limit → delta limit → clamp → calibration (`motor = direction × (cmd − zero_offset)`). MIT torque, tracking, feedback and temperature watchdogs; 10 s staleness timeout |
| Motor command | `/interfacing/motorCMD` | `MotorCmd` per joint with the `control_type` from `arm_actuators.yaml`: MIT position in rad + `mit_kp` / `mit_kd` + gravity feed-forward; servo position in deg |
| CAN | `can_node` | AK MIT: `packAkMitCommand`, extended id `0x800\|id`. GL40 MIT: raw codes from `mit_profiles.yaml` through the DBC. Servo modes: DBC signals. Sent on the CAN socket |
| Feedback | `/interfacing/motorFeedback` → `recording/arm_feedback.py` | Motor-frame deg → command frame (`zero_offset + motor / direction`) → rad; gripper → 0..1. Stale after 0.5 s, the same as joint_command's seed |

### Limits

- **Absolute joint targets only.** This is what the schema's `action` holds. A delta-action policy needs
  its deltas added to the state first.
- **Left arm only**, the default dataset. `pioneer_v1_bimanual` / `_right_arm` policies are refused
  (the action must have 7 values).
- **Every joint needs feedback.** An `active: false` joint gets no commands, and an AK in MIT mode only
  answers when spoken to, so its state is missing and the preflight stops. Single-joint bring-up
  therefore uses the leader teleop or `arm_roundtrip.py`, not a policy.
- **The real arm is slower than sim.** `velocity_max` (`arm_actuators.yaml`) caps each joint at 10–30°/s.
  A policy that moves faster in sim lags on the real arm and sees states it never saw in training.
  Expect a gap until sim rollouts use the same limits.
- **Cameras:** the schema has a real topic only for `wrist_left` (`/camera/color/image_raw`). A policy
  that reads `ego` cannot run on the real arm yet.
