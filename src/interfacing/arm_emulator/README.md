# arm_emulator

Replaces everything below `joint_command` in MuJoCo: `can_node`, the CubeMars drives and the arm. The real
teleop, `joint_command`, recorder and policy runner run on top unmodified, so demos and rollouts go through
the arm's real limits, gains and watchdogs.

```text
teleop --target real --live / deploy/real_arm.py / synthetic_demos.py    real code
        | ArmPose /arm/left/joint_targets
joint_command                                                            real code, emulator's config
        | MotorCmd              ^ MotorFeedback
bridge_node -> drive.py (MIT law, limits, faults, real mit_protocol.cpp)  emulated
        | torque                ^ angles, images
physics.py: the MuJoCo scene at 1 kHz, torque-driven left arm             emulated
```

A result here is not clearance for the real arm: see [What this does not prove](#what-this-does-not-prove).

## Staying apart from the real arm

The commands are the ones that drive the real arm, so the separation matters:

- Keep the arm's 48 V off and the CANable unplugged during emulator sessions.
- The container has `network_mode: none`, no `/dev`, no CAN tools, its own ROS domain, localhost only.
- `safety_guard.py` refuses to start on domain 0, with a CAN interface visible, or with `can_node` or a second
  feedback publisher on the graph.
- `sim_config.py` generates the sim's config at container start from the real YAMLs. It differs only in what it
  declares (wrist and gripper `active`, wrist `gravity_ff_scale`), and it never reaches the real arm's containers.
- Pass the leader as `/dev/serial/by-id/...` only. `/dev/ttyACM*` may be the CANable.

## Run it

```bash
export MONO_DIR=$PWD USER_UID=$(id -u) USER_GID=$(id -g)
C="docker compose -f modules/docker-compose.arm_emulator.yaml"

# Tests, no display, no leader (ARM_EMULATOR_TARGET=core is enough without LeRobot)
$C up -d --build
$C exec arm_emulator bash -c "cd src/interfacing/arm_emulator && python3 -m unittest discover -s tests -t ."
$C exec arm_emulator bash -c "cd src/interfacing/arm_emulator && python3 -m unittest scenarios.test_joint_command"
$C exec arm_emulator bash -c "cd src/interfacing/arm_emulator && python3 -m unittest scenarios.test_recording"
```

### Human demos with the leader

The gripper force is unmeasured, and at the plain linear mapping the cube slips (below), so state the value you use.

```bash
xhost +local:
export LEADER_PORT=/dev/serial/by-id/<the leader board>
C="docker compose -f modules/docker-compose.arm_emulator.yaml -f modules/docker-compose.arm_emulator.teleop.yaml"
$C up -d --build

# terminal 1: the emulator and its window (6 = new cube position)
$C exec arm_emulator python3 -m arm_emulator.run_emulator --viewer --gripper-force-scale 2
# terminal 2: joint_command
$C exec arm_emulator ros2 launch joint_command joint_command.launch.py
# terminal 3: the leader, the same teleop as the real arm. Answer N to the elbow ramp: the arm already starts at home.
$C exec arm_emulator bash -c "cd src/teleop/pioneer_leader_arm_teleop && python3 pioneer_leader_arm_teleop.py --target real --live"
# terminal 4: the recorder. S start, N save, D discard, with this terminal focused.
$C exec arm_emulator bash -c "cd src/robot_learning && python3 -m humanoid_robot_learning.recording.record \
  --schema config/pioneer_emulator_dataset_schema.yaml --task_description 'pick up the cube' --num_episodes 20 --no-auto_start"
```

Between takes, bring the leader home and press 6 in the MuJoCo window. Output: `datasets/pioneer_v1_left_arm/sim_low_level/`.
Add `--cameras wrist_left` to the recorder if the real arm won't have an ego camera when you deploy.

### Scripted demos (no leader, runs on WATcloud)

```bash
$C exec arm_emulator bash -c "cd src/interfacing/arm_emulator && python3 scenarios/synthetic_demos.py --episodes 20 --gripper-force-scale 2"
```

On WATcloud, in a job on a compute node (`slurm-start-dockerd.sh` first), add `-f modules/docker-compose.arm_emulator.gpu.yaml`. Each job has its own ROS graph.

### Train, then evaluate in closed loop

```bash
train-policy --dataset.repo_id=humanoid/pioneer_v1_left_arm --dataset.root=<dataset> \
  --policy.type=act --policy.push_to_hub=false --output_dir=<run>   # plus the device flag: src/robot_learning/README.md

$C exec arm_emulator bash -c "cd src/interfacing/arm_emulator && python3 scenarios/eval_policy.py \
  --policy <run>/checkpoints/last/pretrained_model --episodes 40 --seed 1000 --gripper-force-scale 2"
```

## Files

| Path | Work |
|---|---|
| `arm_emulator/drive.py` | One drive: MIT law, motor limits, AK vs GL II, feedback rounding, heat |
| `arm_emulator/mit_native.py`, `native/` | Binding to the real `mit_protocol.cpp` |
| `arm_emulator/physics.py` | The scene, motor frames, 1 kHz stepping, gripper mapping, load limits |
| `arm_emulator/run_emulator.py` | Entry point; the physics process and its real-time loop |
| `arm_emulator/bridge_node.py` | ROS topics where `can_node` stands, plus control and status topics, in its own process |
| `arm_emulator/camera.py` | Renders cameras in their own process |
| `arm_emulator/faults.py` | Faults to inject: `mute`, `code`, `dead`, `stall`, `zero_shift`, `hot` |
| `arm_emulator/safety_guard.py`, `sim_config.py` | Separation from the real arm; the sim's generated config |
| `config/motor_models.yaml` | Motor numbers; every placeholder is marked `ASSUMED` |
| `scenarios/test_joint_command.py` | Real `joint_command`: normal motion and each fault |
| `scenarios/test_recording.py`, `synthetic_demos.py`, `eval_policy.py` | Recording, scripted demos, closed-loop policy rollouts |
| `scenarios/offline_command.py`, `offline_pick.py` | No-ROS stand-in, for fast physics and grasp checks only |

## What this does not prove

- **`can_node` doesn't run.** Its frame building, DBC handling and non-finite refusal aren't exercised. The next step is a `vcan` stage.
- **Drive firmware is modelled from the manuals**, including fault behaviour, CAN timeout and GL II silence before ENTER.
- **Friction, inertia, heat and grip force are placeholders** until fitted to `arm_roundtrip.py` logs. Grip is the one that decides the task.
- **Images:** the wrist camera is a distortion-free pinhole; the real lens isn't, and the real path still uses the old Sonix calibration.
- **Timing:** `late_ticks` on the status topic counts ticks behind real time. Results from a run with many of them don't count.
- **Nothing breaks in sim.** The load limits are the only signal that a motion would have damaged something, and their limits are assumptions too.

Before any of this reaches the real arm: hardware E-stop on the 48 V supply within reach, and bench checks for AK gains, feedback scaling, joint directions after recalibration, gripper travel and force.
