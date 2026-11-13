# Robot learning: record datasets, train policies

Teleoperate the Pioneer arms, record the demos as a **LeRobot** dataset, train a policy (ACT, SmolVLA, pi0.5, …) on it and run evals.

For human intervention collection, correction aggregation and policy refinement,
see [HG-DAgger](humanoid_robot_learning/hg_dagger/HG_DAGGER.md). Connect your scene and controller through adapters;
the optional MuJoCo/Quest adapter reuses the model and teleop already on main.

**The default robot is `pioneer_bimanual_arm_8dof`** (7-DoF + gripper per arm), recorded in MuJoCo. The old 6-DoF arms
(armv2, `pioneer_v1_*` datasets) remain for Isaac, which has no 8dof USD yet, and for today's real arm; the real arm will
soon be 8dof too. The Quest VR teleop (`src/teleop/quest_controller_teleop`) records both 8dof arms with `--record`
(right B starts / saves a take): see its README.

## 1. Collect demos

Add `--record` to a teleop. The default (8dof arms, MuJoCo, runs on a laptop):

```bash
src/teleop/quest_controller_teleop/start_quest.sh --scene bowling --record --task_description "stand the bottles up"
python3 src/teleop/pioneer_leader_arm_teleop/pioneer_leader_arm_teleop.py --scene peg_insert --record   # leader arm; macOS: mjpython
```

| Teleop | Start a take | Save it (scene resets) | Discard |
|--------|--------------|------------------------|---------|
| Quest | right B (or 7) | right B (or 8) | left Y (or 9) |
| Leader, MuJoCo | 7 | 8 | 9 |
| Leader, Isaac (old arms) | S | N | D (R resets) |

Old 6-DoF arms in Isaac (simulation_isaac container):

```bash
/workspace/isaaclab/isaaclab.sh -p pioneer_leader_arm_teleop.py --target isaac --scene vial_rack --record \
  --task_description "put the vial in the rack" --num_episodes 20
```

More: the [Quest](../teleop/quest_controller_teleop/README.md) and
[leader](../teleop/pioneer_leader_arm_teleop/README.md) READMEs. Output: `<repo>/datasets/<dataset>/sim/` (later
sessions append).

## 2. Data contract

25 fps. Per frame:

| Field | Content |
|-------|---------|
| `observation.state` | measured joint positions + gripper closure (0 open … 1 closed) |
| `action` | commanded joint targets + gripper command |
| `observation.images.<name>` | 640×480 RGB: `ego` + the wrists of the arms recorded |
| `task` | `--task_description` |

**8dof arms (default)**, `config/pioneer_8dof_dataset_schema.yaml`: **8 values** per arm, left first:
`shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow`, `forearm_roll`, `wrist_pitch`, `wrist_yaw` (rad),
`gripper`. Datasets `pioneer_8dof_left_arm` / `_right_arm` / `_bimanual` (the Quest records `_bimanual`).

**Old 6-DoF arms**, `config/pioneer_humanoid_dataset_schema.yaml` (Isaac and today's real arm): **7 values** per arm,
`shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow_pitch`, `elbow_roll`, `wrist_pitch` (rad), `gripper`.
Datasets `pioneer_v1_*`, never mixed with 8dof.

`evaluate.py` picks the robot from the policy's action size (8 = 8dof, 7 = old arms) or `--robot`. Camera poses:
`pioneer_humanoid/arm_8dof_params.py` (8dof), `arm_params.py` (old arms).

## 3. Train

In the `simulation_isaac` container, you can train a end2end policy with methods like ACT like listed below or use any other method to push for a high eval success rate:

```bash
train-policy \
  --dataset.repo_id=humanoid/pioneer_v1_left_arm \
  --dataset.root=/workspace/humanoid/datasets/pioneer_v1_left_arm/sim \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=/workspace/humanoid/outputs/train/pioneer_act
```

Flag pitfalls (`--steps`, not epochs; always `--policy.push_to_hub=false` for local runs) are listed in the [SO101 README](../simulation/so101_vial_task/README.md), which uses the same command.

Our own policy, `humanoid_dit` ([`humanoid_robot_learning/policies/dit`](humanoid_robot_learning/policies/dit)), is a flow-matching diffusion transformer conditioned on CLIP image and text features. Install with `pip install -e .[dit]` and train with `python -m humanoid_robot_learning.train --policy.type=humanoid_dit` plus the arguments above and `--steps=70000 --batch_size=6`. On 540 synthetic peg-insertion demos it succeeds in 80% of 400 sim episodes. At 50k steps: 58-75% across two training seeds (180 demos: 60-64%). The learning rate decays over `--steps`, so set it up front: a finished run can't be extended.

Success rate of any checkpoint on a MuJoCo scene that defines `success` (or multi-step `progress`), left arm, as recorded by the leader-arm teleop. It prints a 95% interval: 40 episodes tell 35% from 65% (about ±15 points); comparing 60% with 70% needs about 400. Use layout seeds the training demos did not (`--seed`; the synthetic demos start at 0).

```bash
python -m humanoid_robot_learning.sim.evaluate --scene peg_insert --policy <run>/checkpoints/last/pretrained_model \
  --episodes 40 --seed 1000 --video_dir eval_videos
```

Sim rollout for a flow-matching policy (pi0 / pi0.5 / SmolVLA) on the push-block scene, driven with Real-Time Chunking (`humanoid_robot_learning/deploy/rtc_driver.py`): [`scripts/pioneer_push_eval_rtc.py`](scripts/pioneer_push_eval_rtc.py). It expects a checkpoint trained on `config/pioneer_humanoid_dataset_schema.yaml` (the old arms' format, recorded by the Isaac leader teleop); `--dry_run` runs it without one. **The VLA is not hooked up yet**: no real checkpoint has been run through it, and its schema, camera keys and action space (absolute vs delta) still need matching. See the `TODO(VLA)` block in the script.

```bash
/workspace/isaaclab/isaaclab.sh -p /workspace/humanoid/src/robot_learning/scripts/pioneer_push_eval_rtc.py \
  --policy_path <checkpoint> --num_episodes 10
```

## 4. Real arm (planned)

Not runnable yet. The plan is the same leader arm and the same schema, saved under `datasets/pioneer_v1_left_arm/real/`:

- [#327](https://github.com/WATonomous/pioneer_humanoid/issues/327) — `pioneer_leader_arm_teleop.py --target real`: dry-run by default, `--live` to command the arm; adds the gripper command path.
- [#328](https://github.com/WATonomous/pioneer_humanoid/issues/328) — real demo recording (cameras, state from motor feedback). Still open there: a separate `humanoid-record` process, or `--target real --record`.

`humanoid-record` (ROS 2, `humanoid_robot_learning/recording/record.py`) records from `/arm/joint_targets`, the gripper motor feedback and the camera topics in the schema (`--cameras wrist_left` for now: `ego` has no real topic yet). Its write path can be tested with no robot:

```bash
pip install -e "src/robot_learning[record]"
humanoid-record --dry_run --sink lerobot,hdf5 --num_episodes 2 --episode_time_s 3
```

| Key | Effect |
|-----|--------|
| S | Start a take |
| N | Stop and save the take, then wait for S |
| D | Delete the last saved take (again to go further back; ignored while recording) |
| ESC | Quit (a take still recording is dropped) |

Each take is its own LeRobot dataset, `real/<session NNN>/take_NNN/`, so D deletes one folder. Combine takes (from one or more sessions) into one dataset before training; `--push_to_hub` also uploads it:

```bash
humanoid-merge-takes datasets/pioneer_v1_left_arm/real/001 datasets/pioneer_v1_left_arm/real/002 \
  --output datasets/pioneer_v1_left_arm/real_merged [--push_to_hub <hf-user>/pioneer_v1_left_arm]
```

`--sink` (`lerobot`, `hdf5`, or both) exists only on `humanoid-record`; the sim leader and keyboard teleop always write LeRobot.
