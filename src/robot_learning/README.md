# Robot learning: record datasets, train policies

Teleoperate the Pioneer arms, record the demos as a **LeRobot** dataset, train a policy (ACT, SmolVLA, pi0.5, …) on it and run evals.

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

### RECAP rollouts and corrections

RECAP-style RL ([pi*0.6](https://arxiv.org/abs/2511.14759)) needs the policy's own rollouts with their outcomes, and
operator corrections that resume where a rollout failed. `--record DIR` saves every evaluated episode as a LeRobot
dataset in the Quest teleop's single-arm format (left arm, 8 values, `ego` + `wrist_left` 640x480, 25 fps, per-frame
step instruction as `task`, `subtask_index`), so rollouts, corrections and demos merge into one training set. The printed
success rate is unchanged. Beside the dataset, `recap/` (`humanoid_robot_learning/recap/sidecar.py`) holds:

| File | One row per | Columns |
|------|-------------|---------|
| `recap/episodes.parquet` | episode | `source` (rollout / correction / demo), `round`, `scene`, `policy`, `advantage`, layout `seed`, `success`, `sim_seconds`, `num_frames`; corrections: `parent_episode`, `parent_root`, `resume_s` |
| `recap/frames/episode_NNNNNN.parquet` | frame | `time`, `progress` (the scene's step index), `success`, `state` (`mj_getState(mjSTATE_INTEGRATION)`: includes `userdata`, where scenes latch their step) |

Every frame is snapshotted: about 2.5 KB a frame for drawer_stow, small next to the videos. `--advantage positive`
appends " Advantage: positive" to the prompt of an advantage-conditioned policy; the recorded `task` stays the plain
instruction. `--round` tags the episodes with the RECAP iteration.

Policies trained with LeRobot's `--rename_map` (pi0.5 expects `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`)
are fed our `ego` / `wrist_left` renders under their names: the map comes from the checkpoint's `train_config.json`, or
`--rename_map '{"observation.images.ego": "observation.images.base_0_rgb", ...}'`. Policy cameras with no sim camera
(the right wrist) are left out for the policy to pad. The recorded rollouts keep our camera names.

```bash
python -m humanoid_robot_learning.sim.evaluate --scene drawer_stow --policy <run>/checkpoints/last/pretrained_model \
  --episodes 40 --seed 1000 --record datasets/recap/drawer_stow --round 0
```

Corrections of the failed rollouts, with the Quest:
[`resume_teleop.py`](../teleop/quest_controller_teleop/README.md#recap-corrections).

### RECAP value function and advantage labels

Task-agnostic (`--scene` names the scene), under `humanoid_robot_learning/recap/`. Input: one LeRobot dataset with its
`recap/` sidecar holding all data so far (demos, every round's rollouts and corrections).

| Step | Command | Writes |
|------|---------|--------|
| Value | `python -m humanoid_robot_learning.recap.value` | `value.pt`: 201-bin distribution over the return normalised to [-1, 0], Monte Carlo targets, privileged sim state |
| Labels | `python -m humanoid_robot_learning.recap.label` | `recap/labels_r<k>.parquet` (value, advantage, positive); `--training_copy`: the dataset with " Advantage: positive/negative" in `task` |

Returns: -1 per step, normalised by the time limit in steps (`--max_s` x fps), failures clipped to -1. Advantage
over `--horizon 50` steps (2 s), ε at the 30th percentile (70% positive), corrections forced positive, the token
dropped on 30% of frames. Demos have no sim snapshot: the value sees their arm state, step index and time only.

```bash
python -m humanoid_robot_learning.recap.value --dataset datasets/recap/r1_all --scene drawer_stow --out outputs/recap/value_r1
python -m humanoid_robot_learning.recap.label --dataset datasets/recap/r1_all --value outputs/recap/value_r1/value.pt \
  --round 1 --training_copy datasets/recap/r1_train
```

Then fine-tune from the base checkpoint on `r1_train` and roll out with `evaluate --advantage positive`.

### RECAP parallel rollouts and merging

Roll a checkpoint out in parallel, then merge rollouts, corrections and demos into the one dataset the value function
and labels read. `python -m humanoid_robot_learning.recap.shards` runs K `sim.evaluate --record` processes on one
machine (one seed range each, keeping 5 GB of memory free; `--merge` merges them). On WATcloud:
[`tools/slurm/rollouts.sbatch`](../../tools/slurm/README.md) (job array) or `eval.sbatch` (a run's checkpoints).
`humanoid-merge-takes` renumbers the RECAP sidecars to the merged episodes (`recap/merge.py`).

```bash
python -m humanoid_robot_learning.recap.shards --out datasets/recap/r1_rollouts --shards 4 --episodes 40 --seed 1000 \
  --round 1 --merge -- --scene drawer_stow --policy <run>/checkpoints/last/pretrained_model
humanoid-merge-takes datasets/drawer_stow_train_v1/sim datasets/recap/r1_rollouts/merged datasets/recap/r1_corrections \
  --output datasets/recap/r1_all
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
