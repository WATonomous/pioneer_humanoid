# Robot learning: record datasets, train policies

Teleoperate the Pioneer arms, record the demos as a **LeRobot** dataset, train a policy (ACT, SmolVLA, pi0.5, …) on it and run evals.

**The robot is `pioneer_bimanual_arm_8dof`** (7-DoF + gripper per arm), recorded in MuJoCo or Isaac. The old 6-DoF arms
(armv2, `pioneer_v1_*` datasets) remain only for today's real arm (and MuJoCo's `--robot armv2`); the real arm will soon be
8dof too. The Quest VR teleop (`src/teleop/quest_controller_teleop`) records both 8dof arms with `--record`
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
| Leader, Isaac | S | N | D (R resets) |

Isaac (simulation_isaac container):

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

**Old 6-DoF arms**, `config/pioneer_humanoid_dataset_schema.yaml` (today's real arm): **7 values** per arm,
`shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow_pitch`, `elbow_roll`, `wrist_pitch` (rad), `gripper`.
Datasets `pioneer_v1_*`, never mixed with 8dof.

`evaluate.py` (MuJoCo) picks the robot from the policy's action size (8 = 8dof, 7 = old arms) or `--robot`. Camera poses:
`pioneer_humanoid/arm_8dof_params.py` (8dof), `arm_params.py` (old arms).

## 3. Train

In the `simulation_isaac` container, you can train a end2end policy with methods like ACT like listed below or use any other method to push for a high eval success rate:

```bash
train-policy \
  --dataset.repo_id=humanoid/pioneer_8dof_left_arm \
  --dataset.root=/workspace/humanoid/datasets/pioneer_8dof_left_arm/sim \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=/workspace/humanoid/outputs/train/pioneer_act
```

Flag pitfalls (`--steps`, not epochs; always `--policy.push_to_hub=false` for local runs) are listed in the [SO101 README](../simulation/so101_vial_task/README.md), which uses the same command.

**Pioneer-1** (`pioneer1`, [`humanoid_robot_learning/policies/pioneer1`](humanoid_robot_learning/policies/pioneer1)) is our policy model (the model, not the robot): language-conditioned, for picking the object an instruction names. A DINOv3 ViT-B/16 encodes every camera. The scene camera gives all of its patch tokens, the wrist cameras a 7x7 pooled grid. The instruction's frozen CLIP text features join them through FiLM and as word tokens, and two fusion layers let each patch attend to the words. The action head is a flow-matching transformer (adaLN-Zero) that cross-attends to this context, trained with real-time chunking. At run time it plans 32 actions, executes 16, and starts the next chunk from the 6 actions it had planned next (`n_action_steps`, `replan_prefix`). 256M parameters, 193M trained. DINOv3 is gated on Hugging Face: request access on [its page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) first. Install with `pip install -e .[pioneer1]` and train with:

```bash
python -m humanoid_robot_learning.train --policy.type=pioneer1 --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=<repo> --dataset.root=<dataset> --dataset.video_backend=pyav \
  --batch_size=32 --steps=60000 --save_freq=10000 --output_dir=<run>
```

On 600 synthetic tabletop-pick demos (two objects on the table, "put the <object> in the bowl"; the scene comes in a follow-up PR) it succeeds in **70% of sim episodes on unseen layouts** (140/200 at 40k steps with the defaults; 60k is no better; 0.37 s/step on an L40S, batch 32); checkpoint [`wilsonchenghy/pioneer1-tabletop-pick`](https://huggingface.co/wilsonchenghy/pioneer1-tabletop-pick), private for now). Two inference defaults get it there:

- **Replan every 16 actions, keeping the next 6 planned actions** (`n_action_steps=16`, `replan_prefix=6`): 165/300 vs 136/300 for whole 32-action chunks. It reacts to what it sees twice as often, and the kept actions make each new chunk continue the motion already under way (the model is trained for this with real-time chunking). Replanning without them breaks the motion: 12/100 every 8 actions.
- **Trained weights, no EMA** (`ema_decay=0`): 140/200 vs 111/200 (better on 34 layouts, worse on 5). Most likely the averaged weights blur the choice between options (which object, when to close), so the arm hesitates.

**vs π0.5:** a full π0.5 fine-tune (3.6B, pretrained; [`wilsonchenghy/pi05-full-finetune-tabletop-pick`](https://huggingface.co/wilsonchenghy/pi05-full-finetune-tabletop-pick), private) on the same demos scores 154/200 on the same layouts (better on 37, worse on 23, p≈0.09, so not clearly better). Pioneer-1 is 14× smaller: it trains on a 16 GB GPU and fits onboard (Jetson-class); π0.5 needs an 80 GB GPU to fine-tune.

Most failures never grasp (the arm stalls near its start) or take the other object; few drop it on the way. The CLIP-encoder version picks the named object in 99.6% of paired layouts. The likely limit is the data: the scripted-expert demos never retry or correct, and longer training doesn't help. If memory is short, `--policy.grad_checkpoint=true` recomputes the encoder and fusion layers in the backward pass (same result, about half the memory).

**pi0.5** trains through LeRobot's own pi0.5 (LeRobot 0.7, `pip install 'lerobot[pi,training]==0.7.0'`, Python ≥ 3.12)
with the same entry point. Fine-tune from `lerobot/pi05_base`, training only the action expert (the VLM stays frozen).
Works for any dataset in the schema's format (one arm or both); sim evaluation runs left-arm policies.

Relative actions (optional, recommended; LeRobot's default is absolute) need relative action stats (first command); for absolute, skip it and drop `--policy.use_relative_actions=true`.
`--steps`, `--batch_size` and `--save_freq` below are examples: pick them for your GPU.

```bash
DATA=datasets/<dataset>/sim     # e.g. datasets/pioneer_8dof_left_arm/sim
lerobot-edit-dataset --repo_id humanoid/<dataset> --root $DATA --new_root ${DATA}_relstats \
  --operation.type recompute_stats --operation.relative_action true --operation.chunk_size 50
python -m humanoid_robot_learning.train --dataset.repo_id=humanoid/<dataset> --dataset.root=${DATA}_relstats \
  --policy.path=lerobot/pi05_base --policy.train_expert_only=true --policy.use_relative_actions=true \
  --policy.gradient_checkpointing=true --policy.push_to_hub=false --policy.device=cuda \
  --steps=6000 --batch_size=8 --save_freq=2000 --output_dir=outputs/train/pi05_<task>
```

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

Sim rollout for a flow-matching policy (pi0 / pi0.5 / SmolVLA) on the push-block scene, driven with Real-Time Chunking (`humanoid_robot_learning/deploy/rtc_driver.py`): [`scripts/pioneer_push_eval_rtc.py`](scripts/pioneer_push_eval_rtc.py). It drives the 8dof left arm and expects a checkpoint trained on `config/pioneer_8dof_dataset_schema.yaml` (8 values: 7 joints + gripper; an old 6-DoF arm checkpoint is refused); `--dry_run` runs it without one. **The VLA is not hooked up yet**: no real checkpoint has been run through it, and its schema, camera keys and action space (absolute vs delta) still need matching. See the `TODO(VLA)` block in the script.

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
