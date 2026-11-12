# Cargo synthetic training and HG-DAgger

For the quick update with randomized bag position/colour and arm starts, see
[the randomized policy workflow](RANDOMIZED_QUICK_UPDATE.md).

For Quest Touch controller interventions, see
[VR HG-DAgger setup and controls](VR_HG_DAGGER.md). Keyboard collection remains
the default; VR is selected with `--teleop vr`.

An initial trained policy is now available. See the
[validated training run and collection command](TRAINING_RUN.md) to start
HG-DAgger using it; the instructions below also cover starting a new training run.

The pipeline uses the Pioneer **left arm**, the **100 g cargo bag**, a
reachable two-station layout, two **320×240 RGB cameras** (`ego`, `wrist_left`),
**25 fps**, and seven absolute actions: six arm joint targets in radians,
then gripper closure in `[0,1]`. The right arm holds its resting pose.

The fixed `ego` camera here is a task overview camera, not the Quest head
camera. Use this runner for both synthetic data and human correction; existing
Quest/keyboard datasets have different observation contracts and cannot be
silently mixed. `cargo_contract.json` records the layout, cameras, mass-asset
hash and action contract. Aggregation and checkpoint loading check it.

## 1. Open the container

From the host repo root:

```bash
ACTIVE_MODULES="simulation_isaac" ./watod up -d
./watod -t simulation_isaac
```

Inside:

```bash
cd /workspace/humanoid
export PYTHONPATH="/workspace/humanoid/src/robot_learning:${PYTHONPATH:-}"
```

Use the repo's simulation image; a bare Python install also needs the
`src/robot_learning[cargo]` extra, but Isaac itself comes from the image.

## 2. Synthetic demonstrations

The generated training dataset lives at:

```text
datasets/nasa_cargo/synthetic_50/
```

The initial dataset is complete: **50 successful demos, 26,552 frames**
(17.7 minutes at 25 fps), selected from 79 logged attempts. A side-by-side
camera preview is in `outputs/nasa_cargo/synthetic_demo_preview.mp4`.

To generate a fresh dataset at another path:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode generate --episodes 50 --max-attempts 200 --seed 0 \
  --dataset-root datasets/nasa_cargo/synthetic_new --headless --device cuda
```

`--episodes` is the requested **total number of successful saved episodes**.
Use `--resume` to continue a matching existing dataset; saved seeds are skipped.
The generator never overwrites a dataset. If it exhausts `--max-attempts`, it
exits nonzero and reports the shortfall rather than claiming completion.

The expert uses URDF IK and privileged bag pose to command the real arm's
joints and gripper. It approaches, pinches the red loop, lifts, carries,
lowers, releases and retreats. Cargo root state is written only at episode
reset; there are no grasp attachments, in-episode bag teleports, commanded
bag forces, or substitute gripper fixtures. Successful episodes must satisfy
CargoTask's lift/settled-placement checks **and** finish with an open gripper.
Failed attempts are excluded from training and logged in `attempts.jsonl`.

`collection_manifest.jsonl` records accepted seeds, phases and outcomes. The
dataset is stored as standard LeRobot videos/parquet with CPU episode buffers
and subprocess video encoding. This avoids multi-GB GPU recording buffers.

Validate metadata and decode sample images:

```bash
/workspace/isaaclab/isaaclab.sh -p -m humanoid_robot_learning.sim.cargo_data \
  validate datasets/nasa_cargo/synthetic_50 --episodes 50
```

## 3. Train overnight

Close Isaac viewers to free GPU memory, then run inside the container:

```bash
mkdir -p outputs/train_logs
nohup bash src/robot_learning/scripts/cargo_train_overnight.sh \
  > outputs/train_logs/cargo_act_v1.log 2>&1 &
tail -f outputs/train_logs/cargo_act_v1.log
```

Defaults: ACT, ImageNet-pretrained ResNet-18, batch 4, AMP, 100,000 optimizer
steps, 50-action prediction chunks with 25 steps executed per query, and a
checkpoint every 5,000 steps. Nothing uploads to Hugging Face or W&B. The
backbone weights are cached in `outputs/cache/torch`; they download from
PyTorch if absent. Runtime depends
on the GPU and video decoding; "overnight" is not a fixed completion time.

The output is:

```text
outputs/train/nasa_cargo_act_v1/checkpoints/last/pretrained_model/
```

Override without editing the script:

```bash
CARGO_STEPS=70000 CARGO_BATCH_SIZE=4 \
CARGO_OUTPUT_DIR=/workspace/humanoid/outputs/train/cargo_act_custom \
bash src/robot_learning/scripts/cargo_train_overnight.sh
```

Always choose a new output folder for a new run. The small
`outputs/train/cargo_pipeline_smoke` checkpoint is an integration test, not
a useful trained policy.
`outputs/train/cargo_full_dataset_smoke` also contains only two optimizer
steps, used to verify training against the complete 50-demo dataset.

## 4. Evaluate without assistance

```bash
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode evaluate --episodes 20 --seed 10000 --headless --device cuda \
  --policy outputs/train/nasa_cargo_act_v1/checkpoints/last/pretrained_model \
  --report outputs/nasa_cargo/eval_v1.json
```

This is the Isaac evaluator for this task; `sim.evaluate` is a separate
MuJoCo evaluator. Success requires settled target placement after a lift and
an open commanded gripper. Evaluation results are checkpoint-specific; a
successful synthetic expert does not establish the learned policy's success.

## 5. Collect human interventions

No VR or leader hardware is required. Omit `--headless` to open the window:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode collect --episodes 50 --seed 2000 --round-id round_001 --device cuda \
  --policy outputs/train/nasa_cargo_act_v1/checkpoints/last/pretrained_model \
  --dataset-root datasets/nasa_cargo/round_001/corrections
```

Focus the Isaac viewport:

| Key | Action |
|---|---|
| H | Take control from the policy; H again saves the correction and returns control |
| P | Freeze/resume physics; queued policy actions are cleared |
| W/S, A/D, Q/E | Translate the left gripper |
| Z/X, T/G, C/V | Rotate about the tool axes |
| Shift | Fine motion |
| K | Toggle gripper |
| N | Save the current human segment and continue |
| Delete | Discard the current unsaved human segment |
| B | Reset arm and bag with the next seed; discard the unsaved segment |
| Esc | Exit; unsaved frames are discarded |

On takeover the expert target is seeded from measured arm pose and the
current gripper command, so it does not home the arm or open a carried bag.
Hand-back clears the ACT action queue and predicts from the new observation.
Policy commands pass through joint limits and a joint-target slew limit.

Only human-controlled observations/actions enter the correction dataset.
Each uninterrupted intervention is a separate LeRobot episode; action chunks
cannot bridge an autonomous interval. Segments are automatically split at
750 frames (30 seconds) to bound CPU memory. `--episodes` counts saved
segments, not task successes. A new output folder is required unless
`--resume` is supplied.

`--mode teleop` uses the same keyboard/camera/recording contract without
loading a policy, useful for adding ordinary human demonstrations.

## 6. Aggregate and train the next policy

Retain the original synthetic demos and add corrections:

```bash
/workspace/isaaclab/isaaclab.sh -p -m humanoid_robot_learning.sim.cargo_data merge \
  datasets/nasa_cargo/synthetic_50 \
  datasets/nasa_cargo/round_001/corrections \
  --output datasets/nasa_cargo/round_001/aggregate

CARGO_DATASET_ROOT=/workspace/humanoid/datasets/nasa_cargo/round_001/aggregate \
CARGO_OUTPUT_DIR=/workspace/humanoid/outputs/train/nasa_cargo_act_v2 \
bash src/robot_learning/scripts/cargo_train_overnight.sh
```

Then evaluate v2 on the same held-out seeds and collect another correction
round. The script trains a fresh model on the aggregate; it does not silently
reuse old normalization statistics. Do not feed autonomous policy actions back
as expert labels, or count human-rescued episodes as autonomous successes.

## Scope and validation

A subsequent [full-frame and physics-replay audit](../../outputs/nasa_cargo/audit/README.md)
decoded every camera frame and reviewed six task stages for both cameras in
every episode. Replaying the saved commands produced 46/50 strict placements;
all 50 lifted and released. The four replay failures show placement sensitivity
to small contact differences. The dataset is suitable for initial fixed-scene
training; this replay result is not a learned-policy success rate.

Verified on the complete dataset: first/last RGB frames of all 50 episodes
decode, ACT completes two optimizer steps and saves a reloadable checkpoint,
and the checkpoint executes in the Isaac evaluator. Two-input aggregation
and a subsequent aggregation round both pass 100-episode decoding checks
using temporary test copies; these copies are excluded from training. Six
unit tests cover authority gating, correction boundaries and compacted-file
metadata. The scene's fixture lift/drop/negative-placement regression passes.
Keyboard takeover behavior is covered by the gate tests; interactive GUI
operation still needs an operator check. The two-step checkpoint has no
measured pick-and-place capability.

This is a small-variation bootstrap curriculum: pickup XY varies by ±5 mm,
yaw by ±0.025 rad, and expert pace by ±8%. The expert's object pose is privileged;
the policy sees only joint state and RGB. The scene's layout is adjusted for
left-arm reach, the receiving tray allows the full-size rigid bag to settle,
and the raised starting pose keeps fingers clear of the pickup object.

The 100 g payload, rigid bag, high-friction simulated contacts, fixed lighting
and narrow randomization do not establish hardware transfer or broad task
generalization. Expand randomization and add real/human demonstrations after
the initial policy works. Keep the same recorded camera/action contract.

The implementation follows the human-gated correction/aggregation loop from
[HG-DAgger](https://arxiv.org/abs/1810.02890). Uncertainty ensembles and automatic
intervention requests are not required by this human-controlled pipeline.
