# Quick randomized cargo update

For a larger, collision-checked spawn region during play or HG-DAgger, see
[the `wide` preset](WIDE_SPAWNS.md). The quick fine-tune described here used
the smaller `varied` range.

**Completed:** one 5,000-step fine-tune, with no success-rate optimization loop.
The final checkpoint lifted in 3/3 sanity trials and completed 2/3 placements;
seed 60002 remained outside the valid tray bounds after release. Three trials
are a pipeline sanity check, not a reliable success-rate estimate.

The `varied` reset profile changes all three requested factors:

- Cargo spawn XY: uniform offsets up to ±1.5 cm on the pickup station.
- Cargo body: varied hue/saturation/brightness, with some original white bags.
  The red handle stays red and the mass stays 100 g.
- Left-arm starting pose: independent joint offsets up to ±0.06 rad (±3.4°)
  around the existing raised home, within physical joint limits. The gripper
  starts open and the right arm stays parked.

Yaw also varies by ±0.04 rad. Randomization happens at reset, not during a
demonstration. The expert commands actual arm/gripper joints; accepted synthetic
episodes must lift, release and settle in the tray. Each episode records its
seed, colour, pose offsets and initial joint targets in the collection manifest.
Colour is an in-memory USD material override; the source cargo asset is unchanged.

## One bounded training pass

The quick update uses 12 new successful randomized demonstrations plus the
original 50, then fine-tunes v1 for 5,000 steps at learning rate 5e-6. It keeps
the pretrained observation/action normalization so the existing weights retain
their meaning. It does not run a hyperparameter search or retry until it reaches
a high task success rate. Remaining failures are intended for human correction.

The new output paths are:

```text
datasets/nasa_cargo/randomized_v2/new_demos
datasets/nasa_cargo/randomized_v2/aggregate
outputs/train/nasa_cargo_act_randomized_v2
```

Logs:

```text
outputs/train_logs/cargo_randomized_generation.log
outputs/train_logs/cargo_randomized_update.log
outputs/train_logs/cargo_randomized_v2.log
outputs/train_logs/cargo_randomized_watch.log
```

`cargo_randomized_update.py` watches the in-flight generator, validates and
merges the demonstrations, then starts the single fine-tune. The original v1
checkpoint and its datasets remain available. A two-step warm-start check passed;
all saved normalization values matched v1, including the tensor shapes used
for state, actions and RGB normalization.
Collection was capped at 12 finalized episodes; the next unsaved attempt was
discarded. These add 6,539 frames to the original set, for 62 episodes and
33,091 frames in the aggregate.

## Use randomized resets for evaluation and VR HG-DAgger

After the quick update finishes, inside the simulation container:

```bash
cd /workspace/humanoid
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode evaluate --randomization varied --episodes 5 --seed 61000 --device cuda \
  --policy outputs/train/nasa_cargo_act_randomized_v2/checkpoints/005000/pretrained_model
```

For human correction, follow [the Quest USB/browser setup](VR_HG_DAGGER.md), then:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode collect --teleop vr --randomization varied --single-task-thread --device cuda \
  --policy outputs/train/nasa_cargo_act_randomized_v2/checkpoints/005000/pretrained_model \
  --dataset-root datasets/nasa_cargo/randomized_v2/round_001/vr_corrections \
  --episodes 30 --seed 62000 --round-id randomized_round_001
```

The right grip selects human authority; left-controller motion and trigger
control the arm and gripper. Release right grip to return to policy without
saving. Right B saves the pending corrections and resets; left Y discards them and resets
to another randomized scene. Keep `--randomization varied` on each run to vary colour,
spawn and starting arm pose; the default `baseline` profile retains the old
white-bag setup for comparison.

The policy observation/action contract is unchanged, so these corrections can
be merged with `randomized_v2/aggregate` for your next HG-DAgger training round.
Use `--seed 60002` to start from the initial conditions of the failed sanity
case; exact contact outcomes can vary between runs.

After collecting corrections, inside the container:

```bash
export PYTHONPATH="/workspace/humanoid/src/robot_learning:${PYTHONPATH:-}"
/workspace/isaaclab/isaaclab.sh -p -m humanoid_robot_learning.sim.cargo_data merge \
  datasets/nasa_cargo/randomized_v2/aggregate \
  datasets/nasa_cargo/randomized_v2/round_001/vr_corrections \
  --output datasets/nasa_cargo/randomized_v2/round_001/aggregate

CARGO_POLICY_PATH=/workspace/humanoid/outputs/train/nasa_cargo_act_randomized_v2/checkpoints/005000/pretrained_model \
CARGO_DATASET_ROOT=/workspace/humanoid/datasets/nasa_cargo/randomized_v2/round_001/aggregate \
CARGO_OUTPUT_DIR=/workspace/humanoid/outputs/train/nasa_cargo_hg_v3 \
bash src/robot_learning/scripts/cargo_finetune_quick.sh
```

The original-data mixture and randomized demonstrations are retained alongside
human corrections. Use a new output directory for each training round.

## Original-policy benchmark

Before the randomized update, the frozen v1 checkpoint completed **47/50**
pick-and-place attempts and **50/50** lifts on fresh baseline starts (seeds
30000–30049, 45 simulated seconds maximum each). Failures were at seeds
30010, 30011 and 30032 after lifting/releasing. Median successful completion
was 17.24 simulated seconds. This 94% placement rate applies to the original
small-jitter white-bag distribution, not the new varied profile.

[Raw baseline results](../../outputs/nasa_cargo/evaluations/policy100000_seed30000_n50.json)
and [summary](../../outputs/nasa_cargo/evaluations/policy100000_seed30000_n50_summary.json).
