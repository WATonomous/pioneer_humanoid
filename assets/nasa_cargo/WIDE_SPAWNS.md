# Wider, reachable cargo spawns

Use **`--randomization wide`** for more visible position variation:

| Scene coordinate | Bag-centre range |
|---|---|
| X (forward from the robot base) | 0.475–0.525 m: ±2.5 cm from the original pickup anchor |
| Y (left of the robot base) | 0.115–0.150 m: −2.5 to +1 cm from the anchor |

This is a 5 × 3.5 cm region, almost twice the 3 × 3 cm area used for the quick
fine-tune. Colours, yaw and initial arm-joint variation remain enabled. The
pickup support becomes 50 × 30 cm, centred at (0.50, 0.13) m; its top stays at
0.48 m and its upper Y edge remains clear of the receiving tray rim. The
nominal anchor, robot base, target tray, cameras and action format stay fixed.
Support dimensions and measured settled spawn positions are recorded in reset
metadata. The original `baseline` and `varied` presets retain their old ranges.

## Play it

Stop the current script with Ctrl+C, then inside the simulation container:

```bash
cd /workspace/humanoid
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode evaluate --randomization wide --episodes 10 --seed 183 --device cuda \
  --policy outputs/train/nasa_cargo_act_randomized_v2/checkpoints/005000/pretrained_model
```

Seeds 183 and 184 demonstrate opposite sides of the X range (about 52.3 cm and
47.6 cm). A fixed seed repeats the first spawn when restarting; successive
episodes use successive seeds. The terminal prints each actual settled bag
position under `[Cargo reset]`.

## Use VR HG-DAgger

Follow [the Quest container/USB/browser setup](VR_HG_DAGGER.md), then run:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode collect --teleop vr --randomization wide --single-task-thread --device cuda \
  --policy outputs/train/nasa_cargo_act_randomized_v2/checkpoints/005000/pretrained_model \
  --dataset-root datasets/nasa_cargo/wide_round_001/vr_corrections \
  --episodes 30 --seed 183 --round-id wide_round_001
```

Hold the right grip to intervene, use the left controller/trigger to correct,
and release the right grip to save and resume the policy. Left Y resets to the
next seed and discards unsaved frames. The wider support is a reset-domain
variation; the recording contract is compatible with the existing aggregate.

The existing policy was trained with the smaller `varied` range. No automatic
retraining was performed for `wide`; failures at the new positions are useful
for human corrections.

## Validation

Unit checks cover 1,000 sampled footprints staying supported and clear of the
tray rim, plus 135 IK approach/grasp/lift targets under physical joint limits.
Four corner cases were also spawned in CUDA physics: all stayed near their
requested positions and all were lifted by the actual arm through contact.
Observed rises were approximately 11.7–12.0 cm. This verifies corner reach and
contact lifting, rather than learned-policy success across the new region.

The first proposed region extended closer to the robot. Physics showed the
bag being pushed away there, so that stand-overlap region was excluded.

[Physics results](../../outputs/nasa_cargo/wide_range_check.json).
