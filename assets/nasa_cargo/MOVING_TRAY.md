# Fixed cargo, moving green tray

Use `--randomization tray` instead of `wide` to randomize the destination.
The cargo starts at (0.50, 0.14) m with zero yaw; its body colour and the arm's
initial joint offsets still vary. Tray centre X varies from 0.25 to 0.45 m and
Y from 0.49 to 0.53 m, in robot-base scene coordinates (X forward, Y left).
This is a **20 cm × 4 cm** destination region. A fixed seed repeats the same
reset; successive episodes/seeds change it.

The earlier 9/13 cm bounds assumed placement with the bag facing straight
ahead. A 25° turn about vertical makes the larger region reachable while the
whole bag still fits within the original rim. The scripted expert uses that
turn for farther goals and preserves the original orientation near the old
goal. It retreats 10 cm above its placement endpoint.
The learned policy and human operator remain free to choose any placement
orientation that passes the original upright, whole-bag containment checks.
The bag is moved only through robot contact; the new yaw parameter never
writes its pose during an episode.

The physical tray floor and all four rim colliders move together at reset.
Task success follows the current tray position, and recordings include
requested/measured tray positions, measured cargo spawn and expert reference
yaw. The recording/camera/action contract remains compatible with existing
data. Cameras and prop dimensions are unchanged.

## Play the policy

Inside the running simulation container, stop the previous runner with Ctrl+C:

```bash
cd /workspace/humanoid
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode evaluate --randomization tray --episodes 2 --seed 72050 --max-seconds 30 --device cuda \
  --policy outputs/train/nasa_cargo_act_randomized_v2/checkpoints/005000/pretrained_model \
  --report outputs/nasa_cargo/evaluations/moving_tray.json
```

Seeds 72050 and 72051 differ by about 18 cm in tray X. The terminal prints
`[Playback] Starting episode 1/2` and `2/2`; a failed episode resets after
30 **simulated** seconds, which can take longer in wall time. Restarting with
the same seed always repeats the same first placement.

The old policy learned a fixed destination, so it may still aim there. These
are new task variations for HG-DAgger; no policy has been retrained here.

## VR corrections

First follow the container/Quest connection instructions in
[VR_HG_DAGGER.md](VR_HG_DAGGER.md), then run:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode collect --teleop vr --randomization tray --single-task-thread --device cuda \
  --policy outputs/train/nasa_cargo_act_randomized_v2/checkpoints/005000/pretrained_model \
  --dataset-root datasets/nasa_cargo/tray_round_001/vr_corrections \
  --episodes 30 --seed 82000 --round-id tray_round_001
```

The scene waits at startup while the server and camera previews stay live.
Enter VR with head and both controllers tracked and both grip buttons released;
control then starts automatically. The HUD says `WAITING FOR VR` until ready.

Hold right grip to intervene; move/grip with the left controller. Release
right grip to return to policy without saving. Press **right B** to save the
unsaved corrections and reset to the next seed. Press **left Y** to discard them and reset to the next
seed, then release right grip to resume. Y overrides B when both arrive together.
The HUD shows both saved segment totals and the unsaved frame count. At 750 unsaved frames, physics pauses
for B/Y rather than automatically saving. Use a new dataset directory for this round.

## Validation

Unit checks cover 1,000 seeded target samples, fixed pickup poses, rim
clearance, the rotated bag footprint and joint-limited placement/carry/retreat
targets. The cargo test suite passes 19 tests with one optional dependency skip
on the host. A failed straight-ahead scripted transfer alone does not prove a
tray position unreachable.

For CUDA physical boundary tests:

```bash
/workspace/isaaclab/isaaclab.sh -p tools/isaac_harness/check_cargo_tray.py \
  --headless --device cuda --report outputs/nasa_cargo/tray_broad_range_check.json
```

The checks verify stable cargo spawns, all five tray colliders, moved-goal
metrics and real robot contact transfers. They measure task reach and physics,
not the learned policy's success rate.

The 20 × 4 cm CUDA boundary check completed all four arm trajectories without
IK rejection and passed collider/reset/goal checks. Two of four scripted
transfers met the strict settled-placement success metric, including the
farthest corner (0.45, 0.53) m. The other two missed placement; no policy
training or success-rate tuning was performed.
See [the boundary report](../../outputs/nasa_cargo/tray_broad_range_check.json).


## Train after a correction round

Stop the collector before training. Validate the recorded segments, merge
with the existing 62-demo aggregate, and fine-tune the current checkpoint.
This preserves the checkpoint's observation/action normalization. Choose new
output directories for every round; use `--resume` only when appending to the
same collection dataset. These commands are instructions, not an automatic
training launch.

```bash
cd /workspace/humanoid
export PYTHONPATH="/workspace/humanoid/src/robot_learning:${PYTHONPATH:-}"
/workspace/isaaclab/isaaclab.sh -p -m humanoid_robot_learning.sim.cargo_data \
  validate datasets/nasa_cargo/tray_round_001/vr_corrections

/workspace/isaaclab/isaaclab.sh -p -m humanoid_robot_learning.sim.cargo_data merge \
  datasets/nasa_cargo/randomized_v2/aggregate \
  datasets/nasa_cargo/tray_round_001/vr_corrections \
  --output datasets/nasa_cargo/tray_round_001/aggregate

CARGO_POLICY_PATH=/workspace/humanoid/outputs/train/nasa_cargo_act_randomized_v2/checkpoints/005000/pretrained_model \
CARGO_DATASET_ROOT=/workspace/humanoid/datasets/nasa_cargo/tray_round_001/aggregate \
CARGO_OUTPUT_DIR=/workspace/humanoid/outputs/train/nasa_cargo_hg_tray_round_001 \
CARGO_STEPS=5000 \
bash src/robot_learning/scripts/cargo_finetune_quick.sh
```

Then evaluate the new policy without interventions on the same tray profile:

```bash
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode evaluate --randomization tray --episodes 20 --seed 81000 \
  --max-seconds 30 --headless --device cuda \
  --policy outputs/train/nasa_cargo_hg_tray_round_001/checkpoints/last/pretrained_model \
  --report outputs/nasa_cargo/evaluations/hg_tray_round_001.json
```

For the next round, collect using that new checkpoint and a new dataset folder,
then merge corrections with this round's aggregate. Collection seeds start at
82000; reserve the 81000-series evaluation seeds for evaluation rather than
human correction. `--episodes 30` in collection means 30 saved intervention
segments, not 30 completed tasks. Intervene before a bad approach, missed
grasp, collision or incorrect placement; complete the useful correction, then
release right grip whenever you want to return control to the policy. Press
right B to save a useful correction and reset, or left Y to discard the unsaved attempt
and reset. Saved segments survive Ctrl+C; all unsaved frames are discarded on
exit, including interventions whose right grip has already been released.


## Restarting before the first correction

If a session exits before its first saved segment, LeRobot leaves an empty
metadata shell without `meta/tasks.parquet`. Restart collection with `--resume`:
the cargo writer verifies zero episode/frame counters and absence of recording
artifacts, preserves that shell in a sibling `.empty-session-*` backup, and
creates a fresh local writer at the original path. Completed sessions append
normally. Inconsistent metadata or recording artifacts are preserved and cause
a local error rather than being overwritten. No Hugging Face login is needed
for these local correction datasets.


## Completed HG-DAgger round 001

The existing ACT checkpoint was fine-tuned once for 5,000 steps on the original
62 synthetic demonstrations plus 30 accepted human corrections (92 segments,
36,401 frames). The saved observation/action normalization was preserved.
The original policy is retained. The new checkpoint is:

`outputs/train/nasa_cargo_hg_tray_round_001/checkpoints/005000/pretrained_model`

On the same 50 seeds (81000–81049), same tray profile and 30-second timeout,
strict placement improved from **19/50 (38%) to 45/50 (90%)**. Both policies
lifted in all 50 trials. Farther-half tray placements improved from 2/26 to
24/26; nearer-half placements improved from 17/24 to 21/24. The final checkpoint
was selected before the after-evaluation; no test-set tuning was performed.
This is one HG-DAgger aggregation/refinement round using ACT, not a literal
reproduction of the paper's driving-policy ensemble or five data-collection
rounds. See [the report](../../outputs/nasa_cargo/tray_round_001/HG_DAGGER_ROUND_001.md)
and [paired results](../../outputs/nasa_cargo/tray_round_001/before_after_comparison.json).

To play the updated policy inside the simulation container:

```bash
cd /workspace/humanoid
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode evaluate --randomization tray --episodes 10 --seed 83000 \
  --max-seconds 30 --device cuda \
  --policy outputs/train/nasa_cargo_hg_tray_round_001/checkpoints/005000/pretrained_model \
  --report outputs/nasa_cargo/evaluations/hg_round001_playback.json
```

For another HG-DAgger round, use the updated checkpoint, a new dataset folder
(e.g. `datasets/nasa_cargo/tray_round_002/vr_corrections`) and fresh collection
seeds. Keep the reserved evaluation seeds for comparisons.
