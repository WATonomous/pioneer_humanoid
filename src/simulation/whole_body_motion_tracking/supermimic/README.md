# SuperMimic on Pioneer

One whole-body tracking policy for the Pioneer humanoid, trained on all 77 [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) motion-capture clips with [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) (MIT) on [mjlab](https://github.com/mujocolab/mjlab), in simulation.

[media/pioneer_one_policy_77_clips.mp4](media/pioneer_one_policy_77_clips.mp4): the policy running, dancing, sparring, hopping and stumbling through a 61 s sequence of moves.

## Results

Benchmark: the first 20 s of every clip. A clip fails when the robot falls (pelvis below 0.5 m or torso tilt above 0.8 rad) or drifts more than 0.8 m from the reference in the ground plane.

| | Clips completed | Mean normalised return |
|---|---|---|
| Hip and knee motors at 21 rad/s (`PIONEER_FAST=1`, the training setting) | **55/77** | 0.79 |
| Same policy, real motor speeds (AKH70-48 hip pitch and knee: 3.67 rad/s) | 51/77 | 0.79 |

Per category (`PIONEER_FAST=1`):

| Completed | Category |
|---|---|
| All | walk 12/12, dance 8/8, aiming 5/5, jumps 3/3, push and stumble 3/3, push and fall 2/2, fight and sports 2/2, push 1/1 |
| Some | obstacles 12/17, fight 2/3, run 2/4, ground 2/5, multiple actions 1/4 |
| None | fall and get up 0/6, sprint 0/2 |

Everyday motions all pass; the failures are floor work (getting up, ground moves) and fast running. Failures come mid-clip (6-20 s in), after the robot has tracked the start.

Training stopped at about 94k iterations of the 320k default, when the curve had flattened. The policy was trained on the whole-body model before the knee range went to 140 deg; the scripts below build the current model, so a retrain uses the current joint limits.

## What changed for Pioneer

`supermimic.patch` (against SuperMimic-Reproduced `d9be385`), all behind `SM_ROBOT=pioneer` (the G1 default is unchanged):

- `robots/pioneer.py`: the robot config. 29 DoF (12 legs, waist, 2 neck, 14 arms; fingers locked in fists), DC motors with each joint's real torque and no-load speed, PD gains from the Isaac Lab locomotion (legs) and arm-swing (arms) configs. The neck is held straight (no reward tracks the head). `PIONEER_FAST=1` raises the hip pitch and knee speed limit to 21 rad/s, a sim-only experiment; `PIONEER_SELFCOL=1` turns on forearm/hand vs trunk collision.
- Terminations: an optional xy-drift reset (`--env.root-xy-threshold`; without it a policy that stands still while the clip walks away never resets) and a reset on non-finite or absurd sim states (the CAD collision boxes occasionally blow up MuJoCo Warp on ground clips).
- The replay buffer drops non-finite rows (one turned the world model to NaN).
- The height and tilt thresholds are config fields, and the benchmark uses the run's own terminations.
- Resuming onto a different clip set keeps the networks and resets the clip sampling statistics, so training can grow from 8 to 24 to 77 clips.
- `PIONEER_MISMATCH_P`: for get-up training, start some lying episodes from a different lying pose.

## Reproduce

Needs a CUDA GPU (trained on an RTX 5070 Ti, 16 GB) and [uv](https://docs.astral.sh/uv/).

```bash
HERE=<this repo>/src/simulation/whole_body_motion_tracking/supermimic
# 1. SuperMimic with the Pioneer patch
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced && cd SuperMimic-Reproduced
git checkout d9be385 && git apply $HERE/supermimic.patch && uv sync
SM=$PWD/.venv/bin/python

# 2. Pioneer models: floating pelvis and fists for retargeting, then the training copy (no gravcomp, named hitboxes)
$SM $HERE/scripts/make_retarget_mjcf.py pioneer_retarget.xml
$SM $HERE/scripts/make_train_mjcf.py pioneer_retarget.xml pioneer_train.xml
export SM_ROBOT=pioneer PIONEER_XML=$PWD/pioneer_train.xml

# 3. Retarget LAFAN1 with GMR (YanjieZe/GMR at 25f56db, installed in its own environment as its README says).
#    The config is G1's turned to Pioneer's frames, recalibrated on a T-pose frame; the script also registers "pioneer" in GMR.
python $HERE/scripts/make_lafan_config.py <GMR>/general_motion_retargeting pioneer_retarget.xml <lafan1>/walk1_subject1.bvh
for f in <lafan1>/*.bvh; do python $HERE/scripts/retarget_lafan.py $f csv/$(basename $f .bvh).csv; done
python $HERE/scripts/lift_out_of_floor.py pioneer_retarget.xml csv/fallAndGetUp*.csv csv/pushAndFall*.csv csv/ground*.csv

# 4. Motion files: 5 Hz smoothing, recorded speed (1), 50 Hz; and the 20 s benchmark set
$SM $HERE/scripts/convert_for_supermimic.py csv data/npz/pioneer_77 1
$SM $HERE/scripts/first_seconds.py data/npz/pioneer_77 data/npz/pioneer_77first20 20
```

Training ran in stages, each resuming the last (`--resume logs/<previous>/ckpt_latest.pt`): 8 clips (dance, fall and get up, fight, fight and sports, jumps, run, walk), then 24, then all 77; about 9.6 h in total. The last stage:

```bash
PIONEER_FAST=1 .venv/bin/sm-train --run-name pioneer_full77 --resume logs/pioneer_24clips/ckpt_latest.pt \
  --data.npz-dir data/npz/pioneer_77 --env.num-envs 4096 --env.anchor-obs \
  --env.root-xy-threshold 0.8 --env.root-height-threshold 0.5 --env.root-tilt-threshold 0.8 \
  --policy-trainer.batch-size 2048 --wm-trainer.batch-size 2048 \
  --schedule.total-iterations 93700 --schedule.checkpoint-interval 4000 --schedule.max-hours 4.5

# benchmark (unset PIONEER_FAST for the real motor speeds)
PIONEER_FAST=1 $SM scripts/eval_benchmark.py --ckpt logs/pioneer_full77/ckpt_latest.pt --npz-dir data/npz/pioneer_77first20
```
