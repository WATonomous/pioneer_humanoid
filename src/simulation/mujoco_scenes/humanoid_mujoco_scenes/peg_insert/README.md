# Peg insertion: scripted expert → learned policies → corrections

A square peg (40 mm, 1 mm total clearance) must be picked up and inserted into a block in front of
the left arm. A privileged, physics-only expert records demonstrations; policies (ACT, flow
transformer) are trained in `src/robot_learning`; expert takeovers from policy states supply
correction data. Results and the experiment history are in
[`docs/research/peg_insert_act_vs_flow.md`](../../../../../docs/research/peg_insert_act_vs_flow.md).

| Module | Verb | What it does |
|---|---|---|
| `scene.py` | — | the MuJoCo scene (`@scene("peg_insert")`) |
| `expert.py` | collect demos | privileged expert (case table in its docstring) + demo recorder CLI |
| `workspace.py` | map / sample | independent peg/hole placements: 1 cm map, held-out cells, layout files |
| `handoffs.py` | — | every correction/diagnostic start state and takeover detector, in one table |
| `eval.py` | eval | unassisted closed-loop evaluation of any policy checkpoint |
| `collect_corrections.py` | collect corrections | expert labels from policy takeovers, prepared tasks, or failed eval traces |
| `dagger.py` | iterate | collect → convert → continue training → eval, until a success target |
| `audit.py` | audit | hole-shift visual response and chunk latency, no training |
| `recording.py` | — | raw episode format shared by demos and corrections |

All commands run from the repo root with `src/pioneer_humanoid`, `src/simulation/mujoco_scenes`
and `src/robot_learning` installed (`pip install -e ...`). Policy tools also need torch + the
pinned LeRobot (`e670ac5daf9b76`). On Linux servers set `MUJOCO_GL=egl`; on macOS use `mjpython`
for `--viewer`.

## The expert

Commands go through the position actuators at **25 Hz** (40 × 1 ms physics steps per command); it
never welds the peg, applies forces or writes object poses after `reset`. It grasps the upper
3 cm so the fingers clear the block, servos the *measured* peg pose (compensating slip and
gravity error), inserts at 6 mm/s and verifies a seated, released, stationary peg. Failed grasps
get three attempts; a stalled insertion gets three retract/realign attempts. Everything else
discards the episode. `guarded=True` (used for corrections from policy states) adds early-jam
aborts and a seating check before opening; on full expert runs it changes nothing
(35/40 either way) and costs ~5 s, so demos don't use it.

```bash
# physics-only check, no cameras
python -m humanoid_mujoco_scenes.peg_insert.expert --episodes 20 --seed 0 --verbose
# record until the collection holds 50 successful demos (rerun the same command to resume)
python -m humanoid_mujoco_scenes.peg_insert.expert --episodes 100 --successes 50 --output datasets/peg_insert_raw
python -m humanoid_robot_learning.raw_to_lerobot --raw datasets/peg_insert_raw --root datasets/pioneer_peg_insert
```

Each episode stores `observation.state` (six measured joint angles + gripper closure 0–1),
`action` (six absolute actuator targets + commanded closure), `phase`, peg/hole poses, and
640×480 `ego` / `wrist_left` PNGs. Frame *t* is the observation before its action, which is held
for 40 ms. `--randomize 0.01` jitters both objects ±1 cm; larger, independent placements come
from `workspace.py`:

```bash
python -m humanoid_mujoco_scenes.peg_insert.workspace map --output outputs/peg_workspace
python -m humanoid_mujoco_scenes.peg_insert.workspace sample --map outputs/peg_workspace/workspace_map.json \
  --split test --counts 20 30 --output outputs/peg_workspace/heldout_layouts.json
python -m humanoid_mujoco_scenes.peg_insert.workspace sample --map outputs/peg_workspace/workspace_map.json \
  --split train --counts 25 75 --output datasets/peg_workspace/layouts.json --record datasets/peg_workspace/raw
```

## Train and evaluate

ACT trains with LeRobot (chunk 50, batch 8–16, ResNet18; see `src/robot_learning/README.md`). To
continue a checkpoint on demos + corrections with phase quotas and without re-estimating its
normalization:

```bash
python -m humanoid_robot_learning.lerobot_finetune \
  --config_path=outputs/train/act/checkpoints/150000/pretrained_model/train_config.json --resume=true \
  --dataset.root=datasets/pioneer_peg_insert --extra-dataset=datasets/corrections_r1 \
  --phase-config=src/robot_learning/config/peg_insert_phases.json --phase-metrics \
  --steps=174000 --batch_size=8 --output_dir=outputs/train/act_r1
```

Evaluation is unassisted: the expert never acts after the handoff, and success requires a peg
that stays seated and released for 0.5 s within 60 simulated seconds.

```bash
python -m humanoid_mujoco_scenes.peg_insert.eval --checkpoint <ckpt>/pretrained_model --device cuda \
  --layouts outputs/peg_workspace/heldout_layouts.json --n-action-steps 50 \
  --report outputs/eval/act_r1/heldout.json --trace-dir outputs/eval/act_r1/traces
# insertion-only diagnostic from a held peg (aligned | offset_positive | offset_negative | tilt_positive | tilt_negative)
python -m humanoid_mujoco_scenes.peg_insert.eval --checkpoint <ckpt> --start-stage insertion \
  --insertion-case tilt_positive --episodes 10 --report outputs/eval/act_r1/tilt_pos.json
```

Use disjoint seed ranges for collection and evaluation, report N, and pick checkpoints by
closed-loop success, not training loss. `--video-dir` writes overview/ego/wrist MP4s in
simulation time; `--trace-dir` saves full physical state per step.

## Corrections

`collect_corrections` records only the expert's actions, from a physical handoff to verified
insertion and release:

```bash
# expert takes over when the policy stalls/tilts/misaligns at the rim
python -m humanoid_mujoco_scenes.peg_insert.collect_corrections --source takeover --trigger stall \
  --checkpoint <ckpt>/pretrained_model --device cuda --seed 3000 --episodes 100 --corrections 10 --output datasets/corr_r1
# expert-prepared starts: 80 broad offsets/heights/tilts, or 24 large tilts
python -m humanoid_mujoco_scenes.peg_insert.collect_corrections --source tasks --tasks broad --output datasets/corr_broad
# restore states from failed eval traces (first rim contact / hover / pre-drop), guarded expert
python -m humanoid_mujoco_scenes.peg_insert.collect_corrections --source traces \
  --eval-report outputs/eval/act_r1/heldout.json --trace-dir outputs/eval/act_r1/traces --output datasets/corr_traces
```

`dagger` chains collection, conversion, continued training and evaluation for several rounds,
skipping any step whose output exists:

```bash
python -m humanoid_mujoco_scenes.peg_insert.dagger --root outputs/dagger/act --checkpoint <ckpt> \
  --base-dataset datasets/pioneer_peg_insert --trigger stall --corrections 10 --steps-per-round 6000 \
  --phase-config src/robot_learning/config/peg_insert_phases.json --eval-layouts outputs/peg_workspace/heldout_layouts.json
```

## Audits

```bash
# does the policy follow a 3/10 mm hole shift seen only in the images? (expert gain ≈ 1.0)
python -m humanoid_mujoco_scenes.peg_insert.audit hole-shift --checkpoint <ckpt> --seeds 1000 1001 1002 --report outputs/audit/hole.json
python -m humanoid_mujoco_scenes.peg_insert.audit latency --checkpoint <ckpt> --device cuda --report outputs/audit/latency.json
```

Cluster job submission (WATcloud) is in [`tools/slurm/README.md`](../../../../../tools/slurm/README.md).
