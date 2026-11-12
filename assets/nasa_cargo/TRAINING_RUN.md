# Trained cargo policy and HG-DAgger handoff

The initial ACT run is complete: **100,000 optimizer steps** on the 50 synthetic
demos. The final checkpoint is suitable for starting keyboard HG-DAgger in the
current cargo scene. No human corrections have been collected by these tests.

```text
outputs/train/nasa_cargo_act_v1/checkpoints/100000/pretrained_model
```

Both `checkpoints/last` and `checkpoints/best_eval` point to `100000`.

## What was checked

Training and headless Isaac evaluation ran in separate processes, with evaluation
starting at an early checkpoint. Training remained finite and completed normally;
the final logged loss was 0.023. Early models were rejected based on actual robot
behavior, rather than assuming a decreasing loss meant the task worked.

| Optimizer step | Placements | Lifts | Test starts |
|---|---|---|---|
| 2,000 | 0/1 | 0/1 | Early test stopped after a complete failed rollout |
| 8,000 | 1/3 | 2/3 | 10000–10002 |
| 14,000 | 2/3 | 3/3 | 10000–10002 |
| 20,000 | 1/3 | 3/3 | 10000–10002 |
| 30,000 | 0/3 | 3/3 | 10000–10002 |
| 40,000 | 3/3 | 3/3 | 10000–10002 |
| 50,000 | 2/3 | 3/3 | 10000–10002 |
| 60,000 | 2/3 | 3/3 | 10000–10002 |
| 70,000 | 3/3 | 3/3 | 10000–10002 |
| 80,000 | 2/3 | 3/3 | 10000–10002 |
| 90,000 | 2/3 | 3/3 | 10000–10002 |
| 100,000 | 3/3 | 3/3 | 10000–10002 |
| 100,000, fresh suite | **10/10** | **10/10** | **11000–11009**, not used for checkpoint selection |

Every rollout was autonomous. Success required a prior lift, the conservative
whole-bag bounds inside the tray, upright supported placement with low velocity
for one second, an open gripper command and measured gripper closure below 0.1.
The fresh trials used the same unchanged scene, payload and reset distribution.

Six unit tests also pass for human authority, preserving a carried grip on
takeover, action-queue resets, correction boundaries, pause/discard and repeated
dataset aggregation. Interactive keyboard operation still needs an operator;
the installed keyboard API was checked against the controller.

Isaac's incomplete SIGTERM shutdown caused an evaluator startup problem during
early testing. Evaluator cleanup now stops its owned process group, and the
monitor checks finite loss, available disk and evaluator progress. The training
run did not need a restart. A measured-open check was also moved into the
evaluator's termination condition so it waits for physical release before success.

## Start human correction collection

Quest controller collection is available through the
[VR HG-DAgger workflow](VR_HG_DAGGER.md). The commands below use the keyboard.

On the host, open the usual GUI-enabled simulation container:

```bash
ACTIVE_MODULES="simulation_isaac" ./watod up -d
./watod -t simulation_isaac
```

Inside:

```bash
cd /workspace/humanoid
/workspace/isaaclab/isaaclab.sh -p \
  src/robot_learning/humanoid_robot_learning/sim/hg_dagger.py \
  --mode collect --episodes 50 --seed 2000 --round-id round_001 --device cuda \
  --policy outputs/train/nasa_cargo_act_v1/checkpoints/100000/pretrained_model \
  --dataset-root datasets/nasa_cargo/round_001/corrections
```

Focus the viewport. **H** takes control; **H** again saves the human segment and
returns control. **K** toggles the gripper; **P** pauses; **B** resets for the next
attempt and discards unsaved frames. W/S, A/D, Q/E translate the gripper; Shift
reduces sensitivity. Only human-controlled frames become training labels.

There is no need to start this initial training run again. See
[the full workflow](HG_DAGGER.md) for correction merging and training a new policy
on the aggregate, using a new output directory.

## Results and logs

- [Fresh evaluation](../../outputs/train/nasa_cargo_act_v1/headless_evaluations/fresh_100000.json)
- [Checkpoint evaluation history](../../outputs/train/nasa_cargo_act_v1/headless_evaluations)
- [Training log](../../outputs/train_logs/cargo_act_v1.log)
- [Background monitor log](../../outputs/train_logs/cargo_watch.log)
- [Fresh evaluation log](../../outputs/train_logs/cargo_final_fresh.log)

The small test suites cover fixed cameras, fixed lighting, the 100 g rigid bag,
pickup jitter of ±5 mm and yaw jitter of ±0.025 rad. These results support an
initial simulated HG-DAgger policy; they do not establish broad generalization
or hardware transfer. Placement was the main failure mode in earlier models.
