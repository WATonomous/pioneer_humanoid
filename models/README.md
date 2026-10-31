# `models/`

Frozen RL policy baselines worth keeping across the team — checked in so they
travel with the repo.

| dir | task / gym id | notes |
|-----|---------------|-------|
| `push/` | `Isaac-Bimanual-Push-Block-v0` | teacher policy (`model_1499.pt`) for the vision-distillation run |
| `push_distill/` | `Isaac-Bimanual-Push-Block-Distill-v0` | distilled vision student (`student_699.pt`) |
| `pioneer_hand_cube/` | `Isaac-Repose-Cube-PioneerHand-v0` | in-hand reorientation (`model_6600.pt`) |
| `pioneer_humanoid_flat/` | `Isaac-Locomotion-Flat-PioneerHumanoid-v0` | flat-ground walking (`model_3900.pt`) |
| `pioneer_humanoid_rough/` | `Isaac-Locomotion-Rough-PioneerHumanoid-v0` | rough-terrain walking (`model_2999.pt`) |
| `pioneer_leg_only_flat/` | `Isaac-Locomotion-Flat-PioneerLegOnly-v0` | Pioneer leg-only humanoid, flat-ground walking (`model_2999.pt`) |
| `pioneer_whole_body_flat/` | `Isaac-Locomotion-Flat-PioneerWholeBody-v0` | Pioneer whole-body humanoid, flat-ground walking with arm swing (`model_2299.pt`) |
| `badminton_shared_student_ppo/` | `Mjlab-Badminton-Shared-Student-PPO-Priv` | bimanual badminton receive on EKF perception (`model_11996.pt`): one single-arm policy drives whichever arm `badminton_shared_chooser/` picks; arms start at rest, grippers closed on a centred racket. 98.4% bank hits (chooser margin 1.0), hits split 49/51 right/left, 89% net clearance; the deployable one |
| `badminton_shared_chooser/` | `Mjlab-Badminton-Shared-Student-PPO` | its critic picks the arm from the robot's own observations (`model_11997.pt`); pass it as `--critic-task`/`--critic-checkpoint` |

**Live training runs go to `outputs/rl/<experiment>/` (gitignored; the badminton
tasks log to `src/simulation/badminton/logs/rsl_rl/<experiment>/`)** — `play.py`
with no `--checkpoint` loads the latest from there. Only promote a checkpoint
here when it's a baseline the team should share; keep it to one `.pt` per task.
