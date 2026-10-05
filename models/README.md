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
| [`pioneer_humanoid_rough_no_stairs_selective_knee_shape/`](pioneer_humanoid_rough_no_stairs_selective_knee_shape/README.md) | `Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0` | normalized non-stair rough baseline (`model_4000.pt`), 19 active rewards; known stepping asymmetry |
| `badminton_teacher/` | `Mjlab-Badminton-Receive-Teacher` | privileged-state receive teacher (`model_5996.pt`), 98.5% bank hits |
| `badminton_student_ppo/` | `Mjlab-Badminton-Receive-Student-PPO` | EKF-perception receive policy (`model_4997.pt`), 98.8% bank hits, 78% net clearance; the deployable one |

**Live training runs go to `outputs/rl/<experiment>/` (gitignored; the badminton
tasks log to `src/simulation/badminton/logs/rsl_rl/<experiment>/`)** — `play.py`
with no `--checkpoint` loads the latest from there. Only promote a checkpoint
here when it's a baseline the team should share; keep it to one `.pt` per task.
