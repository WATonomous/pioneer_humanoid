# humanoid_rl

The RL runners for every task in [`humanoid_rl_tasks`](../humanoid_rl_tasks): PPO training (RSL-RL), playback with video, and diagnostics, in `humanoid_rl/scripts/`.

| Command | Script | Does |
|---|---|---|
| `rl-train` | `train.py` | Trains a task; checkpoints and logs land in `$HUMANOID_ROOT/outputs/rl/` |
| `rl-play` | `play.py` | Runs a checkpoint (`--video` records a clip) |
| | `diagnose_*.py` | Joint, spawn and rollout checks |

Run inside the `simulation_isaac` container, from `$HUMANOID_ROOT`:

```bash
rl-train --task=Isaac-Locomotion-Flat-PioneerLegOnly-v0 --headless
rl-play  --task=Isaac-Locomotion-Flat-PioneerLegOnly-Play-v0 --num_envs 1 --video --checkpoint models/pioneer_leg_only_flat/model_2999.pt
```

Walking tasks, trained models and sim2sim: [Pioneer walking](../humanoid_rl_tasks/humanoid_rl_tasks/locomotion/config/pioneer/README.md).
