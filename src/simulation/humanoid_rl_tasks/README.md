# humanoid_rl_tasks

The Isaac Lab RL tasks: one folder per task under `humanoid_rl_tasks/` (`locomotion`, `inhand`, `push_block`, `pick_place`, `badminton`), each with its env config, rewards and gym registration. Importing the package registers every task. Train and play them with [`humanoid_rl`](../humanoid_rl).

Locomotion (`humanoid_rl_tasks/locomotion/`):
- [Pioneer walking](humanoid_rl_tasks/locomotion/config/pioneer/README.md): leg-only and whole-body Pioneer, flat and uneven ground, trained models, sim2sim in MuJoCo.
- [Old leg-only humanoid](humanoid_rl_tasks/locomotion/locomotion.md) (`config/pioneer_humanoid_v1`) and its [stairs](humanoid_rl_tasks/locomotion/STAIRS.md).

`scripts/` and `tests/`: checks for the old leg-only humanoid's stairs task.
