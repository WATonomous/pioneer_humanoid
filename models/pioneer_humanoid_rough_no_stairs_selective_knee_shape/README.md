# Pioneer humanoid: rough terrain, model4000

Frozen simulation baseline with the original 19 active reward terms. This is the
4,000-index checkpoint selected for the team handoff, not either subsequent
swing-timing or normalized swing-clearance experiment. Existing flat/rough
checkpoint files are preserved. The shared robot now has corrected knee geometry
and nominal stance; the older model3900/model2999 checkpoints have not been
validated against this changed geometry.

| Item | Value |
| --- | --- |
| Checkpoint | `model_4000.pt` (6,890,357 bytes) |
| SHA256 | `d91d583dd73180458c82896dd1cf696e7cdb8d64f393c4604b2a9521d8c8fc03` |
| Train task | `Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0` |
| Play task | `Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0` |
| Stack | provided `simulation_isaac` Docker image; Isaac Lab 2.3.2 / Sim 5.1 / RSL-RL 3.1.2 |
| Training | fresh initialization, seed 43, 6,144 environments, 4,001 updates |
| Network | actor/critic 512/256/128 ELU, both observation normalizers enabled |
| Interface | 235 observation values, 12 joint-position actions, 20 ms policy step |

`Isaac-Velocity-...` names are aliases of the same task configurations. Use the
**SelectiveKneeShape** task with the raw checkpoint: it enables both observation
normalizers and preserves the saved training recipe. Do not substitute an older
unnormalized rough/flat runner.

The portable [manifest](manifest.json) records all 19 active rewards, their
parameters, action/observation ordering, and asset/source fingerprints.
[params/agent.yaml](params/agent.yaml) records the saved runner settings; it is
provenance, not a file automatically loaded by `rl-play` or `rl-train`. Task
registrations provide the actual runtime configuration.

See [VALIDATION.md](VALIDATION.md) for the completed package, CPU, and short
headless runner/physics checks and their limits.

## View with the provided Docker setup

First use the normal watod setup in
[the simulation quickstart](../../docker/simulation/isaac_lab/QUICKSTART.md), then
enter the running `simulation_isaac_dev` service. Inside the container:

```bash
cd "$HUMANOID_ROOT"
sha256sum models/pioneer_humanoid_rough_no_stairs_selective_knee_shape/model_4000.pt
rl-play \
  --task=Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0 \
  --checkpoint="$HUMANOID_ROOT/models/pioneer_humanoid_rough_no_stairs_selective_knee_shape/model_4000.pt" \
  --num_envs=9 --real-time
```

Use `--num_envs=1` for a closer view. A working host display is required for GUI
playback; the simulator runs in the provided Docker environment. Native Windows
viewer launchers and laptop-specific workarounds are deliberately not part of
this bundle. Runtime exports and videos beside the frozen checkpoint are ignored;
playback does not alter the checkpoint.

## Train the same recipe from scratch

This is optional future work; it does not happen when playing the frozen policy.
Inside the container:

```bash
cd "$HUMANOID_ROOT"
rl-train \
  --task=Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0 \
  --headless --num_envs=6144 --seed=43 --max_iterations=4001 --fail_on_nonfinite_grad \
  --run_name=model4000_recipe_reproduction \
  agent.experiment_name=pioneer_humanoid_rough_no_stairs_selective_knee_shape_overnight \
  agent.save_interval=250
```

This preserves the saved population and recipe, not bit-for-bit determinism across
GPU hardware or library versions. Reduce the environment count if memory is
limited; that changes the training population. The original run used a fail-fast
non-finite-gradient check, recorded in the manifest. Stop and diagnose any
non-finite-gradient failure; do not disable that check to keep a corrupt run going.

## Continue this checkpoint

Resume restores model/normalizer/optimizer state; it is not a fresh experiment.
The current trainer discovers resume checkpoints under `outputs/rl/`, so copy the
frozen baseline into an exact named run folder first. This leaves the shared model
untouched:

```bash
cd "$HUMANOID_ROOT"
resume_dir="$HUMANOID_ROOT/outputs/rl/pioneer_humanoid_rough_no_stairs_selective_knee_shape/frozen_model4000"
mkdir -p "$resume_dir"
cp -n models/pioneer_humanoid_rough_no_stairs_selective_knee_shape/model_4000.pt "$resume_dir/model_4000.pt"
sha256sum "$resume_dir/model_4000.pt"
rl-train \
  --task=Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0 \
  --headless --num_envs=6144 --seed=43 --max_iterations=1000 --fail_on_nonfinite_grad \
  --resume=True --load_run=frozen_model4000 --checkpoint=model_4000.pt \
  --run_name=model4000_continuation \
  agent.save_interval=250
```

Check the copied file's hash against the table before continuing; `cp -n` does
not overwrite an existing copy. `--max_iterations=1000` requests 1,000 additional
updates. Do not use implicit "latest checkpoint" selection for this handoff.

## Limitations and next work

The policy walks flat and non-stair rough terrain, but still has visible left/right
stepping asymmetry and can stumble on difficult elevation changes. It is a useful
simulation baseline, not a gait-quality guarantee or hardware-validated controller.
The training terrain mix is 20% plane, 40% random rough, 20% boxes, and 10% each
up/down pyramid slopes; it does not include stair terrain.

Selective self-collision enables only foot/opposite-calf pairs, not every robot
link pair. The knee coordinates use negative flexion. Preserve the matching URDF,
nominal pose, action scales, knee target bounds, observation normalizers, and
terrain-contact sensors when comparing this policy. Do not remove existing reward
terms just to reduce their count; changing the recipe requires a new experiment.

The [stairs task handoff](../../src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion/STAIRS.md)
is a separate training scaffold. This checkpoint is not a trained stairs policy.
