# Native Windows policy viewer

Use this source-only helper when WSL/Docker headless training works but the
reviewer needs a native Windows GUI. Export on CPU in the team's existing Docker
environment; run live physics and rendering in native Windows Isaac Lab.
Prefer matching **Isaac Lab 2.3.2 / Isaac Sim 5.1** on both sides. This contribution
includes no trained policy, checkpoint, or training run.

Supported play tasks are
`Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0` and
`Isaac-Locomotion-Stairs-PioneerHumanoid-Play-v0`. Equivalent `Isaac-Velocity-*`
aliases are accepted. Legacy Flat/plain Rough tasks are not supported by this
helper. The task label alone cannot prove that a checkpoint was trained with
the matching recipe, or that it can walk or climb stairs.

## 1. Export your own trusted checkpoint

From the host, enter the existing team development container with
`./watod -t simulation_isaac`; do not recreate it or interrupt another run.
The following commands run **inside** that container. Use a checkpoint you own
or trust: loading a raw PyTorch checkpoint can execute code.

```bash
cd "$HUMANOID_ROOT"
POLICY_CHECKPOINT="$HUMANOID_ROOT/outputs/rl/pioneer_humanoid_rough_no_stairs_selective_knee_shape/<run>/model_<iteration>.pt"
POLICY_TASK="Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0"
CUDA_VISIBLE_DEVICES="" "$ISAACLAB/isaaclab.sh" -p -u "$RL_RUNNERS/export_policy.py" \
  --checkpoint "$POLICY_CHECKPOINT" --task "$POLICY_TASK" \
  --output outputs/native_viewer/policy.pt \
  --metadata outputs/native_viewer/policy.json
```

Replace the placeholders with the exact saved checkpoint. For a checkpoint
actually trained on the stairs recipe, use its path and set `POLICY_TASK` to
`Isaac-Locomotion-Stairs-PioneerHumanoid-Play-v0` before exporting. Changing the
label does not turn a rough-terrain checkpoint into a stairs policy.

The exporter reconstructs the actor and saved observation normalizers on CPU;
it does not launch Isaac Sim or train. It produces TorchScript plus a **version-3
JSON manifest**, including policy/asset hashes and an inference probe. Create a
fresh export with this source; older local manifests are not accepted.
Existing destinations are protected; choose a new output name or add
`--overwrite` to intentionally replace an earlier export. The exporter uses
restricted `weights_only=True` loading, with no unsafe-pickle fallback.

Share the raw checkpoint separately from the generated `policy.pt` and
`policy.json` pair. Native playback consumes only that pair, not the raw
checkpoint. Copy both files into the recipient's `outputs/native_viewer/`, using
the same PR source and robot assets. Checkpoints, exports, logs, and recordings
belong under git-ignored `outputs/`; none are committed or bundled with the PR.

## 2. Open the native Windows GUI

Install native Windows Isaac Lab with its Isaac Sim environment first. Set
`ISAACLAB_BAT` to your own installation's `isaaclab.bat` in a **Windows** shell.
Replace `Ubuntu`, `your-user`, and the installation path below. A normal local
Windows checkout works too; replace the UNC launcher path with its local path.

Command Prompt:

```bat
set "ISAACLAB_BAT=C:\path\to\IsaacLab\isaaclab.bat"
call "\\wsl.localhost\Ubuntu\home\your-user\humanoid\src\simulation\humanoid_rl\humanoid_rl\scripts\view_native_windows.cmd" ^
  --policy outputs\native_viewer\policy.pt ^
  --metadata outputs\native_viewer\policy.json ^
  --task Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0 ^
  --num_envs 1 --seed 42 --command 0.5 0 0
```

PowerShell:

```powershell
$env:ISAACLAB_BAT = 'C:\path\to\IsaacLab\isaaclab.bat'
$viewer = '\\wsl.localhost\Ubuntu\home\your-user\humanoid\src\simulation\humanoid_rl\humanoid_rl\scripts\view_native_windows.cmd'
& $viewer --policy 'outputs\native_viewer\policy.pt' --metadata 'outputs\native_viewer\policy.json' `
  --task 'Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0' `
  --num_envs 1 --seed 42 --command 0.5 0 0
```

For stairs playback, substitute the matching stairs play ID used for export.
The launcher temporarily maps UNC paths with `pushd`, prepends this checkout's
three Python source packages to `PYTHONPATH`, waits for the viewer, returns its
upstream launcher's exit code, and removes the temporary mapping. Some Isaac Lab
batch versions return zero even after a Python error; inspect the console log,
not just the exit code. It installs no machine-local
helpers. Relative policy, manifest, and video paths are repository-root-relative.
The mapped-drive spelling is needed for USD's relative payloads; keep the launcher
running until the viewer exits.

Do **not** add `--headless` or `--kit_args=--/app/vulkan=false` for this native GUI
path. Keep WSL/Docker headless checks separate from native Windows viewing, and
wait until the GPU is available before opening another simulator.

## Controls, recording, and limits

`--max_steps 0` is the default: run until the window closes or you press **Q** or
**Escape** with the Isaac viewport focused. Add `--max_steps 1000` for a finite
smoke test. `--num_envs 9` gives an overview; `--command VX VY YAW_RATE` sets the
fixed velocity command. `--flat_terrain` is a rough-task-only diagnostic override,
not a different trained policy or a valid option for the stairs task.

To record, append
`--video --video_seconds 20 --video_folder outputs/native_viewer/videos` to a
native launch. Recording enables cameras and stops after the requested simulated
duration; videos stay under ignored `outputs/`.

On failure, check the Windows `ISAACLAB_BAT` path, matching task/manifest, both
exported files, and unchanged robot USD assets. Do not bypass stale-manifest or
hash failures; export again from the matching trusted checkpoint and source.
Send back the full console log, exact command, and Lab/Sim versions.

Run the helper's CPU/source tests inside the existing container:

```bash
CUDA_VISIBLE_DEVICES="" "$ISAACLAB/isaaclab.sh" -p -m unittest discover \
  -s src/simulation/humanoid_rl/tests -v
```

CPU export/probe and source tests validate packaging and inference, not the real
Windows launcher, GPU rendering, or locomotion. Native GUI execution requires
separate Windows verification. A passing inference probe does not guarantee
identical cross-platform physics, stable walking, or successful stair completion.
