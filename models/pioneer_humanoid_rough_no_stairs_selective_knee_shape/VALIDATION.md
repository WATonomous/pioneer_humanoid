# Model4000 handoff checks

Local preparation, 2026-10-05, using the provided Isaac Lab 2.3.2 / Isaac Sim 5.1
image. These checks establish package identity and wiring, not hardware safety
or reliable stair climbing.

- Checkpoint SHA256 matches the manifest; iteration is 4000.
- All 25 policy/normalizer tensors and 51 optimizer-state tensors are finite.
- All 19 active reward names, functions, weights, and parameters match the saved
  model4000 recipe. The later timing reward and clearance replacement are absent.
- Saved runner configuration and 26 asset/source fingerprints were verified.
- All 63 CPU tests passed: 26 knee-shaping, four numerical-guard, and 33 stairs
  geometry/configuration/curriculum tests. No tests were skipped in the image.
- The real stairs validator passed registration, Hydra serialization round-trip,
  generated mesh profiles, baseline preservation, and observation normalization.
  Two environments then ran 20 zero-action physics steps with finite
  observations, rewards, and terrain-contact data (235 observations/12 actions).
- The repository's actual RSL playback workflow loaded this raw checkpoint with
  the matching SelectiveKneeShape Play task, exported JIT and ONNX policies, and
  ran five headless inference steps. The checkpoint hash remained unchanged.
- A separate native-viewer demonstration was recorded with the hash-verified
  model4000 export and inspected at 1, 10, and 18 seconds. The sharing MP4 is
  20.00 seconds, 1,000 frames at 50 fps, 1280×720 H.264. The native recording
  omitted one frame; the sharing version holds its final frame for 0.02 seconds.
  The video is an attachment, not a checked-in simulation result.

The local WSL/Docker headless runs logged graphics initialization and CUDA
shutdown warnings, but their physics/runner assertions completed and processes
exited with status 0. Docker GUI rendering and long-run stability are not thereby
validated. The native demonstration does not prove general rough-terrain success.

Repeat the public CPU tests from the repository root with CPU Torch available:

```bash
CUDA_VISIBLE_DEVICES="" "$PYTHON" -m unittest discover \
  -s src/simulation/humanoid_rl_tasks/tests -p 'test_*.py' -v
```

For the GPU-idle real Isaac stairs validator and eventual locomotion pilot,
follow [STAIRS.md](../../src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion/STAIRS.md).
