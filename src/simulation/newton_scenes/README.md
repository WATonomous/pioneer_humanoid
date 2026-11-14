# `newton_scenes/`

Soft-body scenes in [Newton](https://github.com/newton-physics/newton) (VBD solver, GPU). Plain scripts, not registered
with a teleop scene registry. Tested with newton 1.7.0, warp 1.19.0, mujoco 3.15.0, trimesh 5.2.0.

```bash
pip install "newton[examples]==1.7.0" trimesh "imageio[ffmpeg]"
# NVIDIA driver older than CUDA 13 (nvidia-smi says 12.x)? PyPI's warp is the CUDA 13 build, take the CUDA 12 one:
pip install --force-reinstall --no-deps https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl
```

- `wash_dish.py`: keyboard teleop. The right arm picks up a soft sponge and scrubs a dirty plate in a sink; the dirt
  wipes off where the sponge presses. Runs at about 2.5x real time on a laptop RTX 4060. Keys are in the file's header.
  `--demo out.mp4` runs a scripted grab-and-scrub headless instead.
- `sponge_grip.py`: the gripper squeezes a finely meshed sponge to a 3.5 cm jaw gap and lifts it 10 cm (about 3 min on
  a laptop GPU, not real time).
