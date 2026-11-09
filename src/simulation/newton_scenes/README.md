# `newton_scenes/`

Soft-body scenes in [Newton](https://github.com/newton-physics/newton) (VBD solver, GPU). Plain scripts, not registered
with a teleop scene registry. Tested with newton 1.7.0, warp 1.19.0, mujoco 3.15.0, trimesh 5.2.0.

```bash
pip install newton==1.7.0 mujoco trimesh imageio[ffmpeg]
python sponge_grip.py sponge.mp4   # ~3 min on a laptop GPU; also writes sponge.png
```

- `sponge_grip.py`: the Pioneer gripper squeezes a kitchen sponge to a 3.5 cm jaw gap and lifts it 10 cm.
