"""Render a scene to PNG, or use --interactive to view it without leader hardware.

    MUJOCO_GL=egl python -m humanoid_mujoco_scenes.preview --scene peg_insert --png peg.png
"""
from __future__ import annotations

import argparse

import mujoco

from humanoid_mujoco_scenes import list_scenes, make_model, scene_camera, scene_reset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="bare", help="scene name (an unknown name lists them)")
    parser.add_argument("--png", default="preview.png")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds simulated before the frame")
    parser.add_argument("--interactive", action="store_true", help="Open the MuJoCo viewer")
    args = parser.parse_args()
    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")

    from pioneer_humanoid.arm_params import CAMERA_NAMES
    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    cameras = {name: (480, 640) for name in CAMERA_NAMES} if args.interactive else None
    model = make_model(args.scene, cameras=cameras)
    data = mujoco.MjData(model)
    set_home(model, data)
    reset_hook = scene_reset(args.scene)
    if args.interactive and reset_hook is not None:
        import numpy as np
        reset_hook(model, data, np.random.default_rng())
    mujoco.mj_forward(model, data)
    if args.interactive:
        from mujoco import viewer

        viewer.launch(model, data, show_right_ui=False)
        return
    mujoco.mj_step(model, data, nstep=int(args.settle / model.opt.timestep))

    camera = mujoco.MjvCamera()
    for key, value in (scene_camera(args.scene) or {}).items():
        setattr(camera, key, value)
    from PIL import Image

    with mujoco.Renderer(model, 960, 1280) as renderer:
        renderer.update_scene(data, camera)
        Image.fromarray(renderer.render()).save(args.png)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
