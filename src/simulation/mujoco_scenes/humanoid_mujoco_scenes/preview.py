"""Render a scene to PNG with the arm at home — no viewer, no leader. Checks scenes on a headless box.

    MUJOCO_GL=egl python -m humanoid_mujoco_scenes.preview --scene peg_insert --png peg.png
"""
from __future__ import annotations

import argparse

import mujoco

from humanoid_mujoco_scenes import list_scenes, make_model, scene_camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="bare", help="scene name (an unknown name lists them)")
    parser.add_argument("--png", default="preview.png")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds simulated before the frame")
    args = parser.parse_args()
    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")

    from PIL import Image
    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    model = make_model(args.scene)
    data = mujoco.MjData(model)
    set_home(model, data)
    mujoco.mj_step(model, data, nstep=int(args.settle / model.opt.timestep))

    camera = mujoco.MjvCamera()
    for key, value in (scene_camera(args.scene) or {}).items():
        setattr(camera, key, value)
    with mujoco.Renderer(model, 960, 1280) as renderer:
        renderer.update_scene(data, camera)
        Image.fromarray(renderer.render()).save(args.png)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
