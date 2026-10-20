"""Render a scene to PNG, or use --interactive to view it without leader hardware.

    MUJOCO_GL=egl python -m humanoid_mujoco_scenes.preview --scene peg_insert --png peg.png
"""
from __future__ import annotations

import argparse

import mujoco

from humanoid_mujoco_scenes import list_scenes, make_model, scene_camera, scene_reset, scene_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="bare", help="scene name (an unknown name lists them)")
    parser.add_argument("--png", default="preview.png")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds simulated before the frame")
    parser.add_argument("--interactive", action="store_true", help="Open the MuJoCo viewer")
    args = parser.parse_args()
    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")

    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    model = make_model(args.scene)
    data = mujoco.MjData(model)
    set_home(model, data)
    reset_hook = scene_reset(args.scene)
    if args.interactive and reset_hook is not None:
        import numpy as np
        reset_hook(model, data, np.random.default_rng())
    mujoco.mj_forward(model, data)
    if args.interactive:
        import time
        from mujoco import viewer
        hook = scene_step(args.scene)
        substeps = max(1, round(0.01 / model.opt.timestep))
        camera = scene_camera(args.scene) or {}
        with viewer.launch_passive(model, data) as handle:
            for key, value in camera.items():
                setattr(handle.cam, key, value)
            while handle.is_running():
                tick = time.monotonic()
                with handle.lock():
                    if hook is not None:
                        hook(model, data)
                    mujoco.mj_step(model, data, nstep=substeps)
                handle.sync()
                time.sleep(max(0, substeps * model.opt.timestep - (time.monotonic() - tick)))
        print(f"Viewer stopped; simulated {data.time:.3f} seconds.")
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
