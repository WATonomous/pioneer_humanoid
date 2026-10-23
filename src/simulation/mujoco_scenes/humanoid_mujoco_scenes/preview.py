"""Render a scene to PNG, or use --interactive to view it without leader hardware.

    MUJOCO_GL=egl python -m humanoid_mujoco_scenes.preview --scene peg_insert --png peg.png
    mjpython -m humanoid_mujoco_scenes.preview --interactive --scene matcha  # macOS
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from humanoid_mujoco_scenes import list_scenes, make_model, scene_camera, scene_reset, scene_step


def _reset_scene(scene_name: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    mujoco.mj_resetData(model, data)
    set_home(model, data)
    reset = scene_reset(scene_name)
    if reset is not None:
        reset(model, data, np.random.default_rng())
    mujoco.mj_forward(model, data)


def _launch_interactive(scene_name: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Own the control loop: scene mechanics, physics, viewer synchronization."""
    from mujoco import viewer

    hook = scene_step(scene_name)
    paused, reset_requested = False, False

    def on_key(key: int) -> None:
        nonlocal paused, reset_requested
        if key == 32:  # Space
            paused = not paused
        elif key == 82:  # R
            reset_requested = True

    substeps = max(1, round(0.01 / model.opt.timestep))
    control_dt = substeps * model.opt.timestep
    last_sim_time = data.time
    with viewer.launch_passive(model, data, key_callback=on_key) as handle:
        with handle.lock():
            for key, value in (scene_camera(scene_name) or {}).items():
                setattr(handle.cam, key, value)
        while handle.is_running():
            started = time.monotonic()
            with handle.lock():
                # Native Reset also rewinds simulation time; initialize Python scene state too.
                if reset_requested or data.time < last_sim_time:
                    _reset_scene(scene_name, model, data)
                    reset_requested = False
                if not paused:
                    if hook is not None:
                        hook(model, data)
                    mujoco.mj_step(model, data, nstep=substeps)
                last_sim_time = data.time
            handle.sync()  # Publishes physics state and imports UI controls/perturbations.
            remaining = control_dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="bare", help="scene name (an unknown name lists them)")
    parser.add_argument("--png", default="preview.png")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds simulated before the frame")
    parser.add_argument("--interactive", action="store_true", help="Open the MuJoCo viewer (macOS: run with mjpython)")
    args = parser.parse_args()
    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")

    from pioneer_humanoid.arm_params import CAMERA_NAMES
    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    cameras = {name: (480, 640) for name in CAMERA_NAMES} if args.interactive else None
    model = make_model(args.scene, cameras=cameras)
    data = mujoco.MjData(model)
    if args.interactive:
        _reset_scene(args.scene, model, data)
        _launch_interactive(args.scene, model, data)
        return
    set_home(model, data)
    mujoco.mj_forward(model, data)
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
