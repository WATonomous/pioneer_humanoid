"""Render a scene to PNG, or use --interactive to view it without leader hardware.

    MUJOCO_GL=egl python -m humanoid_mujoco_scenes.preview --scene peg_insert --png peg.png
"""
from __future__ import annotations

import argparse

import mujoco

from humanoid_mujoco_scenes import list_scenes, make_model, scene_camera, scene_reset, scene_step


def _launch_interactive(scene_name: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Run scene mechanics in the managed viewer's physics thread, including on macOS."""
    from mujoco import viewer

    hook = scene_step(scene_name)
    if hook is None:
        viewer.launch(model, data)
        return

    reset_hook = scene_reset(scene_name)
    previous_control = mujoco.get_mjcb_control()
    last_hook_time = float("-inf")
    last_observed_time = float("-inf")

    def control(current_model, current_data):
        nonlocal last_hook_time, last_observed_time
        if previous_control is not None:
            previous_control(current_model, current_data)
        if current_model is not model or current_data is not data:
            return  # The managed viewer also permits loading another model.
        now = current_data.time
        if now < last_observed_time:
            last_hook_time = float("-inf")
            if now == 0:
                from pioneer_humanoid.mujoco_bimanual_arm import set_home
                import numpy as np

                set_home(model, data)
                if reset_hook is not None:
                    reset_hook(model, data, np.random.default_rng())
                # The callback runs during actuation: refresh positions/velocities after reset
                # without recursively entering mj_forward and its control callback.
                mujoco.mj_fwdPosition(model, data)
                mujoco.mj_fwdVelocity(model, data)
        last_observed_time = now
        # Match teleop's 10 ms control cadence; paused mj_forward calls must not advance task state.
        if now - last_hook_time >= 0.01 - 1e-9:
            last_hook_time = now
            hook(current_model, current_data)

    mujoco.set_mjcb_control(control)
    try:
        viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(previous_control)


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
        _launch_interactive(args.scene, model, data)
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
