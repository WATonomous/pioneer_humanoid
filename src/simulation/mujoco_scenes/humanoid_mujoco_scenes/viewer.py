"""View any registered Pioneer MuJoCo scene, without leader hardware.

macOS: mjpython -m humanoid_mujoco_scenes.viewer --scene peg_insert
Linux / Windows: python -m humanoid_mujoco_scenes.viewer --scene peg_insert
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from humanoid_mujoco_scenes import (
    list_scenes, make_model, scene_camera, scene_reset, scene_step,
)
from pioneer_humanoid.mujoco_bimanual_arm import set_home


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="bare", help="Registered scene name (default: bare)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Close after this many seconds (for a launch check)")
    args = parser.parse_args()
    if args.scene not in list_scenes():
        parser.error(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")

    model = make_model(args.scene)
    data = mujoco.MjData(model)
    reset_hook = scene_reset(args.scene)
    step_hook = scene_step(args.scene)
    set_home(model, data)
    if reset_hook:
        reset_hook(model, data, np.random.default_rng())
    mujoco.mj_forward(model, data)
    substeps = max(1, round(0.01 / model.opt.timestep))
    control_dt = substeps * model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for key, value in (scene_camera(args.scene) or {}).items():
            setattr(viewer.cam, key, value)
        print(f"Viewer opened: {args.scene}. Close the window to stop.", flush=True)
        launched = time.monotonic()
        while viewer.is_running():
            start = time.monotonic()
            if args.seconds is not None and start - launched >= args.seconds:
                break
            with viewer.lock():
                if step_hook:
                    step_hook(model, data)
                mujoco.mj_step(model, data, nstep=substeps)
            viewer.sync()
            time.sleep(max(0.0, control_dt - (time.monotonic() - start)))
        print(f"Viewer stopped; simulated {data.time:.3f} seconds.", flush=True)


if __name__ == "__main__":
    main()
