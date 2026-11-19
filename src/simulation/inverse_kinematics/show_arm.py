"""Open the 8-DoF bimanual arm (pioneer_bimanual_arm_8dof) in the MuJoCo viewer.

Needs only `mujoco` and `numpy` (no Isaac Lab). On macOS the viewer must be started with `mjpython`:

    mjpython show_arm.py            # arms hold the home pose
    mjpython show_arm.py --wave     # the left arm moves its shoulder and elbow back and forth

The model is built from the URDF by src/pioneer_humanoid/pioneer_humanoid/mujoco_bimanual_arm_8dof.py
(joint limits and PD position actuators included), so there is no XML file to load.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# <repo>/src/simulation/inverse_kinematics/show_arm.py -> <repo>/src
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "pioneer_humanoid"))

from pioneer_humanoid.arm_8dof_params import DEFAULT_JOINT_POS  # noqa: E402
from pioneer_humanoid.mujoco_bimanual_arm_8dof import arm_spec, set_home  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", action="store_true", help="move the left arm's shoulder pitch and elbow")
    args = parser.parse_args()

    model = arm_spec().compile()
    data = mujoco.MjData(model)
    set_home(model, data)          # every joint and its actuator target at the home pose
    mujoco.mj_forward(model, data)

    print(f"{model.nu} actuators. Left arm joints (limits in degrees):")
    for name in DEFAULT_JOINT_POS:
        if name.startswith("left_"):
            lo, hi = np.degrees(model.joint(name).range)
            print(f"  {name:24s} home {np.degrees(DEFAULT_JOINT_POS[name]):7.1f}   range {lo:7.1f} .. {hi:7.1f}")

    wave = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("left_shoulder_pitch", "left_elbow")}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            started = time.time()
            if args.wave:
                swing = 0.5 * np.sin(2 * np.pi * 0.25 * data.time)       # +-0.5 rad, one cycle every 4 s
                for name, actuator in wave.items():
                    data.ctrl[actuator] = DEFAULT_JOINT_POS[name] + swing
            mujoco.mj_step(model, data)
            viewer.sync()
            spare = model.opt.timestep - (time.time() - started)          # run in real time
            if spare > 0:
                time.sleep(spare)


if __name__ == "__main__":
    main()
