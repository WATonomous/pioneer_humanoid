"""Look at env.reset(): shows the arm in the MuJoCo viewer and resets to a new start + goal every few seconds.

Run it from the inverse_kinematics folder. On macOS the viewer needs mjpython:
    mjpython tests/view_reset.py                  (a new goal every 3 s, until you close the window)
    mjpython tests/view_reset.py --every 5 --seed 2 --physics

Drawn: the GOAL as a big red sphere with its three axes (red x, green y, blue z), and the arm's current TOOL POINT
as a small cyan sphere with its axes. When the cyan point sits on the red one with the axes lined up, the goal is reached.
Without --physics the arm stays exactly as reset() left it. With --physics the simulation runs, so you see it hold
(or sag, or move) under the PD controller.
This does not change env.py: it only stands in for _observation() when that isn't written yet.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env import PolicyIKEnv
from metrics import CONTROL_DT

AXIS_RGBA = ([1.0, 0.1, 0.1, 1.0], [0.1, 0.9, 0.1, 1.0], [0.2, 0.3, 1.0, 1.0])   # x red, y green, z blue


def add_frame(scn, p, R, size, sphere_rgba, axis_len):
    """Add a sphere at p plus three axis arrows (the columns of R) to the user scene."""
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]), p, np.eye(3).ravel(), np.array(sphere_rgba))
    scn.ngeom += 1
    for i in range(3):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(AXIS_RGBA[i]))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, size * 0.35, p, p + axis_len * R[:, i])
        scn.ngeom += 1


def draw(env, scn):
    scn.ngeom = 0
    p_t = env.data.site_xpos[env.arm.tcp_id]
    R_t = env.data.site_xmat[env.arm.tcp_id].reshape(3, 3)
    add_frame(scn, env.p_goal, env.R_goal, 0.02, [1.0, 0.1, 0.1, 0.45], 0.12)      # the goal
    add_frame(scn, p_t, R_t, 0.01, [0.0, 0.9, 0.9, 0.9], 0.08)                      # where the tool point is now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=3.0, help="seconds between resets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--physics", action="store_true", help="run the simulation between resets")
    ap.add_argument("--max-resets", type=int, default=0, help="stop after this many resets (0 = until the window closes)")
    args = ap.parse_args()

    env = PolicyIKEnv(seed=args.seed)
    try:
        env.reset()
    except Exception as err:    # _observation() isn't finished yet (or has a bug): use a stand-in so reset() still works
        print(f"note: _observation() failed ({type(err).__name__}: {err}); using a stand-in observation")
        env._observation = lambda: np.zeros(1)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.15, -0.1]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth, viewer.cam.elevation = 135, -20
        n = 0
        while viewer.is_running() and (args.max_resets == 0 or n < args.max_resets):
            env.reset()
            n += 1
            with viewer.lock():
                draw(env, viewer.user_scn)
            viewer.sync()
            print(f"reset {n}: rho0 = {env.reward.best:7.1f}   goal at {np.round(env.p_goal, 3)}")

            t_end = time.time() + args.every
            while viewer.is_running() and time.time() < t_end:
                if args.physics:
                    env.sim.step(env.z)                                     # one control step, holding the reference z
                    mujoco.mj_forward(env.model, env.data)                  # Sim.step leaves site positions one substep old
                    with viewer.lock():
                        draw(env, viewer.user_scn)
                    viewer.sync()
                    time.sleep(CONTROL_DT)                                  # about real time
                else:
                    time.sleep(0.05)


if __name__ == "__main__":
    main()
