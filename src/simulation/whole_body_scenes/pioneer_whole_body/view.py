"""Look at a scene in MuJoCo's interactive viewer, run its demo live, or write it out as an MJCF file.

    python -m pioneer_whole_body.view --scene bearing_block            # the scene, the robot standing (drag, zoom)
    python -m pioneer_whole_body.view --scene bearing_block --play     # the scripted demo, live (about real time)
    python -m pioneer_whole_body.view --scene bearing_block --tape out/bearing_block.npz   # a recorded run, looped
    python -m pioneer_whole_body.view --scene bare --export pioneer_v2.xml   # an MJCF any MuJoCo viewer opens

Don't open ``assets/pioneer_v2_humanoid/pioneer_humanoid_v2.urdf`` in MuJoCo's viewer directly: MuJoCo's URDF reader
puts the collision boxes in group 0 next to the CAD meshes (they hide them: a robot of grey boxes) and drops the
colours. ``assets/pioneer_v2_humanoid/pioneer_v2.xml`` (``--export`` of ``bare``) is the robot as these scenes use it.
"""
from __future__ import annotations

import argparse
import os
import re
import time

import mujoco

from . import entry, make_model, make_spec
from .robot import ASSET_DIR


def export(scene: str, path: str) -> str:
    """Write the scene as MJCF (static: the scripted mechanisms -- threads, latches -- live in the scene's Python)."""
    spec = make_spec(scene)
    xml = spec.to_xml()
    for body in spec.bodies:                         # bodies given a mass / inertia directly (the thread nuts): the
        if body.mass > 0 and body.name:              # XML writer leaves those out; put them back
            ipos = [0.0 if v != v else v for v in body.ipos]          # (unset: NaN)
            inertial = (f'<inertial pos="{" ".join(map(str, ipos))}" mass="{body.mass}" '
                        f'diaginertia="{" ".join(map(str, body.inertia))}"/>')
            xml = re.sub(rf'(<body name="{re.escape(body.name)}"[^>]*>)', rf"\1{inertial}", xml, count=1)
    xml = re.sub(r"\s*<default>\s*<default>\s*<default/>\s*</default>\s*</default>", "", xml)   # attach leftovers
    meshdir = os.path.relpath(ASSET_DIR, os.path.dirname(os.path.abspath(path)))
    xml = xml.replace('<compiler angle="radian"/>', f'<compiler angle="radian" meshdir="{meshdir}"/>', 1)
    with open(path, "w") as f:
        f.write(xml)
    mujoco.MjModel.from_xml_path(path)              # it loads
    return path


def _viewer(m, d):
    import mujoco.viewer
    v = mujoco.viewer.launch_passive(m, d)
    v.cam.lookat[:] = [0.25, 0.0, 1.0]
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.6, 200.0, -18.0
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bare")
    ap.add_argument("--play", action="store_true", help="run the scripted demo live")
    ap.add_argument("--tape", help="play back a recorded run (run_demo's PREFIX.npz), looped")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--export", help="write the scene as an MJCF file and exit")
    a = ap.parse_args()
    if a.export:
        print(export(a.scene, a.export))
        return

    e = entry(a.scene)
    m = make_model(a.scene)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    v = _viewer(m, d)

    if a.tape:
        from .video import Tape
        tape = Tape.load(a.tape)
        while v.is_running():
            t0 = time.time()
            for i, t in enumerate(tape.t):
                if not v.is_running():
                    return
                d.qpos[:] = tape.qpos[i]
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(max(0.0, t / a.speed - (time.time() - t0)))
        return

    if a.play:
        from .controller import WholeBody
        from .skills import TaskFailed
        t_wall = time.time()

        def frame(wb):                                  # keep the viewer in step with the simulation clock
            if not v.is_running():
                raise SystemExit
            v.sync()
            time.sleep(max(0.0, wb.t / a.speed - (time.time() - t_wall)))

        wb = WholeBody(m, d, scene_step=e.step, on_frame=frame, fps=30.0)
        try:
            e.plan(wb, {})
            print("done")
        except TaskFailed as err:
            print("failed:", err)
    else:                                               # just the scene: the robot holds its pose
        from .controller import WholeBody
        wb = WholeBody(m, d, scene_step=e.step)
        while v.is_running():
            for _ in range(5):
                wb.tick()
            v.sync()
            time.sleep(0.05)
    while v.is_running():
        v.sync()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
