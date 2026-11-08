"""What the controller is asking for, drawn in the MuJoCo viewer.

Text (top-left, small): the reset key. Markers: a sphere at each active arm's wrist target with a
line from the wrist, coloured by lead used.
"""
from __future__ import annotations

import numpy as np

TARGET_RADIUS = 0.015  # m
LINE_WIDTH = 3.0       # px
LEAD_WARN = 0.7        # fraction of the lead limit where the marker turns orange; 1 = target pinned (red)
COLOURS = {"ok": (0.2, 0.9, 0.4, 0.8), "warn": (1.0, 0.6, 0.1, 0.8), "pinned": (1.0, 0.15, 0.15, 0.9)}


def show_text(viewer, text: str) -> None:
    import mujoco

    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_50, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ""))


def draw_targets(viewer, model, data, driver, joint_positions: dict) -> None:
    """Replace the viewer's user geoms with one target sphere and wrist line per active arm."""
    import mujoco

    base = model.body("base_link").id
    rotation, origin = data.xmat[base].reshape(3, 3), data.xpos[base]
    scene = viewer.user_scn
    scene.ngeom = 0
    for side, q in joint_positions.items():
        target = driver.targets[side]
        if not target.active or target.position is None or scene.ngeom + 2 > scene.maxgeom:
            continue
        lead = target.saturation
        colour = np.array(COLOURS["pinned" if lead >= 1 else "warn" if lead >= LEAD_WARN else "ok"], dtype=np.float32)
        wrist = driver.solvers[side].kinematics(q)[0]
        goal, wrist = origin + rotation @ target.position, origin + rotation @ wrist
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, TARGET_RADIUS),
                            goal, np.eye(3).ravel(), colour)
        mujoco.mjv_connector(scene.geoms[scene.ngeom + 1], mujoco.mjtGeom.mjGEOM_LINE, LINE_WIDTH, wrist, goal)
        scene.geoms[scene.ngeom + 1].rgba = colour
        scene.ngeom += 2
