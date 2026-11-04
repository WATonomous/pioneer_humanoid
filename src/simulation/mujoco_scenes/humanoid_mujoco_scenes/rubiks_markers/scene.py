"""Rubik's cube and two whiteboard markers in the workcell.

A 57 mm 3x3 cube (yellow on top) left of centre and an orange and a red whiteboard marker lying on their
sides to its right, all free bodies on the workcell's floor boards.
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene

T = TABLE_TOP_Z

CUBE = 0.057                     # 3x3 Rubik's cube edge
CUBE_POS = (0.53, 0.06)
CUBE_YAW = np.radians(32)      # green face turned toward the open front
# Rubik's colours: yellow top, orange toward the open front (-x), green on the left seen from the front (+y).
FACE_RGB = {
    "+z": [0.98, 0.85, 0.05], "-z": [0.95, 0.95, 0.95],
    "-x": [1.0, 0.42, 0.0], "+x": [0.85, 0.05, 0.08],
    "+y": [0.0, 0.6, 0.25], "-y": [0.05, 0.25, 0.85],
}
MARKER_LEN, MARKER_R = 0.135, 0.009


def _add_rubiks_cube(spec: mujoco.MjSpec) -> None:
    h = CUBE / 2
    body = spec.worldbody.add_body(name="rubiks_cube", pos=[*CUBE_POS, T + h], quat=[np.cos(CUBE_YAW / 2), 0, 0, np.sin(CUBE_YAW / 2)])
    body.add_freejoint(name="rubiks_cube")
    body.add_geom(name="rubiks_core", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h] * 3, mass=0.09,
                  rgba=[0.03, 0.03, 0.03, 1], friction=[0.8, 0.005, 0.0001])
    cell, sticker = CUBE / 3, CUBE / 3 * 0.42
    for face, rgb in FACE_RGB.items():
        sign, axis = (1 if face[0] == "+" else -1), "xyz".index(face[1])
        mat = spec.add_material(name=f"sticker{face}", rgba=[*rgb, 1], specular=0.5, shininess=0.6)
        others = [a for a in range(3) if a != axis]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                pos, size = [0.0] * 3, [0.0] * 3
                pos[axis], size[axis] = sign * (h + 0.0004), 0.0005
                pos[others[0]], pos[others[1]] = i * cell, j * cell
                size[others[0]] = size[others[1]] = sticker
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=mat.name,
                              contype=0, conaffinity=0, mass=0, group=1)


def _add_marker(spec: mujoco.MjSpec, name: str, pos, yaw: float, rgb) -> None:
    """Whiteboard marker lying on its side: white barrel, coloured cap at one end and plug at the other."""
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    body = spec.worldbody.add_body(name=name, pos=[*pos, T + MARKER_R + 0.0005], quat=list(q))
    body.add_freejoint(name=name)
    along = [0.70710678, 0, 0.70710678, 0]   # cylinder axis along the body's x
    half = MARKER_LEN / 2
    b0, b1 = -half + 0.04, half - 0.008   # barrel: from under the cap to the plug
    body.add_geom(name=f"{name}_barrel", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[MARKER_R, (b1 - b0) / 2, 0],
                  pos=[(b0 + b1) / 2, 0, 0], quat=along, rgba=[0.96, 0.96, 0.95, 1], mass=0.012,
                  friction=[0.6, 0.005, 0.0005], condim=6)   # rolling friction: a marker comes to rest
    body.add_geom(name=f"{name}_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[MARKER_R, 0.022, 0],
                  pos=[-half + 0.022, 0, 0], quat=along, rgba=[*rgb, 1], mass=0.004)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[MARKER_R * 0.9, 0, 0], pos=[-half + 0.006, 0, 0],
                  rgba=[*rgb, 1], contype=0, conaffinity=0, mass=0, group=1)
    body.add_geom(name=f"{name}_plug", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[MARKER_R * 0.75, 0.004, 0],
                  pos=[half - 0.004, 0, 0], quat=along, rgba=[*rgb, 1], mass=0.001)
    # printed label band
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[MARKER_R * 1.01, 0.016, 0], pos=[0.0, 0, 0],
                  quat=along, rgba=[*rgb, 1], contype=0, conaffinity=0, mass=0, group=1)


@scene("rubiks_markers", camera=dict(lookat=[0.5, 0.0, T], distance=1.9, azimuth=20, elevation=-30))
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    _add_rubiks_cube(spec)
    _add_marker(spec, "marker_orange", (0.565, -0.17), np.radians(-120), [1.0, 0.38, 0.02])
    _add_marker(spec, "marker_red", (0.485, -0.165), np.radians(-140), [0.88, 0.06, 0.08])
