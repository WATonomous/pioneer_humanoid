"""MuJoCo-native version of the shared Isaac lightbox workcell."""
from __future__ import annotations

from pathlib import Path

import mujoco

from ._register import add_floor

_PROPS = Path(__file__).resolve().parents[4] / "assets" / "props"
_LIGHTBOX_MESH = str(_PROPS / "lightbox.obj")

ROBOT_STAND_LIFT_Z = 1.1997
ROBOT_BASE_X = 0.15
ROBOT_BASE_POS = (ROBOT_BASE_X, 0.0, ROBOT_STAND_LIFT_Z)

TABLE_HEIGHT_M = 30.5 * 0.0254
TABLE_TOP_Z = TABLE_HEIGHT_M
TABLE_CENTER_X = 0.69
TABLE_CENTER_Y = 0.00612
TABLE_DEPTH = 24.0 * 0.0254
TABLE_WIDTH = 60.0 * 0.0254
TABLE_X_MIN = TABLE_CENTER_X - TABLE_DEPTH / 2
TABLE_X_MAX = TABLE_CENTER_X + TABLE_DEPTH / 2
TABLE_Y_HALF = TABLE_WIDTH / 2
_TABLE_TOP_THICKNESS = 0.04

_LIGHTBOX_POS = (TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z)
_LIGHTBOX_ROT = (0.0, 0.70710678, -0.70710678, 0.0)
_LIGHTBOX_SCALE = (1.0, 1.0, 30.5 / 30.0)
WORKCELL_CAMERA = dict(
    lookat=[0.55, 0.0, 0.9],
    distance=2.4,
    azimuth=319,
    elevation=-20,
)


def add_lightbox_workcell(spec: mujoco.MjSpec) -> None:
    """Add the shared floor, CAD table, lightbox, and lighting to ``spec``."""
    add_floor(spec)
    lightbox = spec.add_mesh(name="workcell_lightbox_mesh", file=_LIGHTBOX_MESH, scale=_LIGHTBOX_SCALE)
    spec.worldbody.add_geom(
        name="workcell_table_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TABLE_DEPTH / 2, TABLE_WIDTH / 2, _TABLE_TOP_THICKNESS / 2],
        pos=[TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z - _TABLE_TOP_THICKNESS / 2],
        rgba=[0.0, 0.0, 0.0, 0.0],
    )
    spec.worldbody.add_geom(
        name="workcell_lightbox",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=lightbox.name,
        pos=_LIGHTBOX_POS,
        quat=_LIGHTBOX_ROT,
        rgba=[0.9, 0.9, 0.9, 1.0],
        contype=0,
        conaffinity=0,
    )
