"""Lightbox workcell for MuJoCo: the room every manipulation scene builds on, as in Isaac.

The same rig as Isaac's ``humanoid_rl_tasks.workcell.LightboxWorkcellCfg`` (not imported: that module
needs Isaac): the arm on its floor stand in front of the 30.5-inch table inside the lightbox. The visible
lightbox and table are the Onshape CAD (``assets/props/lightbox.usd``, converted to the STLs in
``assets/props/lightbox_mujoco/``), visual only like in Isaac; the table top collides as a plain box.

MuJoCo scenes keep the arm base at the world origin (``make_model``'s default ``robot_pos``), so every
Isaac x here is shifted by Isaac's ``ROBOT_BASE_X``: the table is where it is relative to the arm.

A scene calls ``add_workcell(spec)`` and puts its objects on ``TABLE_TOP_Z``, inside ``TABLE_X`` and
``TABLE_HALF_Y``.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from ._register import add_floor

MESH_DIR = Path(__file__).resolve().parents[4] / "assets" / "props" / "lightbox_mujoco"

_ISAAC_ROBOT_BASE_X = 0.15          # Isaac's ROBOT_BASE_X; MuJoCo's arm base is at x = 0

# Table: 30.5 inches high, 24 deep (X) by 60 wide (Y), centred where Isaac's lightbox is. Objects stand on the
# lightbox's white bottom panel (3/16 in, scaled like the rest of the CAD), not on the table itself.
TABLE_HEIGHT = 30.5 * 0.0254
BOTTOM_PANEL = 0.1875 * 0.0254 * 30.5 / 30.0
TABLE_TOP_Z = TABLE_HEIGHT + BOTTOM_PANEL      # the work surface
TABLE_CENTER = (0.69 - _ISAAC_ROBOT_BASE_X, 0.00612)
TABLE_X = (TABLE_CENTER[0] - 12 * 0.0254, TABLE_CENTER[0] + 12 * 0.0254)   # front (robot side), back edge
TABLE_HALF_Y = 30 * 0.0254

# Lightbox CAD: X = table width, Y = depth, vertical is -Z, origin at the table top's centre. Same pose
# and scale as Isaac's LIGHTBOX_ROT / LIGHTBOX_SCALE: width along world Y, open face toward the robot.
LIGHTBOX_POS = (*TABLE_CENTER, TABLE_HEIGHT)
LIGHTBOX_QUAT = (0.0, 0.70710678, -0.70710678, 0.0)   # wxyz
LIGHTBOX_SCALE = (1.0, 1.0, 30.5 / 30.0)              # CAD table is 30 in; the real one is 30.5 in
LIGHTBOX_TOP_Z = 1.456   # top of the lightbox frame (world z)

# The box is lit by one light tube (~1 m) hung under the top front rail, slightly right of centre seen
# from the front, diffuser facing down; the white foam-board walls bounce the rest. MuJoCo has
# no area lights, so the tube is a row of wide spotlights along it (soft, overlapping shadows) plus raised
# ambient for the bounce. The CAD's two round fill-light housings in the top corners are not lit.
TUBE_X, TUBE_Y, TUBE_Z = 0.30, -0.06, 1.385
TUBE_HALF_LEN, TUBE_R = 0.48, 0.03
TUBE_LIGHTS = 7                  # spotlights along the tube (with the headlight, OpenGL's 8)
TUBE_DIFFUSE = 0.12              # each; ~0.85 total


def add_workcell(spec: mujoco.MjSpec) -> None:
    """Floor, the lightbox (visual CAD), its tube light and the table top's collision box."""
    add_floor(spec, light=False)   # the lightbox's top panel would shade the table from an overhead light
    world = spec.worldbody
    colors = json.loads((MESH_DIR / "colors.json").read_text())
    for part, rgb in colors.items():
        spec.add_mesh(name=f"lightbox_{part}", file=str(MESH_DIR / f"{part}.stl"), scale=list(LIGHTBOX_SCALE),
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        spec.add_material(name=f"lightbox_{part}", rgba=[*rgb, 1], specular=0.2 if part == "table" else 0.1)
        world.add_geom(name=f"lightbox_{part}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"lightbox_{part}",
                       material=f"lightbox_{part}", pos=list(LIGHTBOX_POS), quat=list(LIGHTBOX_QUAT),
                       contype=0, conaffinity=0, group=1)

    # Table + bottom panel collision (invisible; the CAD above is what you see). Solid down to the floor.
    (x0, x1), (cx, cy) = TABLE_X, TABLE_CENTER
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, TABLE_HALF_Y, TABLE_TOP_Z / 2],
                   pos=[cx, cy, TABLE_TOP_Z / 2], rgba=[0, 0, 0, 0], group=3)
    _add_tube_light(spec)


def _add_tube_light(spec: mujoco.MjSpec) -> None:
    world = spec.worldbody
    spec.add_material(name="tube_housing", rgba=[0.04, 0.04, 0.045, 1], specular=0.3, shininess=0.4)
    spec.add_material(name="tube_diffuser", rgba=[1, 1, 1, 1], emission=1.0)
    rail = LIGHTBOX_TOP_Z - 0.02
    # housing (along y), diffuser strip on its underside, and two hanger clips up to the rail
    world.add_geom(name="tube_housing", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[TUBE_R, TUBE_HALF_LEN, 0],
                   pos=[TUBE_X, TUBE_Y, TUBE_Z], quat=[0.70710678, 0.70710678, 0, 0], material="tube_housing",
                   contype=0, conaffinity=0, group=1)
    world.add_geom(name="tube_diffuser", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.022, TUBE_HALF_LEN, 0.006],
                   pos=[TUBE_X + 0.004, TUBE_Y, TUBE_Z - TUBE_R + 0.003], material="tube_diffuser",
                   contype=0, conaffinity=0, group=1)
    for dy in (-0.33, 0.33):
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.03, (rail - TUBE_Z - TUBE_R) / 2 + 0.002],
                       pos=[TUBE_X, TUBE_Y + dy, (rail + TUBE_Z + TUBE_R) / 2], material="tube_housing",
                       contype=0, conaffinity=0, group=1)
    # the light itself: spots along the tube, down and a little toward the back wall
    for i, y in enumerate(np.linspace(-TUBE_HALF_LEN, TUBE_HALF_LEN, TUBE_LIGHTS)):
        world.add_light(name=f"tube_light{i}", pos=[TUBE_X + 0.01, TUBE_Y + y, TUBE_Z - TUBE_R - 0.01],
                        dir=[0.6, 0, -1], cutoff=88, exponent=0, attenuation=[1, 0, 0],
                        diffuse=[TUBE_DIFFUSE] * 3, specular=[0.05] * 3, castshadow=i % 2 == 1)
    # white-wall bounce: ambient, and the free camera's headlight kept faint
    spec.visual.headlight.ambient = [0.38, 0.38, 0.38]
    spec.visual.headlight.diffuse = [0.08, 0.08, 0.08]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 8192
