"""Lightbox workcell for MuJoCo: the room every manipulation scene builds on, as in Isaac.

The same rig as Isaac's ``humanoid_rl_tasks.workcell.LightboxWorkcellCfg`` (not imported: that module
needs Isaac): the arm on its floor stand in front of the 30.5-inch table inside the lightbox. The visible
lightbox and table are the Onshape CAD (``assets/props/lightbox.usd``, baked to OBJ by
``bake_lightbox.py``), visual only like in Isaac; the table top collides as a plain box.

MuJoCo scenes keep the arm base at the world origin (``make_model``'s default ``robot_pos``), so every
Isaac x here is shifted by Isaac's ``ROBOT_BASE_X``: the table is where it is relative to the arm.

A scene calls ``add_workcell(spec)`` and puts its objects on ``TABLE_TOP_Z``, inside ``TABLE_X`` and
``TABLE_HALF_Y``.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco

from ._register import add_floor

MESH_DIR = Path(__file__).resolve().parents[4] / "assets" / "props" / "lightbox_mujoco"

_ISAAC_ROBOT_BASE_X = 0.15          # Isaac's ROBOT_BASE_X; MuJoCo's arm base is at x = 0

# Table: 30.5 inches high, 24 deep (X) by 60 wide (Y), centred where Isaac's lightbox is.
TABLE_TOP_Z = 30.5 * 0.0254
TABLE_CENTER = (0.69 - _ISAAC_ROBOT_BASE_X, 0.00612)
TABLE_X = (TABLE_CENTER[0] - 12 * 0.0254, TABLE_CENTER[0] + 12 * 0.0254)   # front (robot side), back edge
TABLE_HALF_Y = 30 * 0.0254

# Lightbox CAD: X = table width, Y = depth, vertical is -Z, origin at the table top's centre. Same pose
# and scale as Isaac's LIGHTBOX_ROT / LIGHTBOX_SCALE: width along world Y, open face toward the robot.
LIGHTBOX_POS = (*TABLE_CENTER, TABLE_TOP_Z)
LIGHTBOX_QUAT = (0.0, 0.70710678, -0.70710678, 0.0)   # wxyz
LIGHTBOX_SCALE = (1.0, 1.0, 30.5 / 30.0)              # CAD table is 30 in; the real one is 30.5 in
# The two 90 cm fill lights under the top panel (CAD x = +-0.42 m, 0.605 m above the table top).
FILL_LIGHT_Y = 0.42
FILL_LIGHT_Z = TABLE_TOP_Z + 0.605 * LIGHTBOX_SCALE[2]


def add_workcell(spec: mujoco.MjSpec) -> None:
    """Floor, the lightbox (visual CAD, its two fill lights) and the table top's collision box."""
    add_floor(spec, light=False)   # the lightbox's top panel would shade the table from an overhead light
    world = spec.worldbody
    colors = json.loads((MESH_DIR / "colors.json").read_text())
    for part, rgb in colors.items():
        spec.add_mesh(name=f"lightbox_{part}", file=str(MESH_DIR / f"{part}.obj"), scale=list(LIGHTBOX_SCALE),
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        spec.add_material(name=f"lightbox_{part}", rgba=[*rgb, 1], specular=0.2 if part == "table" else 0.1)
        world.add_geom(name=f"lightbox_{part}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"lightbox_{part}",
                       material=f"lightbox_{part}", pos=list(LIGHTBOX_POS), quat=list(LIGHTBOX_QUAT),
                       contype=0, conaffinity=0, group=1)

    # Table collision (invisible; the CAD table above is what you see). Solid down to the floor.
    (x0, x1), (cx, cy) = TABLE_X, TABLE_CENTER
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, TABLE_HALF_Y, TABLE_TOP_Z / 2],
                   pos=[cx, cy, TABLE_TOP_Z / 2], rgba=[0, 0, 0, 0], group=3)

    for y in (-FILL_LIGHT_Y, FILL_LIGHT_Y):
        world.add_light(name=f"fill_light{'_l' if y > 0 else '_r'}", pos=[cx, cy + y, FILL_LIGHT_Z - 0.02],
                        dir=[0, 0, -1], cutoff=75, exponent=1, diffuse=[0.45, 0.45, 0.45], specular=[0.1, 0.1, 0.1],
                        castshadow=True)
