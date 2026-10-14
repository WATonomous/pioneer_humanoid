"""Lightbox workcell: the default manipulation setup every scene builds on.

The real rig is the pioneer arm on its floor stand in front of a 30.5-inch
table inside the lightbox (Onshape CAD, ``assets/props/lightbox``). This module
owns that room -- floor, light, visible CAD lightbox, invisible table
collision -- plus the arm pose and table-top height, so every manipulation
scene shares one grounding instead of each re-deriving it.

A scene subclasses ``LightboxWorkcellCfg`` and adds only its own objects,
placed on ``TABLE_TOP_Z``::

    @configclass
    class MySceneCfg(LightboxWorkcellCfg):
        widget = RigidObjectCfg(..., init_state=...(pos=(0.5, 0.0, TABLE_TOP_Z)))

Imported by the RL tasks (``push_block``) and by ``humanoid_isaac_scenes``
(which already depends on this package), so the teleop ``--scene`` registry
and RL training see the identical room.
"""
from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

_PROPS = Path(__file__).resolve().parents[4] / "assets" / "props"  # -> <repo>/assets/props
TABLE_USD = str(_PROPS / "table.usd")
LIGHTBOX_USD = str(_PROPS / "lightbox.usd")

# ── arm on its floor stand ───────────────────────────────────────────────────
ROBOT_STAND_LIFT_Z = 1.1997   # base_link lift so the stand's feet reach floor level
ROBOT_BASE_X = 0.15
ROBOT_BASE_POS = (ROBOT_BASE_X, 0.0, ROBOT_STAND_LIFT_Z)

# ── table ────────────────────────────────────────────────────────────────────
# table.usd is 29.5 inches along the local axis that becomes world Z after
# _TABLE_ROT.  Scale that axis to the real table's 30.5-inch height.
TABLE_HEIGHT_M = 30.5 * 0.0254
_TABLE_SOURCE_HEIGHT_IN = 29.5
_TABLE_HEIGHT_SCALE = TABLE_HEIGHT_M / _TABLE_SOURCE_HEIGHT_IN
# The robot stand asset's lowest point lands at world Z=0 after
# ROBOT_STAND_LIFT_Z, and the lightbox legs also begin at Z=0.  Ground the
# table legs and plane on that same floor.
_TABLE_BOTTOM_Z = 0.0
_TABLE_CENTER_Z = _TABLE_BOTTOM_Z + TABLE_HEIGHT_M / 2
TABLE_TOP_Z = _TABLE_BOTTOM_Z + TABLE_HEIGHT_M
GROUND_Z = 0.0

_TABLE_POS = (0.69, 0.00612, _TABLE_CENTER_Z)
_TABLE_ROT = (0.5000000000000001, 0.5, 0.5, 0.49999999999999994)  # wxyz
_TABLE_SCALE = (0.0254, _TABLE_HEIGHT_SCALE, 0.0254)  # SolidWorks inch export; local Y is world height

# Usable table top (world frame): 24-inch depth along X, 60-inch width along Y,
# centred on _TABLE_POS.  The front edge (robot side) is TABLE_X_MIN.
TABLE_X_MIN = _TABLE_POS[0] - 12 * 0.0254
TABLE_X_MAX = _TABLE_POS[0] + 12 * 0.0254
TABLE_Y_HALF = 30 * 0.0254

# ── lightbox (visual only) ───────────────────────────────────────────────────
# Onshape export: X=60-inch table width, Y=24-inch depth, and the assembly's
# vertical direction is -Z. Rotate it so width is world Y, depth is world X,
# the open face points toward -X (the robot), and its legs land at Z=0.
LIGHTBOX_POS = (0.69, 0.00612, TABLE_TOP_Z)
LIGHTBOX_ROT = (0.0, 0.70710678, -0.70710678, 0.0)  # wxyz: X=180 deg, Z=-90 deg
LIGHTBOX_SCALE = (1.0, 1.0, 30.5 / 30.0)  # CAD table is 30 in; physical table is 30.5 in

# Teleop initial view: from the robot's front-right, looking at the table top.
WORKCELL_CAMERA = ([1.6, -1.2, 1.5], [0.55, 0.0, TABLE_TOP_Z])


@configclass
class LightboxWorkcellCfg(InteractiveSceneCfg):
    """Floor + light + lightbox + table collision, with the arm slot ``MISSING``.

    Place the arm at ``ROBOT_BASE_POS`` (the RL env cfg / teleop registry fill
    ``robot`` in) and put scene objects on ``TABLE_TOP_Z``.
    """

    robot: ArticulationCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Preserve the verified table collision invisibly. The converted CAD below
    # supplies the real visible table/lightbox but contains no collision API.
    table_collision = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableCollision",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_TABLE_POS, rot=_TABLE_ROT),
        spawn=UsdFileCfg(
            usd_path=TABLE_USD,
            scale=_TABLE_SCALE,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    lightbox = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Lightbox",
        init_state=AssetBaseCfg.InitialStateCfg(pos=LIGHTBOX_POS, rot=LIGHTBOX_ROT),
        spawn=UsdFileCfg(usd_path=LIGHTBOX_USD, scale=LIGHTBOX_SCALE),
    )
