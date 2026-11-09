"""Lightbox workcell: the default manipulation setup every scene builds on.

The real rig is the pioneer arm on its floor stand in front of a 30.5-inch
table inside the lightbox (Onshape CAD, ``assets/props/lightbox``). This module
owns the rig -- visible CAD lightbox and its light tube, glossy table top, invisible
table collision -- plus the arm pose and table-top height, so every manipulation
scene shares one grounding instead of each re-deriving it.

There are two variants of the rig, the second built on the first:

  * ``LightboxWorkcellCfg``  SIMPLE: table, lightbox (with its tube light) and the robot.
  * ``LabWorkcellCfg``       FULL LAB: the simple one plus the lab around it -- concrete floor, black tiled left wall,
                             glass back wall, the dim dome and the warm overhead room light. It inherits from the
                             simple one, so a change to the rig shows up in both.

A scene subclasses ``LightboxWorkcellCfg`` and adds only its own objects,
placed on ``TABLE_TOP_Z``; mix in ``LabRoomCfg`` to get its full-lab twin::

    @configclass
    class MySceneCfg(LightboxWorkcellCfg):
        widget = RigidObjectCfg(..., init_state=...(pos=(0.5, 0.0, TABLE_TOP_Z)))

    @configclass
    class MyLabSceneCfg(MySceneCfg, LabRoomCfg):
        pass

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
LAB_ROOM_USD = str(_PROPS / "lab_room.usd")  # concrete floor, black tiled left wall, glass back wall (visual only)
# Photos of the view through the lab's glass wall; the back wall shows one of them (picked at random by the vision DR).
LAB_BACK_WALL_TEXTURES = tuple(str(_PROPS / "lab_room" / "textures" / f"glass_view_{i}.jpg") for i in range(3))
LAB_BACK_WALL_TEXTURE_PRIM = "/World/LabRoom/Looks/GlassView/tex"

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

# The box is lit by one light tube (~1 m) hung under the top front rail, slightly right of centre seen from
# the front, diffuser facing down; the white foam-board walls bounce the rest (as mujoco_scenes' workcell).
LIGHTBOX_TOP_Z = 1.456   # top of the lightbox frame
BOTTOM_PANEL = 0.1875 * 0.0254 * 30.5 / 30.0   # the lightbox's white bottom panel on the table (3/16 in)
TUBE_POS = (0.45, -0.06, 1.385)
TUBE_HALF_LEN, TUBE_R = 0.48, 0.03

# Kit's RTX texture streaming loads textures on several worker threads, and with the lab room's three textures it intermittently
# aborts Kit with "unlock() called by non-owning thread" (a lock taken on one thread and released on another; headless or GUI).
# Streaming only helps scenes with huge texture sets, so every sim that loads these scenes passes this as ``RenderCfg.carb_settings``.
RENDER_CARB_SETTINGS = {"/rtx-transient/resourcemanager/enableTextureStreaming": False}

# Teleop initial view: from the robot's front-right, looking at the table top.
WORKCELL_CAMERA = ([1.6, -1.2, 1.5], [0.55, 0.0, TABLE_TOP_Z])
# Same idea for the full-lab scenes, from inside the room (its back wall is at x = 1.10, so the camera above would be behind it).
LAB_WORKCELL_CAMERA = ([-0.7, -1.8, 1.8], [0.8, 0.1, 0.9])


# Warm room light on the ceiling outside the box, 1 m behind the cube (on the robot's side) and 1.5 m above it; its light
# reaches the cube through the open front, partly shadowed by the robot.
ROOM_LIGHT_POS = (0.48 - 1.0, LIGHTBOX_POS[1], TABLE_TOP_Z + 1.5)
ROOM_LIGHT_COLOR = (1.0, 0.8, 0.6)
ROOM_LIGHT_RADIUS = 0.3
ROOM_LIGHT_INTENSITY = 100000.0
DOME_INTENSITY = 400.0
TUBE_INTENSITY = 140000.0


@configclass
class WhiteboardSurfaceCfg(sim_utils.MdlFileCfg):
    """A glossy white dry-erase board, as an Omniverse OmniPBR material (the renderer's full reflection model).

    A high-gloss board is above ~70 GU at 60 deg, i.e. nearly mirror-smooth, and cured melamine resin has a refractive
    index of ~1.5-1.7 (5-7% reflectance at normal incidence): so a very low reflection roughness and full specular level.
    A true dielectric (metallic 0) only reflects ~5-8% of the scene, which vanishes against the bright white box, so
    ``metallic_constant`` is raised to 0.5 to make the reflections read (1.0 is a perfect mirror)."""

    mdl_path: str = "OmniPBR.mdl"
    diffuse_color_constant: tuple[float, float, float] = (0.86, 0.86, 0.87)
    reflection_roughness_constant: float = 0.02
    specular_level: float = 1.0
    metallic_constant: float = 0.5


_BOARD = WhiteboardSurfaceCfg()
_HOUSING = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.04, 0.04, 0.045), roughness=0.5)
_SEAM_Y, _BOARD_HALF_Y, _SEAM_HALF = -0.01, 0.745, 0.002   # the seam is 4 mm wide
_BOARD_X, _BOARD_LEN_X = _TABLE_POS[0], 24 * 0.0254 - 0.008
_BOARD_Z = TABLE_TOP_Z + BOTTOM_PANEL - 0.0003
_HANGER_H = (LIGHTBOX_TOP_Z - 0.02) - (TUBE_POS[2] + TUBE_R) + 0.004
_HANGER_Z = (LIGHTBOX_TOP_Z - 0.02 + TUBE_POS[2] + TUBE_R) / 2


@configclass
class LightboxWorkcellCfg(InteractiveSceneCfg):
    """Floor + light + lightbox + table collision, with the arm slot ``MISSING``.

    Place the arm at ``ROBOT_BASE_POS`` (the RL env cfg / teleop registry fill
    ``robot`` in) and put scene objects on ``TABLE_TOP_Z``.
    """

    robot: ArticulationCfg = MISSING

    # The box is lit by its own tube light; the lab around it is LabRoomCfg.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z)),
        spawn=GroundPlaneCfg(visible=False),
    )
    tube_light = AssetBaseCfg(
        prim_path="/World/tube_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TUBE_POS[0] + 0.004, TUBE_POS[1], TUBE_POS[2] - TUBE_R - 0.02),
                                                rot=(0.70710678, 0.0, 0.0, 0.70710678)),   # its length (local X) along Y
        spawn=sim_utils.CylinderLightCfg(length=2 * TUBE_HALF_LEN, radius=0.012, intensity=TUBE_INTENSITY,
                                         color=(1.0, 0.98, 0.95), treat_as_line=False),
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

    # The table's converted collision has holes in its top (an object at x 0.72, y -0.17 falls through): a plain
    # box under the same top surface fills them.
    table_top_collision = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTopCollision",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_TABLE_POS[0], _TABLE_POS[1], TABLE_TOP_Z - 0.01)),
        spawn=sim_utils.CuboidCfg(size=(TABLE_X_MAX - TABLE_X_MIN, 2 * TABLE_Y_HALF, 0.02), visible=False,
                                  collision_props=sim_utils.CollisionPropertiesCfg()),
    )

    lightbox = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Lightbox",
        init_state=AssetBaseCfg.InitialStateCfg(pos=LIGHTBOX_POS, rot=LIGHTBOX_ROT),
        spawn=UsdFileCfg(usd_path=LIGHTBOX_USD, scale=LIGHTBOX_SCALE),
    )

    # Two glossy white boards on the bottom panel, meeting at a seam (visual only).
    floor_board_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FloorBoardLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_BOARD_X, (_SEAM_Y + _SEAM_HALF + _BOARD_HALF_Y) / 2, _BOARD_Z)),
        spawn=sim_utils.CuboidCfg(size=(_BOARD_LEN_X, _BOARD_HALF_Y - _SEAM_Y - _SEAM_HALF, 0.0008), visual_material=_BOARD),
    )
    floor_board_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FloorBoardRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_BOARD_X, (_SEAM_Y - _SEAM_HALF - _BOARD_HALF_Y) / 2, _BOARD_Z)),
        spawn=sim_utils.CuboidCfg(size=(_BOARD_LEN_X, _BOARD_HALF_Y + _SEAM_Y - _SEAM_HALF, 0.0008), visual_material=_BOARD),
    )
    floor_seam = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FloorSeam",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_BOARD_X, _SEAM_Y, _BOARD_Z - 0.0001)),
        spawn=sim_utils.CuboidCfg(size=(_BOARD_LEN_X, 2 * _SEAM_HALF, 0.0006),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=1.0)),   # black line
    )

    # The light tube's black housing with its glowing diffuser strip underneath, and two hangers up to the rail.
    tube_housing = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TubeHousing",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TUBE_POS, rot=(0.70710678, 0.70710678, 0.0, 0.0)),   # axis along Y
        spawn=sim_utils.CapsuleCfg(radius=TUBE_R, height=2 * TUBE_HALF_LEN, axis="Z", visual_material=_HOUSING),
    )
    tube_diffuser = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TubeDiffuser",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TUBE_POS[0] + 0.004, TUBE_POS[1], TUBE_POS[2] - TUBE_R + 0.003)),
        spawn=sim_utils.CuboidCfg(size=(0.044, 2 * TUBE_HALF_LEN, 0.012), visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 1.0, 1.0), emissive_color=(1.0, 1.0, 1.0))),
    )
    tube_hanger_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TubeHangerLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TUBE_POS[0], TUBE_POS[1] + 0.33, _HANGER_Z)),
        spawn=sim_utils.CuboidCfg(size=(0.024, 0.06, _HANGER_H), visual_material=_HOUSING),
    )
    tube_hanger_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TubeHangerRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TUBE_POS[0], TUBE_POS[1] - 0.33, _HANGER_Z)),
        spawn=sim_utils.CuboidCfg(size=(0.024, 0.06, _HANGER_H), visual_material=_HOUSING),
    )


@configclass
class LabRoomCfg:
    """The lab around the box: concrete floor, black tiled wall on the robot's left (+y), the glass wall behind the box
    (+x), the faint dome and the warm overhead room light. A mixin: add it to any scene class to get its full-lab twin."""

    lab_room = AssetBaseCfg(prim_path="/World/LabRoom", spawn=UsdFileCfg(usd_path=LAB_ROOM_USD))
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=DOME_INTENSITY),
    )
    room_light = AssetBaseCfg(
        prim_path="/World/room_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=ROOM_LIGHT_POS),
        spawn=sim_utils.DiskLightCfg(radius=ROOM_LIGHT_RADIUS, color=ROOM_LIGHT_COLOR, intensity=ROOM_LIGHT_INTENSITY),
    )


@configclass
class LabWorkcellCfg(LightboxWorkcellCfg, LabRoomCfg):
    """FULL LAB: the simple workcell (``LightboxWorkcellCfg``) plus the lab room and the overhead lighting."""
