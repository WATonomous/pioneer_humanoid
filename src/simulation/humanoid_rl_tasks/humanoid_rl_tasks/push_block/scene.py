"""Single source of truth for the push-block scene: geometry + placement.

Placement is the **teleop-verified grounding** (measured via isaac_harness bbox,
see humanoid_isaac_scenes): the arm on its floor stand (``base_link``
lifted ``ROBOT_STAND_LIFT_Z``, feet at floor level), a real work table whose top
sits at ``TABLE_TOP_Z``, and the ramp-box + block on that table top. This
replaces the SO101-inherited grounding (arm at the origin, table top at z=0)
that the RL push env used through mid-2026.

Imported by:
  - ``push_env_cfg.py`` -- the RL env fills ``scene.robot`` / ``scene.ee_frame``
    and layers the MDP on top.
  - ``humanoid_isaac_scenes`` -- the teleop registry fills
    ``scene.robot`` and drops ``ee_frame``.

Every world-Z-dependent MDP constant (``FLOOR_Z``, ``FLOOR_Z_COLLISION``,
``RAMP_BASE_Z``, ``BLOCK_DROP_MIN_Z``) is derived from ``TABLE_TOP_Z`` here, so
the reward / observation terms stay correct against the raised table.
"""
from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from .mdp.utils import BLOCK_HALF_SIZE

# ── shared USD props (src/simulation/assets/) ───────────────────────────────
_PROPS = Path(__file__).resolve().parents[5] / "assets" / "props"  # -> <repo>/assets/props
BLOCK_USD = str(_PROPS / "block.usd")
BOX_USD = str(_PROPS / "box.usd")
TABLE_USD = str(_PROPS / "table.usd")
LIGHTBOX_USD = str(_PROPS / "lightbox.usd")

# ── grounding (teleop-verified; see module docstring) ────────────────────────
ROBOT_STAND_LIFT_Z = 1.1997   # base_link lift so the stand's feet reach floor level
ROBOT_BASE_X = 0.15
ROBOT_BASE_POS = (ROBOT_BASE_X, 0.0, ROBOT_STAND_LIFT_Z)
# table.usd is 29.5 inches along the local axis that becomes world Z after
# _TABLE_ROT.  Scale that axis to the real table's 30.5-inch height while
# keeping the feet at the same world position used by the verified scene.
TABLE_HEIGHT_M = 30.5 * 0.0254
_TABLE_SOURCE_HEIGHT_IN = 29.5
_TABLE_HEIGHT_SCALE = TABLE_HEIGHT_M / _TABLE_SOURCE_HEIGHT_IN
# The robot stand asset's lowest point lands at world Z=0 after
# ROBOT_STAND_LIFT_Z, and the enclosure walls also begin at Z=0.  Ground the
# table legs and plane on that same floor instead of leaving the scene floating.
_TABLE_BOTTOM_Z = 0.0
_TABLE_CENTER_Z = _TABLE_BOTTOM_Z + TABLE_HEIGHT_M / 2
TABLE_TOP_Z = _TABLE_BOTTOM_Z + TABLE_HEIGHT_M
GROUND_Z = 0.0

_TABLE_POS = (0.69, 0.00612, _TABLE_CENTER_Z)
_TABLE_ROT = (0.5000000000000001, 0.5, 0.5, 0.49999999999999994)  # wxyz
_TABLE_SCALE = (0.0254, _TABLE_HEIGHT_SCALE, 0.0254)  # SolidWorks inch export; local Y is world height

# ── block / ramp-box geometry (env frame; robot base at the origin) ──────────
BLOCK_HALF = BLOCK_HALF_SIZE
PUSH_DIR = (1.0, 0.0)

# The converted lightbox/table begins at world X ~= 0.385.  The original task
# geometry was authored around a table that began near X=0.115, so shift the
# entire task (cube, ramp, target, and success bounds) together by 0.27 m.
# Keeping this as one explicit offset prevents the visible props and the RL
# reward geometry from drifting apart again.
TASK_X_OFFSET = 0.27

# box placed corner at (0.27, 0.127) in the original task, yaw -90 deg:
# box-local +y (up the ramp) -> env +x
BOX_POS = (0.27 + TASK_X_OFFSET, 0.127, TABLE_TOP_Z)
BOX_QUAT = (0.70711, 0.0, 0.0, -0.70711)

# block starts on the table in front of the ramp (center at ~(0.21, 0))
BLOCK_INIT_POS = (0.21 - BLOCK_HALF + TASK_X_OFFSET, -BLOCK_HALF, TABLE_TOP_Z)

RAMP_BASE_X = 0.279 + TASK_X_OFFSET  # ramp meets the table
RAMP_TOP_X = 0.308 + TASK_X_OFFSET   # ramp meets the interior floor
RAMP_BASE_Z = TABLE_TOP_Z                    # support-surface height at the ramp base
FLOOR_Z = TABLE_TOP_Z + 0.0063               # visual interior floor (absolute world Z)
# Effective COLLISION floor: box USD's collision surface sits ~5 mm above the
# visual floor, so a settled 50.8 mm block rests at center z ~= FLOOR_Z_COLLISION
# + BLOCK_HALF. Used by the block_on_floor success check; FLOOR_Z stays the
# visual value for the ramp_geometry obs and the scoop penalty.
FLOOR_Z_COLLISION = TABLE_TOP_Z + 0.0115
FLOOR_X_MAX = 0.511 + TASK_X_OFFSET  # interior floor end (back wall)
FLOOR_Y_HALF = 0.114
FLOOR_TARGET = (0.37 + TASK_X_OFFSET, 0.0)  # target point on the interior floor (xy)

BLOCK_DROP_MIN_Z = TABLE_TOP_Z - 0.10  # below this = block fell off the table

# ── spawn curriculum (xy/yaw offsets around the block anchor) ────────────────
FULL_YAW = (0.0, 6.2831853)
SPAWN_STAGES = [
    {"x": (-0.06, 0.02), "y": (-0.06, 0.06), "yaw": FULL_YAW},
    {"x": (-0.10, 0.05), "y": (-0.12, 0.12), "yaw": FULL_YAW},
    {"x": (-0.12, 0.20), "y": (-0.20, 0.20), "yaw": FULL_YAW},
    {"x": (-0.14, 0.30), "y": (-0.26, 0.26), "yaw": FULL_YAW},
]
REPOSITION_START_STAGE = 2
BOX_EXCLUSION = {
    "x_min": RAMP_BASE_X,
    "x_max": 0.524 + TASK_X_OFFSET + BLOCK_HALF,
    "y_abs": 0.127 + BLOCK_HALF,
}

# ── lightbox enclosure (visual only; teleop / quest values) ──────────────────
# Onshape export: X=60-inch table width, Y=24-inch depth, and the assembly's
# vertical direction is -Z. Rotate it so width is world Y, depth is world X,
# the open face points toward -X (the robot), and its legs land at Z=0.
LIGHTBOX_POS = (0.69, 0.00612, TABLE_TOP_Z)
LIGHTBOX_ROT = (0.0, 0.70710678, -0.70710678, 0.0)  # wxyz: X=180 deg, Z=-90 deg
LIGHTBOX_SCALE = (1.0, 1.0, 30.5 / 30.0)  # CAD table is 30 in; physical table is 30.5 in


@configclass
class PushBlockSceneCfg(InteractiveSceneCfg):
    """Block + ramp-box + table + lightbox, on the teleop-verified grounding.

    ``robot`` and ``ee_frame`` are ``MISSING`` -- the RL env cfg and the teleop
    registry each fill them in. ``tiled_camera`` stays ``None`` unless the
    distillation env cfg sets it.
    """

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    tiled_camera: Optional[TiledCameraCfg] = None

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

    # dynamic block to push (corner-origin USD)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BLOCK_INIT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=BLOCK_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # static open box + ramp (collision baked into the USD, no RigidBodyAPI)
    box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BOX_POS, rot=BOX_QUAT),
        spawn=UsdFileCfg(usd_path=BOX_USD),
    )
