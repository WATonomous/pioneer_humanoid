"""Single source of truth for the push-block scene: geometry + placement.

The room -- arm on its floor stand at ``ROBOT_BASE_POS``, the 30.5-inch table
whose top sits at ``TABLE_TOP_Z``, the CAD lightbox -- is the shared lightbox
workcell (``humanoid_rl_tasks.workcell``); this scene only adds the ramp-box +
block on that table top and the MDP geometry around them.

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
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from ..workcell import (  # noqa: F401  (grounding re-exported for push_env_cfg / teleop)
    GROUND_Z,
    ROBOT_BASE_POS,
    ROBOT_BASE_X,
    ROBOT_STAND_LIFT_Z,
    TABLE_TOP_Z,
    LightboxWorkcellCfg,
)
from .mdp.utils import BLOCK_HALF_SIZE

# ── shared USD props (src/simulation/assets/) ───────────────────────────────
_PROPS = Path(__file__).resolve().parents[5] / "assets" / "props"  # -> <repo>/assets/props
BLOCK_USD = str(_PROPS / "block.usd")
BOX_USD = str(_PROPS / "box.usd")

# ── block / ramp-box geometry (env frame; robot base at the origin) ──────────
BLOCK_HALF = BLOCK_HALF_SIZE
PUSH_DIR = (1.0, 0.0)

# The lightbox table's front edge is at world X = TABLE_X_MIN (~0.385).  The
# original task geometry was authored around a table that began near X=0.115,
# so shift the entire task (cube, ramp, target, and success bounds) together.
# Keeping this as one explicit offset prevents the visible props and the RL
# reward geometry from drifting apart again.
TASK_X_OFFSET = 0.27  # ~= TABLE_X_MIN - 0.115

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


@configclass
class PushBlockSceneCfg(LightboxWorkcellCfg):
    """Block + ramp-box on the lightbox workcell (``humanoid_rl_tasks.workcell``).

    ``robot`` and ``ee_frame`` are ``MISSING`` -- the RL env cfg and the teleop
    registry each fill them in. ``tiled_camera`` stays ``None`` unless the
    distillation env cfg sets it.
    """

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    tiled_camera: Optional[TiledCameraCfg] = None

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
