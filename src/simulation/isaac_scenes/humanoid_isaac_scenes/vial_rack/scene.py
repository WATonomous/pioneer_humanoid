"""Vial-rack manipulation scene for the pioneer bimanual arm.

Registered as ``@scene("vial_rack")`` -- teleop picks it up via discovery, no
wiring anywhere else. Built on the lightbox workcell
(``humanoid_rl_tasks.workcell``): arm on its floor stand at ``ROBOT_BASE_POS``,
rack + vials on the 30.5-inch table top at ``TABLE_TOP_Z``. Rack + vial USDs
are the so101 vial task's assets.

``robot`` / ``ee_frame`` are ``MISSING`` -- ``humanoid_isaac_scenes.make_scene_cfg``
plugs in the caller's arm.

FIRST PASS: the vials sit ~0.30 m in front of the arm base -- the same reach
as the push task's block -- with the rack just behind them. It should get a
reach-tuning pass against the arm that actually drives it (keyboard_teleop
drives the LEFT / L-suffix chain) -- driving the scene once is that loop.
"""
from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass

from humanoid_isaac_scenes import scene
from humanoid_rl_tasks.workcell import ROBOT_BASE_POS, ROBOT_BASE_X, TABLE_TOP_Z, LightboxWorkcellCfg

_ASSETS = Path(__file__).resolve().parents[5] / "assets" / "lerobot" / "so101_vial_task" / "usd"
VIAL_RACK_USD = str(_ASSETS / "Vial_rack_simple.usda")
VIAL_USD = str(_ASSETS / "Vial_opaque.usda")

# ── rack + vials (first pass; reach-tune against the LEFT arm) ────────────────
# Offsets from the arm base in x; y is negative = the LEFT arm's side.
RACK_POS = (ROBOT_BASE_X + 0.37, -0.32, TABLE_TOP_Z)
VIAL_INIT_POS = [
    (ROBOT_BASE_X + 0.30, -0.05, TABLE_TOP_Z + 0.03),
    (ROBOT_BASE_X + 0.30, -0.13, TABLE_TOP_Z + 0.03),
    (ROBOT_BASE_X + 0.30, -0.21, TABLE_TOP_Z + 0.03),
]

_VIAL_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_depenetration_velocity=1.0,
    disable_gravity=False,
)


@scene(
    "vial_rack",
    robot_pos=ROBOT_BASE_POS,
    camera=([1.3, -1.3, TABLE_TOP_Z + 0.65], [ROBOT_BASE_X + 0.35, -0.2, TABLE_TOP_Z]),
)
@configclass
class VialRackSceneCfg(LightboxWorkcellCfg):
    """Vial rack + 3 loose vials in the lightbox workcell, for vial-placement teleop."""

    ee_frame: FrameTransformerCfg = MISSING

    # kinematic rack (collidable, not physics-driven)
    vial_rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/VialRack",
        init_state=AssetBaseCfg.InitialStateCfg(pos=RACK_POS),
        spawn=sim_utils.UsdFileCfg(
            usd_path=VIAL_RACK_USD,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    vial_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vial1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=VIAL_INIT_POS[0], rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=VIAL_USD, rigid_props=_VIAL_RIGID_PROPS),
    )
    vial_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vial2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=VIAL_INIT_POS[1], rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=VIAL_USD, rigid_props=_VIAL_RIGID_PROPS),
    )
    vial_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vial3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=VIAL_INIT_POS[2], rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=VIAL_USD, rigid_props=_VIAL_RIGID_PROPS),
    )
