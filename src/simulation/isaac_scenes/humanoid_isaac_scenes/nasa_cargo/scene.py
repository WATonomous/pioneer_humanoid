"""NASA rigid CTB and two reachable stations for Pioneer pick-and-place."""
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from humanoid_isaac_scenes import scene
from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS

ROBOT_POS = (0.0, 0.0, 1.1997)
CAMERA = ((-1.2, -1.8, 1.9), (0.48, 0.0, 0.85))
CARGO_USD = Path(__file__).resolve().parents[5] / "assets/nasa_cargo/ctb.usdc"
TABLE_TOP_Z = 0.48
BODY_HALF_HEIGHT = 0.26679093
PICKUP_XY = (0.50, 0.14)
TARGET_XY = (0.25, 0.495)
TARGET_HALF_XY = (0.28, 0.20)  # clearance for the full-size bag to settle upright
CARGO_POS = (*PICKUP_XY, TABLE_TOP_Z + BODY_HALF_HEIGHT + 0.004)
CARGO_ROT = (1.0, 0.0, 0.0, 0.0)
CARGO_HOME_JOINTS = {**DEFAULT_JOINT_POS, "joint4l": -2.1}


def static_box(path, pos, size, color):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        ),
    )


@scene("nasa_cargo", robot_pos=ROBOT_POS, camera=CAMERA, robot_joint_pos=CARGO_HOME_JOINTS)
@configclass
class NasaCargoSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ground = static_box("Ground", (0.0, 0.0, -0.05), (6.0, 6.0, 0.1), (0.28, 0.30, 0.33))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    pickup_station = static_box("Pickup", (*PICKUP_XY, TABLE_TOP_Z / 2),
                                (0.46, 0.28, TABLE_TOP_Z), (0.28, 0.33, 0.39))
    target_station = static_box("Target", (*TARGET_XY, TABLE_TOP_Z / 2),
                                (0.60, 0.44, TABLE_TOP_Z), (0.12, 0.38, 0.24))
    # Low physical rim; four separate colliders leave the tray interior open.
    target_front = static_box("TargetFront", (TARGET_XY[0] - 0.29, TARGET_XY[1], TABLE_TOP_Z + 0.025),
                              (0.02, 0.44, 0.05), (0.08, 0.24, 0.16))
    target_back = static_box("TargetBack", (TARGET_XY[0] + 0.29, TARGET_XY[1], TABLE_TOP_Z + 0.025),
                             (0.02, 0.44, 0.05), (0.08, 0.24, 0.16))
    target_left = static_box("TargetLeft", (TARGET_XY[0], TARGET_XY[1] + 0.21, TABLE_TOP_Z + 0.025),
                             (0.56, 0.02, 0.05), (0.08, 0.24, 0.16))
    target_right = static_box("TargetRight", (TARGET_XY[0], TARGET_XY[1] - 0.21, TABLE_TOP_Z + 0.025),
                              (0.56, 0.02, 0.05), (0.08, 0.24, 0.16))
    cargo = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cargo",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CARGO_POS, rot=CARGO_ROT),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CARGO_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32, solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
            ),
        ),
    )
