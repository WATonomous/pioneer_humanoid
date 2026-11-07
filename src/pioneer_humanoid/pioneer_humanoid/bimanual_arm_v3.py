"""Isaac Lab articulation config for pioneer_bimanual_arm v3: the 7-DOF arms with wrist pitch + yaw on CubeMars
AK45-10s and the gripper on an AK40-10. The numbers are in arm_params_v3.py; the asset is described in
assets/pioneer_bimanual_arm/README_v3.md.

The USD is converted from urdf/pioneer_bimanual_arm_v3.urdf, with the same options as v2's (usd_v3/config.yaml), inside
the Isaac container:

    ./isaaclab.sh -p scripts/tools/convert_urdf.py \\
        /workspace/humanoid/assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm_v3.urdf \\
        /workspace/humanoid/assets/pioneer_bimanual_arm/usd_v3/pioneer_bimanual_arm_v3.usd \\
        --fix-base --merge-joints --joint-stiffness 100.0 --joint-damping 5.0 --joint-target-type position

The USD is not converted yet: this config was written without Isaac, and loading it is the first check to run.
The converter carries the URDF's joint limits into the USD, so v2's runtime limit patching is not needed here.
v2 (bimanual_arm.py) is unchanged.
"""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .arm_params_v3 import (  # noqa: F401 -- re-exported
    ACTUATOR_GROUPS,
    CAMERAS_WRIST,
    DEFAULT_JOINT_POS,
    EE_BODIES,
    FINGER_BODIES,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CLOSED,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_CLOSED,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_OPEN,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
ARM_V3_USD_PATH = str(_REPO_ROOT / "assets" / "pioneer_bimanual_arm" / "usd_v3" / "pioneer_bimanual_arm_v3.usd")

# the hand link (origin at the wrist centre, where joints 5, 6 and 7 meet) anchors the Jacobian, as link6 did on v2
LEFT_EE_BODY = EE_BODIES["left"]
RIGHT_EE_BODY = EE_BODIES["right"]
LEFT_FINGER_TIP_BODIES = ("link8l", "link9l")
RIGHT_FINGER_TIP_BODIES = ("link8", "link9")
# the claws' link frames are v2's finger frames (same meshes, same joint origins relative to the gripper): the distal
# point is as v2's, 0.02 m along the travel axis (y), the outer side
LEFT_FINGER_DISTAL_TIP_LOCAL = {"link8l": (0.0, 0.02, 0.0), "link9l": (0.0, -0.02, 0.0)}
RIGHT_FINGER_DISTAL_TIP_LOCAL = {"link8": (0.0, -0.02, 0.0), "link9": (0.0, 0.02, 0.0)}

# TPU fingers, as v2 (bimanual_arm.FINGER_MATERIAL)
FINGER_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="max")


def _spawn_v3(prim_path, cfg, translation=None, orientation=None):
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdShade

    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    stage = get_current_stage()
    material_path = f"{prim_path}/finger_material"
    FINGER_MATERIAL.func(material_path, FINGER_MATERIAL)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    for link in FINGER_BODIES:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/{link}")).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics")
    return prim


from isaaclab.sim.utils import clone  # noqa: E402

spawn_bimanual_arm_v3 = clone(_spawn_v3)
spawn_bimanual_arm_v3.__name__ = "spawn_bimanual_arm_v3"            # see bimanual_arm.py: config round-trips by name
spawn_bimanual_arm_v3.__qualname__ = "spawn_bimanual_arm_v3"

BIMANUAL_ARM_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_bimanual_arm_v3,
        usd_path=ARM_V3_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),   # as v2
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=DEFAULT_JOINT_POS),
    actuators={
        name: ImplicitActuatorCfg(
            joint_names_expr=group["joints"],
            stiffness=group["stiffness"],
            damping=group["damping"],
            effort_limit_sim=group["effort_limit"],
            velocity_limit_sim=group["velocity_limit"],
        )
        for name, group in ACTUATOR_GROUPS.items()
    },
)
