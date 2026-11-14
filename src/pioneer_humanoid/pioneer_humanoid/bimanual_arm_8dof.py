"""Isaac articulation config for pioneer_bimanual_arm_8dof: two 7-DoF arms + 1-DoF grippers, the arm being built.

Same model and numbers as the MuJoCo side (mujoco_bimanual_arm_8dof): the URDF (assets/pioneer_bimanual_arm_8dof),
its joint limits, and arm_8dof_params' home pose and PD groups. The URDF is converted to USD on first spawn and
cached in assets/pioneer_bimanual_arm_8dof/usd (gitignored: the USD is over GitHub's 100 MB file limit); it
reconverts when the URDF changes. Finger B (*_gripper_mimic) follows *_gripper by a PhysX mimic joint.
"""
import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.utils import clone

from .arm_8dof_params import ACTUATOR_GROUPS, ARM_ARMATURE, CAMERAS, DEFAULT_JOINT_POS, GRIPPER_STROKE, URDF_PATH
from .arm_params import vertical_aperture
from .bimanual_arm import FINGER_MATERIAL

USD_DIR = URDF_PATH.rsplit("/urdf/", 1)[0] + "/usd"
FINGER_LINKS = [f"{s}_finger_{f}" for s in ("left", "right") for f in ("a", "b")]


def _color_visuals(usd_path: str) -> None:
    """Write the URDF's visual colours into the converted USD: Isaac's URDF importer leaves every mesh white."""
    from pxr import Gf, Sdf

    stem = os.path.splitext(os.path.basename(usd_path))[0]
    layer = Sdf.Layer.FindOrOpen(os.path.join(os.path.dirname(usd_path), "configuration", f"{stem}_base.usd"))
    for link in ET.parse(URDF_PATH).getroot().findall("link"):
        for visual in link.findall("visual"):
            mesh = os.path.splitext(os.path.basename(visual.find("geometry/mesh").get("filename")))[0]
            rgb = [float(c) for c in visual.find("material/color").get("rgba").split()[:3]]
            linear = Gf.Vec3f(*[c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb])
            path = f"/meshes/{mesh}/node_STL_BINARY_/Looks/DefaultMaterial/DefaultMaterial"
            attr = layer.GetAttributeAtPath(f"{path}.inputs:diffuse_color_constant")
            if attr.default != linear:
                attr.default = linear
    if layer.dirty:
        layer.Save()


def _spawn_bimanual_arm_8dof(prim_path, cfg, translation=None, orientation=None):
    from pxr import UsdShade

    from isaaclab.sim.converters import UrdfConverter
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.utils import get_current_stage

    usd_path = UrdfConverter(cfg).usd_path
    _color_visuals(usd_path)
    prim = _spawn_from_usd_file(prim_path, usd_path, cfg, translation, orientation)
    stage = get_current_stage()
    FINGER_MATERIAL.func(f"{prim_path}/finger_material", FINGER_MATERIAL)
    material = UsdShade.Material(stage.GetPrimAtPath(f"{prim_path}/finger_material"))
    for link in FINGER_LINKS:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/{link}")).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics")
    # the importer's mimic is a 25 Hz spring that lets finger B lag 10 mm (0 = rigid), with the limits widened 20 %
    for side in ("left", "right"):
        joint = stage.GetPrimAtPath(f"{prim_path}/joints/{side}_gripper_mimic")
        joint.GetAttribute("physxMimicJoint:rotY:naturalFrequency").Set(0.0)
        joint.GetAttribute("physics:lowerLimit").Set(0.0)
        joint.GetAttribute("physics:upperLimit").Set(GRIPPER_STROKE)
    return prim


spawn_bimanual_arm_8dof = clone(_spawn_bimanual_arm_8dof)
# as bimanual_arm.spawn_bimanual_arm_from_usd: config (de)serialization resolves a callable by module:__name__
spawn_bimanual_arm_8dof.__name__ = spawn_bimanual_arm_8dof.__qualname__ = "spawn_bimanual_arm_8dof"


BIMANUAL_ARM_8DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_bimanual_arm_8dof,
        asset_path=URDF_PATH,
        usd_dir=USD_DIR,
        make_instanceable=True,
        fix_base=True,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=True,   # inverted: IsaacLab passes it to set_parse_mimic, True keeps the mimic
        collider_type="convex_hull",
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(   # force PD as MuJoCo; the actuators below set the gains
            drive_type="force", target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=5.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=DEFAULT_JOINT_POS),
    actuators={
        name: ImplicitActuatorCfg(
            joint_names_expr=group["joints"],
            stiffness=group["stiffness"],
            damping=group["damping"],
            effort_limit_sim=group["effort_limit"],
            velocity_limit_sim=group["velocity_limit"],
            armature=0.0 if "gripper" in name else ARM_ARMATURE,
        )
        for name, group in ACTUATOR_GROUPS.items()
    } | {"gripper_mimic": ImplicitActuatorCfg(joint_names_expr=[".*_gripper_mimic"], stiffness=0.0, damping=0.0)},
)


def make_camera_cfg(name: str, height: int, width: int) -> CameraCfg:
    """Isaac CameraCfg for robot camera ``name`` (arm_8dof_params.CAMERAS), robot at ``{ENV_REGEX_NS}/Robot``."""
    cam = CAMERAS[name]
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + f"{cam.body}/{cam.prim}",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam.focal,
            horizontal_aperture=cam.h_aperture,
            vertical_aperture=vertical_aperture(name, height, width),
            clipping_range=cam.clip,
        ),
        offset=CameraCfg.OffsetCfg(pos=cam.pos, rot=cam.rot, convention="opengl"),
        height=height,
        width=width,
        update_period=0.0,
        data_types=["rgb"],
    )
