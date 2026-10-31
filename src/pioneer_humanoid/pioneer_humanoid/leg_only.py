"""Isaac Lab config of the Pioneer leg-only humanoid (assets/pioneer_leg_only_humanoid), for learning to walk.

The URDF's root frame sits at the soles with -Y forward. Isaac Lab's velocity tasks expect the root at the pelvis with
+X forward, so the URDF is loaded through a copy with a ``base`` root at hip height, turned 90 degrees about Z.
"""
import os
import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files import spawn_from_urdf

_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "pioneer_leg_only_humanoid"))
_OUT_DIR = os.path.abspath(os.path.join(_ASSET_DIR, "..", "..", "outputs", "rl", "assets", "pioneer_leg_only_humanoid"))
HIP_HEIGHT = 0.847917  # hip pitch axes above the soles


def _isaac_urdf(asset: str = "pioneer_leg_only_humanoid", urdf: str = "pioneer_leg_only_humanoid.urdf", legs_only: bool = False) -> str:
    """An Isaac copy of an asset's URDF, re-rooted; ``legs_only`` turns every joint but the 12 leg joints into fixed joints
    (the upper body held rigid in its zero pose: arms hanging at the sides)."""
    asset_dir = os.path.join(_ASSET_DIR, "..", asset)
    src = open(os.path.join(asset_dir, urdf)).read()
    src = re.sub(r'filename="([^"/][^"]*)"', lambda m: f'filename="{os.path.abspath(os.path.join(asset_dir, m.group(1)))}"', src)
    if legs_only:
        src = re.sub(r'(<joint name="(?!Hip_|Knee_|Ankle_)[^"]*" type=)"(revolute|prismatic|continuous)"', r'\1"fixed"', src)
    root = (
        '<link name="base"/>\n'
        '  <joint name="base_to_base_link" type="fixed"><parent link="base"/><child link="base_link"/>'
        f'<origin xyz="0 0 {-HIP_HEIGHT}" rpy="0 0 1.5707963"/></joint>\n  <link name="base_link">'
    )
    src = src.replace('<link name="base_link">', root, 1)
    out_dir = os.path.join(_OUT_DIR, "..", asset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{asset}_isaac.urdf")
    if not os.path.exists(path) or open(path).read() != src:
        with open(path, "w") as f:
            f.write(src)
    return path


def _spawn_coloured(prim_path, cfg, *args, **kwargs):
    """spawn_from_urdf, then the URDF's CAD colours: the importer gives every STL mesh its own white material."""
    from pxr import Gf, Sdf

    prim = spawn_from_urdf(prim_path, cfg, *args, **kwargs)
    name = os.path.splitext(os.path.basename(cfg.asset_path))[0]
    layer = Sdf.Layer.FindOrOpen(os.path.join(cfg.usd_dir, "configuration", f"{name}_base.usd"))
    colours = re.findall(r'meshes/([^"/]+)\.stl".*?<color rgba="([^"]+)"', open(cfg.asset_path).read())
    for mesh, rgba in colours:
        attr = layer.GetAttributeAtPath(f"/meshes/{mesh}/node_STL_BINARY_/Looks/DefaultMaterial/DefaultMaterial.inputs:diffuse_color_constant")
        if attr is not None:
            attr.default = Gf.Vec3f(*map(float, rgba.split()[:3]))
    return prim


LEG_ONLY_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=_spawn_coloured,
        asset_path=_isaac_urdf(),
        usd_dir=os.path.join(_OUT_DIR, "usd"),
        force_usd_conversion=False,
        make_instanceable=True,
        fix_base=False,
        merge_fixed_joints=True,
        root_link_name="base",
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.84),
        # slight crouch, soles level: Hip_F > 0 swings the thigh forward, Knee < 0 bends the knee, Ankle_P = Hip_F + Knee
        joint_pos={"Hip_F_.*": 0.25, "Knee_.*": -0.5, "Ankle_P_.*": -0.25, "Hip_A_.*": 0.0, "Hip_R_.*": 0.0, "Ankle_R_.*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # effort and velocity limits are the motors' (URDF)
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["Hip_F_.*", "Hip_A_.*", "Hip_R_.*", "Knee_.*"],
            effort_limit_sim={"Hip_F_.*": 222.0, "Hip_A_.*": 120.0, "Hip_R_.*": 60.0, "Knee_.*": 222.0},
            velocity_limit_sim={"Hip_F_.*": 3.6652, "Hip_A_.*": 20.944, "Hip_R_.*": 20.42, "Knee_.*": 3.6652},
            stiffness={"Hip_F_.*": 200.0, "Hip_A_.*": 150.0, "Hip_R_.*": 150.0, "Knee_.*": 200.0},
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["Ankle_P_.*", "Ankle_R_.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=20.42,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)


# The whole-body humanoid (arms and hands, 44.7 kg) walking on the same 12 leg joints, everything above the hips held rigid.
WHOLE_BODY_WALKER_CFG = LEG_ONLY_HUMANOID_CFG.replace(
    spawn=LEG_ONLY_HUMANOID_CFG.spawn.replace(
        asset_path=_isaac_urdf("pioneer_whole_body_humanoid", "pioneer_whole_body_humanoid.urdf", legs_only=True),
        usd_dir=os.path.join(_OUT_DIR, "..", "pioneer_whole_body_humanoid", "usd"),
    )
)
