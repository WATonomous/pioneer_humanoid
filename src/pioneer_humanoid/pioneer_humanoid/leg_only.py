"""Isaac Lab config of the Pioneer leg-only humanoid (assets/pioneer_leg_only_humanoid), for learning to walk.

The URDF's root frame sits at the soles with -Y forward. Isaac Lab's velocity tasks expect the root at the pelvis with
+X forward, so the URDF is loaded through a copy with a ``base`` root at hip height, turned 90 degrees about Z.
"""
import math
import os
import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files import spawn_from_urdf

_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "pioneer_leg_only_humanoid"))
_OUT_DIR = os.path.abspath(os.path.join(_ASSET_DIR, "..", "..", "outputs", "rl", "assets", "pioneer_leg_only_humanoid"))
HIP_HEIGHT = 0.847917  # hip pitch axes above the soles


def _isaac_urdf(asset: str = "pioneer_leg_only_humanoid", urdf: str = "pioneer_leg_only_humanoid.urdf", legs_only: bool = False,
                keep: tuple[str, ...] = (), name: str | None = None, pose: dict[str, float] | None = None) -> str:
    """An Isaac copy of an asset's URDF, re-rooted; ``legs_only`` turns every joint but the 12 leg joints (and ``keep``) into
    fixed joints (the upper body held rigid in its zero pose: arms hanging at the sides). ``pose`` (joint -> angle) bakes an
    angle into a joint that becomes fixed, as a rotation of its frame about its axis (the fixed hands' fists)."""
    asset_dir = os.path.join(_ASSET_DIR, "..", asset)
    src = open(os.path.join(asset_dir, urdf)).read()
    src = re.sub(r'filename="([^"/][^"]*)"', lambda m: f'filename="{os.path.abspath(os.path.join(asset_dir, m.group(1)))}"', src)
    for joint, angle in (pose or {}).items():
        src = _bake_angle(src, joint, angle)
    if legs_only:
        free = "|".join(["Hip_", "Knee_", "Ankle_"] + [f'{k}"' for k in keep])
        src = re.sub(rf'(<joint name="(?!{free})[^"]*" type=)"(revolute|prismatic|continuous)"', r'\1"fixed"', src)
    root = (
        '<link name="base"/>\n'
        '  <joint name="base_to_base_link" type="fixed"><parent link="base"/><child link="base_link"/>'
        f'<origin xyz="0 0 {-HIP_HEIGHT}" rpy="0 0 1.5707963"/></joint>\n  <link name="base_link">'
    )
    src = src.replace('<link name="base_link">', root, 1)
    out_dir = os.path.join(_OUT_DIR, "..", asset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name or asset}_isaac.urdf")
    if not os.path.exists(path) or open(path).read() != src:
        with open(path, "w") as f:
            f.write(src)
    return path


def _bake_angle(src: str, joint: str, angle: float) -> str:
    """Give `joint`'s origin the rotation `angle` about its axis (its origin carries no rpy in these URDFs)."""
    block = re.search(rf'<joint name="{joint}" type="[^"]*">.*?</joint>', src, flags=re.S).group(0)
    x, y, z = (float(v) for v in re.search(r'<axis xyz="([^"]*)"', block).group(1).split())
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s_, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    r = [[t * x * x + c, t * x * y - s_ * z, t * x * z + s_ * y],
         [t * x * y + s_ * z, t * y * y + c, t * y * z - s_ * x],
         [t * x * z - s_ * y, t * y * z + s_ * x, t * z * z + c]]
    rpy = (math.atan2(r[2][1], r[2][2]), math.asin(-max(-1.0, min(1.0, r[2][0]))), math.atan2(r[1][0], r[0][0]))
    baked = re.sub(r'(<origin xyz="[^"]*")', rf'\1 rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"', block, count=1)
    return src.replace(block, baked)


# a closed fist: knuckles, middle and tip joints curled, the thumb folded across (all curl positive)
FIST = {
    f"{side}_{finger}{joint}": angle
    for side in ("l", "r")
    for finger in ("", "i_", "r_", "l_")
    for joint, angle in (("mcp", 1.4), ("pip", 1.5), ("dip", 1.2))
} | {f"{side}_th_{joint}": angle for side in ("l", "r") for joint, angle in (("c", 0.6), ("f", 0.9), ("mcp", 0.8), ("ip", 0.9))}


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

# The same with shoulder pitch and elbows free, for a scripted arm swing (humanoid_rl_tasks ... pioneer_leg_only/arm_swing.py).
ARM_SWING_JOINTS = ("l_shoulder_pitch", "r_shoulder_pitch", "l_elbow", "r_elbow")
WHOLE_BODY_ARM_SWING_CFG = WHOLE_BODY_WALKER_CFG.replace(
    spawn=WHOLE_BODY_WALKER_CFG.spawn.replace(
        asset_path=_isaac_urdf("pioneer_whole_body_humanoid", "pioneer_whole_body_humanoid.urdf", legs_only=True, keep=ARM_SWING_JOINTS,
                               name="pioneer_whole_body_humanoid_arm_swing", pose=FIST),
        usd_dir=os.path.join(_OUT_DIR, "..", "pioneer_whole_body_humanoid", "usd_arm_swing"),
    ),
    actuators={
        **WHOLE_BODY_WALKER_CFG.actuators,
        "arms": ImplicitActuatorCfg(
            joint_names_expr=list(ARM_SWING_JOINTS),
            effort_limit_sim={".*_shoulder_pitch": 53.0, ".*_elbow": 22.0},
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)
