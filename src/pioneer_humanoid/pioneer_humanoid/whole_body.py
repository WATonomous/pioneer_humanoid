import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "whole_body_humanoid"
)
_URDF_PATH = os.path.join(_ASSET_DIR, "whole_body_humanoid.urdf")
_USD_DIR = os.path.join(_ASSET_DIR, "usd")

# Mild crouch; negative knee position is hardware-convention flexion.
# With the corrected knee axes, 0.84 m places the foot soles at the terrain
# surface without the calf/foot penetration produced by the old 0.75 m pose.
_SPAWN_HEIGHT = 0.84


WHOLE_BODY_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        usd_dir=_USD_DIR,
        usd_file_name="whole_body_humanoid.usd",
        force_usd_conversion=False,
        make_instanceable=True,
        fix_base=False,
        merge_fixed_joints=True,
        root_link_name="base",
        self_collision=False,
        collider_type="convex_hull",
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=5.0,
            )
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.5,
        ),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, _SPAWN_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "Hip_F_L": 0.12,
            # Zero hip ab/adduction. The old +0.04 was a compensation for the
            # non-mirrored Hip_A_R axis: it adducted the left leg while abducting
            # the right, for a net stance of 0.30 m. With the axes now
            # mirror-consistent, +0.04 adducts *both* legs and narrows the stance
            # to 0.23 m; zero restores the shipped 0.30 m stance and is
            # mirror-symmetric by construction.
            "Hip_A_L": 0.0,
            "Hip_R_L": 0.0,
            "Knee_L": -0.22,
            # The right ankle-pitch axis is now negated in the URDF to match the
            # left, so a matching physical foot pitch takes the same joint sign.
            "Ankle_P_L": -0.10,
            "Ankle_R_L": 0.0,
            "Hip_F_R": 0.12,
            "Hip_A_R": 0.0,
            "Hip_R_R": 0.0,
            "Knee_R": -0.22,
            "Ankle_P_R": -0.10,
            "Ankle_R_R": 0.0,
        },
    ),
    actuators={
        "hip_flexion": ImplicitActuatorCfg(
            joint_names_expr=["Hip_F_.*"],
            effort_limit_sim=222.0,
            velocity_limit_sim=3.6652,
            stiffness=100.0,
            damping=8.0,
        ),
        "hip_abduction": ImplicitActuatorCfg(
            joint_names_expr=["Hip_A_.*"],
            effort_limit_sim=120.0,
            velocity_limit_sim=20.944,
            stiffness=80.0,
            damping=5.0,
        ),
        "hip_rotation": ImplicitActuatorCfg(
            joint_names_expr=["Hip_R_.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=20.42,
            stiffness=60.0,
            damping=4.0,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=["Knee_.*"],
            effort_limit_sim=222.0,
            velocity_limit_sim=3.6652,
            stiffness=100.0,
            damping=8.0,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=["Ankle_.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=20.42,
            stiffness=80.0,
            damping=6.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
