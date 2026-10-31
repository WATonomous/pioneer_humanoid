import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "old_leg_only_humanoid"
)
_URDF_PATH = os.path.join(_ASSET_DIR, "old_leg_only_humanoid.urdf")
_USD_DIR = os.path.join(_ASSET_DIR, "usd")

# Mild crouch; keep feet near ground after bend.
_SPAWN_HEIGHT = 0.75
ANKLE_R = 1.5

WHOLE_BODY_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        usd_dir=_USD_DIR,
        usd_file_name="old_leg_only_humanoid.usd",
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
        # a slight crouch with the soles level: negative Knee bends the knee (its sign fixed Oct 22), positive
        # Ankle_P points the toes down, so Ankle_P = Hip_F + Knee levels the sole and Ankle_R = Hip_A takes the roll out
        joint_pos={
            "Hip_F_L": 0.12,
            "Hip_A_L": 0.04,
            "Hip_R_L": 0.0,
            "Knee_L": -0.22,
            "Ankle_P_L": -0.10,
            "Ankle_R_L": 0.04,
            "Hip_F_R": 0.12,
            "Hip_A_R": 0.04,
            "Hip_R_R": 0.0,
            "Knee_R": -0.22,
            "Ankle_P_R": -0.10,
            "Ankle_R_R": 0.04,
        },
    ),
    
    actuators={
        # AKH70-48: 222 N·m peak, 35 RPM no-load, 48:1, rotor 1170 g·cm²
        "hip_flexion": DCMotorCfg(
            joint_names_expr=["Hip_F_.*"],
            stiffness=267.0,            # 5 Hz
            damping=17.0,               # ζ ≈ 1.0
            armature=0.270,
            friction=2.2,               # datasheet back-drive torque
            saturation_effort=222.0,
            effort_limit=222.0,
            velocity_limit=3.67,
        ),
        # AK80-64 @ 48 V: 120 N·m peak, 75 RPM no-load, 64:1, rotor 564.5 g·cm²
        "hip_abduction": DCMotorCfg(
            joint_names_expr=["Hip_A_.*"],
            stiffness=146.0,            # 4 Hz
            damping=12.0,               # ζ ≈ 1.0
            armature=0.231,
            friction=2.0,               # estimate (not published)
            saturation_effort=120.0,
            effort_limit=120.0,
            velocity_limit=7.85,
        ),
        # AK80-64 @ 48 V (changed from AK60-39)
        "hip_rotation": DCMotorCfg(
            joint_names_expr=["Hip_R_.*"],
            stiffness=146.0,            # 4 Hz
            damping=12.0,               # ζ ≈ 1.0
            armature=0.231,
            friction=2.0,               # estimate (not published)
            saturation_effort=120.0,
            effort_limit=120.0,
            velocity_limit=7.85,
        ),
        # AKH70-48: same motor as hip flexion
        "knee": DCMotorCfg(
            joint_names_expr=["Knee_.*"],
            stiffness=267.0,            # 5 Hz
            damping=17.0,               # ζ ≈ 1.0
            armature=0.270,
            friction=2.2,
            saturation_effort=222.0,
            effort_limit=222.0,
            velocity_limit=3.67,
        ),
        # AK60-39 V3 through linkage: 72 N·m peak, 98 RPM no-load, 39:1, rotor 305 g·cm²
        # Motor values reflected to the joint at ratio r: torque × r, speed ÷ r, inertia × r²
        "ankle": DCMotorCfg(
            joint_names_expr=["Ankle_.*"],
            stiffness=103.0,            # 5 Hz
            damping=6.6,                # ζ ≈ 1.0
            armature=0.0464 * ANKLE_R**2,   # ≈ 0.104
            friction=1.4 * ANKLE_R,         # ≈ 2.1
            saturation_effort=72.0 * ANKLE_R,   # ≈ 108
            effort_limit=72.0 * ANKLE_R,
            velocity_limit=10.3 / ANKLE_R,      # ≈ 6.87
        ),
    },
    


    soft_joint_pos_limit_factor=0.95,
)