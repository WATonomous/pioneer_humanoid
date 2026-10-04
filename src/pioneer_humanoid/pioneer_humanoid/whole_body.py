import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from .actuators import DelayedTrapezoidalPDActuatorCfg

_ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "whole_body_humanoid"
)
_URDF_PATH = os.path.join(_ASSET_DIR, "whole_body_humanoid.urdf")
_USD_DIR = os.path.join(_ASSET_DIR, "usd")

# Mild crouch; keep feet near ground after bend.
_SPAWN_HEIGHT = 0.75


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
            "Hip_A_L": 0.04,
            "Hip_R_L": 0.0,
            "Knee_L": -0.22,
            "Ankle_P_L": 0.10,
            "Ankle_R_L": 0.0,
            "Hip_F_R": 0.12,
            "Hip_A_R": 0.04,
            "Hip_R_R": 0.0,
            "Knee_R": -0.22,
            "Ankle_P_R": 0.10,
            "Ankle_R_R": 0.0,
        },
    ),
    # NOTE: acceleration_limit and min_delay/max_delay below are PLACEHOLDER estimates
    # (accel ~= velocity_limit_sim / 0.15s; delay ~= 0-4 control steps @ dt=0.005s),
    # not values fit against real hardware. See PR description for the required
    # step/sine sysID sweep per joint before these should be trusted for training a
    # policy intended for real deployment.
    actuators={
        "hip_flexion": DelayedTrapezoidalPDActuatorCfg(
            joint_names_expr=["Hip_F_.*"],
            effort_limit_sim=222.0,
            velocity_limit_sim=3.6652,
            stiffness=100.0,
            damping=8.0,
            acceleration_limit=24.43,
            min_delay=0,
            max_delay=4,
        ),
        "hip_abduction": DelayedTrapezoidalPDActuatorCfg(
            joint_names_expr=["Hip_A_.*"],
            effort_limit_sim=120.0,
            velocity_limit_sim=20.944,
            stiffness=80.0,
            damping=5.0,
            acceleration_limit=139.63,
            min_delay=0,
            max_delay=4,
        ),
        "hip_rotation": DelayedTrapezoidalPDActuatorCfg(
            joint_names_expr=["Hip_R_.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=20.42,
            stiffness=60.0,
            damping=4.0,
            acceleration_limit=136.13,
            min_delay=0,
            max_delay=4,
        ),
        "knee": DelayedTrapezoidalPDActuatorCfg(
            joint_names_expr=["Knee_.*"],
            effort_limit_sim=222.0,
            velocity_limit_sim=3.6652,
            stiffness=100.0,
            damping=8.0,
            acceleration_limit=24.43,
            min_delay=0,
            max_delay=4,
        ),
        "ankle": DelayedTrapezoidalPDActuatorCfg(
            joint_names_expr=["Ankle_.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=20.42,
            stiffness=80.0,
            damping=6.0,
            acceleration_limit=136.13,
            min_delay=0,
            max_delay=4,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
