import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from .delayed_dc_motor_cfg import DelayedDCMotorCfg
from .leg_params import JOINT_MOTOR, check_gains

_ASSET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "whole_body_humanoid"
)
_URDF_PATH = os.path.join(_ASSET_DIR, "whole_body_humanoid.urdf")
_USD_DIR = os.path.join(_ASSET_DIR, "usd")

# Mild crouch; keep feet near ground after bend.
_SPAWN_HEIGHT = 0.75

# PhysX hard speed clamp = this x datasheet no-load speed (D15). Safety net only: the DCMotor
# torque-speed clip does the realistic limiting, and its braking torque is already at -peak
# below this speed.
_VELOCITY_LIMIT_SIM_MARGIN = 1.5

# Command delay in whole physics steps (5 ms each, D6), drawn per env at reset.
_MIN_DELAY = 0
_MAX_DELAY = 2


def _leg_actuator(joints: list[str], kp: float, kd: float) -> DelayedDCMotorCfg:
    """DelayedDCMotorCfg for leg joints that share one motor model, with limits from leg_params.

    joints: names without the _L/_R suffix, e.g. ["Ankle_P", "Ankle_R"].
    kp, kd: joint-side gains, checked against the motor's MIT range (r = 1 for now, D10).
    """
    names = [f"{joint}_{side}" for joint in joints for side in ("L", "R")]
    motors = {JOINT_MOTOR[name] for name in names}
    if len(motors) != 1:
        raise ValueError(f"{joints} use different motors: {sorted(m.name for m in motors)}")
    motor = motors.pop()
    for name in names:
        check_gains(name, kp, kd)
    return DelayedDCMotorCfg(
        joint_names_expr=[f"{joint}_.*" for joint in joints],
        stiffness=kp,
        damping=kd,
        effort_limit=motor.peak_torque,  # flat cap at peak (D13; x r at the ankle, r = 1)
        saturation_effort=motor.saturation_effort,  # chart line extended to zero speed (D14)
        velocity_limit=motor.torque_speed_limit,  # where the chart line reaches 0 Nm (D14)
        velocity_limit_sim=_VELOCITY_LIMIT_SIM_MARGIN * motor.no_load_speed,  # D15
        # effort_limit_sim left unset: explicit actuators default to 1e9 (no PhysX cap); the clip above caps.
        armature=motor.armature,  # rotor inertia x gear_ratio^2
        friction=motor.backdrive_torque,  # Nm on Isaac Sim 5.x; first guess until Task 11
        min_delay=_MIN_DELAY,
        max_delay=_MAX_DELAY,
    )


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
    # Gain set A (D5): kp unchanged, kd capped at the MIT max of 5 (D4).
    actuators={
        "hip_flexion": _leg_actuator(["Hip_F"], kp=100.0, kd=5.0),  # kd was 8
        "hip_abduction": _leg_actuator(["Hip_A"], kp=80.0, kd=5.0),
        "hip_rotation": _leg_actuator(["Hip_R"], kp=60.0, kd=4.0),
        "knee": _leg_actuator(["Knee"], kp=100.0, kd=5.0),  # kd was 8
        "ankle": _leg_actuator(["Ankle_P", "Ankle_R"], kp=80.0, kd=5.0),  # kd was 6
    },
    soft_joint_pos_limit_factor=0.95,
)
