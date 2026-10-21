"""Simulator-neutral leg motor parameters: CubeMars datasheet specs, joint -> motor map, MIT gain check.

Plain Python (no Isaac imports), like arm_params.py, so tests and sysid tools run on the host and
whole_body.py (Isaac Lab) can build its actuator configs from the same numbers.

Values are stored in datasheet units (rpm, g*cm^2) so each one can be checked against the manual
by eye; SI conversions are properties. All torques and speeds are at the output shaft, 48 V.
"""
import math
from dataclasses import dataclass

_RPM_TO_RAD_S = 2.0 * math.pi / 60.0  # 1 rev = 2*pi rad, 1 min = 60 s
_GCM2_TO_KGM2 = 1e-7  # 1 g = 1e-3 kg, 1 cm^2 = 1e-4 m^2


@dataclass(frozen=True)
class MotorSpec:
    name: str
    gear_ratio: float  # rotor turns per output turn
    peak_torque: float  # Nm; MIT mode can output up to this (D13 effort cap)
    rated_torque: float  # Nm; thermal (time-averaged) limit, Task 5 metric
    no_load_speed_rpm: float  # rpm
    rotor_inertia_gcm2: float  # g*cm^2, rotor only (motor side of the gearbox)
    backdrive_torque: float  # Nm; first guess for friction until Task 11 measures it
    # MIT frame encoding ranges (symmetric: min = -max). These set the scale of the packed
    # integer fields, NOT what the motor can physically do (e.g. AKH70-48 t_max 280 > peak 222).
    p_max: float  # rad
    v_max: float  # rad/s
    t_max: float  # Nm
    kp_max: float = 500.0  # MIT kp range upper bound (Nm/rad); lower bound 0
    kd_max: float = 5.0  # MIT kd range upper bound (Nm*s/rad); lower bound 0

    @property
    def no_load_speed(self) -> float:
        """rad/s."""
        return self.no_load_speed_rpm * _RPM_TO_RAD_S

    @property
    def rotor_inertia(self) -> float:
        """kg*m^2, motor side."""
        return self.rotor_inertia_gcm2 * _GCM2_TO_KGM2

    @property
    def armature(self) -> float:
        """kg*m^2 at the output shaft. The rotor spins gear_ratio times faster than the output,
        so its inertia appears gear_ratio^2 bigger there."""
        return self.rotor_inertia * self.gear_ratio**2


# Values from the plan's motor fact sheet (manuals: AK V3.2.0 §4.2, AK driver V1.0.18 §5.3).
AKH70_48 = MotorSpec(
    name="AKH70-48", gear_ratio=48, peak_torque=222.0, rated_torque=74.0,
    no_load_speed_rpm=35.0, rotor_inertia_gcm2=1170.0, backdrive_torque=2.22,
    p_max=12.56, v_max=5.0, t_max=280.0,
)
AK80_64 = MotorSpec(  # hip yaw (D1); AK 2.0 protocol
    name="AK80-64 KV80", gear_ratio=64, peak_torque=120.0, rated_torque=48.0,
    no_load_speed_rpm=75.0, rotor_inertia_gcm2=564.5, backdrive_torque=4.7,
    p_max=12.5, v_max=8.0, t_max=144.0,
)
AK60_39 = MotorSpec(  # ankle, via linkage (D10)
    name="AK60-39 V3.0 KV80", gear_ratio=39, peak_torque=72.0, rated_torque=24.0,
    no_load_speed_rpm=98.0, rotor_inertia_gcm2=305.0, backdrive_torque=1.4,
    p_max=12.56, v_max=10.0, t_max=80.0,
)

_MOTOR_BY_JOINT = {
    "Hip_F": AKH70_48,
    "Knee": AKH70_48,
    "Hip_A": AK80_64,
    "Hip_R": AK80_64,
    "Ankle_P": AK60_39,
    "Ankle_R": AK60_39,
}

# URDF joint name -> motor. Both legs use the same motors.
JOINT_MOTOR = {f"{joint}_{side}": motor for joint, motor in _MOTOR_BY_JOINT.items() for side in ("L", "R")}


def check_gains(joint_name: str, kp: float, kd: float, r: float = 1.0) -> None:
    """Raise ValueError if joint-side gains can't be sent to the joint's motor in MIT mode.

    kp, kd are joint-side (what the sim config uses). r is the linkage ratio
    (joint torque = motor torque * r); 1.0 for direct drive and for the ankle placeholder (D10).
    The motor sees kp / r^2 and kd / r^2: the motor turns r times further than the joint, and its
    torque is multiplied by r again on the way out.
    can_node clamps out-of-range gains on the real robot only; the sim would silently use them.
    """
    motor = JOINT_MOTOR[joint_name]
    if r <= 0:
        raise ValueError(f"{joint_name}: linkage ratio r must be > 0, got {r}")
    if kp < 0 or kd < 0:
        raise ValueError(f"{joint_name}: gains must be >= 0, got kp={kp}, kd={kd}")
    motor_kp, motor_kd = kp / r**2, kd / r**2
    if motor_kp > motor.kp_max:
        raise ValueError(f"{joint_name}: motor kp {motor_kp:g} > {motor.name} max {motor.kp_max:g} (r={r:g})")
    if motor_kd > motor.kd_max:
        raise ValueError(f"{joint_name}: motor kd {motor_kd:g} > {motor.name} max {motor.kd_max:g} (r={r:g})")
