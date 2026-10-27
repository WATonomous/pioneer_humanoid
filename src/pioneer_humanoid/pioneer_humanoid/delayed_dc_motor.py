"""DelayedDCMotor: delayed PD actuator with a DC-motor torque-speed limit (#335 Task 3).

Combines two Isaac Lab 2.3.2 actuators by multiple inheritance:
- DelayedPDActuator: delays the targets by a whole number of physics steps, drawn per env at reset.
- DCMotor: clips torque with a linear torque-speed curve (saturation_effort, velocity_limit, effort_limit).

Method resolution order (MRO), i.e. the order `super()` walks through:
    DelayedDCMotor -> DelayedPDActuator -> DCMotor -> IdealPDActuator -> ActuatorBase
so compute() runs: delay targets -> DCMotor saves joint_vel -> PD math -> DCMotor._clip_effort.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

_MISSING_TYPE = type(MISSING)  # class of the "field not set" placeholder

from isaaclab.actuators import DCMotor, DelayedPDActuator

if TYPE_CHECKING:  # only for type hints; avoids a circular import with the cfg file
    from .delayed_dc_motor_cfg import DelayedDCMotorCfg

# Limits the torque-speed clip needs, with what each one means.
# None is rejected as well as MISSING: for velocity_limit / effort_limit, Isaac treats None as
# "use the USD value", which silently takes the URDF limits (wrong for these motors, D4/D13).
REQUIRED_LIMITS = {
    "saturation_effort": "stall torque in Nm (torque available at zero speed)",
    "velocity_limit": "no-load speed in rad/s (where available torque reaches 0)",
    "effort_limit": "flat torque cap in Nm (peak torque, D13)",
}


class DelayedDCMotor(DelayedPDActuator, DCMotor):
    """Delayed PD actuator whose output torque is limited by a DC-motor torque-speed curve."""

    cfg: DelayedDCMotorCfg

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        # Check before super().__init__: DCMotor divides by saturation_effort there, so a missing value
        # would otherwise crash with an unclear TypeError ('Tensor' / '_MISSING_TYPE').
        # isinstance, not `is MISSING`: configclass copies field values, so the placeholder may be
        # a different _MISSING_TYPE object than dataclasses.MISSING (identity check fails, type check works).
        missing = [
            name for name in REQUIRED_LIMITS if getattr(cfg, name) is None or isinstance(getattr(cfg, name), _MISSING_TYPE)
        ]
        if missing:
            details = "\n".join(f"  - {name}: {REQUIRED_LIMITS[name]}" for name in missing)
            raise ValueError(
                f"DelayedDCMotorCfg for joints {cfg.joint_names_expr} is missing required limits:\n{details}\n"
                "Set them explicitly (from leg_params), not via the URDF/USD fallback."
            )
        super().__init__(cfg, *args, **kwargs)
