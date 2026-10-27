"""Config for DelayedDCMotor (#335 Task 3).

Inherits the fields of both parents:
- DelayedPDActuatorCfg: min_delay, max_delay (whole physics steps).
- DCMotorCfg: saturation_effort (stall torque, Nm). velocity_limit and effort_limit are required too.
The robot builds the actuator with `cfg.class_type(...)`.
"""

from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.utils import configclass

from .delayed_dc_motor import DelayedDCMotor


@configclass
class DelayedDCMotorCfg(DelayedPDActuatorCfg, DCMotorCfg):
    """Configuration for :class:`DelayedDCMotor`."""

    class_type: type = DelayedDCMotor
