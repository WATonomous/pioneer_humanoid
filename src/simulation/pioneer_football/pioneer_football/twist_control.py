"""Convert a heading or waypoint into explicit body translation and yaw intent."""

import torch

from .approach import wrap_angle


# Slow translation through large turns and allow a genuine stationary rotation toward the target heading.
def heading_twist(heading, target_heading, speed, yaw_limit=0.8):
    error = wrap_angle(target_heading - heading)
    forward = speed * ((0.6 - error.abs()) / 0.3).clamp(0, 1)
    return torch.stack(
        (
            forward * heading.cos(),
            forward * heading.sin(),
            (2 * error).clamp(-yaw_limit, yaw_limit),
        ),
        dim=1,
    )


# Preserve braking room at a waypoint instead of imposing a nonzero minimum walking speed.
def waypoint_twist(position, heading, target, maximum=0.5, yaw_limit=0.8):
    delta = target - position
    distance = delta.norm(dim=1)
    speed = torch.sqrt(0.8 * distance).clamp(max=maximum)
    bearing = torch.where(
        distance > 1e-6, torch.atan2(delta[:, 1], delta[:, 0]), heading
    )
    return heading_twist(heading, bearing, speed, yaw_limit)
