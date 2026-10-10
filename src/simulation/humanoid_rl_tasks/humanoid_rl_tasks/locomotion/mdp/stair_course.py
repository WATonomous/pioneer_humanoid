"""Completion and curriculum for a forward up–walk–down stair course.

Only the stairs task imports these terms. All coordinates are relative to the
terrain's spawn origin, so tiles can be translated freely in the training map.
"""

from __future__ import annotations

import torch


def course_complete(
    env,
    goal_distance: float,
    lateral_limit: float = 2.0,
    minimum_height: float = 0.65,
    tilt_limit: float = 0.8,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Reach the flat exit upright, inside the course (not just the top)."""
    data = env.scene[asset_name].data
    position = data.root_pos_w - env.scene.env_origins
    gravity = data.projected_gravity_b
    finite = torch.isfinite(position).all(dim=1) & torch.isfinite(gravity).all(dim=1)
    tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
    return (
        finite
        & (position[:, 0] >= goal_distance)
        & (position[:, 1].abs() <= lateral_limit)
        # The goal is on the flat exit at spawn height, not on elevated stairs.
        & (position[:, 2] >= minimum_height)
        & (tilt <= tilt_limit)
        # XY tilt alone is also zero when fully upside-down.
        & (gravity[:, 2] < 0.0)
    )


def course_out_of_bounds(
    env,
    lateral_limit: float = 2.0,
    backward_limit: float = -0.7,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Reset before the robot can leave its lane or bypass the stair flights."""
    position = env.scene[asset_name].data.root_pos_w - env.scene.env_origins
    return (
        ~torch.isfinite(position).all(dim=1)
        | (position[:, 1].abs() > lateral_limit)
        | (position[:, 0] < backward_limit)
    )


def complete_stair_course_levels(
    env,
    env_ids,
    goal_distance: float,
    lateral_limit: float = 2.0,
    minimum_height: float = 0.65,
    tilt_limit: float = 0.8,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Promote a full upright traversal; demote unsuccessful completed episodes.

    The initial reset has no rollout and must not alter terrain levels. A fall
    on the same frame as crossing the goal is not a success. No reward, asset,
    observation, actuator, or PPO parameter is modified by this curriculum.
    """
    completed = course_complete(env, goal_distance, lateral_limit, minimum_height, tilt_limit, asset_name)
    has_rollout = env.episode_length_buf[env_ids] > 0
    move_up = completed[env_ids] & ~env.termination_manager.terminated[env_ids] & has_rollout
    move_down = ~move_up & has_rollout
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up=move_up, move_down=move_down)
    return terrain.terrain_levels.float().mean()
