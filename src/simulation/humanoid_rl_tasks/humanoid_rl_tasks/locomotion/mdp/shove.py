"""Shoving for the Onshape course, with the applied disturbance logged.

The inherited Isaac Lab ``push_by_setting_velocity`` event adds a random
velocity to the root, but it is invisible from outside: the policy's next
action partially cancels it within a step or two, so watching the root's
velocity afterwards under-reports the shove that was actually applied.

This wrapper logs the applied disturbance (the change the event itself made to
env 0's root velocity) so a recording can be checked for "did the disturbance
fire, how hard, and where". It is a no-op for the training itself -- the
physics and the sampled shove are exactly the inherited event's.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.events import push_by_setting_velocity


def push_robot_logged(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict,
    asset_cfg=None,
    log_env: int = 0,
) -> None:
    """Apply the inherited push and log what env ``log_env`` received.

    Args:
        env: The environment.
        env_ids: The environments the event applies to.
        velocity_range: Per-axis ranges, as the inherited event takes them.
        asset_cfg: The asset to push (the robot).
        log_env: Which environment to report on (0 = the one in frame).
    """
    if asset_cfg is not None:
        asset = env.scene[asset_cfg.name]
        before = asset.data.root_vel_w[log_env].clone()
        push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
        after = asset.data.root_vel_w[log_env]
        dv = (after - before).tolist()
        if max(abs(dv[0]), abs(dv[1])) > 1e-4:
            pos = asset.data.root_pos_w[log_env].tolist()
            print(f"[SHOVE] t={env.common_step_counter * env.step_dt:.2f}s "
                  f"dv=({dv[0]:+.3f},{dv[1]:+.3f}) v=({after[0]:+.2f},{after[1]:+.2f}) "
                  f"root_y={pos[1]:+.2f}", flush=True)
    else:
        push_by_setting_velocity(env, env_ids, velocity_range)
