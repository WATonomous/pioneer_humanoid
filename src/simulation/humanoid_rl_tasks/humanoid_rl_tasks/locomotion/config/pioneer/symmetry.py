"""Left-right mirror of the Pioneer leg-only humanoid's observations and actions, for rsl_rl's symmetry augmentation.

Each observation group ("policy", and "critic" if there is one) is mirrored term by term, by name, as in
leggedrobotics/foothold-prediction-model. Joints come in left/right pairs (Isaac Lab orders them Hip_F_L, Hip_F_R, ...)
and the URDF mirrors each left axis onto its right partner, so mirroring a joint vector is a plain swap with no sign
change. The gait clock's mirror is the other leg's phase, half a cycle on: sin and cos change sign.
"""
import math

import torch
from tensordict import TensorDict

_SIGNS = {
    "base_lin_vel": (1, -1, 1),
    "base_ang_vel": (-1, 1, -1),
    "projected_gravity": (1, -1, 1),
    "velocity_commands": (1, -1, -1),
}
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")


def _swap(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, 6, 2).flip(-1).reshape(-1, 12)


def _mirror_group(env, obs: torch.Tensor, group: str) -> torch.Tensor:
    manager = env.observation_manager
    out, idx = obs.clone(), 0
    for name, shape in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
        dim = math.prod(shape)
        seg = obs[:, idx : idx + dim]
        if name in _SIGNS:
            out[:, idx : idx + dim] = seg * torch.tensor(_SIGNS[name], device=obs.device, dtype=obs.dtype)
        elif name in _JOINT_TERMS:
            out[:, idx : idx + dim] = _swap(seg)
        elif name == "gait_clock":
            out[:, idx : idx + dim] = -seg
        else:
            raise ValueError(f"no mirror rule for observation term '{name}' in group '{group}'")
        idx += dim
    return out


@torch.no_grad()
def mirror(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Returns (obs, actions) doubled: the originals, then their mirror images."""
    env = env.unwrapped
    actions_aug = None if actions is None else torch.cat([actions, _swap(actions)])
    if obs is None:
        return None, actions_aug
    n = obs.batch_size[0]
    aug = {key: torch.cat([obs[key], _mirror_group(env, obs[key], key) if key in ("policy", "critic") else obs[key]])
           for key in obs.keys()}
    return TensorDict(aug, batch_size=[2 * n], device=obs.device), actions_aug
