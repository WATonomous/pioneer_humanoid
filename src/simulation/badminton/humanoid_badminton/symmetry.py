"""Left/right mirror of observations and actions for rsl_rl's Symmetry
extension (PPO symmetry_cfg.data_augmentation_func).

The task is mirror-symmetric about the plane x = 0: the arm bases sit at
x = +/-0.163, the court and net are centred on x = 0, and the left arm is
the right arm's mirror image under q_left = -q_right for every joint
(scripts check: racket position error 0 m, face normal M n_R . n_L = 1 over
random joint draws; joint ranges map onto each other the same way). The
mirror M = diag(-1, 1, 1) therefore maps a valid state to a valid state,
and the mirrored action is the mirrored arm doing the mirrored motion.

Without it, the arm pick has no reason to be symmetric: shots near the
centre line can be hit by either arm, and the cold teacher (cold1) gave
the whole centre band to the right arm (64% of hits). Training on each
mini-batch plus its mirror, with the mirror loss, pushes the policy toward
pi(M s) = M pi(s), so mirrored shots go to the mirrored arm.

Per observation term (the layouts in env_cfg / mdp / perception_command):

  joint_pos, joint_vel   [right 6, left 6] -> [-left, -right]
  face_state             [p_R, n_R, p_L, n_L] -> [M p_L, M n_L, M p_R, M n_R]
  actions                [select, right 6, left 6] -> [-select, -left, -right]
  arm_assign             [right, left] -> [left, right]
  shuttle                [p, v, traj points] (all 3-vectors) -> x negated
  shuttle_uncertainty    standard deviations and a norm: unchanged
  intercept              [p*, time] -> x of p* negated
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

N_ARM = 6


def _swap_arms(n_per_arm: int) -> list[int]:
    return list(range(n_per_arm, 2 * n_per_arm)) + list(range(n_per_arm))


def _term_map(name: str, dim: int) -> tuple[list[int], list[float]]:
    """(source index, sign) per output element of one term:
    mirrored[i] = sign[i] * obs[perm[i]]."""
    if name in ("joint_pos", "joint_vel"):
        assert dim == 2 * N_ARM, (name, dim)
        return _swap_arms(N_ARM), [-1.0] * dim
    if name == "face_state":
        assert dim == 12, (name, dim)
        sign = [-1.0, 1.0, 1.0] * 4
        return _swap_arms(6), sign
    if name == "actions":
        assert dim == 1 + 2 * N_ARM, (name, dim)
        return [0] + [1 + i for i in _swap_arms(N_ARM)], [-1.0] * dim
    if name == "arm_assign":
        assert dim == 2, (name, dim)
        return [1, 0], [1.0, 1.0]
    if name == "shuttle":
        assert dim % 3 == 0, (name, dim)
        return list(range(dim)), [-1.0 if i % 3 == 0 else 1.0
                                  for i in range(dim)]
    if name == "shuttle_uncertainty":
        return list(range(dim)), [1.0] * dim
    if name == "intercept":
        assert dim == 4, (name, dim)
        return list(range(dim)), [-1.0, 1.0, 1.0, 1.0]
    raise KeyError(f"no mirror rule for observation term {name!r}")


def group_maps(obs_manager) -> dict[str, tuple[list[int], list[float]]]:
    """Flat (perm, sign) per observation group from the manager's terms."""
    out = {}
    for group, names in obs_manager.active_terms.items():
        perm, sign, off = [], [], 0
        for name, shape in zip(names, obs_manager.group_obs_term_dim[group]):
            assert len(shape) == 1, (group, name, shape)
            p, s = _term_map(name, shape[0])
            perm += [off + i for i in p]
            sign += s
            off += shape[0]
        out[group] = (perm, sign)
    return out


ACTION_PERM, ACTION_SIGN = _term_map("actions", 1 + 2 * N_ARM)

_cache: dict = {}


def _tensors(env, device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    key = (id(env), str(device))
    if key not in _cache:
        maps = group_maps(env.unwrapped.observation_manager)
        obs_t = {g: (torch.tensor(p, device=device),
                     torch.tensor(s, device=device)) for g, (p, s) in maps.items()}
        _cache[key] = (obs_t, torch.tensor(ACTION_PERM, device=device),
                       torch.tensor(ACTION_SIGN, device=device))
    return _cache[key]


def mirror_obs_actions(env, obs: TensorDict | None = None,
                       actions: torch.Tensor | None = None):
    """rsl_rl data_augmentation_func: returns [original; mirrored] stacked
    along the batch dimension, for observations and actions (None in, None
    out)."""
    obs_out = act_out = None
    if obs is not None:
        obs_t, _, _ = _tensors(env, obs.device)
        mirrored = {}
        for group in obs.keys():
            perm, sign = obs_t[group]
            mirrored[group] = obs[group][..., perm] * sign
        mirrored = TensorDict(mirrored, batch_size=obs.batch_size,
                              device=obs.device)
        obs_out = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        _, perm, sign = _tensors(env, actions.device)
        act_out = torch.cat([actions, actions[..., perm] * sign], dim=0)
    return obs_out, act_out
