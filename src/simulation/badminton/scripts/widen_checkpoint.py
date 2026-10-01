"""Warm-start a bimanual PPO checkpoint from a single-arm (right) one.

  uv run scripts/widen_checkpoint.py --group teacher \\
      --old ../../../models/badminton_teacher/model_5996.pt \\
      --out logs/rsl_rl/badminton_teacher/init_bimanual/model_0.pt

The single-arm policy saw 6-joint proprioception and one racket face, with
the stand at arm.base_x = -0.245 (right arm swing plane on x = 0). The
bimanual env has 12 joints, two faces, an arm-assignment one-hot, and the
stand centred at x = 0. The new actor is the old one twice, side by side
(hidden widths doubled, block-diagonal):

  block A  the old network on the right arm's view: world x inputs shifted
           by OLD_BASE_X, so the right arm sees the geometry it was trained
           on; drives the right-arm outputs
  block B  the same network on the mirrored view (x -> -x about the stand
           centre, left joints times arm.left_mirror, then the same shift);
           drives the left-arm outputs through left_mirror

The mirrored view is a fixed linear map of the observation, so it is folded
into block B's first layer and bias (exact, given the normaliser). The new
input normaliser holds the old statistics mapped into the new layout, so
block A's first layer is a plain copy. The assignment one-hot gets zero
weights. The critic keeps its width; its input is widened like block A
(zero weights on left-arm columns). The Adam moments are dropped (their
shapes changed); resume from the output with --agent.resume True.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aero  # noqa: E402

# arm.base_x of the single-arm scene. The right arm sat 0.245 m further
# left (-x) than it does now, so an old world x = new world x + OLD_BASE_X.
OLD_BASE_X = -0.245

# observation layouts, in observation-manager order: (term, old dim, new dim)
# per-arm terms: the old block is the right arm, the new one right + left
LAYOUT = {
    "teacher": [("joint_pos", 6, 12), ("joint_vel", 6, 12),
                ("face_state", 6, 12), ("actions", 6, 12),
                ("arm_assign", 0, 2), ("shuttle", 30, 30),
                ("intercept", 4, 4)],
    "student": [("joint_pos", 6, 12), ("joint_vel", 6, 12),
                ("face_state", 6, 12), ("actions", 6, 12),
                ("arm_assign", 0, 2), ("shuttle", 30, 30),
                ("shuttle_uncertainty", 7, 7)],
}


def column_maps(group: str, mirror: list[float]):
    """For each old column j and each block, the new column k, the sign and
    the shift with o_old_j = sign * o_new_k + shift. Returns (block_a,
    block_b, n_old, n_new, assign_cols); each block is a list of (k, sign,
    shift) indexed by j."""
    a, b = [], []
    k0 = 0
    assign_cols = []
    for term, n_old, n_new in LAYOUT[group]:
        for j in range(n_old):
            if term in ("joint_pos", "joint_vel", "actions"):
                a.append((k0 + j, 1.0, 0.0))
                b.append((k0 + 6 + j, float(mirror[j]), 0.0))
            elif term == "face_state":
                # per face [pos(3), normal(3)]; x components mirror
                is_x = j in (0, 3)
                shift = OLD_BASE_X if j == 0 else 0.0
                a.append((k0 + j, 1.0, shift))
                b.append((k0 + 6 + j, -1.0 if is_x else 1.0, shift))
            elif term in ("shuttle", "intercept"):
                # shuttle: [p(3), v(3), traj 8x(3)] -> x at j % 3 == 0;
                # positions are p and traj (all but v at 3..5)
                # intercept: [p*(3), time]
                if term == "shuttle":
                    is_x = j % 3 == 0
                    is_pos = not (3 <= j < 6)
                else:
                    is_x = j == 0
                    is_pos = j < 3
                shift = OLD_BASE_X if (is_x and is_pos) else 0.0
                a.append((k0 + j, 1.0, shift))
                b.append((k0 + j, -1.0 if is_x else 1.0, shift))
            else:   # shuttle_uncertainty: unsigned stds and a jump norm
                a.append((k0 + j, 1.0, 0.0))
                b.append((k0 + j, 1.0, 0.0))
        if term == "arm_assign":
            assign_cols = list(range(k0, k0 + n_new))
        k0 += n_new
    n_old = sum(t[1] for t in LAYOUT[group])
    return a, b, n_old, k0, assign_cols


def new_normalizer(old: dict, prefix: str, block_a, block_b, n_new, assign_cols):
    """Old stats in the new layout: right columns are the inverse of block
    A's map, left-arm columns the inverse of block B's; assignment one-hot
    gets mean 0.5, var 0.25."""
    mu = old[prefix + "_mean"][0]
    var = old[prefix + "_var"][0]
    new_mu = torch.zeros(n_new)
    new_var = torch.ones(n_new)
    for blk in (block_b, block_a):          # block A wins on shared columns
        for j, (k, sign, shift) in enumerate(blk):
            new_mu[k] = (mu[j] - shift) / sign
            new_var[k] = var[j]
    new_mu[assign_cols] = 0.5
    new_var[assign_cols] = 0.25
    return {prefix + "_mean": new_mu.unsqueeze(0),
            prefix + "_var": new_var.unsqueeze(0),
            prefix + "_std": new_var.sqrt().unsqueeze(0),
            prefix + "count": old[prefix + "count"].clone()}


def fold_first_layer(W, b, old_sd, new_norm, prefix, block, n_new, eps):
    """First-layer weights/bias over the new normalised input z' that
    reproduce W z_old + b, where z_old is the old normalisation of the
    block's view of the observation."""
    mu = old_sd[prefix + "_mean"][0]
    sig = old_sd[prefix + "_std"][0] + eps
    mu_n = new_norm[prefix + "_mean"][0]
    sig_n = new_norm[prefix + "_std"][0] + eps
    W_new = torch.zeros(W.shape[0], n_new)
    b_new = b.clone()
    for j, (k, sign, shift) in enumerate(block):
        a_j = sign * sig_n[k] / sig[j]
        b_j = (sign * mu_n[k] + shift - mu[j]) / sig[j]
        W_new[:, k] += W[:, j] * a_j
        b_new += W[:, j] * b_j
    return W_new, b_new


def widen_actor(sd: dict, group: str, mirror, eps: float) -> dict:
    a, b, n_old, n_new, assign_cols = column_maps(group, mirror)
    assert sd["mlp.0.weight"].shape[1] == n_old, (
        f"checkpoint actor has {sd['mlp.0.weight'].shape[1]} inputs, "
        f"layout {group} expects {n_old}")
    pre = "obs_normalizer."
    norm = new_normalizer(sd, pre, a, b, n_new, assign_cols)
    out = dict(norm)
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    first, last = layers[0], layers[-1]
    W0, b0 = sd[f"mlp.{first}.weight"], sd[f"mlp.{first}.bias"]
    Wa, ba = fold_first_layer(W0, b0, sd, norm, pre, a, n_new, eps)
    Wb, bb = fold_first_layer(W0, b0, sd, norm, pre, b, n_new, eps)
    out[f"mlp.{first}.weight"] = torch.cat([Wa, Wb], dim=0)
    out[f"mlp.{first}.bias"] = torch.cat([ba, bb])
    for i in layers[1:-1]:
        W, bias = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
        out[f"mlp.{i}.weight"] = torch.block_diag(W, W)
        out[f"mlp.{i}.bias"] = torch.cat([bias, bias])
    # output: block A -> right joints, block B -> left joints * mirror
    W, bias = sd[f"mlp.{last}.weight"], sd[f"mlp.{last}.bias"]
    m = torch.tensor(mirror, dtype=W.dtype).unsqueeze(-1)
    out[f"mlp.{last}.weight"] = torch.block_diag(W, W * m)
    out[f"mlp.{last}.bias"] = torch.cat([bias, bias * m.squeeze(-1)])
    std = sd["distribution.std_param"]
    out["distribution.std_param"] = torch.cat([std, std])
    return out


def widen_critic(sd: dict, group: str, mirror, eps: float) -> dict:
    a, b, n_old, n_new, assign_cols = column_maps(group, mirror)
    assert sd["mlp.0.weight"].shape[1] == n_old
    pre = "obs_normalizer."
    norm = new_normalizer(sd, pre, a, b, n_new, assign_cols)
    out = {k: v.clone() for k, v in sd.items() if k.startswith("mlp.")}
    out.update(norm)
    out["mlp.0.weight"], out["mlp.0.bias"] = fold_first_layer(
        sd["mlp.0.weight"], sd["mlp.0.bias"], sd, norm, pre, a, n_new, eps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="single-arm PPO checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", required=True, choices=sorted(LAYOUT),
                    help="observation group of the old actor (teacher for "
                         "the teacher, student for student PPO)")
    ap.add_argument("--critic-group", default="teacher",
                    help="observation group of the critic (teacher for both "
                         "PPO tasks)")
    ap.add_argument("--eps", type=float, default=1e-2,
                    help="EmpiricalNormalization eps (rsl_rl default)")
    args = ap.parse_args()

    mirror = aero.load_params()["arm"]["left_mirror"]
    ckpt = torch.load(args.old, map_location="cpu", weights_only=False)
    new = {
        "actor_state_dict": widen_actor(ckpt["actor_state_dict"], args.group,
                                        mirror, args.eps),
        "critic_state_dict": widen_critic(ckpt["critic_state_dict"],
                                          args.critic_group, mirror, args.eps),
        # Adam moments are shape-bound and dropped; the param groups (same
        # tensor count, lr) stay so mjlab's resume can load the optimizer
        "optimizer_state_dict": {
            "state": {},
            "param_groups": ckpt["optimizer_state_dict"]["param_groups"]},
        "iter": 0,
        "infos": {"widened_from": os.path.abspath(args.old)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(new, args.out)
    shapes = {k: tuple(v.shape) for k, v in new["actor_state_dict"].items()
              if k.startswith("mlp.") and k.endswith("weight")}
    print(f"wrote {args.out}: actor {shapes}")


if __name__ == "__main__":
    main()
