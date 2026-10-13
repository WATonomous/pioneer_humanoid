"""Warm-start an arm-select PPO checkpoint from a rule-assigned bimanual one.

  uv run scripts/add_arm_select.py \\
      --old ../../../models/badminton_bimanual_teacher/model_2999.pt \\
      --out logs/rsl_rl/badminton_teacher/init_arm_select/model_0.pt

The rule-assigned bimanual env had 12 actions (joint targets). The
arm-select env prepends one action, arm_select (> 0 right, <= 0 left, locked
on the first tick with |output| > control.arm_select_threshold), so:

  actor output   a new first row with zero weights and bias: the mean pick
                 is 0 and the exploration noise (--select-std) picks at
                 random until PPO learns the sign
  observation    the "actions" term (last action) gains the arm_select
                 column at its start; actor and critic get a zero-weight
                 input column there, normaliser mean 0 / var 1

Everything else is copied. The Adam moments are dropped (their shapes
changed); resume from the output with --agent.resume True.
"""

from __future__ import annotations

import argparse
import os

import torch

# first column of the "actions" observation term, in both groups:
# joint_pos (12), joint_vel (12), face_state (12) come before it
ACTIONS_COL = 36


def insert_col(t: torch.Tensor, col: int, value: float) -> torch.Tensor:
    pad = torch.full((*t.shape[:-1], 1), value, dtype=t.dtype)
    return torch.cat([t[..., :col], pad, t[..., col:]], dim=-1)


def widen_inputs(sd: dict) -> dict:
    out = {k: v.clone() for k, v in sd.items()}
    out["mlp.0.weight"] = insert_col(sd["mlp.0.weight"], ACTIONS_COL, 0.0)
    pre = "obs_normalizer."
    for key, value in ((pre + "_mean", 0.0), (pre + "_var", 1.0),
                       (pre + "_std", 1.0)):
        out[key] = insert_col(sd[key], ACTIONS_COL, value)
    return out


def widen_actor(sd: dict, select_std: float) -> dict:
    out = widen_inputs(sd)
    last = max(int(k.split(".")[1]) for k in sd if k.startswith("mlp."))
    W, b = sd[f"mlp.{last}.weight"], sd[f"mlp.{last}.bias"]
    out[f"mlp.{last}.weight"] = torch.cat([torch.zeros(1, W.shape[1]), W])
    out[f"mlp.{last}.bias"] = torch.cat([torch.zeros(1), b])
    std = sd["distribution.std_param"]
    out["distribution.std_param"] = torch.cat(
        [torch.full((1,), select_std, dtype=std.dtype), std])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True,
                    help="rule-assigned bimanual PPO checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select-std", type=float, default=0.5,
                    help="initial exploration std of arm_select")
    args = ap.parse_args()

    ckpt = torch.load(args.old, map_location="cpu", weights_only=False)
    actor = ckpt["actor_state_dict"]
    n_out = actor["distribution.std_param"].shape[0]
    assert n_out == 12, f"expected a 12-action bimanual actor, got {n_out}"
    new = {
        "actor_state_dict": widen_actor(actor, args.select_std),
        "critic_state_dict": widen_inputs(ckpt["critic_state_dict"]),
        # Adam moments are shape-bound and dropped; the param groups (same
        # tensor count, lr) stay so mjlab's resume can load the optimizer
        "optimizer_state_dict": {
            "state": {},
            "param_groups": ckpt["optimizer_state_dict"]["param_groups"]},
        "iter": 0,
        "infos": {"arm_select_from": os.path.abspath(args.old)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(new, args.out)
    shapes = {k: tuple(v.shape) for k, v in new["actor_state_dict"].items()
              if k.startswith("mlp.") and k.endswith("weight")}
    print(f"wrote {args.out}: actor {shapes}")


if __name__ == "__main__":
    main()
