"""Widen a foothold-task checkpoint for the 8-row (one row behind) patch.

The foothold scan used to return 51 values per foot -- [x, y, 49 heights] for a
7 x 7 patch. With ``patch_rows_behind=1`` it returns 58: the extra row is
flattened *first*, so the per-foot block is [x, y, 7 new heights, 49 old
heights]. The policy input grows 152 -> 166 and the critic 845 -> 859 (the
critic's foothold block starts at 743: 48 base + 693 height map + 2 gait clock).

This script rebuilds the first layer of both networks at the new width, copying
each old column to its new position and zero-filling the 14 new ones, so the
loaded policy behaves exactly like the source until training moves the new
weights. Optimizer state is dropped on purpose: an enlarged first layer cannot
be resumed, so pair this with ``train.py --warm_start_checkpoint``.

    isaaclab.sh -p tools/onshape_course/widen_foothold_obs.py \\
        --checkpoint outputs/rl/.../model_3549.pt --out /tmp/model_3549_8row.pt
"""

import argparse

import torch

OLD_ACTOR = 152
NEW_ACTOR = 166
OLD_CRITIC = 845
NEW_CRITIC = 859
FOOTHOLD_OLD = 51  # 2 xy + 49 heights, per foot
EXTRA = 7         # one extra row of 7 heights, per foot


def _new_column(old_col: int, foothold_start: int) -> int:
    """Where an old input column lives in the widened layout."""
    if old_col < foothold_start:
        return old_col  # head inputs (base state, joints, actions, gait clock) do not move
    rel = old_col - foothold_start
    foot, within = divmod(rel, FOOTHOLD_OLD)
    return foothold_start + foot * (FOOTHOLD_OLD + EXTRA) + within + (EXTRA if within >= 2 else 0)


def widen(weight: torch.Tensor, old_n: int, new_n: int, foothold_start: int) -> torch.Tensor:
    assert weight.shape[1] == old_n, f"expected {old_n} input columns, got {weight.shape[1]}"
    out = torch.zeros(weight.shape[0], new_n, dtype=weight.dtype)
    for col in range(old_n):
        out[:, _new_column(col, foothold_start)] = weight[:, col]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    if "actor.0.weight" not in state or "critic.0.weight" not in state:
        raise SystemExit("checkpoint has no actor/critic first layers; wrong file?")

    state["actor.0.weight"] = widen(state["actor.0.weight"], OLD_ACTOR, NEW_ACTOR, foothold_start=50)
    state["critic.0.weight"] = widen(state["critic.0.weight"], OLD_CRITIC, NEW_CRITIC, foothold_start=743)

    ck["model_state_dict"] = state
    ck.pop("optimizer_state_dict", None)
    ck["iter"] = 0
    torch.save(ck, args.out)
    print(f"[WIDEN] {args.checkpoint} -> {args.out}")
    print(f"[WIDEN] actor.0.weight {tuple(state['actor.0.weight'].shape)}  "
          f"critic.0.weight {tuple(state['critic.0.weight'].shape)}")


if __name__ == "__main__":
    main()
