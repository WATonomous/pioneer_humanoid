"""Expand a 12-joint walker checkpoint to the 16-joint arm-swing robot.

The arm-swing asset keeps ``l_shoulder_pitch``, ``r_shoulder_pitch``, ``l_elbow``
and ``r_elbow`` movable, so the policy needs 4 extra actions and 12 extra
observations (4 joint positions, 4 joint velocities, 4 previous actions).

The observation/action layout is per-joint blocks in *articulation order*, and
the arm-swing asset interleaves the arm joints among the legs (it follows the
URDF joint order). Columns therefore have to be remapped by joint name -- simply
appending the new columns shifts every leg observation and makes the warm-started
policy shuffle its legs.

The four new action rows are zero-initialised (weight and bias), which commands
the arm joints' default pose (0 rad: arms hanging at the sides).

Run inside the Isaac container:

    isaaclab.sh -p tools/rl_sessions/expand_walker_checkpoint.py \\
        --src outputs/rl/<run>/model_2150.pt \\
        --dst outputs/rl/<run>/model_2150_arm_swing.pt
"""
import argparse

import torch

# Articulation joint order of the two assets (USD/URDF order, see the env's
# "Resolved joint names" log line or robot.joint_names).
WALKER_JOINTS = [
    "Hip_F_L", "Hip_F_R", "Hip_A_L", "Hip_A_R", "Hip_R_L", "Hip_R_R",
    "Knee_L", "Knee_R", "Ankle_P_L", "Ankle_P_R", "Ankle_R_L", "Ankle_R_R",
]
ARM_SWING_JOINTS = [
    "Hip_F_L", "Hip_F_R", "l_shoulder_pitch", "r_shoulder_pitch",
    "Hip_A_L", "Hip_A_R", "l_elbow", "r_elbow",
    "Hip_R_L", "Hip_R_R", "Knee_L", "Knee_R",
    "Ankle_P_L", "Ankle_P_R", "Ankle_R_L", "Ankle_R_R",
]

BASE = 12      # base_lin_vel, base_ang_vel, projected_gravity, velocity_commands
SCAN = 187     # height_scan columns (unchanged, always last)
ARM_ACTION_LOG_STD = -1.5  # arms start with a modest exploration std (scale is 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="12-joint walker checkpoint")
    parser.add_argument("--dst", required=True, help="output 16-joint checkpoint")
    args = parser.parse_args()

    old_names, new_names = WALKER_JOINTS, ARM_SWING_JOINTS
    old_n, new_n = len(old_names), len(new_names)
    old_obs = BASE + 3 * old_n + SCAN
    new_obs = BASE + 3 * new_n + SCAN
    assert old_n == 12 and new_n == 16, "this tool is specific to the 12 -> 16 expansion"

    # new obs column -> old obs column (None = new term for an arm joint)
    perm = list(range(BASE))
    for block in range(3):  # joint_pos, joint_vel, actions
        old_off, new_off = BASE + block * old_n, BASE + block * new_n
        for name in new_names:
            perm.append(old_off + old_names.index(name) if name in old_names else None)
    perm += [BASE + 3 * old_n + k for k in range(SCAN)]
    assert len(perm) == new_obs

    ckpt = torch.load(args.src, map_location="cpu")
    sd = ckpt["model_state_dict"]

    def remap_columns(w):
        out = torch.zeros(w.shape[0], new_obs, dtype=w.dtype)
        for j, src in enumerate(perm):
            if src is not None:
                out[:, j] = w[:, src]
        return out

    def remap_rows(w):
        out = torch.zeros(new_n, w.shape[1], dtype=w.dtype)
        for j, name in enumerate(new_names):
            if name in old_names:
                out[j] = w[old_names.index(name)]
        return out

    sd["actor.0.weight"] = remap_columns(sd["actor.0.weight"])
    sd["critic.0.weight"] = remap_columns(sd["critic.0.weight"])
    sd["actor.6.weight"] = remap_rows(sd["actor.6.weight"])
    sd["actor.6.bias"] = remap_rows(sd["actor.6.bias"].unsqueeze(1)).squeeze(1)

    # normalisers: same column permutation; new arm dims start at mean 0 / std 1
    for group in ("actor_obs_normalizer", "critic_obs_normalizer"):
        for stat in ("_mean", "_var", "_std"):
            key = f"{group}.{stat}"
            old = sd[key]
            new = torch.full((1, new_obs), 0.0 if stat == "_mean" else 1.0, dtype=old.dtype)
            for j, src in enumerate(perm):
                if src is not None:
                    new[0, j] = old[0, src]
            sd[key] = new

    old_log_std = sd["log_std"]
    new_log_std = torch.full((new_n,), ARM_ACTION_LOG_STD, dtype=old_log_std.dtype)
    for j, name in enumerate(new_names):
        if name in old_names:
            new_log_std[j] = old_log_std[old_names.index(name)]
    sd["log_std"] = new_log_std

    torch.save(ckpt, args.dst)

    # verify the copied entries survived the permutation
    orig = torch.load(args.src, map_location="cpu")["model_state_dict"]
    for j, name in enumerate(new_names):
        if name in old_names:
            i = old_names.index(name)
            assert torch.allclose(sd["actor.6.bias"][j], orig["actor.6.bias"][i])
            assert torch.allclose(sd["log_std"][j], orig["log_std"][i])
    for j, src in enumerate(perm):
        if src is not None:
            assert torch.allclose(sd["actor.0.weight"][:, j], orig["actor.0.weight"][:, src])

    print(f"expanded {args.src} -> {args.dst}")
    print(f"  obs {old_obs} -> {new_obs}, actions {old_n} -> {new_n}")
    for j, name in enumerate(new_names):
        state = "copied" if name in old_names else "fresh (zero weight/bias)"
        print(f"  action {j:2d} {name:20s} {state}")


if __name__ == "__main__":
    main()
