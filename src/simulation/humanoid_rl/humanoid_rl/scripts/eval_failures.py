# Copyright (c) 2022-2025, The Isaac Lab Project Developers / WATonomous.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a push-block teacher for many episodes at full difficulty and record how every episode ended.

Full difficulty = all domain randomization and the final spawn range from step 0 (the same as the
distillation env). For each finished episode it writes spawn pose, final block pose, how the episode
ended and which success condition failed, then prints where the failures cluster.

Example:
  $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/eval_failures.py --task Isaac-Bimanual-Push-Block-v0 --headless \\
    --teacher outputs/rl/push_bimanual/<run>/model_<N>.pt --num_envs 512 --episodes 2048
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record how every push-block episode ends.")
parser.add_argument("--task", type=str, default="Isaac-Bimanual-Push-Block-v0")
parser.add_argument("--teacher", type=str, required=True, help="Path to an RSL-RL PPO .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episodes", type=int, default=2048, help="Stop after at least this many finished episodes.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_dr", action="store_true", help="Nominal physics: no physics DR events and no observation noise.")
parser.add_argument("--narrow_spawn", action="store_true", help="Start-of-curriculum spawn range (SPAWN_STAGES[0]) instead of the final one.")
parser.add_argument("--episode_s", type=float, default=None, help="Episode length in seconds (task default if omitted).")
parser.add_argument("--tag", type=str, default="", help="Suffix for the CSV name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import humanoid_rl_tasks  # noqa: F401
from humanoid_rl_tasks.push_block.mdp.rewards import block_on_floor
from humanoid_rl_tasks.push_block.mdp.utils import BLOCK_HALF_SIZE, block_center_w
from humanoid_rl_tasks.push_block.push_env_cfg import SPAWN_RAMP_TARGET
from humanoid_rl_tasks.push_block.scene import RAMP_BASE_X, SPAWN_STAGES
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.modules import ActorCritic

TERMINATIONS = ("time_out", "block_off_course", "object_dropping")


def _load_teacher_actor(checkpoint: str, obs_dim: int, act_dim: int, device: str):
    """Same loader as distill_push.py: ActorCritic sized like BimanualPushBlockPPORunnerCfg, actor weights only."""
    import inspect

    kwargs = dict(num_actions=act_dim, actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64],
                  activation="elu", init_noise_std=1.0)
    if "obs_groups" in inspect.signature(ActorCritic.__init__).parameters:
        from tensordict import TensorDict

        obs = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
        ac = ActorCritic(obs=obs, obs_groups={"policy": ["policy"], "critic": ["policy"]}, **kwargs).to(device)
    else:
        ac = ActorCritic(num_actor_obs=obs_dim, num_critic_obs=obs_dim, **kwargs).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ac.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=False)
    ac.eval()
    return ac.actor


def _block_state(env, success_params: dict) -> dict:
    """Block pose and success conditions for every env, right now."""
    obj = env.scene["object"]
    center = block_center_w(obj) - env.scene.env_origins
    w, x, y, z = obj.data.root_quat_w.unbind(-1)
    return dict(
        x=center[:, 0], y=center[:, 1], z=center[:, 2],
        yaw=torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        tilt=torch.acos(torch.clamp(1 - 2 * (x * x + y * y), -1.0, 1.0)),
        speed=torch.norm(obj.data.root_lin_vel_w, dim=1),
        success=block_on_floor(env, **success_params) > 0.5,
    )


def _failed_conditions(row: dict, p: dict) -> str:
    bad = []
    if not (p["x_min"] < row["final_x"] < p["x_max"]):
        bad.append("x")
    if not abs(row["final_y"]) < p["y_half"]:
        bad.append("y")
    if not abs(row["final_z"] - (p["floor_z"] + BLOCK_HALF_SIZE)) < p["z_tol"]:
        bad.append("height")
    if not row["speed"] < p["rest_speed"]:
        bad.append("moving")
    return "+".join(bad)


def _where_it_ended(row: dict, p: dict) -> str:
    """One label for where a time-out failure's block ended up."""
    x, y = row["final_x"], row["final_y"]
    if x < RAMP_BASE_X - 0.01:
        side = "right (-y)" if y < -0.127 else "left (+y)" if y > 0.127 else "centre"
        return f"short of the ramp, {side}"
    if x < p["x_min"]:
        return "on the ramp"
    if abs(y) >= p["y_half"]:
        return "past the platform sideways, right (-y)" if y < 0 else "past the platform sideways, left (+y)"
    if row["tilt"] > math.radians(10):
        return "on the floor but tilted > 10 deg"
    return "on the floor, wrong height or still moving"


def _pct(n: int, total: int) -> str:
    return f"{n:5d} ({100.0 * n / max(total, 1):4.1f}%)"


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    success_params = {k: v for k, v in env_cfg.curriculum.success.params.items() if k not in ("window", "min_episode_len")}
    # full difficulty from step 0: no ramps, final spawn range
    env_cfg.curriculum.dr_ramp = None
    env_cfg.curriculum.spawn_ramp = None
    env_cfg.events.reset_object_position.params["pose_range"] = dict(SPAWN_STAGES[0] if args_cli.narrow_spawn else SPAWN_RAMP_TARGET)
    if args_cli.episode_s:
        env_cfg.episode_length_s = args_cli.episode_s
    if args_cli.no_dr:
        for name in ("dr_block_material", "dr_finger_material", "dr_block_mass", "dr_arm_mass", "dr_arm_gains"):
            setattr(env_cfg.events, name, None)
        env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    device = u.device
    n = u.num_envs

    obs_dict, _ = env.reset()
    teacher = _load_teacher_actor(args_cli.teacher, obs_dict["policy"].shape[-1], u.action_manager.total_action_dim, device)
    steps = torch.zeros(n, device=device)
    need_spawn = torch.ones(n, dtype=torch.bool, device=device)
    spawn = torch.zeros(n, 3, device=device)
    rows = []

    while len(rows) < args_cli.episodes:
        state = _block_state(u, success_params)  # state before this step = the state one control step before an episode ends
        if need_spawn.any():
            spawn[need_spawn] = torch.stack([state["x"], state["y"], state["yaw"]], 1)[need_spawn]
            need_spawn[:] = False
        with torch.no_grad():
            actions = teacher(obs_dict["policy"])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        steps += 1
        dones = terminated | truncated
        if dones.any():
            reasons = {name: u.termination_manager.get_term(name) for name in TERMINATIONS}
            for i in dones.nonzero().flatten().tolist():
                reason = next((nm for nm in TERMINATIONS[1:] if reasons[nm][i]), "time_out")
                row = dict(
                    spawn_x=spawn[i, 0].item(), spawn_y=spawn[i, 1].item(), spawn_yaw_deg=math.degrees(spawn[i, 2].item()) % 90.0,
                    final_x=state["x"][i].item(), final_y=state["y"][i].item(), final_z=state["z"][i].item(),
                    final_yaw_deg=math.degrees(state["yaw"][i].item()) % 90.0, tilt=state["tilt"][i].item(),
                    speed=state["speed"][i].item(), steps=int(steps[i].item()), success=bool(state["success"][i].item()), reason=reason,
                )
                row["failed_conditions"] = "" if row["success"] else _failed_conditions(row, success_params)
                rows.append(row)
            steps[dones] = 0
            need_spawn[dones] = True

    out_dir = os.path.join(os.environ.get("HUMANOID_ROOT") or os.getcwd(), "outputs", "eval")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"failures_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{args_cli.tag}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    ok = [r for r in rows if r["success"]]
    bad = [r for r in rows if not r["success"]]
    se = math.sqrt(len(ok) / total * (1 - len(ok) / total) / total)
    print(f"\nEVAL {total} episodes from {args_cli.teacher}")
    print(f"EVAL success {len(ok) / total:.4f} +/- {1.96 * se:.4f} (95%), failures {len(bad)}  [state taken one control step (0.02 s) before the episode ended]")
    print("EVAL how the failures ended:")
    for reason in TERMINATIONS:
        print(f"EVAL   {reason:18s} {_pct(sum(r['reason'] == reason for r in bad), len(bad))}")
    timeouts = [r for r in bad if r["reason"] == "time_out"]
    print(f"EVAL where the {len(timeouts)} time-out failures ended (first-failing condition of the success test):")
    counts: dict = {}
    for r in timeouts:
        counts[_where_it_ended(r, success_params)] = counts.get(_where_it_ended(r, success_params), 0) + 1
    for label, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"EVAL   {label:46s} {_pct(c, len(timeouts))}")
    print("EVAL which success conditions fail (all failures; x/y = outside the platform, height, moving):")
    cond: dict = {}
    for r in bad:
        cond[r["failed_conditions"] or "none"] = cond.get(r["failed_conditions"] or "none", 0) + 1
    for label, c in sorted(cond.items(), key=lambda kv: -kv[1])[:6]:
        print(f"EVAL   {label:20s} {_pct(c, len(bad))}")

    def rate_by(key, edges, label):
        print(f"EVAL success rate by {label}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            seg = [r for r in rows if lo <= r[key] < hi]
            if seg:
                print(f"EVAL   {lo:+.3f} .. {hi:+.3f}: {sum(r['success'] for r in seg) / len(seg):.3f}  (n={len(seg)})")

    rate_by("spawn_y", [-0.13, -0.08, -0.03, 0.03, 0.08, 0.13], "spawn y (m; -y is the robot's right)")
    rate_by("spawn_x", [-0.0, 0.40, 0.44, 0.48, 0.52, 0.60], "spawn x (m)")
    rate_by("spawn_yaw_deg", [0, 22.5, 45, 67.5, 90.0001], "spawn yaw (deg, mod 90)")
    if timeouts:
        ys = sorted(r["final_y"] for r in timeouts)
        xs = sorted(r["final_x"] for r in timeouts)
        mid = len(ys) // 2
        print(f"EVAL time-out failures: median final x {xs[mid]:.3f}, median final y {ys[mid]:.3f}; "
              f"spawn y of failures mean {sum(r['spawn_y'] for r in timeouts) / len(timeouts):+.3f} vs successes {sum(r['spawn_y'] for r in ok) / max(len(ok), 1):+.3f}")
    print(f"EVAL per-episode records: {path}\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
