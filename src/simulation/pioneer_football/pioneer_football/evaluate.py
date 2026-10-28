"""Evaluate a checkpoint on fixed stand/walk/turn/stop commands, not reward alone."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from . import mdp
from .task import make_env_cfg, make_rl_cfg

SCENARIOS = (
    "stand",
    "forward",
    "left",
    "right",
    "turn_left",
    "turn_right",
    "walk_turn_stop",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--episodes", type=int, default=112)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = p.parse_args()
    if min(args.num_envs, args.episodes) <= 0:
        p.error("Counts must be positive")
    cfg = make_env_cfg(play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = min(args.num_envs, args.episodes)
    cfg.episode_length_s = 10.0
    cfg.auto_reset = False
    # Keep training observation noise for an honest baseline evaluation.
    cfg.observations["actor"].enable_corruption = True
    cmd_cfg = cfg.commands["base_velocity"]
    cmd_cfg.resampling_time_range = (1e6, 1e6)
    raw = ManagerBasedRlEnv(cfg, device=args.device)
    env = RslRlVecEnvWrapper(raw, clip_actions=1.0)
    try:
        runner = MjlabOnPolicyRunner(
            env, asdict(make_rl_cfg()), log_dir=None, device=args.device
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        obs = env.get_observations()
        n = cfg.scene.num_envs
        device = args.device
        counts = torch.zeros(n, device=device)
        xy = torch.zeros_like(counts)
        yaw = torch.zeros_like(counts)
        effort = torch.zeros_like(counts)
        speed = torch.zeros_like(counts)
        starts = raw.scene["robot"].data.root_link_pos_w.clone()
        cmd = raw.command_manager.get_term("base_velocity")
        scenario_ids = torch.arange(n, device=device) % len(SCENARIOS)
        trial_ids = list(range(n))
        next_trial = n
        commands = torch.tensor(
            [
                [0, 0, 0],
                [0.3, 0, 0],
                [0, 0.15, 0],
                [0, -0.15, 0],
                [0, 0, 0.4],
                [0, 0, -0.4],
                [0.3, 0, 0],
            ],
            device=device,
            dtype=torch.float32,
        )
        rows = []
        while len(rows) < args.episodes:
            with torch.inference_mode():
                cmd.vel_command_b[:] = commands[scenario_ids]
                cmd.is_standing_env[:] = False
                elapsed = raw.episode_length_buf * raw.step_dt
                seq = scenario_ids == 6
                cmd.vel_command_b[seq & (elapsed >= 3) & (elapsed < 6), 2] = 0.3
                cmd.vel_command_b[seq & (elapsed >= 6)] = 0
                obs = env.get_observations()
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Nonfinite policy output")
                obs, reward, done, info = env.step(actions)
                if not torch.isfinite(reward).all():
                    raise RuntimeError("Nonfinite reward")
                xy += mdp.tracking_error(raw)
                yaw += mdp.yaw_error(raw)
                counts += 1
                effort = torch.maximum(effort, mdp.effort_fraction(raw))
                speed = torch.maximum(speed, mdp.speed_fraction(raw))
                position = raw.scene["robot"].data.root_link_pos_w.clone()
                for idx in done.nonzero().flatten().tolist():
                    if trial_ids[idx] >= args.episodes:
                        continue
                    timeout = bool(info["time_outs"][idx]) and not bool(
                        raw.reset_terminated[idx]
                    )
                    mean_xy = float(xy[idx] / counts[idx])
                    mean_yaw = float(yaw[idx] / counts[idx])
                    rows.append(
                        {
                            "trial_id": trial_ids[idx],
                            "scenario": SCENARIOS[int(scenario_ids[idx])],
                            "survived": timeout,
                            "duration_s": float(counts[idx] * raw.step_dt),
                            "velocity_error_mps": mean_xy,
                            "yaw_error_rps": mean_yaw,
                            "displacement_m": float(
                                (position[idx, :2] - starts[idx, :2]).norm()
                            ),
                            "peak_effort_fraction": float(effort[idx]),
                            "peak_speed_fraction": float(speed[idx]),
                            "tracking_pass": timeout
                            and mean_xy < 0.10
                            and mean_yaw < 0.2,
                        }
                    )
                    trial_ids[idx] = next_trial
                    scenario_ids[idx] = next_trial % len(SCENARIOS)
                    next_trial += 1
                ids = done.nonzero().flatten()
                if len(ids):
                    raw.reset(env_ids=ids)
                    counts[ids] = 0
                    xy[ids] = 0
                    yaw[ids] = 0
                    effort[ids] = 0
                    speed[ids] = 0
                    starts[ids] = raw.scene["robot"].data.root_link_pos_w[ids]
        rows.sort(key=lambda row: row["trial_id"])
        result = {
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "episodes": rows,
            "survival_rate": sum(r["survived"] for r in rows) / len(rows),
            "tracking_pass_rate": sum(r["tracking_pass"] for r in rows) / len(rows),
            "criteria": "10 seconds without fall; mean planar velocity error < .10 m/s; mean yaw error < .2 rad/s. Engineering diagnostic, not hardware certification.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print({k: v for k, v in result.items() if k != "episodes"})
    finally:
        env.close()


if __name__ == "__main__":
    main()
