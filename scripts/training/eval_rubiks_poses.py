"""Evaluate the ACT cube policy on reproducibly perturbed MuJoCo starts."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from humanoid_robot_learning.sim.evaluate import CONTROL_DT, HOLD_S, SceneEnv, load_policy, wilson
from lerobot.utils.control_utils import predict_action


FPS = 25
MAX_SECONDS = 8
CUBE_XY_M = 0.02
CUBE_YAW_DEG = 10.0
ARM_JOINT_DEG = 3.0


def perturb_start(env: SceneEnv, seed: int, mode: str) -> dict:
    rng = np.random.default_rng(seed)
    cube_start = env.model.joint("rubiks_cube").qposadr[0]
    cube = env.data.qpos[cube_start:cube_start + 7]
    xy = rng.uniform(-CUBE_XY_M, CUBE_XY_M, size=2)
    yaw_deg = float(rng.uniform(-CUBE_YAW_DEG, CUBE_YAW_DEG))
    arm_delta_deg = rng.uniform(-ARM_JOINT_DEG, ARM_JOINT_DEG, size=6)
    # Always draw in the same order so isolated perturbations match the joint benchmark.
    if mode == "arm":
        xy[:] = 0
        yaw_deg = 0.0
    elif mode == "cube":
        arm_delta_deg[:] = 0
    cube[:2] += xy
    # This scene's cube starts with an identity quaternion.
    half_yaw = np.deg2rad(yaw_deg) / 2
    cube[3:] = [np.cos(half_yaw), 0, 0, np.sin(half_yaw)]
    home = env.data.qpos[env.arm_qpos]
    arm = np.clip(home + np.deg2rad(arm_delta_deg), env.arm_lo, env.arm_hi)
    env.data.qpos[env.arm_qpos] = arm
    env.data.ctrl[env.arm_acts] = arm
    mujoco.mj_forward(env.model, env.data)
    return {"cube_xy_m": xy.tolist(), "cube_yaw_deg": yaw_deg, "arm_joint_delta_deg": np.rad2deg(arm - home).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--action-steps", type=int, help="ACT actions to execute before replanning")
    control.add_argument("--ensemble-coeff", type=float, help="Enable ACT temporal ensembling with this coefficient")
    parser.add_argument("--perturb", choices=("both", "cube", "arm"), default="both")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed must be nonnegative")
    if args.ensemble_coeff is not None and not math.isfinite(args.ensemble_coeff):
        parser.error("--ensemble-coeff must be finite")

    device = torch.device("cuda")
    policy, pre, post = load_policy(args.policy, str(device))
    if args.action_steps is not None:
        if policy.config.type != "act" or policy.config.temporal_ensemble_coeff is not None:
            parser.error("--action-steps requires ACT with temporal ensembling disabled")
        if not 1 <= args.action_steps <= policy.config.chunk_size:
            parser.error(f"--action-steps must be between 1 and {policy.config.chunk_size}")
        # The network still predicts the trained chunk; select_action queues only this prefix.
        policy.config.n_action_steps = args.action_steps
    if args.ensemble_coeff is not None:
        if policy.config.type != "act":
            parser.error("--ensemble-coeff requires ACT")
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = args.ensemble_coeff
        policy.temporal_ensembler = ACTTemporalEnsembler(args.ensemble_coeff, policy.config.chunk_size)
    cameras = {
        key.removeprefix("observation.images."): tuple(feature.shape[1:])
        for key, feature in policy.config.image_features.items()
    }
    env = SceneEnv("rubiks_cube", cameras)
    control_steps = round(1 / (FPS * CONTROL_DT))
    trials = []
    try:
        for seed in [None, *range(args.seed, args.seed + args.episodes)]:
            env.reset(0 if seed is None else seed)
            offsets = {} if seed is None else perturb_start(env, seed, args.perturb)
            policy.reset()
            held = 0.0
            for _ in range(MAX_SECONDS * FPS):
                action = predict_action(
                    env.observe(), policy, device, pre, post, use_amp=False, task="sim leader teleop demonstration"
                )
                env.step(action.reshape(-1).cpu().numpy(), control_steps)
                held = held + 1 / FPS if env.success(env.model, env.data) else 0.0
                if held >= HOLD_S:
                    break
            trial = {
                "seed": seed,
                "success": held >= HOLD_S,
                "elapsed_s": round(float(env.data.time), 2),
                "peak_lift_cm": round(float(env.data.userdata[0]) * 100, 2),
                **offsets,
            }
            print(json.dumps(trial), flush=True)
            if seed is None:
                reference = trial
                if args.action_steps is None and args.ensemble_coeff is None and not reference["success"]:
                    raise RuntimeError("The reference starting pose did not reproduce the known successful rollout")
            else:
                trials.append(trial)
    finally:
        env.close()

    successes = sum(trial["success"] for trial in trials)
    interval = wilson(successes, len(trials))
    rate = successes / len(trials)
    args.output.write_text(json.dumps({
        "policy": args.policy,
        "action_steps": policy.config.n_action_steps,
        "ensemble_coeff": policy.config.temporal_ensemble_coeff,
        "perturbation": args.perturb,
        "reference": reference,
        "successes": successes,
        "success_rate": rate,
        "wilson_95": interval,
        "trials": trials,
    }, indent=2) + "\n")
    print(f"Varied starts: {successes}/{len(trials)} = {rate:.1%} (95% Wilson CI {interval[0]:.1%}–{interval[1]:.1%})")


if __name__ == "__main__":
    main()
