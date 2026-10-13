"""Roll out a trained LeRobot policy in a MuJoCo scene and report the success rate.

    MUJOCO_GL=egl python sim_policy_eval.py --scene peg_insert \
        --policy outputs/train/pioneer_act_peg_insert/checkpoints/last/pretrained_model --episodes 20

Same sim, cameras, state and action layout as ``synthetic_teleop.py --record`` (TeleopSim), so the policy
sees what it was trained on: every 40 ms it gets the 7-value state and the camera images and returns the
7-value action (6 joint targets in rad + gripper closure), which is held for the next 4 control ticks.
Layouts come from ``--seed`` + index; keep them apart from the training seeds (recording starts at 0).
Success is the scene's own check (synthetic_tasks.TASKS), evaluated as soon as it holds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch_backbone  # noqa: E402,F401  (so checkpoints trained with the patch-token backbone load)
from synthetic_teleop import TeleopSim  # noqa: E402  (also puts the repo packages on sys.path)


def observe(sim):
    """(state, camera images) as recorded: 6 joint angles + mean finger closure, one image per camera."""
    import numpy as np
    data = sim.data
    closure = sum((data.qpos[q] - o) / (c - o) for q, o, c in zip(sim.grip_qpos, sim.grip_open, sim.grip_closed)) / len(sim.grip_qpos)
    state = np.append(data.qpos[sim.arm_qpos], min(max(closure, 0.0), 1.0)).astype(np.float32)
    return state, sim.render_at(data.qpos.copy())


RENDER_SIZE = (480, 640)   # the recorded cameras (dataset schema); a policy trained on downscale_dataset.py output gets them scaled


def policy_frame(sim, policy, preprocessor, postprocessor, device, fps, instruction, use_amp=False, sizes=None):
    """One 1/fps frame under the policy: observe, predict, hold the action for the control ticks. Returns the images.
    ``sizes``: {camera: (height, width)} the policy expects, if smaller than the renders (area-scaled, as the dataset was)."""
    import numpy as np
    from lerobot.utils.control_utils import predict_action

    model, data = sim.model, sim.data
    state, images = observe(sim)
    seen = images
    if sizes:
        import cv2
        seen = {n: img if img.shape[:2] == tuple(sizes[n]) else cv2.resize(img, tuple(sizes[n])[::-1], interpolation=cv2.INTER_AREA)
                for n, img in images.items()}
    obs = {"observation.state": state, **{f"observation.images.{n}": img for n, img in seen.items()}}
    action = predict_action(obs, policy, device, preprocessor, postprocessor, use_amp, task=instruction,
                            robot_type="pioneer_v1_left_arm")
    action = action.squeeze(0).cpu().numpy().astype(float)
    g = min(max(action[6], 0.0), 1.0)
    lo, hi = model.jnt_range[[model.joint(j).id for j in sim.joints]].T
    data.ctrl[sim.arm_acts] = np.clip(action[:6], lo, hi)
    data.ctrl[sim.grip_acts] = [o + g * (c - o) for o, c in zip(sim.grip_open, sim.grip_closed)]
    for _ in range(round(1.0 / fps / sim.control_dt)):
        if sim.hook is not None:
            sim.hook(model, data)
        sim.mujoco.mj_step(model, data, nstep=sim.substeps)
    return images


def rollout(scene, seed, policy, preprocessor, postprocessor, device, cameras, fps, max_s, hold_s, video=None, use_amp=False):
    """One episode; returns (success, sim seconds, furthest step reached, steps in the scene)."""
    import numpy as np
    from synthetic_tasks import TASKS

    task = TASKS[scene]
    sim = TeleopSim(scene, seed, {n: RENDER_SIZE for n in cameras})
    model, data = sim.model, sim.data
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    best, total, held, ok = 0, 0, 0.0, False
    instruction = task.instruction
    for frame in range(int(max_s * fps)):
        if sim.progress is not None:
            index, total, instruction = sim.progress(model, data)
            best = max(best, int(index))
        images = policy_frame(sim, policy, preprocessor, postprocessor, device, fps, instruction, use_amp, sizes=cameras)
        if video is not None:
            video.append_data(np.concatenate([images[n] for n in sorted(images)], axis=1))
        # Success must hold for hold_s (not a peg bouncing through the hole).
        held = held + 1.0 / fps if task.success(model, data) else 0.0
        if held >= hold_s:
            ok = True
            break
    if sim.progress is not None:
        index, total, _ = sim.progress(model, data)
        best = max(best, int(index))
    sim_t = data.time
    sim.close()
    return ok, sim_t, best, int(total)


def load_policy(path, device: str, n_action_steps: int = 0, temporal_ensemble_coeff: float | None = None):
    """(policy, preprocessor, postprocessor, cfg, cameras {name: (height, width)}) from a pretrained_model directory."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    path = Path(path).resolve()
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    if temporal_ensemble_coeff is not None:
        cfg.temporal_ensemble_coeff, cfg.n_action_steps = temporal_ensemble_coeff, 1
    elif n_action_steps:
        cfg.n_action_steps = n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": device}})
    cameras = {key.removeprefix("observation.images."): tuple(ft.shape[-2:]) if ft.shape[0] == 3 else tuple(ft.shape[:2])
               for key, ft in cfg.input_features.items() if key.startswith("observation.images.")}
    return policy, preprocessor, postprocessor, cfg, cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LeRobot policy in a Pioneer MuJoCo scene.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--policy", required=True, help="pretrained_model directory of a checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000, help="first layout seed (training takes start at 0)")
    parser.add_argument("--max_s", type=float, default=60.0, help="time limit per episode (sim seconds)")
    parser.add_argument("--hold_s", type=float, default=0.4, help="success must hold this long")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_action_steps", type=int, default=0, help="override the policy's n_action_steps")
    parser.add_argument("--temporal_ensemble_coeff", type=float, default=None, help="ACT: temporal ensembling (sets n_action_steps=1)")
    parser.add_argument("--videos", type=int, default=3, help="save the first N episodes as mp4 next to --out")
    parser.add_argument("--out", type=str, default=None, help="results JSON (default: <policy>/../eval_<scene>.json)")
    args = parser.parse_args()

    import torch

    path = Path(args.policy).resolve()
    policy, preprocessor, postprocessor, cfg, cameras = load_policy(path, args.device, args.n_action_steps, args.temporal_ensemble_coeff)
    fps = 25
    out = Path(args.out) if args.out else path.parent / f"eval_{args.scene}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    results, t0 = [], time.time()
    for i in range(args.episodes):
        seed, video = args.seed + i, None
        if i < args.videos:
            import imageio
            video = imageio.get_writer(out.with_name(f"{out.stem}_seed{seed}.mp4"), fps=fps, macro_block_size=1)
        ok, sim_t, best, total = rollout(args.scene, seed, policy, preprocessor, postprocessor, torch.device(args.device),
                                         cameras, fps, args.max_s, args.hold_s, video)
        if video is not None:
            video.close()
        results.append({"seed": seed, "success": ok, "sim_s": round(sim_t, 2), "steps_done": best, "steps": total})
        done = sum(r["success"] for r in results)
        print(f"[EVAL] seed {seed}: {'ok  ' if ok else 'FAIL'} {sim_t:5.1f} s  step {best}/{total}   "
              f"running {done}/{len(results)} ({time.time() - t0:.0f} s wall)", flush=True)
    rate = sum(r["success"] for r in results) / max(1, len(results))
    summary = {"scene": args.scene, "policy": str(path), "episodes": len(results), "success_rate": rate,
               "mean_steps_done": sum(r["steps_done"] for r in results) / max(1, len(results)),
               "n_action_steps": cfg.n_action_steps, "results": results}
    out.write_text(json.dumps(summary, indent=2))
    print(f"[EVAL] {args.scene}: success {rate:.0%} ({sum(r['success'] for r in results)}/{len(results)})  -> {out}")


if __name__ == "__main__":
    main()
