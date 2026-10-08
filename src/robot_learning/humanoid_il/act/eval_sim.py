"""Run a trained ACT policy closed-loop in tidy_table and score it with the scene's own checks. Sim only.

    cd src/robot_learning/humanoid_il/act
    MUJOCO_GL=egl python eval_sim.py <checkpoint>/pretrained_model --episodes 20 [--video out.mp4]

Episodes use seeds from --seed (default 100000) so they are layouts no training demo came from (demo seeds
start at 0). Prints per-episode steps done / tidiness / success and the totals. The policy acts at the dataset's
25 fps; each action is held for the 4 control steps (10 ms each) until the next one, as recorded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2] / "pioneer_humanoid"))
sys.path.insert(0, str(_HERE.parents[2] / "simulation" / "mujoco_scenes"))

from tidy_sim import CONTROL_DT, RECORD_EVERY, TidySim  # noqa: E402


def load_policy(path: Path, device: str):
    import torch  # noqa: F401
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(path))
    policy.config.device = device
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def observation(sim: TidySim, image_keys: list[str], task: str) -> dict:
    import torch

    obs = {"observation.state": torch.from_numpy(sim.state()), "task": task}
    images = sim.images()
    for key in image_keys:
        cam = key[len("observation.images."):]
        obs[key] = torch.from_numpy(images[cam]).permute(2, 0, 1).float() / 255.0
    env = sim.env_state()
    if env is not None:
        obs["observation.environment_state"] = torch.from_numpy(env)
    return obs, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="…/checkpoints/<step>/pretrained_model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max_seconds", type=float, default=90.0, help="sim time per episode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video", type=Path, default=None, help="write the episodes' top-camera view here (mp4)")
    args = parser.parse_args()

    import torch

    policy, pre, post = load_policy(args.checkpoint, args.device)
    image_keys = [k for k in policy.config.input_features if k.startswith("observation.images.")]
    shapes = {k[len("observation.images."):]: tuple(policy.config.input_features[k].shape[1:]) for k in image_keys}
    sim = TidySim(shapes)
    from humanoid_mujoco_scenes.tidy_table import scene as S

    frames, results = [], []
    for ep in range(args.episodes):
        seed = args.seed + ep
        sim.reset(seed)
        policy.reset()
        action = sim.action()
        for step in range(int(args.max_seconds / CONTROL_DT)):
            if step % RECORD_EVERY == 0:
                index, total, task = sim.step_info()
                obs, images = observation(sim, image_keys, task)
                with torch.inference_mode():
                    action = post(policy.select_action(pre(obs))).squeeze(0).cpu().numpy()
                if args.video is not None and "top" in images:
                    frames.append(images["top"])
            sim.act(action)
            index, total, _ = sim.step_info()
            if index >= total and np.linalg.norm(sim.state()[:6] - sim.home) < 0.05 and step % RECORD_EVERY == 0:
                break                                  # all binned and back home
        status = S.episode_status(sim.model, sim.data)
        results.append(status)
        print(f"[EVAL] seed {seed}: steps {status['step']}/{status['total']} tidiness {status['tidiness']:.0%} "
              f"success {status['success']} dropped {status['dropped']} toppled {status['toppled']} "
              f"early {status['early']}", flush=True)
    n = len(results)
    print(f"[EVAL] success {sum(r['success'] for r in results)}/{n}, mean tidiness "
          f"{np.mean([r['tidiness'] for r in results]):.0%}, steps done {sum(r['step'] for r in results)}/"
          f"{sum(r['total'] for r in results)}")
    if args.video is not None and frames:
        import imageio

        imageio.mimsave(args.video, frames, fps=25, macro_block_size=1)
        print(f"[EVAL] video: {args.video}")
    sim.close()


if __name__ == "__main__":
    main()
