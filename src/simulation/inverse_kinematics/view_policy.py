"""Watch a trained PPO policy reach goals in the MuJoCo viewer. On macOS the viewer needs mjpython:

    ~/mj-venv/bin/mjpython view_policy.py                              # the newest model in runs/
    ~/mj-venv/bin/mjpython view_policy.py runs/ppo_1m.zip --episodes 5

Runs the evaluation setup (no noise, one goal per episode, 160 steps) in real time with the deterministic policy.
Red sphere + axes = the goal, cyan sphere + axes = where the tool point is. After each episode it prints whether the
episode was a strict success (final-second error within 1 mm and 0.2 deg) and the final position error.
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import EnvConfig
from gym_env import PolicyIKGymEnv
from metrics import CONTROL_DT
from tests.view_reset import draw            # the goal / tool-point markers
from train import stats_path                 # where train.py saves a model's observation scaling


def newest_model():
    models = sorted(Path("runs").glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise SystemExit("no model given and none found in runs/ (train one with train.py)")
    return str(models[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="a .zip saved by train.py (default: the newest in runs/)")
    ap.add_argument("--episodes", type=int, default=0, help="stop after this many (0 = until the window closes)")
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--pause", type=float, default=1.5, help="seconds to wait between episodes")
    args = ap.parse_args()

    path = args.model or newest_model()
    model = PPO.load(path, device="cpu")
    genv = PolicyIKGymEnv(EnvConfig(), seed=args.seed)
    env = genv.env
    print(f"model: {path}")
    stats = None                    # models trained with --normalize need the same observation scaling here
    if Path(stats_path(path)).exists():
        stats = VecNormalize.load(stats_path(path), DummyVecEnv([lambda: PolicyIKGymEnv(EnvConfig(), seed=0)]))
        stats.training, stats.norm_reward = False, False
        print("using the saved observation normalization")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.15, -0.1]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth, viewer.cam.elevation = 135, -20
        n = 0
        while viewer.is_running() and (args.episodes == 0 or n < args.episodes):
            obs, _ = genv.reset()
            n += 1
            done, info = False, {}
            while viewer.is_running() and not done:
                t0 = time.time()
                action, _ = model.predict(stats.normalize_obs(obs) if stats else obs, deterministic=True)
                obs, _, terminated, truncated, info = genv.step(action)
                done = terminated or truncated
                with viewer.lock():
                    draw(env, viewer.user_scn)
                viewer.sync()
                time.sleep(max(0.0, CONTROL_DT - (time.time() - t0)))          # about real time
            if info.get("is_success") is not None:
                print(f"episode {n}: {'SUCCESS' if info['is_success'] else 'miss   '}   "
                      f"final position error {1000 * info['final_pos_err']:7.1f} mm, "
                      f"rotation {np.degrees(info['final_rot_err']):6.2f} deg")
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
