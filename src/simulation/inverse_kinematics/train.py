"""A simple PPO training run on PolicyIK (task A5).

    ~/mj-venv/bin/python train.py                       # 50,000 steps
    ~/mj-venv/bin/python train.py --steps 1000000 --out runs/ppo_1m
    ~/mj-venv/bin/python train.py --resume runs/ppo_1m --steps 1000000 --out runs/ppo_2m --seed 1   # keep training

Trains with the paper's setup (observation noise 0.01, a new goal every 4 s, 12 s rollouts), saves the model, then
scores it on clean 160-step episodes with one goal each (the evaluation setup).

PPO settings (all options, see --help). The defaults are more cautious than Stable-Baselines3's, because with its
defaults the policy updates were far too big (approx_kl 0.2-1 instead of about 0.02) and the action noise collapsed:
  --target-kl     stop an update early when the policy has moved more than 1.5x this
  --n-epochs, --n-steps, --batch-size, --lr   fewer passes over bigger batches with a smaller learning rate
  --ent-coef      a bonus for keeping the actions random, so exploration doesn't collapse (the paper uses 0.01)
  --normalize     scale each observation channel by its running mean and std (the paper does this too); the scaling is
                  saved next to the model as <out>_vecnorm.pkl and view_policy.py loads it
"""
import argparse
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import EnvConfig
from gym_env import PolicyIKGymEnv


def stats_path(model_path):
    """Where the observation scaling of a model is saved: <model>_vecnorm.pkl (without any .zip)."""
    p = str(model_path)
    return (p[:-4] if p.endswith(".zip") else p) + "_vecnorm.pkl"


class SaveLatest(BaseCallback):
    """Every `every` steps, save the model (and its observation scaling) to `path`, replacing the last save.
    A long run can then be interrupted, watched with view_policy.py while it trains, or continued with --resume."""

    def __init__(self, path, every, venv=None):
        super().__init__()
        self.path, self.every, self.venv, self.last = path, every, venv, 0

    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self.model.save(self.path)
            if self.venv is not None:
                self.venv.save(stats_path(self.path))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000, help="training steps")
    ap.add_argument("--out", default="runs/ppo_policyik", help="where to save the model (.zip is added)")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="with --resume, use a new seed or the same goals repeat")
    ap.add_argument("--resume", metavar="MODEL", help="continue training from a saved model (.zip optional)")
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate (SB3's default is 3e-4)")
    ap.add_argument("--n-steps", type=int, default=8192, help="steps collected before each update (SB3: 2048)")
    ap.add_argument("--batch-size", type=int, default=1024, help="minibatch size (SB3: 64)")
    ap.add_argument("--n-epochs", type=int, default=4, help="passes over each batch (SB3: 10)")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="exploration bonus (SB3: 0). The paper uses 0.01, but with actions clipped to +-1 it made the "
                         "action noise GROW to std 2 (every sampled action saturated), so it is off by default")
    ap.add_argument("--std-init", type=float, default=0.4,
                    help="starting std of the action noise (SB3: 1.0, which already saturates the +-1 clipping)")
    ap.add_argument("--reset-std", action="store_true", help="with --resume: set the std back to --std-init first")
    ap.add_argument("--target-kl", type=float, default=0.03, help="early-stop an update beyond 1.5x this; 0 = off")
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                    help="normalize the observations (on by default; --no-normalize to turn off)")
    ap.add_argument("--normalize-reward", action="store_true", help="also scale the rewards (off by default)")
    ap.add_argument("--near-goal-prob", type=float, default=0.5,
                    help="fraction of episodes that start close to their goal, to practice the final correction and holding")
    ap.add_argument("--edge-goal-prob", type=float, default=0.25,
                    help="fraction of goals redrawn to have a joint within 5%% of a limit (the hardest goals for the policy)")
    ap.add_argument("--hard-start-prob", type=float, default=0.25,
                    help="fraction of (non-near-goal) starts with 1-3 joints pinned at a limit, to practice getting out")
    ap.add_argument("--band-goals", action="store_true",
                    help="goals only from two bands of distance to the joint limits: --center-goal-prob of them 20%%+ away, the rest 5-20%%")
    ap.add_argument("--center-goal-prob", type=float, default=0.5, help="with --band-goals: share of goals 20%%+ from every limit")
    ap.add_argument("--save-every", type=int, default=500_000, help="save to --out every this many steps (0 = only at the end)")
    args = ap.parse_args()

    if args.resume and stats_path(args.resume) == stats_path(args.out):
        raise SystemExit("--out is the model you are resuming from: pick a new name so it isn't overwritten")

    cfg = EnvConfig(obs_noise=0.01, resample_goals=True, max_steps=360, near_goal_prob=args.near_goal_prob,
                    edge_goal_prob=args.edge_goal_prob, hard_start_prob=args.hard_start_prob,
                    band_goals=args.band_goals, center_goal_prob=args.center_goal_prob)
    venv = DummyVecEnv([lambda: Monitor(PolicyIKGymEnv(cfg, seed=args.seed))])    # Monitor gives the episode reward log

    normalize = args.normalize
    if args.resume and normalize and not Path(stats_path(args.resume)).exists():
        # A model trained without normalization expects raw observations, so normalizing now would break it.
        print(f"note: {stats_path(args.resume)} not found (the model was trained without normalization): staying off")
        normalize = False
    if normalize and args.resume:
        venv = VecNormalize.load(stats_path(args.resume), venv)          # keep the running mean and std
        venv.training, venv.norm_reward = True, args.normalize_reward
    elif normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=args.normalize_reward, clip_obs=10.0)

    hp = dict(learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
              ent_coef=args.ent_coef, target_kl=args.target_kl or None)
    print(f"PPO settings: {hp}, observation normalization: {normalize}")
    if args.resume:
        # Loads the weights and optimizer state. The environment is not stored in the file, so it is passed in here,
        # and the settings above replace the ones the model was saved with.
        model = PPO.load(args.resume, env=venv, device="cpu", **hp)
        print(f"resuming from {args.resume} at step {model.num_timesteps}")
        if args.reset_std:
            model.policy.log_std.data.fill_(float(np.log(args.std_init)))
    else:
        model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=1,    # 2 hidden layers of 64
                    policy_kwargs=dict(log_std_init=float(np.log(args.std_init))), **hp)
    saver = SaveLatest(args.out, args.save_every, venv if normalize else None) if args.save_every else None
    model.learn(total_timesteps=args.steps, reset_num_timesteps=not args.resume, callback=saver)   # a resumed run keeps counting
    model.save(args.out)
    print(f"saved {args.out}.zip")
    if normalize:
        venv.save(stats_path(args.out))
        print(f"saved {stats_path(args.out)}")
        venv.training, venv.norm_reward = False, False         # evaluation must not change the scaling

    # Evaluation: no noise, one goal per episode, 160 steps. Different seed from training.
    eval_env = PolicyIKGymEnv(EnvConfig(), seed=1000 + args.seed)
    successes, errors = 0, []
    for _ in range(args.eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(venv.normalize_obs(obs) if normalize else obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        successes += info["is_success"]
        errors.append(info["final_pos_err"])
    errors.sort()
    print(f"eval: {successes}/{args.eval_episodes} strict successes (within 1 mm and 0.2 deg), "
          f"median final position error {1000 * errors[len(errors) // 2]:.1f} mm")


if __name__ == "__main__":
    main()
