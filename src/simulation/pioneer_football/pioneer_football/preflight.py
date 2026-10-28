"""Check the selected device and execute real environment transitions."""

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv

from .model import JOINTS
from .task import make_env_cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        p.error("Counts must be positive")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA unavailable: check NVIDIA driver and install with uv sync --extra cu128."
            )
        info = torch.cuda.get_device_properties(args.device)
        print(
            f"GPU: {info.name}; VRAM: {info.total_memory / 2**30:.1f} GiB; torch CUDA: {torch.version.cuda}"
        )
        x = torch.randn(64, 64, device=args.device)
        assert torch.isfinite(x @ x).all()
    cfg = make_env_cfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    # Force timeouts to verify terminal reporting and automatic resets.
    cfg.episode_length_s = 0.1
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        obs, _ = env.reset()
        assert list(env.scene["robot"].joint_names) == list(JOINTS)
        feet = env.scene["feet"]
        assert [n.rsplit("/", 1)[-1] for n in feet.primary_names] == [
            "l_foot",
            "r_foot",
        ]
        assert obs["actor"].shape == (args.num_envs, 48)
        assert env.action_manager.total_action_dim == 12
        resets = 0
        for step in range(args.steps):
            action = torch.zeros((args.num_envs, 12), device=args.device)
            if step % 2:
                action.uniform_(-0.1, 0.1)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            resets += int((terminated | truncated).sum())
        if args.steps >= 5:
            assert resets > 0
        print(
            f"PASS: {args.steps} transitions x {args.num_envs} environments, {resets} episode endings; finite observations/rewards."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
