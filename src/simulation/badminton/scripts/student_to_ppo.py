"""Turn a distilled student checkpoint into a PPO warm-start checkpoint.

  uv run scripts/student_to_ppo.py --student <distill model.pt> \\
      --out logs/rsl_rl/badminton_shared_student_ppo_priv/<run-name>/model_0.pt

Builds the Student-PPO runner, copies the distilled student's weights into
the actor (same MLPModel layout: obs normaliser + MLP + scalar Gaussian
std), leaves the critic and optimiser fresh, and saves in rsl_rl PPO format so training can
--agent.resume from it. Needs the GPU sim to size the networks.
"""

import argparse
import os
import sys
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import humanoid_badminton  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

TASK = "Mjlab-Badminton-Shared-Student-PPO-Priv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--task", default=TASK,
                    help="student-PPO task (Mjlab-Badminton-Shared-Student-PPO for the chooser)")
    args = ap.parse_args()

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = 16
    agent_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=args.device)

    ckpt = torch.load(args.student, map_location=args.device,
                      weights_only=False)
    sd = ckpt["student_state_dict"]
    missing, unexpected = runner.alg._raw_actor.load_state_dict(sd, strict=False)
    assert not unexpected, f"student keys the actor lacks: {unexpected}"
    assert not missing, f"actor keys the student lacks: {missing}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    runner.save(args.out)
    print(f"wrote {args.out} (actor <- {args.student}, critic/optimizer fresh)")


if __name__ == "__main__":
    main()
