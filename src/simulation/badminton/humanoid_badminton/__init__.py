"""mjlab (MuJoCo Warp) training tasks for the badminton receive env.

Importing this package registers three tasks (bimanual: two arms, a racket
in each hand, one arm assigned per shot):

  Mjlab-Badminton-Receive-Teacher   PPO on privileged observations (true
                                    shuttle state + true trajectory prior)
  Mjlab-Badminton-Receive-Student   distills the trained teacher into the
                                    student observation set (EKF-tracked
                                    shuttle + noisy trajectory prior)
  Mjlab-Badminton-Receive-Student-PPO PPO fine-tune of the distilled student

Run from the stationary/ directory (the package imports aero/launcher/
perception_torch from there):

  uv run scripts/train_rl.py Mjlab-Badminton-Receive-Teacher \
      --env.scene.num-envs 4096
  uv run scripts/play_rl.py Mjlab-Badminton-Receive-Teacher --viewer viser
"""

import os
import sys

# aero/launcher/perception_torch live in stationary/, one level up.
_STATIONARY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STATIONARY not in sys.path:
    sys.path.insert(0, _STATIONARY)

from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from rsl_rl.runners import DistillationRunner  # noqa: E402

from humanoid_badminton.env_cfg import make_env_cfg  # noqa: E402
from humanoid_badminton.rl_cfg import (make_distill_cfg, make_student_ppo_cfg,  # noqa: E402
                                       make_teacher_ppo_cfg)


class BadmintonDistillationRunner(DistillationRunner):
    """rsl_rl DistillationRunner with mjlab's None-option stripping.

    MjlabOnPolicyRunner pops None-valued cnn_cfg/distribution_cfg/rnn_*
    from the "actor"/"critic" model dicts before rsl_rl sees them, but the
    distillation cfg names its models "student"/"teacher", so the raw
    runner passes cnn_cfg=None into MLPModel and crashes."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        for key in ("student", "teacher"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
                if train_cfg[key].get("rnn_type") is None:
                    for opt in ("rnn_type", "rnn_hidden_dim",
                                "rnn_num_layers"):
                        train_cfg[key].pop(opt, None)
        super().__init__(env, train_cfg, log_dir, device)


register_mjlab_task(
    "Mjlab-Badminton-Receive-Teacher",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_teacher_ppo_cfg(),
)

register_mjlab_task(
    "Mjlab-Badminton-Receive-Student",
    env_cfg=make_env_cfg(assign_source="ekf"),
    play_env_cfg=make_env_cfg(play=True, assign_source="ekf"),
    rl_cfg=make_distill_cfg(),
    runner_cls=BadmintonDistillationRunner,
)

register_mjlab_task(
    "Mjlab-Badminton-Receive-Student-PPO",
    env_cfg=make_env_cfg(assign_source="ekf"),
    play_env_cfg=make_env_cfg(play=True, assign_source="ekf"),
    rl_cfg=make_student_ppo_cfg(),
)
