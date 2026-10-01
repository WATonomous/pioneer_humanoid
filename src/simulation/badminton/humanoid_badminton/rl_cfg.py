"""RSL-RL configs: teacher PPO, then teacher->student distillation.

Phase 1 (Mjlab-Badminton-Receive-Teacher): PPO where both actor and critic
see the privileged "teacher" group.

Phase 2 (Mjlab-Badminton-Receive-Student): rsl_rl DistillationRunner. Load
the trained teacher with --agent.resume True --agent.load-run <teacher-run>;
Distillation.load restores only the teacher weights by default, and the
student (on the "student" group) learns to match its actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl.config import (RslRlBaseRunnerCfg, RslRlModelCfg,
                             RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg)

HIDDEN = (512, 256, 128)
# bimanual actor: twice the single-arm width, so scripts/widen_checkpoint.py
# can warm-start it as two block-diagonal copies of the single-arm actor
# (right arm, mirrored left arm). The critic keeps HIDDEN.
ACTOR_HIDDEN = (1024, 512, 256)


def make_teacher_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=3000,
        save_interval=100,
        experiment_name="badminton_teacher",
        obs_groups={"actor": ("teacher",), "critic": ("teacher",)},
        actor=RslRlModelCfg(
            hidden_dims=ACTOR_HIDDEN,
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution",
                              "init_std": 0.5, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=HIDDEN, obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=3e-4,
            # The early-run collapses at 0.005 (runs 1-2) were caused by the
            # broken arm mount (run 6), not the coefficient: with
            # no reachable reward, entropy was the only pressure. On the fixed
            # world 0.01 overshot late-run — std drifted 0.42 -> 0.85 after
            # the kernels saturated and precision (d@t*) paid for it (run 7).
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4),
    )


def make_student_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """Phase 3: PPO fine-tune of the distilled student. Asymmetric
    actor-critic — the actor sees the noisy "student" group (what the robot
    has), the critic the privileged "teacher" group (training-only, so it
    may cheat). Distillation cannot teach hedging under an ambiguous
    estimate (the label is the clairvoyant action); optimising the
    student's own return can. Warm-start the actor with
    scripts/student_to_ppo.py, then --agent.resume True."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=1500,
        save_interval=100,
        experiment_name="badminton_student_ppo",
        obs_groups={"actor": ("student",), "critic": ("teacher",)},
        actor=RslRlModelCfg(
            hidden_dims=ACTOR_HIDDEN,
            obs_normalization=True,
            # init_std is overwritten by the distilled student's std (0.1)
            distribution_cfg={"class_name": "GaussianDistribution",
                              "init_std": 0.1, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=HIDDEN, obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            # fine-tune: a third of the teacher's lr so the first updates
            # (fresh critic, meaningless advantages) do not wreck the init
            learning_rate=1e-4,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4),
    )


@dataclass
class RslRlDistillationAlgorithmCfg:
    num_learning_epochs: int = 1
    gradient_length: int = 15
    learning_rate: float = 1e-3
    max_grad_norm: float | None = 1.0
    loss_type: str = "mse"
    optimizer: str = "adam"
    class_name: str = "Distillation"


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Runner cfg shaped for rsl_rl Distillation.construct_algorithm: model
    cfgs under "student"/"teacher", algorithm.class_name = Distillation."""

    class_name: str = "DistillationRunner"
    student: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=ACTOR_HIDDEN,
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution",
                              "init_std": 0.1, "std_type": "scalar"}))
    teacher: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=ACTOR_HIDDEN,
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution",
                              "init_std": 0.5, "std_type": "scalar"}))
    algorithm: RslRlDistillationAlgorithmCfg = field(
        default_factory=RslRlDistillationAlgorithmCfg)


def make_distill_cfg() -> RslRlDistillationRunnerCfg:
    return RslRlDistillationRunnerCfg(
        num_steps_per_env=24,
        max_iterations=1500,
        save_interval=100,
        experiment_name="badminton_student",
        resume=True,                       # teacher checkpoint required
        load_run=".*badminton_teacher.*",
        obs_groups={"student": ("student",), "teacher": ("teacher",)},
    )
