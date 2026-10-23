import gymnasium as gym

from . import agents

# Intermediate configuration classes supply the final recipe's inheritance,
# but only the final new rough recipe and stairs scaffold are public tasks.
_ENVS = (
    ("Rough", "rough_env_cfg:PioneerHumanoidRoughEnvCfg", "PioneerHumanoidRoughPPORunnerCfg", ""),
    ("Rough", "rough_env_cfg:PioneerHumanoidRoughEnvCfg_PLAY", "PioneerHumanoidRoughPPORunnerCfg", "-Play"),
    (
        "RoughNoStairsSelectiveKneeShape",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg",
        "PioneerHumanoidRoughNoStairsSelectiveKneeShapePPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairsSelectiveKneeShape",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsSelectiveKneeShapePPORunnerCfg",
        "-Play",
    ),
    ("Flat", "flat_env_cfg:PioneerHumanoidFlatEnvCfg", "PioneerHumanoidFlatPPORunnerCfg", ""),
    ("Flat", "flat_env_cfg:PioneerHumanoidFlatEnvCfg_PLAY", "PioneerHumanoidFlatPPORunnerCfg", "-Play"),
)

for _terrain, _env_cfg, _agent_cfg, _suffix in _ENVS:
    _kwargs = {
        "env_cfg_entry_point": f"{__name__}.{_env_cfg}",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_agent_cfg}",
    }
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-{_terrain}-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs=_kwargs,
        )

# Separate runner module keeps the stairs scaffold additive: existing rough/flat
# registrations and their PPO defaults are unchanged.
for _suffix, _env_class in (
    ("", "PioneerHumanoidStairsEnvCfg"),
    ("-Play", "PioneerHumanoidStairsEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-Stairs-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.stairs_ppo_cfg:PioneerHumanoidStairsPPORunnerCfg",
            },
        )

# Step-count curriculum (stage 1: 1..4-step flights, warm-started walkers).
# Reuses the stairs PPO recipe; only the env (terrain + episode budget) differs.
for _suffix, _env_class in (
    ("", "PioneerHumanoidStairsStepCountEnvCfg"),
    ("-Play", "PioneerHumanoidStairsStepCountEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-StepCount-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.stairs_ppo_cfg:PioneerHumanoidStairsPPORunnerCfg",
            },
        )

# Bridge stage (1..12 steps at fixed 8 cm). Same runner; env differs.
for _suffix, _env_class in (
    ("", "PioneerHumanoidStairsStepCountMidEnvCfg"),
    ("-Play", "PioneerHumanoidStairsStepCountMidEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-StepCountMid-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.stairs_ppo_cfg:PioneerHumanoidStairsPPORunnerCfg",
            },
        )

# Onshape staircase course (user STEP mesh as terrain). PLAY first for visual
# verification; training follows on the same terrain.
for _suffix, _env_class in (
    ("", "PioneerHumanoidOnshapeCourseEnvCfg"),
    ("-Play", "PioneerHumanoidOnshapeCourseEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-OnshapeCourse-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.onshape_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.stairs_ppo_cfg:PioneerHumanoidStairsPPORunnerCfg",
            },
        )
# Spiral staircase (the user's Onshape spiral). Same recipe as the straight
# Onshape course, but every course term is measured along the path: the
# height is the progress coordinate, the tangent is the direction.
for _suffix, _env_class in (
    ("", "PioneerHumanoidOnshapeSpiralEnvCfg"),
    ("-Play", "PioneerHumanoidOnshapeSpiralEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-OnshapeSpiral-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.onshape_spiral_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.onshape_spiral_ppo_cfg:PioneerHumanoidOnshapeSpiralPPORunnerCfg",
            },
        )
# Up-drill (ascent to walkway) and down-drill (walkway to exit). Same runner.
for _terrain, _env_class, _play_class in (
    ("StairsUp", "PioneerHumanoidStairsUpEnvCfg", "PioneerHumanoidStairsUpEnvCfg_PLAY"),
    ("StairsDown", "PioneerHumanoidStairsDownEnvCfg", "PioneerHumanoidStairsDownEnvCfg_PLAY"),
):
    for _suffix, _cls in (("", _env_class), ("-Play", _play_class)):
        for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
            gym.register(
                id=f"{_prefix}-{_terrain}-PioneerHumanoid{_suffix}-v0",
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:{_cls}",
                    "rsl_rl_cfg_entry_point": f"{agents.__name__}.stairs_ppo_cfg:PioneerHumanoidStairsPPORunnerCfg",
                },
            )
