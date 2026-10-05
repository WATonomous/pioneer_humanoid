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
