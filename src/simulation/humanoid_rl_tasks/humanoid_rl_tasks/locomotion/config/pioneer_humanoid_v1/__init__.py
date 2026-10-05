import gymnasium as gym

from . import agents

_ENVS = (
    ("Rough", "rough_env_cfg:PioneerHumanoidRoughEnvCfg", "PioneerHumanoidRoughPPORunnerCfg", ""),
    ("Rough", "rough_env_cfg:PioneerHumanoidRoughEnvCfg_PLAY", "PioneerHumanoidRoughPPORunnerCfg", "-Play"),
    (
        "RoughNoStairs",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsEnvCfg",
        "PioneerHumanoidRoughNoStairsPPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairs",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsPPORunnerCfg",
        "-Play",
    ),
    (
        "RoughNoStairsGaitTune",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsGaitTuneEnvCfg",
        "PioneerHumanoidRoughNoStairsPPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairsGaitTune",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsGaitTuneEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsPPORunnerCfg",
        "-Play",
    ),
    (
        "RoughNoStairsFootTune",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsFootTuneEnvCfg",
        "PioneerHumanoidRoughNoStairsPPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairsFootTune",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsFootTuneEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsPPORunnerCfg",
        "-Play",
    ),
    (
        "RoughNoStairsFootTuneFine",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsFootTuneFineEnvCfg",
        "PioneerHumanoidRoughNoStairsPPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairsFootTuneFine",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsFootTuneFineEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsPPORunnerCfg",
        "-Play",
    ),
    (
        "RoughNoStairsFootTuneFineNormalizedScratch",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsFootTuneFineEnvCfg",
        "PioneerHumanoidRoughNoStairsFootTuneFineNormalizedScratchPPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairsFootTuneFineNormalizedScratch",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsFootTuneFineEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsFootTuneFineNormalizedScratchPPORunnerCfg",
        "-Play",
    ),
    (
        "RoughNoStairsFootTuneFineKneeAxisFixedNormalizedScratch",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsFootTuneFineKneeAxisFixedEnvCfg",
        "PioneerHumanoidRoughNoStairsFootTuneFineKneeAxisFixedNormalizedScratchPPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairsFootTuneFineKneeAxisFixedNormalizedScratch",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsFootTuneFineKneeAxisFixedEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsFootTuneFineKneeAxisFixedNormalizedScratchPPORunnerCfg",
        "-Play",
    ),
    (
        "RoughNoStairsKneeAxisFixedLegClearance",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsKneeAxisFixedLegClearanceEnvCfg",
        "PioneerHumanoidRoughNoStairsFootTuneFineKneeAxisFixedNormalizedScratchPPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairsKneeAxisFixedLegClearance",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsKneeAxisFixedLegClearanceEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsFootTuneFineKneeAxisFixedNormalizedScratchPPORunnerCfg",
        "-Play",
    ),
    (
        "RoughNoStairsKneeAxisFixedSelectiveSelfCollision",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsKneeAxisFixedSelectiveSelfCollisionEnvCfg",
        "PioneerHumanoidRoughNoStairsSelectiveSelfCollisionNormalizedScratchPPORunnerCfg",
        "",
    ),
    (
        "RoughNoStairsKneeAxisFixedSelectiveSelfCollision",
        "rough_env_cfg:PioneerHumanoidRoughNoStairsKneeAxisFixedSelectiveSelfCollisionEnvCfg_PLAY",
        "PioneerHumanoidRoughNoStairsSelectiveSelfCollisionNormalizedScratchPPORunnerCfg",
        "-Play",
    ),
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
