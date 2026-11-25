import gymnasium as gym

from . import agents

# Register the final walking recipe and Onshape stair courses alongside legacy tasks.
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

# Straight staircase with learned arm swing (shoulders + elbows, 16 actions).
# Warm-started from the rough checkpoint with network surgery (12 -> 16 actions).
for _suffix, _env_class in (
    ("", "PioneerHumanoidOnshapeCourseArmSwingEnvCfg"),
    ("-Play", "PioneerHumanoidOnshapeCourseArmSwingEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-StairsStraightArmSwing-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.onshape_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.onshape_ppo_cfg:PioneerHumanoidOnshapePPORunnerCfg",
            },
        )
# Spiral staircase with learned arm swing (shoulders + elbows, 16 actions).
# Warm-started from rough checkpoint with network surgery (12 -> 16 actions).
for _suffix, _env_class in (
    ("", "PioneerHumanoidOnshapeSpiralArmSwingEnvCfg"),
    ("-Play", "PioneerHumanoidOnshapeSpiralArmSwingEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-StairsSpiralArmSwing-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.onshape_spiral_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.onshape_spiral_ppo_cfg:PioneerHumanoidOnshapeSpiralPPORunnerCfg",
            },
        )
# Straight staircase with foothold-conditioned perception (the "Watch Your Step"
# port): the policy sees predicted next footholds + 7x7 patches, the critic sees
# the full height map, and the arms are swung on the gait clock (12 leg actions).
# Warm-startable from the rough foothold walker without surgery.
for _suffix, _env_class in (
    ("", "PioneerHumanoidStairsFootholdEnvCfg"),
    ("-Play", "PioneerHumanoidStairsFootholdEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-StairsStraightFoothold-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.onshape_foothold_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.onshape_foothold_ppo_cfg:PioneerHumanoidStairsFootholdPPORunnerCfg",
            },
        )

# Spiral staircase with foothold-conditioned perception: same perception and
# action layout as the straight foothold task, on the helicoid course.
for _suffix, _env_class in (
    ("", "PioneerHumanoidSpiralFootholdEnvCfg"),
    ("-Play", "PioneerHumanoidSpiralFootholdEnvCfg_PLAY"),
):
    for _prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
        gym.register(
            id=f"{_prefix}-StairsSpiralFoothold-PioneerHumanoid{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.onshape_spiral_foothold_env_cfg:{_env_class}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.onshape_spiral_foothold_ppo_cfg:PioneerHumanoidSpiralFootholdPPORunnerCfg",
            },
        )
