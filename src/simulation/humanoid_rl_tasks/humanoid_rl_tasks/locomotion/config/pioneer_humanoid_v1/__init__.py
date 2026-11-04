import gymnasium as gym

from . import agents

_ENVS = (
    ("Rough", "rough_env_cfg:PioneerHumanoidRoughEnvCfg", "PioneerHumanoidRoughPPORunnerCfg", ""),
    ("Rough", "rough_env_cfg:PioneerHumanoidRoughEnvCfg_PLAY", "PioneerHumanoidRoughPPORunnerCfg", "-Play"),
    ("Flat", "flat_env_cfg:PioneerHumanoidFlatEnvCfg", "PioneerHumanoidFlatPPORunnerCfg", ""),
    ("Flat", "flat_env_cfg:PioneerHumanoidFlatEnvCfg_PLAY", "PioneerHumanoidFlatPPORunnerCfg", "-Play"),
    # #335 gain set B (D5); the IDs above use gain set A.
    ("Rough", "rough_env_cfg:PioneerHumanoidRoughGainsBEnvCfg", "PioneerHumanoidRoughGainsBPPORunnerCfg", "-GainsB"),
    ("Rough", "rough_env_cfg:PioneerHumanoidRoughGainsBEnvCfg_PLAY", "PioneerHumanoidRoughGainsBPPORunnerCfg", "-GainsB-Play"),
    ("Flat", "flat_env_cfg:PioneerHumanoidFlatGainsBEnvCfg", "PioneerHumanoidFlatGainsBPPORunnerCfg", "-GainsB"),
    ("Flat", "flat_env_cfg:PioneerHumanoidFlatGainsBEnvCfg_PLAY", "PioneerHumanoidFlatGainsBPPORunnerCfg", "-GainsB-Play"),
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
