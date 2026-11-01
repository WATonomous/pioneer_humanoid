import gymnasium as gym

from . import agents

for _name, _cfg, _runner in (("Flat-PioneerLegOnly", "flat_env_cfg:PioneerLegOnlyFlatEnvCfg", "PioneerLegOnlyFlatPPORunnerCfg"),
                             ("Flat-PioneerWholeBody", "flat_env_cfg:PioneerWholeBodyFlatEnvCfg", "PioneerWholeBodyFlatPPORunnerCfg"),
                             ("RoughBlind-PioneerLegOnly", "rough_env_cfg:PioneerLegOnlyRoughEnvCfg", "PioneerLegOnlyRoughPPORunnerCfg"),
                             ("RoughPerceptive-PioneerLegOnly", "rough_env_cfg:PioneerLegOnlyPerceptiveEnvCfg",
                              "PioneerLegOnlyPerceptivePPORunnerCfg"),
                             ("RoughFoothold-PioneerLegOnly", "foothold_env_cfg:PioneerLegOnlyFootholdEnvCfg",
                              "PioneerLegOnlyFootholdPPORunnerCfg"),
                             ("RoughFoothold-PioneerWholeBody", "foothold_env_cfg:PioneerWholeBodyFootholdEnvCfg",
                              "PioneerWholeBodyFootholdPPORunnerCfg")):
    for _suffix in ("", "_PLAY"):
        gym.register(
            id=f"Isaac-Locomotion-{_name}{'-Play' if _suffix else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{_cfg}{_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner}",
            },
        )

gym.register(
    id="Isaac-Locomotion-RoughFoothold-PioneerWholeBody-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.foothold_env_cfg:PioneerWholeBodyFootholdEnvCfg_DEMO",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PioneerWholeBodyFootholdPPORunnerCfg",
    },
)
