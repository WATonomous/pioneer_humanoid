import gymnasium as gym

from . import agents

for _name, _cfg, _runner in (("Flat", "PioneerLegOnlyFlatEnvCfg", "PioneerLegOnlyFlatPPORunnerCfg"),
                             ("Gait", "PioneerLegOnlyGaitEnvCfg", "PioneerLegOnlyGaitPPORunnerCfg"),
                             ("GaitSym", "PioneerLegOnlyGaitEnvCfg", "PioneerLegOnlyGaitSymPPORunnerCfg"),
                             ("GaitWholeBody", "PioneerWholeBodyGaitEnvCfg", "PioneerWholeBodyGaitPPORunnerCfg"),
                             ("GaitArmSwing", "PioneerWholeBodyArmSwingEnvCfg", "PioneerWholeBodyGaitPPORunnerCfg")):
    for _suffix in ("", "_PLAY"):
        gym.register(
            id=f"Isaac-Locomotion-{_name}-PioneerLegOnly{'-Play' if _suffix else ''}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.flat_env_cfg:{_cfg}{_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner}",
            },
        )
