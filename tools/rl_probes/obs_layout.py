import os, sys
sys.path.insert(0, "/workspace/humanoid/src/simulation/humanoid_rl/humanoid_rl/scripts")
from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True, "device": "cuda:0"}).app
import gymnasium as gym, humanoid_rl_tasks  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
T = "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0"
env = RslRlVecEnvWrapper(gym.make(T, cfg=parse_env_cfg(T, device="cuda:0", num_envs=2))); u = env.unwrapped
om = u.observation_manager
for g in ("policy",):
    names = om.active_terms[g]; dims = om.group_obs_term_dim[g]
    i = 0
    for n, d in zip(names, dims):
        w = int(d[0]) if len(d) == 1 else int(__import__("math").prod(d))
        print(f"[O] {g} {n:24s} dims {d}  slice [{i}:{i + w}]"); i += w
    print(f"[O] total {i}")
print("[O] OK", flush=True); env.close(); app.close()
