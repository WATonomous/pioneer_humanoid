"""Frame-to-frame jitter of each policy observation term.

The spiral policy's action signal changes 7.5x more per step than the straight
course's (RMS delta 2.5 vs 0.34 in normalised action units), which is what
"shaky legs" looks like numerically. This measures which *input* moves that
much per step: an input that swings hard every 20 ms is what the network is
reacting to.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src/simulation/humanoid_rl/humanoid_rl/scripts"))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_rl_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import cli_args  # noqa: E402

TASK = os.environ.get("TASK_OVERRIDE", "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")
CKPT = os.environ.get("CKPT_OVERRIDE", "/workspace/humanoid/models/pioneer_humanoid_spiral_foothold/model_299_gait.pt")
N, STEPS = 4, 1200

SLOTS = (
    (1.551, -0.777, 0.816, 1.5708), (0.213, 1.500, 1.417, 3.0005), (-1.504, -0.182, 2.017, 4.8331),
    (1.361, -0.666, 2.857, 7.3987), (-1.192, 0.935, 3.817, 10.3309), (1.500, -0.213, 5.017, 13.9960),
)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
u = env.unwrapped
if "Spiral" in TASK:
    term = u.event_manager.get_term_cfg("reset_base")
    term.params["slots"] = SLOTS
    term.params["xy_jitter"] = 0.03
    term.params["yaw_jitter"] = 0.03


class A:
    pass


a = A()
for k, v in dict(task=TASK, num_envs=N, checkpoint=None, device="cuda:0", distributed=False,
                 seed=5, max_iterations=None, resume=False, run_name=None, logger=None,
                 load_run=None, load_checkpoint=None, video=False, video_length=200).items():
    setattr(a, k, v)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, a)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(CKPT, load_optimizer=False)
policy = runner.get_inference_policy(device=u.device)

om = u.observation_manager
names = om.active_terms["policy"]
dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["policy"]]
print("[J] terms", list(zip(names, dims)))

prev = None
acc = {n: [] for n in names}
act_acc = []
prev_act = None
obs = env.get_observations()
with torch.inference_mode():
    for step in range(STEPS):
        action = policy(obs)
        obs, *_ = env.step(action)
        done = (u.reset_terminated | u.reset_time_outs).tolist()
        full = om.compute()["policy"].detach().cpu()
        start = 0
        for n, d in zip(names, dims):
            block = full[:, start:start + d]
            start += d
            if prev is not None:
                keep = [i for i in range(N) if not done[i] and step > 50]
                if keep:
                    acc[n].append(float((block[keep] - prev[n][keep]).square().mean().sqrt()))
        prev = {n: full[:, s:s + d] for n, d, s in
                zip(names, dims, [sum(dims[:i]) for i in range(len(dims))])}
        if prev_act is not None:
            keep = [i for i in range(N) if not done[i] and step > 50]
            if keep:
                act_acc.append(float((action.detach().cpu()[keep] - prev_act[keep]).square().mean().sqrt()))
        prev_act = action.detach().cpu()

print(f"[J] action delta RMS {torch.tensor(act_acc).mean():.3f}")
for n in names:
    if acc[n]:
        print(f"[J] {n:28s} delta RMS {torch.tensor(acc[n]).mean():.4f}")
print("[J] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
