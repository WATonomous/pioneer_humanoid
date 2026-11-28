"""Render each garment, textured and settled on the table, into one labelled grid image.

  isaac_run.sh grid render_grid.py --garments Top_Short_Seen_0,Pant_Short_Seen_4 --out /lehome/cloth/grid
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--garments", required=True)
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import os as _os, sys as _sys, traceback as _tb  # noqa: E402
_sys.excepthook = lambda t, v, tb: (_tb.print_exception(t, v, tb), _os._exit(1))

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import torch  # noqa: E402

LEHOME = os.environ.get("LEHOME_ROOT", "/lehome")
os.environ.setdefault("LEHOME_DATA", f"{LEHOME}/data")
import humanoid_garment_fold.tasks  # noqa: E402,F401
from humanoid_garment_fold.tasks.garment_pioneer_cfg import GarmentPioneerEnvCfg  # noqa: E402

os.makedirs(args.out, exist_ok=True)
for g in args.garments.split(","):
    path = os.path.join(args.out, f"{g}.png")
    if os.path.exists(path):
        continue
    cfg = GarmentPioneerEnvCfg()
    cfg.garment_name = g
    cfg.garment_cfg_base_path = f"{LEHOME}/assets/objects/Challenge_Garment"
    cfg.sim.use_fabric = True
    cfg.decimation = 1
    cfg.sim.render_interval = 3
    cfg.episode_length_s = 1e4
    cfg.scene_camera.width, cfg.scene_camera.height = 640, 480
    env = gym.make("Humanoid-GarmentFold-Bimanual-Pioneer-v0", cfg=cfg).unwrapped
    env.initialize_obs()
    env._get_observations = lambda: {}
    env._get_rewards = lambda: torch.zeros(1, device=env.device)
    env.reset()
    env.scene_camera.set_world_poses_from_view(torch.tensor([[0.0, -0.45, 1.15]], device=env.device),
                                               torch.tensor([[0.0, -0.02, 0.52]], device=env.device))
    ids = torch.cat([env._l_arm, env._l_grip, env._r_arm, env._r_grip])
    act = env.robot.data.joint_pos[0, ids].clone().unsqueeze(0)
    for _ in range(240):
        env.step(act)
    imageio.imwrite(path, env.scene_camera.data.output["rgb"][0].cpu().numpy()[..., :3])
    print("GRID", json.dumps({"garment": g, "textures": [os.path.basename(v) for v in env.object.visual_usd_paths]}), flush=True)
    env.close()
app.close()
