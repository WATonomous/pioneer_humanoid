"""Drop a garment on the table, arms still, and watch it for 10 s: does it drift or shrink on its own?

  CLOTH_CFG=/lehome/cloth/x.yaml isaac_run.sh idle cloth_idle.py --garment Pant_Short_Seen_4 --out /lehome/cloth/idle/x
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--garment", default="Pant_Short_Seen_4")
p.add_argument("--pose_ep", type=int, default=100, help="LeHome episode whose garment pose to use")
p.add_argument("--out", required=True)
p.add_argument("--run_start", action="store_true", help="arms in the fold run's start pose instead of the default")
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

LEHOME = os.environ.get("LEHOME_ROOT", "/lehome")
os.environ.setdefault("LEHOME_DATA", f"{LEHOME}/data")
import humanoid_garment_fold.tasks  # noqa: E402,F401
from humanoid_garment_fold.tasks.garment_pioneer_cfg import GarmentPioneerEnvCfg  # noqa: E402

from pioneer_humanoid.arm_8dof_params import GRIPPER_STROKE  # noqa: E402
from retarget import IK, load_episode, plan_track  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cfg = GarmentPioneerEnvCfg()
cfg.garment_name = args.garment
cfg.garment_cfg_base_path = f"{LEHOME}/assets/objects/Challenge_Garment"
cfg.particle_cfg_path = os.environ.get("CLOTH_CFG") or cfg.particle_cfg_path
cfg.sim.use_fabric = os.environ.get("FABRIC", "1") == "1"
cfg.decimation = 1
cfg.sim.render_interval = 3
cfg.episode_length_s = 1e4
cfg.scene_camera.width, cfg.scene_camera.height = 640, 480
env = gym.make("Humanoid-GarmentFold-Bimanual-Pioneer-v0", cfg=cfg).unwrapped
env.initialize_obs()
get_obs = env._get_observations
env._get_rewards = lambda: torch.zeros(1, device=env.device)
if not os.environ.get("NO_RESET"):
    env.reset()
if args.pose_ep >= 0:
    env.object.set_all_pose({"Garment": np.array(load_episode(args.pose_ep, "pant_short")["pose"], dtype=np.float32)})
env.scene_camera.set_world_poses_from_view(torch.tensor([[0.75, 0.55, 1.05]], device=env.device),
                                           torch.tensor([[0.0, -0.08, 0.55]], device=env.device))
cloth = env.object._cloth_prim_view
ids = torch.cat([env._l_arm, env._l_grip, env._r_arm, env._r_grip])
act = env.robot.data.joint_pos[0, ids].clone().unsqueeze(0)
if args.run_start:
    E, ik = load_episode(args.pose_ep, "pant_short"), IK()
    q = {a: plan_track(ik, a, E[a][:1], w_down=E[a + "_wdown"][:1])[0][0] for a in ("left", "right")}
    act = torch.tensor(np.concatenate([q["left"], [0.0], q["right"], [0.0]]), dtype=torch.float32, device=env.device).unsqueeze(0)
    jp = env.robot.data.default_joint_pos.clone()
    jp[0, ids] = act[0] * torch.tensor([1.0] * 7 + [GRIPPER_STROKE] + [1.0] * 7 + [GRIPPER_STROKE], device=env.device)
    env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
frames, log, x_prev = [], [], None
for n in range(360):   # 12 s at 30 fps, 3 physics steps each
    for _ in range(3):
        env.step(act)
    x = cloth.get_world_positions()[0].cpu().numpy()
    if n % 30 == 0:
        move = 0.0 if x_prev is None else float(np.linalg.norm(x - x_prev, axis=1).mean() * 1000)   # mm the cloth moved in the last second
        x_prev = x.copy()
        log.append(dict(t=n / 30, move_mm=round(move, 2), cx=round(float(x[:, 0].mean()), 3), cy=round(float(x[:, 1].mean()), 3),
                        zmax=round(float(x[:, 2].max()), 3), size_x=round(float(np.ptp(x[:, 0])), 3),
                        size_y=round(float(np.ptp(x[:, 1])), 3)))
        print("IDLE", json.dumps(log[-1]), flush=True)
    obs = get_obs()
    frames.append(np.hstack([obs["observation.images.top_rgb"][..., :3], env.scene_camera.data.output["rgb"][0].cpu().numpy()[..., :3]]))
imageio.mimsave(os.path.join(args.out, "idle.mp4"), frames, fps=30)
def _xf():
    from pxr import UsdGeom
    import omni.usd
    st = omni.usd.get_context().get_stage()
    out = {}
    for pr in st.Traverse():
        if str(pr.GetPath()).startswith(env.object.usd_prim_path) and pr.IsA(UsdGeom.Xformable):
            out[str(pr.GetPath())] = np.array(UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(0)).tolist()
            print("XF", pr.GetPath(), pr.GetTypeName(), np.round(np.array(out[str(pr.GetPath())]), 3).tolist(), flush=True)
    return json.dumps(out)


cam = env.top_camera if hasattr(env, "top_camera") else env.scene["top_camera"]
np.savez(os.path.join(args.out, "proj.npz"), x=x, img=obs["observation.images.top_rgb"][..., :3],
         M=_xf(), K=cam.data.intrinsic_matrices[0].cpu().numpy(), pos=cam.data.pos_w[0].cpu().numpy(), quat=cam.data.quat_w_ros[0].cpu().numpy())
json.dump(log, open(os.path.join(args.out, "idle.json"), "w"))
os._exit(0)
