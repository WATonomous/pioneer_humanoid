"""Grasp probe: the right gripper descends onto the settled shirt at one point, closes, lifts 10 cm. Reports, per fingertip
depth (relative to the table top), how far the gripper closed and how much cloth came up."""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--depths", default="0.004,0.0,-0.003")
p.add_argument("--point", default="0.16,0.005")
p.add_argument("--open", default="0.0", help="gripper command while approaching (0 open, 1 closed), one per depth or one for all")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

os.environ.setdefault("PIONEER_SRC", "/workspace/humanoid/src/pioneer_humanoid")
os.environ.setdefault("LEHOME_DATA", "/lehome/data")
import humanoid_garment_fold.tasks  # noqa: E402,F401
from humanoid_garment_fold.tasks.garment_pioneer_cfg import GarmentPioneerEnvCfg  # noqa: E402

from retarget import IK, TABLE_Z  # noqa: E402

GI = json.load(open("/lehome/data/top_short_merged/meta/garment_info.json"))
cfg = GarmentPioneerEnvCfg()
cfg.garment_name = "Top_Short_Seen_0"
cfg.garment_cfg_base_path = "/lehome/assets/objects/Challenge_Garment"
cfg.sim.use_fabric = True
cfg.decimation = cfg.sim.render_interval = 3
cfg.episode_length_s = 1e4
cfg.scene_camera.width, cfg.scene_camera.height = 640, 480
env = gym.make("Humanoid-GarmentFold-Bimanual-Pioneer-v0", cfg=cfg).unwrapped
env.initialize_obs()
env.scene_camera.set_world_poses_from_view(torch.tensor([[0.45, 0.25, 0.75]], device=env.device),
                                           torch.tensor([[0.15, 0.0, 0.55]], device=env.device))
robot = env.robot
ids = torch.cat([env._l_arm, env._l_grip, env._r_arm, env._r_grip])
ik = IK()
px, py = map(float, args.point.split(","))
out = []
depths = list(map(float, args.depths.split(",")))
opens = list(map(float, args.open.split(",")))
opens = opens * len(depths) if len(opens) == 1 else opens
for depth, op in zip(depths, opens):
    env.reset()
    env.object.set_all_pose({"Garment": np.array(GI["Top_Short_Seen_0"]["0"]["object_initial_pose"], dtype=np.float32)})
    qL = ik.home["left"]
    ik.d.qpos[ik.qadr["right"]] = ik.home["right"]
    tcp_z = TABLE_Z + 0.025 + depth
    rng = np.random.default_rng(0)
    best = None
    for k in range(12):
        ik.d.qpos[ik.qadr["right"]] = ik.home["right"] if k == 0 else rng.uniform(*ik.lim["right"].T)
        q, e = ik.solve("right", np.array([px, py, tcp_z]), iters=300, w_down=1.0, w_home=0.0)
        if best is None or e < best[1] - 1e-4:
            best = (q, e)
    ik.d.qpos[ik.qadr["right"]] = best[0]
    down, e1 = ik.solve("right", np.array([px, py, tcp_z]), iters=100, w_down=1.0, w_home=0.0)
    above, _ = ik.solve("right", np.array([px, py, tcp_z + 0.05]), iters=100, w_down=1.0, w_home=0.0)
    ik.d.qpos[ik.qadr["right"]] = down
    lift, _ = ik.solve("right", np.array([px, py, tcp_z + 0.10]), iters=100, w_down=1.0, w_home=0.0)

    def act(qR, g):
        return torch.tensor(np.concatenate([qL, [0.0], qR, [g]]), dtype=torch.float32, device=env.device).unsqueeze(0)
    for _ in range(40):   # settle with the gripper above
        env.step(act(above, op))
    for k in range(15):
        env.step(act(above + (down - above) * (k + 1) / 15, op))
    trace = []
    for _ in range(int(os.environ.get("CLOSE_FRAMES", "10"))):
        env.step(act(down, 1.0))
        trace.append(round(float(robot.data.joint_pos[0, env._r_grip].item()) * 1000, 1))
    bi = robot.body_names.index("right_link7")
    w, xq, yq, zq = robot.data.body_quat_w[0, bi].cpu().tolist()
    Rz = np.array([2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)])
    Rr = np.array([[1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)], [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)], [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)]])
    tip = robot.data.body_pos_w[0, bi].cpu().numpy() + Rr @ np.array([-0.0014, 0.0, -0.246])
    fa = robot.data.body_pos_w[0, robot.body_names.index("right_finger_a")].cpu().numpy()
    print("CLOSE_TRACE", depth, trace, "tip z", round(float(tip[2]), 4), "tilt", round(float(np.degrees(np.arccos(Rr[2, 2]))), 1), flush=True)
    grip = float(robot.data.joint_pos[0, env._r_grip].item())
    x0 = env.object.get_current_mesh_points()[0]
    x0 = x0.cpu().numpy() if hasattr(x0, "cpu") else np.asarray(x0)
    for k in range(20):
        env.step(act(down + (lift - down) * (k + 1) / 20, 1.0))
    for _ in range(10):
        env.step(act(lift, 1.0))
    x = env.object.get_current_mesh_points()[0]
    x = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    dz = x[:, 2] - x0[:, 2]
    rise = float(np.percentile(dz, 99.5))
    n_up = int((dz > 0.03).sum())
    imageio.imwrite(f"/lehome/logs/probe_{depth:+.3f}_{op:.2f}.png", env.scene_camera.data.output["rgb"][0].cpu().numpy()[..., :3])
    out.append(dict(depth=depth, ik_err_mm=round(e1 * 1000, 1), grip_closed_mm=round(grip * 1000, 1),
                    open=op, cloth_rise_mm=round(rise * 1000, 1), particles_up_3cm=n_up, cloth_top=round(float(x[:, 2].max()), 3)))
    print("PROBE", out[-1], flush=True)
json.dump(out, open("/lehome/logs/probe.json", "w"), indent=1)
env.close()
app.close()
