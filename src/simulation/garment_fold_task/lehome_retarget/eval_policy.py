"""Evaluate a trained policy (e.g. Pioneer-1) on the garment fold in the same Isaac env, cameras and grasp as the
retargeted data: LeHome garment poses, 1/30 s per action, the picker grasp on a fresh close at the table, LeHome's
success check.

  isaac_run.sh NAME eval_policy.py --policy /lehome/p1/run/checkpoints/last/pretrained_model --episodes 225:250 \
      --out /lehome/out/eval_p1 [--video 3]
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True)
p.add_argument("--kind", default="top_short")
p.add_argument("--episodes", default="225:250", help="LeHome episode range a:b (garment poses), one garment")
p.add_argument("--out", required=True)
p.add_argument("--max_frames", type=int, default=600)
p.add_argument("--stretch", type=int, default=3)
p.add_argument("--video", type=int, default=0)
p.add_argument("--task", default="fold the garment")
p.add_argument("--flat_cloth", action="store_true", help="no garment textures (data recorded before they loaded)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import os as _os, sys as _sys, traceback as _tb  # noqa: E402
_sys.excepthook = lambda t, v, tb: (_tb.print_exception(t, v, tb), _os._exit(1))   # an uncaught error exits instead of waiting on Isaac's error dialog

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

LEHOME = os.environ.get("LEHOME_ROOT", "/lehome")
os.environ.setdefault("LEHOME_DATA", f"{LEHOME}/data")
import humanoid_garment_fold.tasks  # noqa: E402,F401
from humanoid_garment_fold.tasks.garment_pioneer_cfg import GarmentPioneerEnvCfg  # noqa: E402
from humanoid_garment_fold.utils.success_checker_garment import success_checker_garment_fold  # noqa: E402
from humanoid_robot_learning.sim.evaluate import load_policy  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from pioneer_humanoid.arm_8dof_params import GRIPPER_STROKE, TCP_LOCAL  # noqa: E402

from retarget import IK, TCP_GRASP_Z, load_episode, plan_track  # noqa: E402

a0, a1 = map(int, args.episodes.split(":"))
EPS = list(range(a0, a1))
GARMENT = load_episode(EPS[0], args.kind)["garment"]
PIN_RADIUS = 0.018
TIP_LOCAL = np.array([TCP_LOCAL[0], TCP_LOCAL[1], -0.240])
os.makedirs(args.out, exist_ok=True)
LOG = open(os.path.join(args.out, f"eval_{a0}_{a1}.jsonl"), "a")

cfg = GarmentPioneerEnvCfg()
cfg.garment_name = GARMENT
cfg.particle_cfg_path = os.environ.get("CLOTH_CFG") or cfg.particle_cfg_path
if args.flat_cloth:
    import humanoid_garment_fold.tasks.garment_pioneer_env as gpe
    _fix = gpe.fix_garment_asset_paths
    gpe.fix_garment_asset_paths = lambda *a: (_fix(*a), a[1].__setattr__("visual_usd_paths", []))[0]
cfg.garment_cfg_base_path = f"{LEHOME}/assets/objects/Challenge_Garment"
cfg.sim.use_fabric = True
cfg.decimation = 1
cfg.sim.render_interval = args.stretch
cfg.episode_length_s = 1e4
cfg.scene_camera.width, cfg.scene_camera.height = (640, 480) if args.video else (64, 48)
env = gym.make("Humanoid-GarmentFold-Bimanual-Pioneer-v0", cfg=cfg).unwrapped
env.initialize_obs()
robot = env.robot
get_obs = env._get_observations
env._get_observations = lambda: {}
env._get_rewards = lambda: torch.zeros(1, device=env.device)
if args.video:
    env.scene_camera.set_world_poses_from_view(torch.tensor([[0.75, 0.55, 1.05]], device=env.device),
                                               torch.tensor([[0.0, -0.08, 0.55]], device=env.device))
ids = torch.cat([env._l_arm, env._l_grip, env._r_arm, env._r_grip])
SCALE = torch.tensor([1.0] * 7 + [GRIPPER_STROKE] + [1.0] * 7 + [GRIPPER_STROKE], device=env.device)
cloth = env.object._cloth_prim_view
# LeHome's check points index the garment mesh's vertices; on the GPU the cloth is PhysX's particles, which merge
# duplicate vertices on some garments. Map each check vertex to its particle (same rest position) once.
def _check_particles():
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(env.garment_config.asset_path)
    mesh = next(p_ for p_ in st.Traverse() if p_.GetTypeName() == "Mesh")
    pts = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
    o = env.object
    world = o.transform_points(pts, np.asarray(o.init_pos, float), np.asarray(o.init_ori, float), np.asarray(o.init_scale, float))
    part = np.asarray(o.initial_points_positions.cpu() if hasattr(o.initial_points_positions, "cpu") else o.initial_points_positions).reshape(-1, 3)
    idx, gap = [], []
    for c in o.check_points:
        d = np.linalg.norm(part - world[c], axis=1)
        idx.append(int(np.argmin(d)))
        gap.append(float(d.min()))
    print("CHECK_MAP", len(pts), "vertices ->", len(part), "particles, max gap", round(max(gap), 5), flush=True)
    return idx


CHECK = _check_particles()
env.object.check_points = CHECK   # the vendored success checker reads these
DT = env.physics_dt
policy, pre, post = load_policy(args.policy, "cuda")
ik = IK()


def link7(side):
    bi = robot.body_names.index(f"{side}_link7")
    pw = robot.data.body_pos_w[0, bi].cpu().numpy().astype(np.float64)
    w, x, y, z = robot.data.body_quat_w[0, bi].cpu().tolist()
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return pw, R


def physics_step(act, pins):
    env.step(act)
    if not pins:
        return
    x, v = cloth.get_world_positions(), cloth.get_velocities()
    for side, (idx, local) in pins.items():
        pw, R = link7(side)
        new = torch.as_tensor(pw + local @ R.T, dtype=x.dtype, device=x.device)
        v[0, idx] = (new - x[0, idx]) / DT
        x[0, idx] = new
    cloth.set_world_positions(x)
    cloth.set_velocities(v)


for ep in EPS:
    t0 = time.time()
    E = load_episode(ep, args.kind)
    start = {a: plan_track(ik, a, E[a][:1], w_down=E[a + "_wdown"][:1])[0][0] for a in ("left", "right")}
    env.reset()
    env.object.set_all_pose({"Garment": np.array(E["pose"], dtype=np.float32)})
    jp = robot.data.default_joint_pos.clone()
    a_start = torch.tensor(np.concatenate([start["left"], [0.0], start["right"], [0.0]]), dtype=torch.float32, device=env.device)
    jp[0, ids] = a_start * SCALE
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    for _ in range(67 * args.stretch):   # as the data: 7 frames after reset + 60 to settle the cloth
        env.step(a_start.unsqueeze(0))
    policy.reset()
    pins, prev_cmd, armed, grasps, frames = {}, {"left": 1.0, "right": 1.0}, {"left": False, "right": False}, 0, []
    for n in range(args.max_frames):
        obs = get_obs()
        o = {"observation.state": obs["observation.state"].astype(np.float32),
             "observation.images.top": obs["observation.images.top_rgb"][..., :3],
             "observation.images.wrist_left": obs["observation.images.left_rgb"][..., :3],
             "observation.images.wrist_right": obs["observation.images.right_rgb"][..., :3]}
        action = predict_action(o, policy, torch.device("cuda"), pre, post, use_amp=False, task=args.task)
        act = torch.as_tensor(action, dtype=torch.float32, device=env.device).reshape(1, -1)
        if ep - a0 < args.video:
            frames.append(np.hstack([o["observation.images.top"], env.scene_camera.data.output["rgb"][0].cpu().numpy()[..., :3]]))
        for _ in range(args.stretch):
            physics_step(act, pins)
        for side, k in (("left", 7), ("right", 15)):
            cmd = float(act[0, k])
            pw, R = link7(side)
            if cmd > 0.5 and prev_cmd[side] <= 0.5:   # a fresh close near the table may grasp; a close in the air never does
                armed[side] = (pw + R @ np.asarray(TCP_LOCAL))[2] < TCP_GRASP_Z + 0.025
            grip = float(robot.data.joint_pos[0, env._l_grip if side == "left" else env._r_grip].item()) / GRIPPER_STROKE
            if cmd > 0.5 and armed[side] and side not in pins and grip > 0.6:
                tip = pw + R @ TIP_LOCAL
                x = cloth.get_world_positions()[0].cpu().numpy()
                dist = np.linalg.norm(x - tip, axis=1)
                idx = np.nonzero(dist < PIN_RADIUS)[0]
                if len(idx) < 20:
                    idx = np.argsort(dist)[:30]
                    idx = idx[dist[idx] < 2 * PIN_RADIUS]
                if len(idx):
                    pins[side] = (torch.as_tensor(idx, device=env.device), (x[idx] - pw) @ R)
                    grasps += 1
                armed[side] = False
            elif cmd < 0.5:
                armed[side] = False
                if side in pins:
                    idx = pins.pop(side)[0]
                    v = cloth.get_velocities()
                    v[0, idx] = 0.0
                    cloth.set_velocities(v)
            prev_cmd[side] = cmd
    pins = {}
    for _ in range(30):
        env.step(act)
    for _ in range(50):
        res = success_checker_garment_fold(env.object, env.garment_loader.get_garment_type(GARMENT))
        if isinstance(res, dict):
            break
    success = bool(res.get("success", False)) if isinstance(res, dict) else False
    checks = [[bool(d.get("passed")), round(float(d.get("value", -1)), 2)] for d in res.get("details", {}).values()] if isinstance(res, dict) else None
    if frames:
        imageio.mimsave(os.path.join(args.out, f"eval_ep{ep:03d}_{'ok' if success else 'fail'}.mp4"), frames, fps=30)
    rec = dict(ep=ep, garment=GARMENT, success=success, grasps=grasps, checks=checks, total_s=round(time.time() - t0, 1))
    print("EVAL", json.dumps(rec), flush=True)
    LOG.write(json.dumps(rec) + "\n")
    LOG.flush()

LOG.close()
env.close()
app.close()
