"""Replay LeHome garment-fold demos on the pioneer_bimanual_arm_8dof in the Isaac garment env; record the folds that
pass LeHome's success check as a LeRobot dataset (pioneer_8dof joint layout, 30 fps, top + both wrist cameras).

Per demo: LeHome's garment and its initial pose; their SO-101 tool-point path (forward kinematics) mapped to our tool
point (retarget.py) and tracked with IK on the 8dof arm; their gripper open/close timing.
- Each demo frame is 1/30 s of sim (3 physics steps of 1/90 s); LeHome stepped 1/90 s per frame, faster than our joints.
- The demo clock waits for the arm (tool point within ARRIVE, at most MAX_WAIT frames) and for a gripper to close.
- Grasp: a closing gripper holds the cloth particles between its fingertips (moved with the fingers every physics step)
  until it opens. Our gripper's friction pinch alone does not lift LeHome's cloth (10 g particles at 2 g).

  isaac_run.sh NAME run_retarget.py --episodes 0:25 --out /lehome/out/top_short [--video N] [--no_record]
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--kind", default="top_short")
p.add_argument("--episodes", default="0:25", help="LeHome episode range a:b (one garment = 25 episodes)")
p.add_argument("--out", default="/lehome/out/top_short")
p.add_argument("--stretch", type=int, default=3, help="physics steps (1/90 s) per demo frame")
p.add_argument("--video", type=int, default=0, help="also write an mp4 (top + external view) for the first N episodes")
p.add_argument("--no_record", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

os.environ.setdefault("PIONEER_SRC", "/workspace/humanoid/src/pioneer_humanoid")
os.environ.setdefault("LEHOME_DATA", "/lehome/data")
import humanoid_garment_fold.tasks  # noqa: E402,F401
from humanoid_garment_fold.tasks.garment_pioneer_cfg import GarmentPioneerEnvCfg  # noqa: E402
from pioneer_humanoid.arm_8dof_params import GRIPPER_STROKE, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, TCP_LOCAL  # noqa: E402

from humanoid_garment_fold.utils.success_checker_garment import success_checker_garment_fold  # noqa: E402

from retarget import IK, TCP_GRASP_Z, load_episode, plan_track, snap_grasps  # noqa: E402

a0, a1 = map(int, args.episodes.split(":"))
EPS = list(range(a0, a1))
garments = {load_episode(e, args.kind)["garment"] for e in EPS}
assert len(garments) == 1, f"one garment per process, got {garments}"
GARMENT = garments.pop()
NAMES = LEFT_ARM_JOINTS + ["left_gripper"] + RIGHT_ARM_JOINTS + ["right_gripper"]
CAMS = {"top": "observation.images.top_rgb", "wrist_left": "observation.images.left_rgb",
        "wrist_right": "observation.images.right_rgb"}
ARRIVE, MAX_WAIT, GRASP_DWELL, RELEASE_DWELL = 0.008, 45, 8, 8
PIN_RADIUS = 0.025          # m around the fingertip midpoint
TIP_LOCAL = np.array([TCP_LOCAL[0], TCP_LOCAL[1], -0.240])   # fingertip midpoint, 6 mm up from the tips (link7 frame)
os.makedirs(args.out, exist_ok=True)
LOG = open(os.path.join(args.out, f"log_{a0}_{a1}.jsonl"), "a")

cfg = GarmentPioneerEnvCfg()
cfg.garment_name = GARMENT
cfg.garment_cfg_base_path = "/lehome/assets/objects/Challenge_Garment"
cfg.sim.use_fabric = True   # with False the headless cameras never see the physics move (frozen frames)
cfg.decimation = 1          # we step physics ourselves (the grasp acts every physics step), render once per frame
cfg.sim.render_interval = args.stretch
cfg.episode_length_s = 1e4
cfg.scene_camera.width, cfg.scene_camera.height = (640, 480) if args.video else (64, 48)   # dev-only external view
env = gym.make("Humanoid-GarmentFold-Bimanual-Pioneer-v0", cfg=cfg).unwrapped
env.initialize_obs()
robot = env.robot
get_obs = env._get_observations
env._get_observations = lambda: {}                                     # read once per frame below, not per step
env._get_rewards = lambda: torch.zeros(1, device=env.device)          # its success check per step is not needed
if args.video:
    env.scene_camera.set_world_poses_from_view(torch.tensor([[0.75, 0.55, 1.05]], device=env.device),
                                               torch.tensor([[0.0, -0.08, 0.55]], device=env.device))
ids = torch.cat([env._l_arm, env._l_grip, env._r_arm, env._r_grip])
SCALE = torch.tensor([1.0] * 7 + [GRIPPER_STROKE] + [1.0] * 7 + [GRIPPER_STROKE], device=env.device)
cloth = env.object._cloth_prim_view
CHECK = list(env.object.check_points)
DT = env.physics_dt

ds = None
if not args.no_record:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    feats = {k: {"dtype": "float32", "shape": (16,), "names": NAMES} for k in ("observation.state", "action")}
    for k in CAMS:
        feats[f"observation.images.{k}"] = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    root = os.path.join(args.out, "dataset", GARMENT)
    if os.path.isdir(root):
        ds = LeRobotDataset(repo_id=f"humanoid/pioneer_8dof_lehome_{args.kind}", root=root)
    else:
        ds = LeRobotDataset.create(repo_id=f"humanoid/pioneer_8dof_lehome_{args.kind}", fps=30, root=root,
                                   robot_type="pioneer_8dof", use_videos=True, image_writer_threads=4, features=feats)
done = set()
for line in open(os.path.join(args.out, f"log_{a0}_{a1}.jsonl")):
    rec = json.loads(line)
    if "success" in rec:
        done.add(rec["ep"])

ik = IK()


def link7(side):
    bi = robot.body_names.index(f"{side}_link7")
    pw = robot.data.body_pos_w[0, bi].cpu().numpy().astype(np.float64)
    w, x, y, z = robot.data.body_quat_w[0, bi].cpu().tolist()
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return pw, R


def tcp_world(side):
    pw, R = link7(side)
    return pw + R @ np.asarray(TCP_LOCAL)


def action_row(qL, gL, qR, gR):
    return torch.tensor(np.concatenate([qL, [gL], qR, [gR]]), dtype=torch.float32, device=env.device).unsqueeze(0)


def physics_step(act, pins):
    """One 1/90 s step; then the held particles follow their gripper (position and velocity)."""
    env.step(act)
    if not pins:
        return
    x = cloth.get_world_positions()
    v = cloth.get_velocities()
    for side, (idx, local) in pins.items():
        pw, R = link7(side)
        new = torch.as_tensor(pw + local @ R.T, dtype=x.dtype, device=x.device)
        v[0, idx] = (new - x[0, idx]) / DT
        x[0, idx] = new
    cloth.set_world_positions(x)
    cloth.set_velocities(v)


for ep in EPS:
    if ep in done:
        continue
    t0 = time.time()
    try:
        E = load_episode(ep, args.kind)
        g = {"left": E["left_closed"].astype(float), "right": E["right_closed"].astype(float)}
        start = {}
        for a in ("left", "right"):   # start pose only; the full path is planned once the cloth has settled
            q, _, _ = plan_track(ik, a, E[a][:1], w_down=E[a + "_wdown"][:1])
            start[a] = q[0]

        env.reset()
        env.object.set_all_pose({"Garment": np.array(E["pose"], dtype=np.float32)})
        jp = robot.data.default_joint_pos.clone()
        jp[0, ids] = action_row(start["left"], 0.0, start["right"], 0.0)[0] * SCALE
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        hold = action_row(start["left"], g["left"][0], start["right"], g["right"][0])
        for _ in range(7 * args.stretch):   # as LeHome (20 physics steps holding the arms after reset); a multiple of stretch keeps renders on frame ends
            env.step(hold)
        for _ in range(60 * args.stretch):  # our grasps aim at the settled cloth: let it land first (not recorded)
            env.step(hold)
        shifts = snap_grasps(E, cloth.get_world_positions()[0].cpu().numpy()[CHECK], args.kind)
        plan = {a: plan_track(ik, a, E[a], w_down=E[a + "_wdown"]) for a in ("left", "right")}
        qL, qR = plan["left"][0], plan["right"][0]
        ik_err = max(plan[a][1].max() for a in plan)
        t_plan = time.time() - t0

        frames, f, waited, dwell, n, pins, lift, grasps, glog = [], 0, 0, 0, 0, {}, 0.0, 0, []
        closed_on = {"left": False, "right": False}
        off = {"left": np.zeros(3), "right": np.zeros(3)}
        held_since = {"left": False, "right": False}
        rdwell = 0
        while f < len(qL):
            act = action_row(qL[f], g["left"][f], qR[f], g["right"][f])
            obs = get_obs()
            if ds is not None:
                fr = {"observation.state": obs["observation.state"].astype(np.float32),
                      "action": act[0].cpu().numpy().astype(np.float32), "task": "fold the garment"}
                for k, src in CAMS.items():
                    fr[f"observation.images.{k}"] = obs[src][..., :3]
                ds.add_frame(fr)
            if ep - a0 < args.video:
                frames.append(np.hstack([obs["observation.images.top_rgb"][..., :3],
                                         env.scene_camera.data.output["rgb"][0].cpu().numpy()[..., :3]]))
            for _ in range(args.stretch):
                physics_step(act, pins)
            n += 1
            # grasp / release
            for side in ("left", "right"):
                grip = float(robot.data.joint_pos[0, env._l_grip if side == "left" else env._r_grip].item()) / GRIPPER_STROKE
                if g[side][f] > 0.5 and side not in pins and grip > 0.6:
                    pw, R = link7(side)
                    tip = pw + R @ TIP_LOCAL
                    x = cloth.get_world_positions()[0].cpu().numpy()
                    dist = np.linalg.norm(x - tip, axis=1)
                    idx = np.nonzero(dist < PIN_RADIUS)[0]
                    if len(idx) < 20:
                        idx = np.argsort(dist)[:30]
                        idx = idx[dist[idx] < 2 * PIN_RADIUS]
                    if len(idx) and closed_on[side]:
                        pins[side] = (torch.as_tensor(idx, device=env.device), (x[idx] - pw) @ R)
                        held_since[side] = True
                        grasps += 1
                        glog.append([side, f, int(len(idx)), np.round(tip, 3).tolist(), np.round(x[CHECK][:, :2], 3).tolist()])
                    if ep - a0 < args.video and closed_on[side]:
                        print("CLOSE", side, f, "n", len(idx), "tip", np.round(tip, 3).tolist(), "target", np.round(E[side][f], 3).tolist(),
                              "tcp", np.round(tcp_world(side), 3).tolist(), "nearest", round(float(dist.min()), 3), "waited", waited, flush=True)
                    closed_on[side] = False
                elif g[side][f] < 0.5 and side in pins:
                    idx_rel = pins.pop(side)[0]
                    v = cloth.get_velocities()
                    v[0, idx_rel] = 0.0       # let go without flinging the held patch
                    cloth.set_velocities(v)
                    cp = cloth.get_world_positions()[0].cpu().numpy()[CHECK]
                    glog.append([side, f, "release", np.round(tcp_world(side), 3).tolist(), np.round(cp[:, :2], 3).tolist()])
            if n % 10 == 0:
                lift = max(lift, float(cloth.get_world_positions()[0, :, 2].max()))
            # the demo clock waits, before a gripper closes, for that arm to arrive and then for the gripper to close
            closing = [side for side, gg in g.items() if f >= 1 and gg[f] > gg[f - 1]]
            for side in closing:   # a fresh close at the table may grasp; LeHome also re-closes its grippers in the air
                closed_on[side] = bool(E[side][f][2] < TCP_GRASP_Z + 0.015)
            err = max([np.linalg.norm(tcp_world(side) - E[side][f]) for side in closing if closed_on[side]] or [0.0])
            arrived = err < ARRIVE or waited >= MAX_WAIT
            for side in [s_ for s_ in closing if closed_on[s_]]:   # the arm sags under gravity: shift the IK target by the measured error until it arrives
                if not arrived:
                    off[side] += 0.6 * (E[side][f] - tcp_world(side))
                    q_plan = qL if side == "left" else qR
                    ik.d.qpos[ik.qadr[side]] = q_plan[f]
                    q_plan[f] = ik.solve(side, E[side][f] + off[side], iters=30, w_down=float(E[side + "_wdown"][f]), w_home=0.0)[0]
            grasping = any(closed_on[s_] for s_ in closing)
            releasing = [side for side, gg in g.items() if f >= 1 and gg[f] < gg[f - 1] and held_since[side]]
            if releasing and rdwell < RELEASE_DWELL:   # hold still while the gripper opens: moving off drags the cloth back
                rdwell += 1
                continue
            for side in releasing:
                held_since[side] = False
            rdwell = 0
            if arrived and not (grasping and dwell < GRASP_DWELL):
                f, waited, dwell = f + 1, 0, 0
                off = {"left": np.zeros(3), "right": np.zeros(3)}
            else:
                waited += 1
                dwell += int(grasping and arrived)
        pins = {}
        for _ in range(30):  # let the cloth settle before judging
            env.step(act)
        for _ in range(50):   # the vendored checker only evaluates every 50th call (False otherwise)
            res = success_checker_garment_fold(env.object, env.garment_loader.get_garment_type(GARMENT))
            if isinstance(res, dict):
                break
        success = bool(res.get("success", False)) if isinstance(res, dict) else bool(res)
        checks = [[d.get("description", k)[:40], bool(d.get("passed")), round(float(d.get("value", d.get("distance", -1)) or -1), 3)]
                  for k, d in res.get("details", {}).items()] if isinstance(res, dict) else None
        if ds is not None:
            if success:
                ds.save_episode()
            else:
                ds.clear_episode_buffer()
        if frames:
            imageio.mimsave(os.path.join(args.out, f"ep{ep:03d}_{'ok' if success else 'fail'}.mp4"), frames, fps=30)
        rec = dict(ep=ep, garment=E["garment"], success=success, frames=n, demo_frames=len(qL), grasps=grasps,
                   lift=round(lift, 3), ik_err_mm=round(ik_err * 1000, 1), plan_s=round(t_plan, 1),
                   total_s=round(time.time() - t0, 1), shifts=shifts, grasp_log=glog,
                   checks=checks)
    except Exception as exc:  # keep going: one bad episode must not stop the night's run
        traceback.print_exc()
        if ds is not None:
            ds.clear_episode_buffer()
        rec = dict(ep=ep, error=repr(exc)[:300], total_s=round(time.time() - t0, 1))
    print("EPISODE", json.dumps(rec), flush=True)
    LOG.write(json.dumps(rec) + "\n")
    LOG.flush()

if ds is not None and hasattr(ds, "finalize"):
    ds.finalize()
LOG.close()
env.close()
app.close()
