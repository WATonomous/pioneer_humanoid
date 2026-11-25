"""Tower of Hanoi for an LLM agent: the 8dof bimanual arm in the ``hanoi`` scene, driven one command at a time.

The sim is paused between commands (nothing moves while the agent thinks). Each command runs to completion and
returns one JSON line on tcp://127.0.0.1:PORT; ``agent_cli.py`` is the client (see README.md):
  state                       fixed layout, both fingertip points and grippers; ring poses only with --privileged
  move  arm x y z             fingertip point to (x, y, z), gripper pointing down, straight line
  grip  arm open|close        reports whether something is held
  look  cams name             PNGs from the film / top / wrist cameras, with each camera's calibration
  caption text                a line shown in the video
  done                        close the video and the log, exit

Everything goes to outputs/agent_control/RUN/: video.mp4, log.jsonl (every command, its reply and the true ring
poses, which the agent never sees in camera-only mode) and the look images.
"""
import argparse
import json
import os
import socket
import time
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--run", default="run", help="output folder name under outputs/agent_control")
p.add_argument("--privileged", action="store_true", help="state also gives every ring's pose")
p.add_argument("--label", default="", help="title bar of the video, e.g. the model's name")
p.add_argument("--speedup", type=int, default=1, help="video plays this many times faster than sim time")
p.add_argument("--port", type=int, default=7777)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from humanoid_isaac_scenes import make_scene_cfg  # noqa: E402
from humanoid_isaac_scenes.hanoi.scene import (BOARD_TOP, DISK_H, DISKS, PEG_H, PEGS, TOP, TRAY)  # noqa: E402
from humanoid_rl_tasks.workcell import ROBOT_BASE_POS  # noqa: E402
from pioneer_humanoid.arm_8dof_params import (CAMERAS, DEFAULT_JOINT_POS, GRIPPER_STROKE, LEFT_ARM_JOINTS,  # noqa: E402
                                              RIGHT_ARM_JOINTS, TCP_LOCAL)
from pioneer_humanoid.bimanual_arm_8dof import BIMANUAL_ARM_8DOF_CFG, vertical_aperture  # noqa: E402
from pioneer_humanoid.mujoco_bimanual_arm_8dof import arm_spec  # noqa: E402

ROOT = Path(os.environ.get("HUMANOID_ROOT", Path(__file__).resolve().parents[3]))
OUT = ROOT / "outputs" / "agent_control" / args.run
TIP_DROP = 0.022                            # fingertip centre below the TCP (m)
ARMS = {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS}
FILM = ([0.97, -0.22, 1.22], [0.36, 0.0, 0.90])   # video camera: beside the robot, looking into the open face


def look_at(eye, at):   # wxyz of an OpenGL camera (looks along its -Z, +Y up) at eye looking at `at`, world +Z up
    f = np.asarray(at, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, -f], axis=1)
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (float(w), float(x), float(y), float(z))


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def robot_cam(name, h=480, w=640):   # TiledCamera: a plain Camera fails the cameras_enabled check in the Isaac image
    c = CAMERAS[name]
    return TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/" + f"{c.body}/{c.prim}", height=h, width=w, data_types=["rgb"],
                          offset=TiledCameraCfg.OffsetCfg(pos=c.pos, rot=c.rot, convention="opengl"),
                          spawn=sim_utils.PinholeCameraCfg(focal_length=c.focal, horizontal_aperture=c.h_aperture,
                                                           vertical_aperture=vertical_aperture(name, h, w), clipping_range=c.clip))


cfg = make_scene_cfg("hanoi", BIMANUAL_ARM_8DOF_CFG)
cfg.film = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Film", height=720, width=1280, data_types=["rgb"],
                          offset=TiledCameraCfg.OffsetCfg(pos=tuple(FILM[0]), rot=look_at(*FILM), convention="opengl"),
                          spawn=sim_utils.PinholeCameraCfg(focal_length=14, clipping_range=(0.05, 20.0)))
ROBOT_CAMS = {"top": "ego", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}
for _k, _n in ROBOT_CAMS.items():
    setattr(cfg, _k, robot_cam(_n))
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, render_interval=1))
scene = InteractiveScene(cfg)
sim.reset()
robot = scene["robot"]
dev = robot.device
jid = {a: robot.find_joints(js, preserve_order=True)[0] for a, js in ARMS.items()}
gid = {a: robot.find_joints([f"{a}_gripper"])[0] for a in ARMS}
eid = {a: robot.find_bodies([f"{a}_link7"])[0][0] for a in ARMS}
disks = {n: scene[f"disk_{n}"] for n in DISKS}
cam_body = {k: robot.find_bodies([CAMERAS[n].body])[0][0] for k, n in ROBOT_CAMS.items()}

# ── IK: MuJoCo kinematics of the same arm, base at the workcell pose; fingertip point, gripper down, yaw free ──
spec = arm_spec()
spec.body("base_link").pos = ROBOT_BASE_POS
for a in ARMS:
    s = spec.body(f"{a}_link7").add_site()
    s.name, s.pos = f"{a}_tip", (TCP_LOCAL[0], TCP_LOCAL[1], TCP_LOCAL[2] - TIP_DROP)
M = spec.compile()
D = mujoco.MjData(M)
QA = {a: np.array([M.jnt_qposadr[M.joint(j).id] for j in js]) for a, js in ARMS.items()}
DA = {a: np.array([M.jnt_dofadr[M.joint(j).id] for j in js]) for a, js in ARMS.items()}
LIM = {a: M.jnt_range[[M.joint(j).id for j in js]] for a, js in ARMS.items()}
for j, v in DEFAULT_JOINT_POS.items():
    D.qpos[M.jnt_qposadr[M.joint(j).id]] = v
cmd = {a: D.qpos[QA[a]].copy() for a in ARMS}          # commanded arm joints
grip_cmd = {a: 0.0 for a in ARMS}


def rot(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])   # link7 +z up


def ik(a, p, yaw, q0, iters=60):
    D.qpos[QA[a]] = q0
    R = rot(yaw)
    site, body = M.site(f"{a}_tip").id, M.body(f"{a}_link7").id
    jp, jr = np.zeros((3, M.nv)), np.zeros((3, M.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(M, D)
        mujoco.mj_comPos(M, D)
        Rc = D.xmat[body].reshape(3, 3)
        ep, er = p - D.site_xpos[site], 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        mujoco.mj_jacSite(M, D, jp, None, site)
        mujoco.mj_jacBody(M, D, None, jr, body)
        J, e = np.vstack([jp[:, DA[a]], 0.3 * jr[:, DA[a]]]), np.concatenate([ep, 0.3 * er])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(7), J.T @ e)
        D.qpos[QA[a]] = np.clip(D.qpos[QA[a]] + np.clip(dq, -.1, .1), *LIM[a].T)
    mujoco.mj_kinematics(M, D)
    Rc = D.xmat[body].reshape(3, 3)
    pe = float(np.linalg.norm(p - D.site_xpos[site]))
    re = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Rc) - 1) / 2, -1, 1))))
    return D.qpos[QA[a]].copy(), pe, re


def fk_tip(a, q):
    D.qpos[QA[a]] = q
    mujoco.mj_kinematics(M, D)
    Rb = D.xmat[M.body(f"{a}_link7").id].reshape(3, 3)
    return D.site_xpos[M.site(f"{a}_tip").id].copy(), np.arctan2(Rb[1, 0], Rb[0, 0])


# ── stepping, video ──
caption, t_sim, n_steps = "", 0.0, 0
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "log.jsonl", "a")
video = imageio.get_writer(str(OUT / "video.mp4"), fps=30, codec="libx264", quality=8)
FONT, FONT_B = ImageFont.load_default(size=27), ImageFont.load_default(size=32)


def overlay(img):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    w, h = im.size
    dr.rectangle([0, 0, w, 50], fill=(25, 25, 30))
    dr.text((20, 8), args.label, font=FONT_B, fill=(255, 255, 255))
    dr.text((w - 330, 11), f"sim time {t_sim:5.1f} s" + (f"   {args.speedup}x" if args.speedup > 1 else ""), font=FONT, fill=(220, 220, 220))
    dr.rectangle([0, h - 58, w, h], fill=(25, 25, 30))
    dr.text((20, h - 46), caption, font=FONT, fill=(255, 255, 255))
    return np.asarray(im)


def apply_targets():
    tgt = robot.data.joint_pos_target.clone()
    for a in ARMS:
        tgt[0, jid[a]] = torch.tensor(cmd[a], device=dev, dtype=tgt.dtype)
        tgt[0, gid[a]] = grip_cmd[a]
    robot.set_joint_position_target(tgt)


def step(n=1, film=True):
    global t_sim, n_steps
    for _ in range(n):
        apply_targets()
        scene.write_data_to_sim()
        n_steps += 1
        render = film and n_steps % (3 * args.speedup) == 0   # 30 fps of sim time at dt 0.01
        sim.step(render=render)
        scene.update(sim.get_physics_dt())
        t_sim += sim.get_physics_dt()
        if render:
            video.append_data(overlay(scene["film"].data.output["rgb"][0].cpu().numpy()[..., :3]))


def tip_now(a):
    pos, quat = robot.data.body_pos_w[0, eid[a]].cpu().numpy(), robot.data.body_quat_w[0, eid[a]].cpu().numpy()
    return pos + quat_mat(quat) @ np.array([TCP_LOCAL[0], TCP_LOCAL[1], TCP_LOCAL[2] - TIP_DROP])


def gap(a):   # gripper closing travel per finger (m): 0 open, GRIPPER_STROKE fully shut
    return float(robot.data.joint_pos[0, gid[a][0]])


def holding(a):   # told to close but stopped short: something is between the fingers
    return grip_cmd[a] > 0 and gap(a) < GRIPPER_STROKE - 0.004


YAWS = np.radians(np.arange(-180, 180, 15))


def move(a, target, speed=0.15):
    target = np.asarray(target, float)
    if target[2] < TOP + 0.004:
        return {"ok": False, "error": "BELOW_TABLE", "min_z": round(TOP + 0.004, 4)}
    p0, y0 = fk_tip(a, cmd[a])
    best = None
    for yw in sorted(YAWS, key=lambda v: abs((v - y0 + np.pi) % (2 * np.pi) - np.pi)):   # nearest yaw first
        q, pe, re = ik(a, target, yw, cmd[a], iters=120)
        if pe < 0.004 and re < 4:
            best = yw
            break
    if best is None:
        q, pe, re = ik(a, target, y0, cmd[a], iters=200)
        return {"ok": False, "error": "IK_UNREACHABLE", "reach_error_mm": round(pe * 1000, 1),
                "hint": "this arm cannot point straight down there; left arm works at y > +0.03, right at y < -0.03, x 0.40-0.60"}
    dyaw = (best - y0 + np.pi) % (2 * np.pi) - np.pi
    D.qpos[QA[a]] = cmd[a]
    mujoco.mj_kinematics(M, D)
    if D.xmat[M.body(f"{a}_link7").id].reshape(3, 3)[2, 2] < 0.98:   # not pointing down yet (start pose): joint-space move
        qg, q0 = q.copy(), cmd[a].copy()
        n = max(int(np.abs(qg - q0).max() / 0.8 / 0.01), 10)
        for k in range(1, n + 1):
            s = k / n
            cmd[a] = q0 + s * s * (3 - 2 * s) * (qg - q0)
            step()
    else:
        n = max(int(np.linalg.norm(target - p0) / speed / 0.01), int(abs(dyaw) / 1.5 / 0.01), 10)
        q = cmd[a].copy()
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)
            q, pe, re = ik(a, p0 + s * (target - p0), y0 + s * dyaw, q, iters=40)
            if pe > 0.003 or re > 3:
                q, pe, re = ik(a, p0 + s * (target - p0), y0 + s * dyaw, q, iters=300)
            if pe > 0.01 or re > 6:
                step(40)
                return {"ok": False, "error": "PATH_UNREACHABLE", "stopped_at": np.round(tip_now(a), 4).tolist(),
                        "hint": "the straight line to the target leaves the reachable band; go via an intermediate point"}
            cmd[a] = q
            step()
    step(40)
    return {"ok": True, "arrival_error_mm": round(float(np.linalg.norm(tip_now(a) - target)) * 1000, 1),
            "tip": np.round(tip_now(a), 4).tolist(), "holding": holding(a)}


def grip(a, how):
    grip_cmd[a] = GRIPPER_STROKE if how == "close" else 0.0
    step(70)
    return {"ok": True, "grip_travel_mm": round(gap(a) * 1000, 1), "holding": holding(a)}


def truth():   # ring geometric centres (a ring may lie flipped), top heights and tilts
    out = {}
    for n in DISKS:
        up = quat_mat(disks[n].data.root_quat_w[0].cpu().numpy())[:, 2]
        c = disks[n].data.root_pos_w[0].cpu().numpy() + up * DISK_H / 2
        out[n] = {"center": [round(float(v), 4) for v in c], "top_z": round(float(c[2]) + DISK_H / 2, 4),
                  "tilt_deg": round(float(np.degrees(np.arccos(min(1.0, abs(up[2]))))), 1)}
    return out


def solved():   # every ring flat on peg C, stacked largest first
    t = truth()
    xy = np.array(PEGS["C"])
    return all(np.linalg.norm(np.array(t[n]["center"][:2]) - xy) < 0.006 and t[n]["tilt_deg"] < 5
               and abs(t[n]["center"][2] - (BOARD_TOP + DISK_H * (i + 0.5))) < 0.004 for i, n in enumerate(DISKS))


def state():
    out = {"board_top_z": round(BOARD_TOP, 4), "disk_height": DISK_H, "peg_height": round(PEG_H, 3),
           "disk_diameters": DISKS, "pegs": {k: [*xy, round(BOARD_TOP, 4)] for k, xy in PEGS.items()},
           "tray_T": [*TRAY, round(BOARD_TOP, 4)],
           "arms": {a: {"tip": np.round(tip_now(a), 4).tolist(), "grip_travel_mm": round(gap(a) * 1000, 1), "holding": holding(a)}
                    for a in ARMS}}
    if args.privileged:
        out["disks"] = truth()
    return out


def look(cams, name):
    step(1, film=False)
    sim.render()
    scene.update(0.0)
    paths, calib = [], {}
    for c in cams:
        d = scene[c].data
        img = d.output["rgb"][0].cpu().numpy()[..., :3]
        s = 0.5 if c == "film" else 1.0   # the film image is saved at half size
        if c in ROBOT_CAMS:   # the sensor's own pose is read from USD, which the moving links never update
            b, cm = cam_body[c], CAMERAS[ROBOT_CAMS[c]]
            Rb = quat_mat(robot.data.body_quat_w[0, b].cpu().numpy())
            pos = robot.data.body_pos_w[0, b].cpu().numpy() + Rb @ np.asarray(cm.pos)
            R = Rb @ quat_mat(cm.rot) @ np.diag([1.0, -1.0, -1.0])   # OpenGL offset -> ROS optical frame
        else:
            pos, R = d.pos_w[0].cpu().numpy(), quat_mat(d.quat_w_ros[0].cpu().numpy())
        # ROS optical frame: +x image right, +y image down, +z forward
        K = d.intrinsic_matrices[0].cpu().numpy() * s
        calib[c] = {"pos": [round(float(v), 4) for v in pos], "image_right": np.round(R[:, 0], 3).tolist(),
                    "image_down": np.round(R[:, 1], 3).tolist(), "forward": np.round(R[:, 2], 3).tolist(),
                    "fx": round(float(K[0, 0]), 1), "fy": round(float(K[1, 1]), 1), "cx": round(float(K[0, 2]), 1), "cy": round(float(K[1, 2]), 1)}
        f = OUT / f"{name}_{c}.png"
        Image.fromarray(img).resize((int(img.shape[1] * s), int(img.shape[0] * s))).save(f)
        paths.append(str(f.relative_to(ROOT)))
    return {"ok": True, "images": paths, "calibration": calib}


# ── startup: arms straight into the default pose (spawned hanging, they would swing up into the table's underside) ──
q0 = robot.data.default_joint_pos.clone()
for a in ARMS:
    q0[0, jid[a]] = torch.tensor(cmd[a], device=dev, dtype=q0.dtype)
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
step(100, film=False)
for a, y in (("left", 0.20), ("right", -0.20)):
    move(a, [0.50, y, TOP + 0.20])

srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port))
srv.listen(1)
print(f"LISTENING {args.port}, writing {OUT}", flush=True)
running = True
while running and app.is_running():
    conn, _ = srv.accept()
    f = conn.makefile("rw")
    for line in f:
        t0 = time.time()
        try:
            c = json.loads(line)
            op = c.get("op")
            if op == "state":
                r = state()
            elif op == "move":
                r = move(c["arm"], [c["x"], c["y"], c["z"]])
            elif op == "grip":
                r = grip(c["arm"], c["how"])
            elif op == "look":
                r = look(c.get("cams", ["film"]), c.get("name", "look"))
            elif op == "caption":
                caption = c["text"]
                r = {"ok": True}
            elif op == "done":
                r, running = {"ok": True, "sim_time_s": round(t_sim, 1)}, False
            else:
                r = {"ok": False, "error": f"unknown op {op}"}
        except Exception as e:   # report, keep serving
            r = {"ok": False, "error": repr(e)}
        LOG.write(json.dumps({"wall": time.time(), "cmd": line.strip(), "result": r, "truth": truth(), "solved": solved(),
                              "sim_t": round(t_sim, 2), "wall_s": round(time.time() - t0, 2)}) + "\n")
        LOG.flush()
        f.write(json.dumps(r) + "\n")
        f.flush()
        if not running:
            break
    conn.close()
video.close()
print(f"DONE solved={solved()} sim_time={t_sim:.1f}s", flush=True)
app.close()
