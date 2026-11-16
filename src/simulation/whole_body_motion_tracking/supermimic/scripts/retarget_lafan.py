"""Headless LAFAN1 BVH -> Pioneer retargeting with GMR. Saves a LAFAN-style CSV (root xyz, root quat xyzw, 29 joints, 30 fps
as recorded) and optionally an MP4 of the robot (offscreen, MUJOCO_GL=egl), plus joint-limit / speed stats.

  python retarget_lafan.py <bvh> <out.csv> [--video out.mp4]
"""
import argparse

import imageio
import mujoco
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

p = argparse.ArgumentParser()
p.add_argument("bvh")
p.add_argument("out")
p.add_argument("--video")
p.add_argument("--seconds", type=float)
p.add_argument("--robot", default="pioneer", help="pioneer_nolimits: every joint +-180 deg, for visuals only")
args = p.parse_args()

frames, height = load_bvh_file(args.bvh, format="lafan1")
fps = 30
if args.seconds:
    frames = frames[: int(args.seconds * fps)]
gmr = GMR(src_human="bvh_lafan1", tgt_robot=args.robot, actual_human_height=height)
gmr.update_targets(frames[0])
root = gmr.human_body_to_task2[gmr.human_root_name].transform_target_to_world
q0 = gmr.configuration.q.copy()
q0[3:7], q0[:3] = root.wxyz_xyz[:4], root.wxyz_xyz[4:]
gmr.configuration.update(q0)
qs = np.array([gmr.retarget(f).copy() for f in frames])

# GMR puts the foot link origin (4.4 cm above the sole) on the human ankle, so the soles sink a few cm: lift each frame
# until its lowest foot collision point is on the floor (lift only, so running and jumping keep their flight phases)
m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
d = mujoco.MjData(m)
feet = [g for g in range(m.ngeom) if m.geom_group[g] == 3 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) in ("l_foot", "r_foot")]
corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
lows = []
for q in qs:
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    lows.append(min((corners * m.geom_size[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min() for g in feet))
lift = np.maximum(0.0, -np.array(lows))
qs[:, 2] += np.convolve(np.pad(lift, 2, mode="edge"), np.ones(5) / 5, mode="valid")
print(f"feet lifted out of the floor: median {100 * np.median(lift):.1f} cm, max {100 * lift.max():.1f} cm")

out = np.concatenate([qs[:, :3], qs[:, [4, 5, 6, 3]], qs[:, 7:]], axis=1)
np.savetxt(args.out, out, delimiter=",", fmt="%.6f")

names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
dq = np.abs(np.diff(qs[:, 7:], axis=0)) * fps
print(f"{len(qs)} frames, {len(qs) / fps:.1f} s; root height {qs[:, 2].min():.2f}-{qs[:, 2].max():.2f} m")
for i, n in enumerate(names):
    at_lim = np.mean((qs[:, 7 + i] <= lo[i] + 1e-3) | (qs[:, 7 + i] >= hi[i] - 1e-3))
    print(f"  {n:18s} {qs[:, 7 + i].min():+.2f}..{qs[:, 7 + i].max():+.2f} (limit {lo[i]:+.2f}..{hi[i]:+.2f})"
          f"  at limit {100 * at_lim:3.0f}%  max speed {np.percentile(dq[:, i], 99):.1f} rad/s (p99)")

if args.video:
    r = mujoco.Renderer(m, 480, 640)
    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 135
    with imageio.get_writer(args.video, fps=fps) as w:
        for q in qs:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [q[0], q[1], 0.8]
            r.update_scene(d, cam)
            w.append_data(r.render())
    print("video", args.video)
