"""Squeeze test: two soft Fin Ray fingers (finray.py) close on a fixed object, then hold. Prints the grip force and how
far the contact face is pushed back along its length (both averaged over the hold), and optionally renders the finger plane.

  MUJOCO_GL=egl python squeeze.py cyl|box|card [--video out.mp4] [--travel 0.012]
"""
import argparse
import time

import imageio
import mujoco
import numpy as np

import finray

p = argparse.ArgumentParser()
p.add_argument("obj", choices=["cyl", "box", "card"])
p.add_argument("--video", default=None)
p.add_argument("--travel", type=float, default=0.012, help="each carriage past first touch (m)")
p.add_argument("--z", type=float, default=-0.02, help="object height in the finger frame")
p.add_argument("--dt", type=float, default=1e-2)
p.add_argument("--seconds", type=float, default=3.0)
args = p.parse_args()

GAP = 0.020  # contact face at y = +-0.020 of each finger frame
obj = {"cyl": '<geom type="capsule" size="0.015 0.03" euler="0 90 0" rgba="0.3 0.5 0.9 1"/>',
       "box": '<geom type="box" size="0.03 0.012 0.012" rgba="0.3 0.5 0.9 1"/>',
       "card": '<geom type="box" size="0.03 0.0004 0.027" rgba="0.3 0.5 0.9 1"/>'}[args.obj]
half = {"cyl": 0.015, "box": 0.012, "card": 0.0004}[args.obj]
ba, wa, info = finray.finger_xml("fa", +1, dt=args.dt)
bb, wb, _ = finray.finger_xml("fb", -1, dt=args.dt)
y0 = half + GAP + 0.002  # carriage start: faces 2 mm off the object
xml = f"""<mujoco><default><default class="tpu"><geom rgba="0.95 0.45 0.1 1" condim="4" friction="1.2 0.02 0.001" contype="1" conaffinity="0" density="0"/></default></default>
<option timestep="{args.dt}" gravity="0 0 0" integrator="implicitfast" solver="Newton" iterations="50" cone="elliptic"/>
<visual><global offwidth="1280" offheight="720" fovy="8"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
<worldbody>
  <light pos="0.3 -0.2 0.4" dir="-0.6 0.4 -0.7" diffuse="0.6 0.6 0.6"/>
  <body name="obj" pos="0 0 {args.z}">{obj}</body>
  <body name="fa_carriage" pos="0 {-y0} 0"><joint name="ja" type="slide" axis="0 1 0" damping="5"/>
    <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>{ba}</body>
  <body name="fb_carriage" pos="0 {y0} 0"><joint name="jb" type="slide" axis="0 -1 0" damping="5"/>
    <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>{bb}</body>
</worldbody>
<equality>{wa}{wb}</equality>
<actuator><position joint="ja" kp="3000" name="ma"/><position joint="jb" kp="3000" name="mb"/></actuator>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
print(f"nv {m.nv}; per finger: {info['links']} hinged links, {info['welds']} welds", flush=True)
ja = m.joint("ja").qposadr[0]
mujoco.mj_forward(m, d)
face = [m.body(b).id for b in info["face"]]
fz, fy0 = d.xpos[face, 2].copy(), d.xpos[face, 1].copy()
if args.video:
    ren = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, -0.005]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 0, 0
    frames = []
force, push, wall = [], [], 0.0
for k in range(int(args.seconds / args.dt)):
    d.ctrl[:] = (0.002 + args.travel) * min(1.0, d.time / (args.seconds * 0.8))  # close over 80% of the time, then hold
    t0 = time.time()
    mujoco.mj_step(m, d)
    wall += time.time() - t0
    if not np.isfinite(d.qpos).all():
        raise SystemExit(f"diverged at t {d.time:.3f}")
    if d.time > 0.8 * args.seconds:
        force.append(-d.actuator_force[0])
        push.append(d.xpos[face, 1] - fy0 - d.qpos[ja])  # face motion relative to its carriage (- = pushed back)
    if args.video and len(frames) < d.time * 30:
        ren.update_scene(d, cam)
        frames.append(ren.render())
o, back = np.argsort(fz), np.mean(push, 0)
print(f"hold: grip {abs(np.mean(force)):.2f} N (std {np.std(force):.2f}), carriage travel {(d.qpos[ja] - 0.002) * 1000:.1f} mm")
print("face pushed back (mm) at z -65/-50/-35/-20/-5/10/25/40 mm:",
      " ".join(f"{np.interp(z, fz[o], back[o]) * -1000:.1f}" for z in (-0.065, -0.05, -0.035, -0.02, -0.005, 0.01, 0.025, 0.04)))
print(f"{args.seconds} s simulated in {wall:.2f} s ({args.seconds / wall:.1f}x real time, physics only)")
if args.video:
    ren.close()
    imageio.mimsave(args.video, frames, fps=30, quality=8)
    print("wrote", args.video)
