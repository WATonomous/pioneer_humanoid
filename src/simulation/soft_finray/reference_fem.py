"""Validation reference for finray.py (not a second finger): the same squeeze test with the finger as a 2D finite-element
model (MuJoCo flex, plane-stress TPU on a 0.75 mm pixel mesh of the profile). Accurate but ~0.014x real time; finray.BEAM_E
was fitted to its cylinder result.

  python reference_fem.py cyl|box     # ~4 min each
"""
import argparse
import time

import mujoco
import numpy as np

import finray

p = argparse.ArgumentParser()
p.add_argument("obj", choices=["cyl", "box", "card"])
p.add_argument("--travel", type=float, default=0.012)
p.add_argument("--z", type=float, default=-0.02)
p.add_argument("--seconds", type=float, default=3.0)
args = p.parse_args()
H, DT = 0.00075, 1e-3


def mesh():
    """Every inside pixel of the profile -> two triangles; shared pixel corners are the nodes."""
    I, ys, zs = finray.profile(H)
    vid = -np.ones((I.shape[0] + 1, I.shape[1] + 1), int)
    verts, tris = [], []
    for i, j in np.argwhere(I):
        c = []
        for di, dj in ((0, 0), (0, 1), (1, 1), (1, 0)):
            if vid[i + di, j + dj] < 0:
                vid[i + di, j + dj] = len(verts)
                verts.append((ys[j] - H / 2 + dj * H, zs[i] - H / 2 + di * H))
            c.append(vid[i + di, j + dj])
        tris += [(c[0], c[1], c[2]), (c[0], c[2], c[3])]
    return np.array(verts), np.array(tris)


def finger_xml(name, side):
    V, T = mesh()
    V = V * [side, 1]
    if side < 0:
        T = T[:, ::-1]
    e1, e2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
    area = np.zeros(len(V))
    for k in range(3):
        np.add.at(area, T[:, k], 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 3)
    mass = np.maximum(area * finray.WIDTH * finray.TPU_RHO, 1e-6) * 100  # mass scaling: quasi-static, better conditioned
    pinned = V[:, 1] > finray.MOUNT_Z
    rows = np.round(V[:, 1] / H).astype(int)  # contact skin: the outermost node of each row on the face side, and the tip
    skin = {np.nonzero(rows == r)[0][np.argmax(V[rows == r, 0] * side)] for r in np.unique(rows)}
    skin |= set(np.nonzero(V[:, 1] < V[:, 1].min() + 2.5 * H)[0])
    bodies, bnames, vloc = [], [], []
    for k, (y, z) in enumerate(V):
        if pinned[k]:
            bnames.append(f"{name}_carriage")
            vloc.append(f"0 {y:.6f} {z:.6f}")
            continue
        bodies.append(f'<body name="{name}_n{k}" pos="0 {y:.6f} {z:.6f}" gravcomp="1"><joint type="slide" axis="0 1 0"/>'
                      f'<joint type="slide" axis="0 0 1"/><inertial pos="0 0 0" mass="{mass[k]:.3e}" diaginertia="1e-9 1e-9 1e-9"/>'
                      + (f'<geom class="skin" fromto="{-finray.WIDTH / 2} 0 0 {finray.WIDTH / 2} 0 0"/>' if k in skin else "") + '</body>')
        bnames.append(f"{name}_n{k}")
        vloc.append("0 0 0")
    flex = (f'<flex name="{name}" dim="2" radius="0.0006" body="{" ".join(bnames)}" vertex="{" ".join(vloc)}" '
            f'element="{" ".join(" ".join(map(str, t)) for t in T)}"><contact contype="0" conaffinity="0" selfcollide="none"/>'
            f'<elasticity young="{finray.TPU_E}" poisson="0.45" thickness="{finray.WIDTH}" elastic2d="stretch" damping="0.0005"/></flex>')
    return "".join(bodies), flex, V


obj = {"cyl": '<geom type="capsule" size="0.015 0.03" euler="0 90 0"/>', "box": '<geom type="box" size="0.03 0.012 0.012"/>',
       "card": '<geom type="box" size="0.03 0.0004 0.027"/>'}[args.obj]
half = {"cyl": 0.015, "box": 0.012, "card": 0.0004}[args.obj]
ba, fa, V = finger_xml("fa", +1)
bb, fb, _ = finger_xml("fb", -1)
y0 = half + 0.020 + 0.002
xml = f"""<mujoco><default><default class="skin"><geom type="capsule" size="0.0006" condim="4" friction="1.2 0.02 0.001" solref="-20000 -5" contype="1" conaffinity="0"/></default></default>
<option timestep="{DT}" gravity="0 0 0" integrator="discrete" solver="CG" cone="elliptic"/>
<worldbody>
  <body name="obj" pos="0 0 {args.z}">{obj}</body>
  <body name="fa_carriage" pos="0 {-y0} 0"><joint name="ja" type="slide" axis="0 1 0" damping="5"/>
    <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>{ba}</body>
  <body name="fb_carriage" pos="0 {y0} 0"><joint name="jb" type="slide" axis="0 -1 0" damping="5"/>
    <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>{bb}</body>
</worldbody>
<deformable>{fa}{fb}</deformable>
<actuator><position joint="ja" kp="3000"/><position joint="jb" kp="3000"/></actuator>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
ja = m.joint("ja").qposadr[0]
v0 = m.flex_vertadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "fa")]
face = np.nonzero(V[:, 0] > 0.0185)[0]
fz, fy0 = V[face, 1], V[face, 0] - y0
force, push, wall = [], [], 0.0
for k in range(int(args.seconds / DT)):
    d.ctrl[:] = (0.002 + args.travel) * min(1.0, d.time / (args.seconds * 0.8))
    t0 = time.time()
    mujoco.mj_step(m, d)
    wall += time.time() - t0
    if d.time > 0.8 * args.seconds and k % 20 == 0:
        force.append(-d.actuator_force[0])
        push.append(d.flexvert_xpos[v0 + face, 1] - fy0 - d.qpos[ja])
o, back = np.argsort(fz), np.mean(push, 0)
print(f"hold: grip {abs(np.mean(force)):.2f} N (std {np.std(force):.2f}), carriage travel {(d.qpos[ja] - 0.002) * 1000:.1f} mm")
print("face pushed back (mm) at z -65/-50/-35/-20/-5/10/25/40 mm:",
      " ".join(f"{np.interp(z, fz[o], back[o]) * -1000:.1f}" for z in (-0.065, -0.05, -0.035, -0.02, -0.005, 0.01, 0.025, 0.04)))
print(f"{args.seconds} s simulated in {wall:.0f} s ({args.seconds / wall:.3f}x real time)")
