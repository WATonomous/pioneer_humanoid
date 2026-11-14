"""The Pioneer gripper squeezing and lifting a kitchen sponge in Newton VBD.

The gripper (link6 + both fingers, kinematics from the arm's URDF) floats: lower over the sponge, close to a 3.5 cm jaw
gap, lift 10 cm, hold. The fingers are kinematic bodies; the sponge is a voxel tet mesh of open-cell foam (6 x 9 x 3.5
cm, E 30 kPa, 25 kg/m^3). Writes the video, a still and a RESULT line (squeeze, lift, how much the sponge moves while
held still).

  python sponge_grip.py out.mp4
"""
import argparse
import re
import tempfile
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
import trimesh
import warp as wp

import newton
import newton.viewer
from newton.solvers import SolverVBD

REPO = Path(__file__).resolve().parents[3]
URDF = REPO / "assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm.urdf"

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--gap", type=float, default=0.035, help="commanded jaw gap at the fingertips (m); the sponge is 6 cm wide")
p.add_argument("--tip_h", type=float, default=0.01, help="fingertip height above the table at the grasp (m)")
# squeezing 6 cm of coarse foam tets to 2.5 cm inverts them and the sponge buzzes forever; these four together keep it still
p.add_argument("--fric_eps", type=float, default=1e-3, help="m/s; VBD's default 1e-2 lets anything slower than 1 cm/s slide")
p.add_argument("--obj_damp", type=float, default=0.1)
p.add_argument("--contact_ke", type=float, default=2e5)
p.add_argument("--contact_kd", type=float, default=1.0)
p.add_argument("--mu", type=float, default=1.0)
p.add_argument("--pitch", type=float, default=0.004, help="sponge voxel size (m)")
p.add_argument("--substeps", type=int, default=20)
p.add_argument("--iterations", type=int, default=40)
p.add_argument("--seconds", type=float, default=6.0)
p.add_argument("--size", type=int, nargs=2, default=[1280, 720])
p.add_argument("--camera", type=float, nargs=5, default=[0.16, -0.20, 0.16, -18.0, 128.0], help="x y z (m) pitch yaw")
args = p.parse_args()


def load_arm():
    src = URDF.read_text()
    src = re.sub(r"<gazebo.*?</gazebo>|<transmission.*?</transmission>", "", src, flags=re.S)
    src = src.replace("package://armv2URDF/meshes/", f"{URDF.parent.parent}/meshes/")
    src = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler balanceinertia="true" discardvisual="false"/></mujoco>', src, count=1)
    with tempfile.NamedTemporaryFile("w", suffix=".urdf") as f:
        f.write(src)
        f.flush()
        return mujoco.MjModel.from_xml_path(f.name)


m = load_arm()
d = mujoco.MjData(m)
L6, L7, L8 = (m.body(n).id for n in ("link6", "link7", "link8"))
J7, J8 = (m.joint(j).qposadr[0] for j in ("joint7", "joint8"))
TCP = np.array([-0.0069, 0.0, -0.2299])  # jaw centre, in link6
TIP_BELOW_TCP = 0.0253
mujoco.mj_kinematics(m, d)
R_DOWN = d.xmat[L6].reshape(3, 3).copy()  # zero pose: gripper vertical, fingers close along world x
TCP0 = d.xpos[L6] + R_DOWN @ TCP
SPONGE_R = np.array([0.03, 0.045, 0.0175])  # half sizes: 6 cm between the jaws, 9 cm long, 3.5 cm thick
SPONGE_POS = np.array([0.0, 0.0, SPONGE_R[2] + 0.001])


def closed_for_gap(gap):  # measured on this gripper: tip gap 8.02 cm open, -0.5 cm at closed=1
    return float(np.clip((0.0802 - gap) / 0.0852, 0.0, 1.0))


def gripper_poses(tcp, closed):
    """World (pos, R) of link6, link7, link8 for the tool point at `tcp`, gripper vertical, `closed` 0 open .. 1 shut."""
    d.qpos[:] = 0.0
    d.qpos[J7], d.qpos[J8] = -0.041 * closed, 0.041 * closed
    mujoco.mj_kinematics(m, d)
    return [(d.xpos[b] - TCP0 + tcp, d.xmat[b].reshape(3, 3).copy()) for b in (L6, L7, L8)]


def quat_xyzw(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return np.array([q[1], q[2], q[3], q[0]])


def visual_meshes(b):
    """The body's visual mesh geoms, in the body frame."""
    out = []
    for g in np.nonzero(m.geom_bodyid == b)[0]:
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g]:
            continue
        md = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[md]:m.mesh_vertadr[md] + m.mesh_vertnum[md]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[md]:m.mesh_faceadr[md] + m.mesh_facenum[md]]
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, md), v @ Rg.reshape(3, 3).T + m.geom_pos[g], f))
    return out


def voxel_tets(shape, origin, pitch):
    """A filled box of voxels, 5 tets per voxel (alternating split so neighbouring cubes share faces)."""
    even = [(0, 1, 2, 4), (1, 3, 2, 7), (1, 2, 4, 7), (2, 4, 7, 6), (1, 4, 5, 7)]
    odd = [(1, 0, 3, 5), (0, 2, 3, 6), (0, 3, 5, 6), (3, 5, 6, 7), (0, 5, 4, 6)]
    corner_of, verts, tets = {}, [], []
    for i, j, k in np.ndindex(*shape):
        c = []
        for n in range(8):
            key = (i + (n & 1), j + (n >> 1 & 1), k + (n >> 2 & 1))
            if key not in corner_of:
                corner_of[key] = len(verts)
                verts.append(key)
            c.append(corner_of[key])
        tets += [[c[t] for t in tet] for tet in (even if (i + j + k) % 2 == 0 else odd)]
    v, t = origin + np.array(verts, float) * pitch, np.array(tets)
    a, b, c, d_ = (v[t[:, n]] for n in range(4))
    neg = np.einsum("ij,ij->i", b - a, np.cross(c - a, d_ - a)) < 0
    t[neg] = t[neg][:, [0, 2, 1, 3]]
    return v, t


# (time, tool point, closed): settle, lower, close, lift, hold
open_c, shut_c = 0.0, closed_for_gap(args.gap)
grasp = np.array([0.0, 0.0, TIP_BELOW_TCP + args.tip_h])
up = grasp + [0, 0, 0.08]
K = [(0.0, up, open_c), (0.5, up, open_c), (1.5, grasp, open_c), (2.5, grasp, shut_c), (2.8, grasp, shut_c),
     (4.3, grasp + [0, 0, 0.10], shut_c), (6.0, grasp + [0, 0, 0.10], shut_c)]


def target(t):
    for (t0, p0, c0), (t1, p1, c1) in zip(K, K[1:]):
        if t <= t1:
            s = 0.5 - 0.5 * np.cos(np.pi * np.clip((t - t0) / (t1 - t0), 0, 1))
            return p0 + s * (p1 - p0), c0 + s * (c1 - c0)
    return K[-1][1], K[-1][2]


builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.02), wp.quat_identity()), hx=0.25, hy=0.25, hz=0.02,
                      color=(0.62, 0.55, 0.45))
sv, st = voxel_tets(np.ceil(2 * SPONGE_R / args.pitch).astype(int), -SPONGE_R, args.pitch)
E, nu = 3e4, 0.1  # open-cell foam: soft, nearly no lateral bulge
s0 = builder.particle_count
builder.add_soft_mesh(pos=wp.vec3(*SPONGE_POS), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0), vertices=sv.tolist(),
                      indices=st.flatten().tolist(), density=25.0, k_mu=E / (2 * (1 + nu)),
                      k_lambda=E * nu / ((1 + nu) * (1 - 2 * nu)), k_damp=args.obj_damp, particle_radius=0.001,
                      color=(0.95, 0.8, 0.2))
s1 = builder.particle_count

for b, (pb, Rb) in zip((L6, L7, L8), gripper_poses(*target(0.0))):
    nb = builder.add_body(xform=wp.transform(wp.vec3(*pb), wp.quat(*quat_xyzw(Rb))), is_kinematic=True, label=m.body(b).name)
    meshes = visual_meshes(b)
    for name, v, f in meshes:
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision="print" in name)
        builder.add_shape_mesh(nb, mesh=newton.Mesh(v.astype(np.float32), f.flatten().astype(np.int32), compute_inertia=False), cfg=cfg,
                               color=(0.95, 0.45, 0.1) if "finger" in name else (0.88, 0.88, 0.9))
    if b != L6:  # contact through the finger's convex hull: the foam can't sink between the fin-ray ribs
        hull = trimesh.Trimesh(next(v for name, v, f in meshes if name.endswith("finger"))).convex_hull  # outward winding
        builder.add_shape_mesh(nb, mesh=newton.Mesh(hull.vertices.astype(np.float32), hull.faces.flatten().astype(np.int32),
                                                    compute_inertia=False),
                               cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, is_visible=False))

builder.color()
model = builder.finalize()
model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = args.contact_ke, args.contact_kd, args.mu
for arr, val in ((model.shape_material_mu, args.mu), (model.shape_material_ke, args.contact_ke)):
    a = arr.numpy()
    a[:] = val
    arr.assign(a)

state0, state1, control = model.state(), model.state(), model.control()
pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.002)
contacts = pipeline.contacts()
solver = SolverVBD(model, iterations=args.iterations, integrate_with_external_rigid_solver=True, friction_epsilon=args.fric_eps,
                   particle_enable_self_contact=True, particle_self_contact_margin=0.002, particle_self_contact_gap=0.001,
                   particle_topological_contact_filter_threshold=2, particle_rest_shape_contact_exclusion_radius=0.004,
                   particle_vertex_contact_buffer_size=32, particle_edge_contact_buffer_size=64)
DT = 1.0 / 60.0 / args.substeps


def pose_array(poses):
    return np.array([np.concatenate([pb, quat_xyzw(Rb)]) for pb, Rb in poses], dtype=np.float32)


pose_prev = pose_array(gripper_poses(*target(0.0)))
viewer = newton.viewer.ViewerGL(width=args.size[0], height=args.size[1], headless=True)
viewer.set_model(model)
viewer.show_particles = False

x_rest = state0.particle_q.numpy()[s0:s1].copy()
w0 = np.ptp(x_rest[:, 0])
frames, hold_speed, min_width, t, wall = [], [], w0, 0.0, time.time()
while t < args.seconds - 1e-9:
    for _ in range(args.substeps):
        t += DT
        pose_new = pose_array(gripper_poses(*target(t)))
        vel = np.zeros((len(pose_new), 6), np.float32)
        vel[:, :3] = (pose_new[:, :3] - pose_prev[:, :3]) / DT
        state0.body_q.assign(wp.array(pose_prev, dtype=wp.transform))  # VBD reads the motion from state_in to state_out
        state0.body_qd.assign(wp.array(vel, dtype=wp.spatial_vector))
        state1.body_q.assign(wp.array(pose_new, dtype=wp.transform))
        state1.body_qd.assign(state0.body_qd)
        pipeline.collide(state0, contacts)
        state0.clear_forces()
        solver.step(state0, state1, control, contacts, DT)
        state0, state1 = state1, state0
        pose_prev = pose_new
    xs = state0.particle_q.numpy()[s0:s1]
    if t > K[-2][0] + 0.7:
        hold_speed.append(np.linalg.norm(state0.particle_qd.numpy()[s0:s1], axis=1))
    mid = np.abs(xs[:, 2] - np.median(xs[:, 2])) < 0.004  # the sponge's waist, where the jaws are
    if mid.sum() > 4:
        min_width = min(min_width, np.ptp(xs[mid, 0]))
    viewer.set_camera(wp.vec3(*args.camera[:3]), pitch=args.camera[3], yaw=args.camera[4])
    viewer.begin_frame(t)
    viewer.log_state(state0)
    viewer.end_frame()
    frames.append(viewer.get_frame().numpy())

lifted = float(xs[:, 2].min())
print(f"RESULT gap_cmd={100 * args.gap:.1f}cm sponge_width={100 * w0:.2f}cm squeeze={100 * (1 - min_width / w0):.0f}% "
      f"lifted={100 * lifted:.1f}cm held={lifted > 0.05} hold_speed_rms={100 * np.sqrt(np.mean(np.square(hold_speed))):.2f}cm/s "
      f"({time.time() - wall:.0f} s wall)", flush=True)
imageio.mimsave(args.out, frames, fps=60, macro_block_size=8)
imageio.imwrite(str(Path(args.out).with_suffix(".png")), frames[int(2.7 * 60)])
