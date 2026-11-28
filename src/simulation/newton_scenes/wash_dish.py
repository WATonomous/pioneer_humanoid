"""Keyboard teleop in Newton: the Pioneer right arm scrubs a dirty plate in a sink with a soft sponge.

The arm is kinematic (IK on the URDF in MuJoCo, solved every frame); the sponge is a VBD soft body squeezed by the
fingers; dirt spots on the plate wipe off where the sponge presses on them. Keys (Newton viewer window focused):

  I / K   tool forward / back        J / L   left / right        U / O   up / down
  N / M   wrist yaw                  G       open / close the gripper
  R       reset                      Shift   fine motion
  (W A S D Q E, arrows, mouse: viewer camera.  H: hide the UI.  Space: pause.)

  python wash_dish.py                          # teleop
  python wash_dish.py --demo out.mp4           # scripted grab-and-scrub, headless, writes a video
"""
import argparse
import re
import tempfile
import time
from pathlib import Path

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
p.add_argument("--demo", default=None, help="run the scripted demo headless and write this video")
p.add_argument("--substeps", type=int, default=10)
p.add_argument("--iterations", type=int, default=10)
p.add_argument("--pitch", type=float, default=0.008, help="sponge voxel size (m)")
p.add_argument("--gap", type=float, default=0.035, help="jaw gap when closed (m); the sponge is 6.4 cm wide")
args = p.parse_args()

FPS = 60
DT = 1.0 / FPS / args.substeps

# --- the arm: MuJoCo model of the URDF for kinematics only ---


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
BASE_POS, BASE_YAW = np.array([0.0, -0.45, 0.38]), np.pi / 2  # on the counter's back edge, arms reaching +y
R_BASE = np.array([[np.cos(BASE_YAW), -np.sin(BASE_YAW), 0], [np.sin(BASE_YAW), np.cos(BASE_YAW), 0], [0, 0, 1]])
ARMS = {"R": (["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"], ("joint7", "joint8"), "link6"),
        "L": (["joint1L", "joint2l", "joint3l", "joint4l", "joint5l", "joint6l"], ("joint7l", "joint8l"), "link6l")}
QADR = {a: np.array([m.joint(j).qposadr[0] for j in js]) for a, (js, _, _) in ARMS.items()}
DADR = {a: np.array([m.joint(j).dofadr[0] for j in js]) for a, (js, _, _) in ARMS.items()}
LIM = {a: m.jnt_range[[m.joint(j).id for j in js]] for a, (js, _, _) in ARMS.items()}
FADR = {a: [m.joint(j).qposadr[0] for j in fj] for a, (_, fj, _) in ARMS.items()}
L6 = {a: m.body(b).id for a, (_, _, b) in ARMS.items()}
TCP = {"R": np.array([-0.0069, -0.0070, -0.2299]), "L": np.array([-0.0069, 0.0070, -0.2299])}  # jaw centre in link6
mujoco.mj_kinematics(m, d)
R_DOWN = {a: d.xmat[L6[a]].reshape(3, 3).copy() for a in ARMS}  # zero pose: gripper hanging straight down
for a in ARMS:
    m.jnt_limited[[m.joint(j).id for j in ARMS[a][0]]] = 1


def rotz(t):
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def ik(a, target, yaw, iters=8, w=0.05):
    """Damped least squares on arm a's 6 joints: the jaw centre to `target` (world), the gripper's long axis pointing
    down (5 cm per radian, so the tool point wins), and a weak pull toward `yaw`: this 6-DOF arm can't point straight
    down at every yaw."""
    tgt = R_BASE.T @ (target - BASE_POS)
    yaw_x = (rotz(yaw) @ R_DOWN[a])[:, 0]
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        R6 = d.xmat[L6[a]].reshape(3, 3)
        p = d.xpos[L6[a]] + R6 @ TCP[a]
        za = R6[:, 2]
        rot = np.cross(za, R_DOWN[a][:, 2]) + 0.3 * np.dot(np.cross(R6[:, 0], yaw_x), za) * za
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, p, L6[a])
        J = np.vstack([jp[:, DADR[a]], w * jr[:, DADR[a]]])
        step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[tgt - p, w * rot])
        step *= min(1.0, 0.2 / max(np.abs(step).max(), 1e-9))
        d.qpos[QADR[a]] = np.clip(d.qpos[QADR[a]] + step, LIM[a][:, 0], LIM[a][:, 1])
    return np.linalg.norm(tgt - p)


def ik_from_scratch(a, target, yaw):
    """Best of several random starts (the zero pose is singular), keeping the one that points down best."""
    rng, best = np.random.default_rng(0), (np.inf, None)
    for _ in range(16):
        d.qpos[QADR[a]] = rng.uniform(LIM[a][:, 0], LIM[a][:, 1])
        for _ in range(10):
            ik(a, target, yaw, 10, w=0.0)
        for _ in range(20):
            err = ik(a, target, yaw, 10)
        tilt = 1.0 - d.xmat[L6[a]].reshape(3, 3)[:, 2] @ R_DOWN[a][:, 2]
        if err < 0.005 and tilt < best[0]:
            best = (tilt, d.qpos[QADR[a]].copy())
    d.qpos[QADR[a]] = best[1]


def closed_for_gap(gap):  # measured on this gripper: tip gap 8.02 cm open, -0.5 cm at closed=1
    return float(np.clip((0.0802 - gap) / 0.0852, 0.0, 1.0))


def set_grip(a, closed):
    d.qpos[FADR[a][0]], d.qpos[FADR[a][1]] = -0.041 * closed, 0.041 * closed


def quat_xyzw(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return np.array([q[1], q[2], q[3], q[0]])


def visual_meshes(b):
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


BODIES = [b for b in range(1, m.nbody) if visual_meshes(b)]


def body_poses():
    mujoco.mj_kinematics(m, d)
    return np.array([np.r_[R_BASE @ d.xpos[b] + BASE_POS, quat_xyzw(R_BASE @ d.xmat[b].reshape(3, 3))] for b in BODIES],
                    dtype=np.float32)


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


# --- the scene: counter (top at z 0) with a sink where the right arm reaches straight down, a plate in it, a sponge beside ---
SINK_C, SINK_H, SINK_D = np.array([0.32, -0.12]), np.array([0.11, 0.10]), 0.08  # centre, half size, depth
SINK_BOTTOM = -SINK_D
PLATE_R, PLATE_T = 0.075, 0.008
PLATE_TOP = SINK_BOTTOM + PLATE_T
SPONGE_H = np.array([0.032, 0.048, 0.016])  # half sizes: 6.4 x 9.6 x 3.2 cm
SPONGE_POS = np.array([0.30, 0.07, SPONGE_H[2] + 0.002])

builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
counter, steel = (0.86, 0.84, 0.80), (0.70, 0.72, 0.75)


def static_box(c, h, color):
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(*c), wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2], color=color)


top = 0.03  # counter slab thickness
cx0, cx1, cy0, cy1 = -0.50, 0.60, -0.62, 0.25
sx0, sx1, sy0, sy1 = SINK_C[0] - SINK_H[0], SINK_C[0] + SINK_H[0], SINK_C[1] - SINK_H[1], SINK_C[1] + SINK_H[1]
for x0, x1, y0, y1 in ((cx0, sx0, cy0, cy1), (sx1, cx1, cy0, cy1), (sx0, sx1, cy0, sy0), (sx0, sx1, sy1, cy1)):  # around the hole
    static_box([(x0 + x1) / 2, (y0 + y1) / 2, -top / 2], [(x1 - x0) / 2, (y1 - y0) / 2, top / 2], counter)
w = 0.004  # sink: steel tub under the hole
static_box([*SINK_C, SINK_BOTTOM - w], [SINK_H[0], SINK_H[1], w], steel)
for sgn in (-1, 1):
    static_box([SINK_C[0] + sgn * (SINK_H[0] - w), SINK_C[1], -SINK_D / 2], [w, SINK_H[1], SINK_D / 2], steel)
    static_box([SINK_C[0], SINK_C[1] + sgn * (SINK_H[1] - w), -SINK_D / 2], [SINK_H[0], w, SINK_D / 2], steel)
plate = trimesh.creation.cylinder(radius=PLATE_R, height=PLATE_T, sections=48)
builder.add_shape_mesh(-1, xform=wp.transform(wp.vec3(*SINK_C, SINK_BOTTOM + PLATE_T / 2), wp.quat_identity()),
                       mesh=newton.Mesh(plate.vertices.astype(np.float32), plate.faces.flatten().astype(np.int32), compute_inertia=False),
                       color=(0.55, 0.72, 0.85))  # blue-glazed plate

sv, st = voxel_tets(np.round(2 * SPONGE_H / args.pitch).astype(int), -SPONGE_H, args.pitch)
E, nu = 3e4, 0.1  # open-cell foam
s0 = builder.particle_count
builder.add_soft_mesh(pos=wp.vec3(*SPONGE_POS), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0), vertices=sv.tolist(),
                      indices=st.flatten().tolist(), density=25.0, k_mu=E / (2 * (1 + nu)),
                      k_lambda=E * nu / ((1 + nu) * (1 - 2 * nu)), k_damp=0.1, particle_radius=0.001, color=(0.95, 0.8, 0.2))
s1 = builder.particle_count

# arm: every link a kinematic body; only the right fingers touch the sponge, through their convex hulls
HOME = {"R": (np.array([0.30, -0.05, 0.20]), 0.0), "L": (np.array([-0.30, -0.15, 0.25]), 0.0)}
for a in ARMS:
    ik_from_scratch(a, *HOME[a])
Q_HOME = d.qpos.copy()
pose0 = body_poses()
FINGERS = {m.body("link7").id, m.body("link8").id}
for name, v, f in visual_meshes(0):  # the base, on the world body; its stand cut off at the counter top
    v = v.copy()
    v[:, 2] = np.maximum(v[:, 2], -BASE_POS[2])
    builder.add_shape_mesh(-1, xform=wp.transform(wp.vec3(*BASE_POS), wp.quat(*quat_xyzw(R_BASE))),
                           mesh=newton.Mesh(v.astype(np.float32), f.flatten().astype(np.int32), compute_inertia=False),
                           cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
                           color=(0.88, 0.88, 0.9))
for b, pq in zip(BODIES, pose0):
    nb = builder.add_body(xform=wp.transform(wp.vec3(*pq[:3]), wp.quat(*pq[3:])), is_kinematic=True, label=m.body(b).name)
    meshes = visual_meshes(b)
    for name, v, f in meshes:
        builder.add_shape_mesh(nb, mesh=newton.Mesh(v.astype(np.float32), f.flatten().astype(np.int32), compute_inertia=False),
                               cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
                               color=(0.95, 0.45, 0.1) if "finger" in name else (0.88, 0.88, 0.9))
    if b in FINGERS:
        hull = trimesh.Trimesh(next(v for name, v, f in meshes if name.endswith("finger"))).convex_hull
        builder.add_shape_mesh(nb, mesh=newton.Mesh(hull.vertices.astype(np.float32), hull.faces.flatten().astype(np.int32), compute_inertia=False),
                               cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, is_visible=False))

builder.color()
model = builder.finalize()
MU, KE, KD = 1.0, 2e5, 1.0
model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = KE, KD, MU
for arr, val in ((model.shape_material_mu, MU), (model.shape_material_ke, KE)):
    a_ = arr.numpy()
    a_[:] = val
    arr.assign(a_)

state0, state1, control = model.state(), model.state(), model.control()
pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.002)
contacts = pipeline.contacts()
solver = SolverVBD(model, iterations=args.iterations, integrate_with_external_rigid_solver=True,
                   friction_epsilon=1e-3)  # m/s: the default 1e-2 lets anything slower than 1 cm/s slide
NB = len(BODIES)
traj = wp.zeros((args.substeps + 1) * NB, dtype=wp.transform)


@wp.kernel
def set_kinematic(traj: wp.array(dtype=wp.transform), i: int, nb: int, dt: float, q_in: wp.array(dtype=wp.transform),
                  qd_in: wp.array(dtype=wp.spatial_vector), q_out: wp.array(dtype=wp.transform)):
    b = wp.tid()
    t0 = traj[i * nb + b]
    t1 = traj[(i + 1) * nb + b]
    q_in[b] = t0
    q_out[b] = t1  # VBD reads a kinematic body's motion from state_in to state_out
    qd_in[b] = wp.spatial_vector((wp.transform_get_translation(t1) - wp.transform_get_translation(t0)) / dt, wp.vec3(0.0))


def simulate():
    global state0, state1
    for i in range(args.substeps):
        wp.launch(set_kinematic, NB, inputs=[traj, i, NB, DT, state0.body_q, state0.body_qd, state1.body_q])
        pipeline.collide(state0, contacts)
        state0.clear_forces()
        solver.step(state0, state1, control, contacts, DT)
        state0, state1 = state1, state0


graph = None
if wp.get_device().is_cuda and args.substeps % 2 == 0:  # an even count keeps state0/state1 where the graph left them
    traj.assign(np.tile(pose0, (args.substeps + 1, 1)))
    simulate()  # compile first
    with wp.ScopedCapture() as cap:
        simulate()
    graph = cap.graph

# dirt: spots on the plate that the sponge wipes off by pressing on them
rng = np.random.default_rng(0)
ang, rad = rng.uniform(0, 2 * np.pi, 400), PLATE_R * 0.85 * np.sqrt(rng.uniform(0, 1, 400))
DIRT = np.c_[SINK_C[0] + rad * np.cos(ang), SINK_C[1] + rad * np.sin(ang), np.full(400, PLATE_TOP - 0.0015)]
dirty = np.ones(len(DIRT), bool)


def wipe(xs):
    near = xs[xs[:, 2] < PLATE_TOP + 0.004]  # sponge particles pressing on the plate
    if len(near):
        hit = np.min(np.linalg.norm(DIRT[:, None, :2] - near[None, :, :2], axis=-1), axis=1) < 0.007
        dirty[hit] = False


viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=bool(args.demo))
viewer.set_model(model)
viewer.show_particles = False
CAM = (wp.vec3(1.05, 0.45, 0.75), -33.0, -140.0)
viewer.set_camera(*CAM)

target, yaw, closed, closed_cmd = HOME["R"][0].copy(), HOME["R"][1], 0.0, 0.0
SHUT = closed_for_gap(args.gap)


def reset():
    global state0, state1, target, yaw, closed, closed_cmd
    s = model.state()
    state0.particle_q.assign(s.particle_q)
    state0.particle_qd.zero_()
    target, yaw, closed, closed_cmd = HOME["R"][0].copy(), HOME["R"][1], 0.0, 0.0
    d.qpos[:] = Q_HOME
    dirty[:] = True


# scripted demo: (time, tool point, yaw, closed)
grasp_z = SPONGE_POS[2] + 0.004
scrub = [SINK_C + 0.045 * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 6 * np.pi, 13)]
sx, sy = SPONGE_POS[:2]
DEMO = [(0.0, [0.30, -0.05, 0.20], 0.0, 0.0), (1.0, [sx, sy, 0.15], 0.0, 0.0), (2.2, [sx, sy, grasp_z], 0.0, 0.0),
        (3.0, [sx, sy, grasp_z], 0.0, SHUT), (4.2, [sx, sy, 0.14], 0.0, SHUT), (6.0, [*SINK_C, 0.10], 0.0, SHUT),
        (7.2, [*scrub[0], PLATE_TOP + 0.010], 0.0, SHUT)]
for k, xy in enumerate(scrub[1:]):
    DEMO.append((7.2 + 0.6 * (k + 1), [*xy, PLATE_TOP + 0.010], 0.0, SHUT))
DEMO.append((DEMO[-1][0] + 1.2, [*SINK_C, 0.12], 0.0, SHUT))


def demo_target(t):
    for (t0, p0, y0, c0), (t1, p1, y1, c1) in zip(DEMO, DEMO[1:]):
        if t <= t1:
            s = 0.5 - 0.5 * np.cos(np.pi * np.clip((t - t0) / (t1 - t0), 0, 1))
            return np.array(p0) + s * (np.array(p1) - np.array(p0)), y0 + s * (y1 - y0), c0 + s * (c1 - c0)
    return np.array(DEMO[-1][1]), DEMO[-1][2], DEMO[-1][3]


frames, t, prev = [], 0.0, pose0
g_was = r_was = False
wall0, sim_wall = time.time(), 0.0
while viewer.is_running() and (not args.demo or t < DEMO[-1][0] + 0.5):
    tick = time.time()
    if args.demo:
        target, yaw, closed = demo_target(t + 1.0 / FPS)
    elif not viewer.is_paused():
        v = (0.05 if viewer.is_key_down("shift") else 0.20) / FPS  # m per frame
        k = viewer.is_key_down
        target += v * np.array([k("j") - k("l"), k("i") - k("k"), k("u") - k("o")], float)  # camera faces -y: J is +x
        target = np.clip(target, [0.0, -0.35, SINK_BOTTOM + 0.005], [0.50, 0.20, 0.40])
        yaw += (k("n") - k("m")) * 1.5 / FPS
        if k("g") and not g_was:
            closed_cmd = SHUT if closed_cmd == 0.0 else 0.0
        if k("r") and not r_was:
            reset()
        g_was, r_was = k("g"), k("r")
        closed += np.clip(closed_cmd - closed, -1.0 / FPS, 1.0 / FPS)  # about 1 s to close
    if not viewer.is_paused():
        ik("R", target, yaw)
        set_grip("R", closed)
        new = body_poses()
        new[np.sum(new[:, 3:] * prev[:, 3:], axis=1) < 0, 3:] *= -1  # same quaternion hemisphere as the last frame
        s = np.linspace(0, 1, args.substeps + 1)[:, None, None]
        tr = (1 - s) * prev[None] + s * new[None]
        tr[..., 3:] /= np.linalg.norm(tr[..., 3:], axis=-1, keepdims=True)
        traj.assign(tr.reshape(-1, 7))
        tw = time.time()
        if graph:
            wp.capture_launch(graph)
        else:
            simulate()
        xs = state0.particle_q.numpy()[s0:s1]
        sim_wall += time.time() - tw
        wipe(xs)
        prev = new
        t += 1.0 / FPS
    viewer.begin_frame(t)
    viewer.log_state(state0)
    n_d = int(dirty.sum())
    viewer.log_points("dirt", wp.array(DIRT[dirty], dtype=wp.vec3) if n_d else None, 0.004,
                      wp.array(np.tile([0.45, 0.30, 0.12], (max(n_d, 1), 1)), dtype=wp.vec3))
    viewer.end_frame()
    if args.demo:
        frames.append(viewer.get_frame().numpy())
        if len(frames) % 60 == 0:
            print(f"t {t:.1f} s  sponge z {100 * xs[:, 2].min():.1f}..{100 * xs[:, 2].max():.1f} cm  clean "
                  f"{100 * (1 - dirty.mean()):.0f}%  sim {t / sim_wall:.2f}x realtime", flush=True)
    else:
        time.sleep(max(0.0, 1.0 / FPS - (time.time() - tick)))  # no faster than real time

if args.demo:
    import imageio

    print(f"RESULT clean={100 * (1 - dirty.mean()):.0f}% sim_speed={t / sim_wall:.2f}x realtime "
          f"wall={time.time() - wall0:.0f}s", flush=True)
    imageio.mimsave(args.demo, frames, fps=FPS, macro_block_size=8)
    imageio.imwrite(str(Path(args.demo).with_suffix(".png")), frames[int(len(frames) * 0.75)])
