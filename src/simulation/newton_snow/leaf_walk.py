"""Pioneer whole-body humanoid walking along a path covered in autumn maple leaves (Newton XPBD rigid bodies), with
leaves drifting down from above; path-traced with OVRTX in front of a background USD (autumn_scene.py).

The walking runs in MuJoCo as in snow_walk.py; every 20 ms the robot's body poses (and finite-difference velocities) are
written into Newton as kinematic bodies, whose feet and shins kick the leaves. One-way: leaves do not push the robot.
Air: each substep damps leaf velocities (a leaf falls at ~1 m/s) and adds a sideways flutter to airborne leaves.

  python leaf_walk.py <policy.pt> <out.mp4> --bg out/autumn.usda [--seconds 12] [--leaves 2600] [--falling 260]
"""
import argparse
import os
import re
import sys

import imageio
import mujoco
import numpy as np
import torch
import warp as wp

import newton
import newton.viewer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, f"{REPO}/src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion/config/pioneer")
from sim2sim_mujoco import DEFAULT, EFFORT, JOINTS, KD, KP  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("policy")
p.add_argument("out")
p.add_argument("--bg", default=None)
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--vx", type=float, default=0.4)
p.add_argument("--leaves", type=int, default=2600)
p.add_argument("--falling", type=int, default=260)
p.add_argument("--size", type=int, nargs=2, default=[1280, 720])
p.add_argument("--spp", type=int, default=16)
p.add_argument("--theme", choices=["maple", "cherry"], default="maple", help="cherry: pink blossom petals instead of maple leaves")
p.add_argument("--gl", action="store_true", help="OpenGL preview instead of path tracing")
args = p.parse_args()

# --- MuJoCo walker: whole-body arm-swing robot, as snow_walk.py ---
SUB = 16
ARMS = {"l_shoulder_pitch": 53.0, "r_shoulder_pitch": 53.0, "l_elbow": 22.0, "r_elbow": 22.0}
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot/pioneer_whole_body_humanoid/pioneer_whole_body_humanoid_arm_swing_isaac.urdf")).read()
src = src.replace("/workspace/humanoid/", f"{REPO}/")
src = re.sub(r"<mujoco>.*?</mujoco>", '<mujoco><compiler balanceinertia="true" discardvisual="false"/></mujoco>', src, flags=re.S)
spec = mujoco.MjSpec.from_string(src)
for geom in spec.geoms:
    mesh = geom.type == mujoco.mjtGeom.mjGEOM_MESH
    geom.contype, geom.conaffinity, geom.group = 0, 0 if mesh else 1, 1 if mesh else 3
spec.option.timestep = 0.02 / SUB
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
spec.worldbody.first_body().add_freejoint()
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
for name in JOINTS + list(ARMS):
    kind = name[:-2]
    kp, kd, effort = (KP[kind], KD[kind], EFFORT[kind]) if name in JOINTS else (40.0, 2.0, ARMS[name])
    j = spec.joint(name)
    j.armature, j.damping = 0.01, [0.0, 0.0, 0.0]
    spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=[kp] + [0] * 9, biasprm=[0, -kp, -kd] + [0] * 7,
                      forcelimited=True, forcerange=[-effort, effort])
mj = spec.compile()
data = mujoco.MjData(mj)
policy = torch.jit.load(args.policy).eval()
qadr = np.array([mj.joint(j).qposadr[0] for j in JOINTS])
vadr = np.array([mj.joint(j).dofadr[0] for j in JOINTS])
act_id = np.array([mj.actuator(j).id for j in JOINTS])
default = np.array([DEFAULT.get(j[:-2], 0.0) for j in JOINTS])
data.qpos[2] = 0.84
data.qpos[qadr] = default
mujoco.mj_forward(mj, data)

# --- Newton: kinematic robot + leaves ---
builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
KICKERS = ("l_foot", "r_foot", "l_ankle_cross", "r_ankle_cross", "l_shank_foot", "r_shank_foot")
vis = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
robot_mj = []
for b in range(1, mj.nbody):
    name = mj.body(b).name
    nb = builder.add_body(label=name, is_kinematic=True)
    robot_mj.append(b)
    for g in np.nonzero(mj.geom_bodyid == b)[0]:
        q = mj.geom_quat[g]
        xf = wp.transform(wp.vec3(*mj.geom_pos[g]), wp.quat(q[1], q[2], q[3], q[0]))
        if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = mj.geom_dataid[g]
            v0, nv, f0, nf = mj.mesh_vertadr[m], mj.mesh_vertnum[m], mj.mesh_faceadr[m], mj.mesh_facenum[m]
            mesh = newton.Mesh(mj.mesh_vert[v0:v0 + nv].copy(), mj.mesh_face[f0:f0 + nf].flatten().copy(), compute_inertia=False)
            builder.add_shape_mesh(nb, xform=xf, mesh=mesh, cfg=vis, color=tuple(float(c) for c in mj.geom_rgba[g, :3]))
        elif mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and name in KICKERS:
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, is_visible=False, mu=0.6)
            builder.add_shape_box(nb, xform=xf, hx=float(mj.geom_size[g, 0]), hy=float(mj.geom_size[g, 1]), hz=float(mj.geom_size[g, 2]), cfg=cfg)
robot_mj = np.array(robot_mj)
n_robot = len(robot_mj)
builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, is_visible=False))


def maple_leaf(size, rng):
    """A five-lobed maple leaf, `size` m across, gently cupped: (vertices, triangle indices)."""
    th = np.linspace(0, 2 * np.pi, 61)[:-1]
    lobes = np.abs(np.cos(2.5 * (th - np.pi / 2))) ** 0.7
    r = size / 2 * (0.42 + 0.58 * lobes) * (1 - 0.35 * np.clip(-np.sin(th), 0, 1) ** 3)  # shorter base lobes
    r *= 1 + 0.06 * rng.standard_normal(len(th))
    x, y = r * np.cos(th), r * np.sin(th)
    z = 0.12 * (x ** 2 + y ** 2) / size + 0.004 * rng.standard_normal(len(th))
    verts = np.concatenate([[[0.0, 0.0, 0.0]], np.stack([x, y, z], 1)]).astype(np.float32)
    n = len(th)
    tris = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)], np.int32)
    tris = np.concatenate([tris, tris[:, ::-1]])  # two-sided
    return verts, tris.flatten()


def petal(size, rng):
    """A cherry-blossom petal, `size` m long: a rounded oval with a notch at the tip, slightly cupped."""
    th = np.linspace(0, 2 * np.pi, 41)[:-1]
    x, y = size / 2 * np.cos(th), size / 3.2 * np.sin(th)
    x = x - size / 2 * 0.25 * np.exp(-((th % (2 * np.pi)) ** 2 + 0.0) / 0.02)  # the notch at th = 0
    z = 0.25 * (x ** 2 + 2 * y ** 2) / size
    verts = np.concatenate([[[0.0, 0.0, 0.0]], np.stack([x, y, z], 1)]).astype(np.float32)
    tris = np.array([[0, 1 + i, 1 + (i + 1) % 40] for i in range(40)], np.int32)
    return verts, np.concatenate([tris, tris[:, ::-1]]).flatten()


rng = np.random.default_rng(7)
PALETTE = np.array([[0.62, 0.09, 0.03], [0.78, 0.22, 0.04], [0.86, 0.42, 0.06], [0.84, 0.6, 0.1], [0.5, 0.26, 0.08], [0.7, 0.14, 0.05]])
if args.theme == "cherry":
    PALETTE = np.array([[0.95, 0.5, 0.66], [0.9, 0.42, 0.6], [0.98, 0.62, 0.75], [1.0, 0.72, 0.82], [0.86, 0.36, 0.55]])
    shapes = [petal(rng.uniform(0.02, 0.03), rng) for _ in range(12)]
else:
    shapes = [maple_leaf(rng.uniform(0.07, 0.11), rng) for _ in range(12)]
leaf_cfg = newton.ModelBuilder.ShapeConfig(density=40.0, mu=0.7, restitution=0.0, is_visible=False)
length = args.vx * args.seconds
n_ground, n_fall = args.leaves, args.falling
for i in range(n_ground + n_fall):
    if i < n_ground:  # a carpet along the path, a few layers deep
        pos = (rng.uniform(-1.0, length + 2.2), rng.uniform(-1.4, 1.4), rng.uniform(0.004, 0.05))
        rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(0, 0.15, 2), 1.0)), float(rng.uniform(0, 2 * np.pi)))
        rot = wp.mul(rot, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(0, 2 * np.pi))))
    else:  # drifting down from the canopy over the whole clip
        pos = (rng.uniform(-1.5, length + 2.5), rng.uniform(-2.0, 2.5), rng.uniform(1.5, 1.5 + 1.0 * args.seconds))
        rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(0, 1, 3))), float(rng.uniform(0, 2 * np.pi)))
    b = builder.add_body(xform=wp.transform(wp.vec3(*pos), rot), label=f"leaf_{i}")
    v, t = shapes[i % len(shapes)]
    s = float(np.ptp(v[:, 0])) / 2
    builder.add_shape_box(b, hx=s * 0.8, hy=s * (0.55 if args.theme == "cherry" else 0.8), hz=0.0015 if args.theme == "cherry" else 0.0025, cfg=leaf_cfg)  # collision: a thin plate
    colour = PALETTE[rng.integers(len(PALETTE))] * rng.uniform(0.8, 1.1)
    builder.add_shape_mesh(b, mesh=newton.Mesh(v, t, compute_inertia=False), cfg=vis, color=tuple(float(c) for c in np.clip(colour, 0, 1)))
model = builder.finalize()
state, state_1 = model.state(), model.state()
control = model.control()
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()
solver = newton.solvers.SolverXPBD(model, iterations=6)
print(f"bodies {model.body_count}: robot {n_robot}, {args.theme} {n_ground} on the ground + {n_fall} falling")


@wp.kernel
def air(qd: wp.array(dtype=wp.spatial_vector), q: wp.array(dtype=wp.transform), first: int, t: float, dt: float):
    """Leaf aerodynamics, roughly: strong drag (a leaf falls at about 1 m/s), and a sideways flutter while airborne."""
    i = wp.tid() + first
    v = wp.spatial_top(qd[i])
    w = wp.spatial_bottom(qd[i])
    p = wp.transform_get_translation(q[i])
    if p[2] > 0.06:
        drag = 9.0
        phase = float(i) * 1.7
        v = v + dt * wp.vec3(1.2 * wp.sin(2.1 * t + phase) + 0.4, 1.0 * wp.cos(1.7 * t + 0.6 * phase), 0.0)
        w = w + dt * wp.vec3(6.0 * wp.sin(3.0 * t + phase), 6.0 * wp.cos(2.6 * t + phase), 0.0)
    else:
        drag = 2.0
    k = wp.exp(-drag * dt)
    qd[i] = wp.spatial_vector(v * k, w * wp.exp(-3.0 * dt))


prev_pose = None


def sync_robot(dt):
    """Kinematic robot bodies: poses from MuJoCo, velocities by finite difference (what the leaves feel)."""
    global prev_pose
    pose = np.concatenate([data.xpos[robot_mj], data.xquat[robot_mj][:, [1, 2, 3, 0]]], 1).astype(np.float32)
    lin = np.zeros((n_robot, 3), np.float32) if prev_pose is None else (pose[:, :3] - prev_pose[:, :3]) / dt
    ang = data.cvel[robot_mj][:, :3].astype(np.float32)  # world-frame angular velocity (rotational part of cvel)
    prev_pose = pose
    bq, bqd = state.body_q.numpy(), state.body_qd.numpy()
    bq[:n_robot] = pose
    bqd[:n_robot] = np.concatenate([lin, ang], 1)
    state.body_q.assign(bq)
    state.body_qd.assign(bqd)


if args.gl:
    viewer = newton.viewer.ViewerGL(width=args.size[0], height=args.size[1], headless=True)
else:
    viewer = newton.viewer.ViewerRTX(width=args.size[0], height=args.size[1], headless=True, async_rendering=False,
                                     environment="none" if args.bg else "default",
                                     render_settings={"omni:rtx:pt:samplesPerPixel": ("uint", args.spp),
                                                      "omni:rtx:pt:denoising:enabled": ("bool", True)})
    if args.bg:
        viewer.add_background_usd(args.bg)
viewer.set_model(model)
viewer.show_ground = False

SUBSTEPS = 8
sub_dt = 0.02 / SUBSTEPS


FEET = [list(robot_mj).index(mj.body(n).id) for n in ("l_foot", "r_foot")]


DBG = []


def foot_wake():
    """Approximate air wake of a swinging foot: leaves within ~15 cm of a foot moving faster than 0.4 m/s get part of
    its velocity and some lift (the solver's contacts alone only shove the leaves the foot actually touches)."""
    q, qd = state.body_q.numpy(), state.body_qd.numpy()
    leaves = slice(n_robot, n_robot + n_ground + n_fall)
    for f in FEET:
        fp, fv = q[f, :3], qd[f, :3]
        speed = np.linalg.norm(fv)
        DBG.append((speed, fp[2], int((np.linalg.norm(q[leaves, :3] - fp, axis=1) < 0.15).sum())))
        if speed < 0.4:
            continue
        d = np.linalg.norm(q[leaves, :3] - fp, axis=1)
        near = d < 0.18
        if near.any():
            gain = (1 - d[near] / 0.18)[:, None] * rng.uniform(0.5, 1.0, (near.sum(), 1))
            kick = gain * (fv * np.array([1.0, 1.0, 0.3]) + np.array([0, 0, 1.5 * speed]))
            idx = np.nonzero(near)[0] + n_robot
            qd[idx, :3] += kick
            qd[idx, 3:] += rng.normal(0, 4.0, (len(idx), 3))
    state.body_qd.assign(qd)


def physics(t):
    global state, state_1
    for _ in range(SUBSTEPS):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, state_1, control, contacts, sub_dt)
        state, state_1 = state_1, state
        wp.launch(air, dim=n_ground + n_fall, inputs=[state.body_qd, state.body_q, n_robot, t, sub_dt])


# let the carpet settle before the robot sets off
sync_robot(0.02)
for k in range(50):
    physics(-1.0 + 0.02 * k)
    sync_robot(0.02)
settled = state.body_q.numpy()[n_robot:n_robot + n_ground, :3].copy()
max_h = np.zeros(n_ground)

command = np.array([args.vx, 0.0, 0.0])
action = np.zeros(12)
frames, t = [], 0.0
for step in range(int(args.seconds / 0.02)):
    rot = data.xmat[1].reshape(3, 3)
    ang = data.qvel[3:6]
    lin = rot.T @ data.qvel[:3] + np.cross(ang, mj.body_ipos[1])
    phase = 2 * np.pi * (step * 0.02 % 1.0)
    obs = np.concatenate([lin, ang, rot.T @ np.array([0, 0, -1.0]), command, data.qpos[qadr] - default, data.qvel[vadr], action,
                          [np.sin(phase), np.cos(phase)]])
    with torch.no_grad():
        action = policy(torch.from_numpy(obs.astype(np.float32))[None]).numpy()[0].astype(float)
    data.ctrl[act_id] = default + 0.25 * action
    swing = 0.35 * min(abs(command[0]) / 0.8, 1.0) * np.sin(2 * np.pi * (step * 0.02 % 1.0 - 0.55 / 2 + 0.25))
    for name, target in (("l_shoulder_pitch", -swing), ("r_shoulder_pitch", swing), ("l_elbow", -0.3), ("r_elbow", -0.3)):
        data.ctrl[mj.actuator(name).id] = target
    for _ in range(SUB):
        mujoco.mj_step(mj, data)
    sync_robot(0.02)
    foot_wake()
    physics(t)
    max_h = np.maximum(max_h, state.body_q.numpy()[n_robot:n_robot + n_ground, 2])
    t += 0.02
    if data.qpos[2] < 0.4:
        print(f"fell at {t:.2f} s")
        break
    if step % 2:
        continue
    base = data.qpos[:3].copy()
    if t < args.seconds / 2:  # ahead and to the right, looking back at the robot
        viewer.set_camera(wp.vec3(base[0] + 2.5, base[1] - 2.0, 1.1), pitch=-6, yaw=141)
    else:  # side shot travelling with it
        viewer.set_camera(wp.vec3(base[0] + 0.5, base[1] - 3.0, 1.0), pitch=-6, yaw=99)
    viewer.begin_frame(t)
    viewer.log_state(state)
    viewer.end_frame()
    frames.append(viewer.get_frame().numpy())
    if step % 50 == 0:
        print(f"t {t:.1f} s, base x {base[0]:.2f} m", flush=True)

q = state.body_q.numpy()[n_robot:n_robot + n_ground, :3]
d = np.array(DBG); print("foot speed p50/p90/max", np.percentile(d[:, 0], [50, 90, 100]).round(2), "foot z min/max", d[:, 1].min().round(3), d[:, 1].max().round(3), "near>0 frac", (d[:, 2] > 0).mean().round(2), "near when fast", d[d[:, 0] > 0.4, 2].mean().round(1))
print(f"ground leaves moved > 10 cm: {(np.linalg.norm(q - settled, axis=1) > 0.1).sum()}, lifted > 10 cm at some point: {(max_h > 0.1).sum()}, highest {max_h.max():.2f} m")
imageio.mimsave(args.out, frames, fps=25, macro_block_size=8)
for k in (len(frames) // 3, 2 * len(frames) // 3, -1):
    imageio.imwrite(args.out.replace(".mp4", f"_{k % len(frames)}.png"), frames[k])
print(f"wrote {args.out}: {len(frames)} frames, walked {data.qpos[0]:.2f} m")
