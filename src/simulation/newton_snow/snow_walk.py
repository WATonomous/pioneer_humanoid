"""Pioneer humanoid (leg-only, or whole-body with the scripted arm swing and fists) walking through MPM snow (Newton implicit MPM), with falling flakes.

The walking runs in MuJoCo exactly as sim2sim_mujoco.py (the policy trained in Isaac Lab on flat ground); every 20 ms the
robot's body poses are written into a Newton model whose feet are kinematic colliders for the snow. One-way coupling:
the feet push the snow, the snow does not push back.

  python snow_walk.py <policy.pt> <out.mp4> [--seconds 10] [--vx 0.4] [--depth 0.08]
"""
import argparse
import os
import sys

import imageio
import mujoco
import numpy as np
import torch
import warp as wp

import newton
import newton.viewer
from newton.solvers import SolverImplicitMPM

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, f"{REPO}/src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion/config/pioneer")
import re  # noqa: E402

from sim2sim_mujoco import DEFAULT, EFFORT, JOINTS, KD, KP, build  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("policy")
p.add_argument("out")
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--vx", type=float, default=0.4)
p.add_argument("--depth", type=float, default=0.05, help="snow depth (m)")
p.add_argument("--dx", type=float, default=0.03, help="MPM voxel size (m)")
p.add_argument("--ppc", type=float, default=3.0, help="particles per cell along each axis")
p.add_argument("--flakes", type=int, default=15000)
p.add_argument("--whole_body", action="store_true", help="the arm-swing whole-body robot (robot/make_robot.sh)")
p.add_argument("--view", choices=["front", "back", "tour"], default="front")
p.add_argument("--rtx", action="store_true", help="OVRTX path-traced rendering instead of the OpenGL viewer")
p.add_argument("--no_scenery", action="store_true")
p.add_argument("--bg", default=None, help="background USD for --rtx (night_scene.py); replaces the built-in ground and scenery")
p.add_argument("--slab_y", type=float, nargs=2, default=None, help="lateral extent of the simulated snow (m)")
p.add_argument("--spp", type=int, default=16, help="path-tracing samples per pixel")
p.add_argument("--size", type=int, nargs=2, default=[1280, 720])
p.add_argument("--slab_x1", type=float, default=None, help="far end of the simulated slab (m)")
p.add_argument("--record", default=None, help="physics only (no renderer): save each video frame's bodies and particles in this dir")
p.add_argument("--replay", default=None, help="render only: path-trace the frames --record saved (fits the 8 GB GPU)")
p.add_argument("--material", choices=["snow", "sand", "gravel"], default="snow")
args = p.parse_args()

# --- MuJoCo walker (the proven sim2sim setup) ---
SUB = 16
ARMS = {"l_shoulder_pitch": 53.0, "r_shoulder_pitch": 53.0, "l_elbow": 22.0, "r_elbow": 22.0}  # effort limits (leg_only.py)


def build_whole_body(timestep):
    """As sim2sim_mujoco.build, from the Isaac copy of the whole-body URDF (already re-rooted at the hips, everything but
    the legs and the four arm-swing joints fixed, fists baked in); arms on PD 40 / 2 as WHOLE_BODY_ARM_SWING_CFG."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot/pioneer_whole_body_humanoid/pioneer_whole_body_humanoid_arm_swing_isaac.urdf")).read().replace("/workspace/humanoid/", f"{REPO}/")
    src = re.sub(r"<mujoco>.*?</mujoco>", '<mujoco><compiler balanceinertia="true" discardvisual="false"/></mujoco>', src, flags=re.S)
    spec = mujoco.MjSpec.from_string(src)
    for geom in spec.geoms:
        mesh = geom.type == mujoco.mjtGeom.mjGEOM_MESH
        geom.contype, geom.conaffinity, geom.group = 0, 0 if mesh else 1, 1 if mesh else 3
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.first_body().add_freejoint()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    for name in JOINTS + list(ARMS):
        kind = name[:-2]
        kp, kd, effort = (KP[kind], KD[kind], EFFORT[kind]) if name in JOINTS else (40.0, 2.0, ARMS[name])
        joint = spec.joint(name)
        joint.armature, joint.damping = 0.01, [0.0, 0.0, 0.0]
        spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=[kp] + [0] * 9, biasprm=[0, -kp, -kd] + [0] * 7,
                          forcelimited=True, forcerange=[-effort, effort])
    return spec.compile()


mj = build_whole_body(0.02 / SUB) if args.whole_body else build(False)
data = mujoco.MjData(mj)
policy = torch.jit.load(args.policy).eval()
n_obs = next(policy.parameters()).shape[1]
qadr = np.array([mj.joint(j).qposadr[0] for j in JOINTS])
vadr = np.array([mj.joint(j).dofadr[0] for j in JOINTS])
act_id = np.array([mj.actuator(j).id for j in JOINTS])
default = np.array([DEFAULT.get(j[:-2], 0.0) for j in JOINTS])
data.qpos[2] = 0.84
data.qpos[qadr] = default
mujoco.mj_forward(mj, data)


def robot_from_mujoco(builder):
    """The robot in Newton, built from MuJoCo's compiled model (exact body frames, geom poses, meshes): one body per
    MuJoCo body, posed every step from MuJoCo. Only the foot collision boxes touch the snow."""
    feet = {"l_foot", "r_foot", "l_ankle_cross", "r_ankle_cross"}
    bodies = []
    for b in range(1, mj.nbody):
        name = mj.body(b).name
        nb = builder.add_body(label=name)
        bodies.append(b)
        for g in np.nonzero(mj.geom_bodyid == b)[0]:
            q = mj.geom_quat[g]
            xf = wp.transform(wp.vec3(*mj.geom_pos[g]), wp.quat(q[1], q[2], q[3], q[0]))
            rgb = tuple(float(c) for c in mj.geom_rgba[g, :3])
            if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = mj.geom_dataid[g]
                v0, nv, f0, nf = mj.mesh_vertadr[m], mj.mesh_vertnum[m], mj.mesh_faceadr[m], mj.mesh_facenum[m]
                mesh = newton.Mesh(mj.mesh_vert[v0:v0 + nv].copy(), mj.mesh_face[f0:f0 + nf].flatten().copy(), compute_inertia=False)
                cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
                builder.add_shape_mesh(nb, xform=xf, mesh=mesh, cfg=cfg, color=rgb)
            elif mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and name in feet:
                cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, is_visible=False)
                builder.add_shape_box(nb, xform=xf, hx=float(mj.geom_size[g, 0]), hy=float(mj.geom_size[g, 1]), hz=float(mj.geom_size[g, 2]), cfg=cfg)
    return np.array(bodies)


builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
mj_of_newton = robot_from_mujoco(builder)
feet = [i for i, b in enumerate(mj_of_newton) if mj.body(b).name in ("l_foot", "r_foot", "l_ankle_cross", "r_ankle_cross")]
names = [mj.body(b).name for b in mj_of_newton]
builder.add_ground_plane(color=(0.86, 0.88, 0.93))
if not args.bg:
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.01), wp.quat_identity()), hx=60.0, hy=60.0, hz=0.01, color=(0.84, 0.87, 0.92),
                          cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False))
SolverImplicitMPM.register_custom_attributes(builder)


def scenery(builder, length):
    """Visual only: snow-capped pines scattered on both sides of the path and a wooden fence along its far side."""
    vis = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    rng = np.random.default_rng(3)
    trees = [(rng.uniform(-6, length + 8), side * rng.uniform(3.5, 12.0)) for side in (-1, 1) for _ in range(28)]
    for x, y in trees:
        h = rng.uniform(2.2, 4.2)
        green = tuple(float(c) for c in np.array([0.09, 0.23, 0.15]) * rng.uniform(0.8, 1.15))
        builder.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(x, y, 0.25 * h), wp.quat_identity()), radius=0.09 * h / 3,
                                   half_height=0.25 * h, color=(0.33, 0.22, 0.14), cfg=vis)
        for k, (z, r) in enumerate(((0.45, 0.42), (0.68, 0.32), (0.88, 0.2))):  # three tiers, each with a snowy top
            hh = 0.2 * h
            builder.add_shape_cone(-1, xform=wp.transform(wp.vec3(x, y, z * h), wp.quat_identity()), radius=r * h, half_height=hh, color=green, cfg=vis)
            builder.add_shape_cone(-1, xform=wp.transform(wp.vec3(x, y, z * h + 0.55 * hh), wp.quat_identity()), radius=0.45 * r * h,
                                   half_height=0.45 * hh, color=(0.93, 0.95, 0.98), cfg=vis)
    y_fence = 2.4
    for x in np.arange(-4.0, length + 6.0, 1.6):  # posts and two rails
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(x, y_fence, 0.45), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.45, color=(0.45, 0.3, 0.18), cfg=vis)
    for z in (0.45, 0.75):
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(length / 2 + 1.0, y_fence, z), wp.quat_identity()), hx=length / 2 + 5.0, hy=0.025,
                              hz=0.05, color=(0.5, 0.34, 0.2), cfg=vis)


if not args.no_scenery and not args.bg:
    scenery(builder, args.vx * args.seconds)

lo = np.array([-1.0, args.slab_y[0] if args.slab_y else (-1.7 if args.view == "tour" else -0.9), 0.0])
hi = np.array([args.slab_x1 or args.vx * args.seconds + (2.8 if args.view == "front" else 1.5), args.slab_y[1] if args.slab_y else 0.9, args.depth])
res = np.ceil(args.ppc * (hi - lo) / args.dx).astype(int)
cell = (hi - lo) / res
builder.add_particle_grid(pos=wp.vec3(lo), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=res[0] + 1, dim_y=res[1] + 1,
                          dim_z=res[2] + 1, cell_x=cell[0], cell_y=cell[1], cell_z=cell[2], mass=float(np.prod(cell) * (300.0 if args.material == "snow" else 1600.0)),
                          jitter=float(cell.max()), radius_mean=float(cell.max() * 0.5))
model = builder.finalize()
print(f"bodies {model.body_count} ({[names[b] for b in feet]} touch the snow), snow particles {model.particle_count}")

cfg = SolverImplicitMPM.Config()
cfg.voxel_size = args.dx
cfg.tolerance = 1e-5
cfg.transfer_scheme = "pic"
cfg.grid_type = "sparse"
cfg.strain_basis = "P0"
cfg.max_iterations = 50
cfg.critical_fraction = 0.0
cfg.air_drag = 1.0
cfg.collider_velocity_mode = "backward"
state = model.state()
# fresh snow (values from Newton's snowball example): compacts under load and keeps the print
model.mpm.particle_Jp.fill_(0.975)
state.mpm.particle_Jp.fill_(0.975)
model.mpm.young_modulus.fill_(1.4e6)
model.mpm.poisson_ratio.fill_(0.3)
model.mpm.friction.fill_(0.5)
model.mpm.damping.fill_(0.01)
model.mpm.yield_pressure.fill_(1.4e5)
model.mpm.tensile_yield_ratio.fill_(0.2)
model.mpm.yield_stress.fill_(0.0)
model.mpm.hardening.fill_(5.0)
model.mpm.dilatancy.fill_(0.0)
if args.material in ("sand", "gravel"):  # dry sand (Newton's granular example): cohesionless Drucker-Prager, friction 0.68
    model.mpm.particle_Jp.fill_(1.0)
    state.mpm.particle_Jp.fill_(1.0)
    model.mpm.young_modulus.fill_(1.0e15)
    model.mpm.friction.fill_(0.68 if args.material == "sand" else 0.9)  # angular gravel locks up more
    model.mpm.yield_pressure.fill_(1.0e12)
    model.mpm.tensile_yield_ratio.fill_(0.0)
    model.mpm.hardening.fill_(0.0)
    q = model.particle_q.numpy()  # pin a 6 cm rim (zero mass = kinematic) so the cohesionless slab's sides don't slump
    rim = (np.minimum(q[:, :2] - lo[:2], hi[:2] - q[:, :2]) < 0.06).any(1)
    mass = model.particle_mass.numpy(); mass[rim] = 0.0; model.particle_mass.assign(mass)
mpm = None if args.replay else SolverImplicitMPM(model, config=cfg)  # after the material: the solver reads it at construction


def sync_bodies():
    pose = np.concatenate([data.xpos[mj_of_newton], data.xquat[mj_of_newton][:, [1, 2, 3, 0]]], 1).astype(np.float32)
    state.body_q.assign(wp.array(pose, dtype=wp.transform))


sync_bodies()
if mpm: mpm.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=state.body_q)

if args.record:
    os.makedirs(args.record, exist_ok=True)
    viewer = type("NoViewer", (), {"set_camera": lambda *a, **k: None, "__bool__": lambda self: False})()
elif args.rtx:
    viewer = newton.viewer.ViewerRTX(width=args.size[0], height=args.size[1], headless=True, async_rendering=False,
                                     environment="none" if args.bg else "default",
                                     render_settings={"omni:rtx:pt:samplesPerPixel": ("uint", args.spp),
                                                      "omni:rtx:pt:denoising:enabled": ("bool", True)})
    if args.bg:
        viewer.add_background_usd(args.bg)
else:
    viewer = newton.viewer.ViewerGL(width=args.size[0], height=args.size[1], headless=True)
if viewer:
    viewer.set_model(model)
    viewer.show_particles = True
    viewer.show_ground = False
if viewer and not args.rtx:
    viewer.renderer.sky_upper = (0.34, 0.42, 0.56)
    viewer.renderer.sky_lower = (0.70, 0.75, 0.83)
    viewer.renderer.background_color = viewer.renderer.sky_upper
draw_r = wp.array(model.particle_radius.numpy() * 1.9)
SNOW = [0.8, 0.82, 0.86] if args.bg else [0.97, 0.98, 1.0]  # with --bg: the ground's snow albedo, so the two read as one field
print(f'SLAB {lo[0]:.3f} {hi[0]:.3f} {lo[1]:.3f} {hi[1]:.3f} {args.depth}')
DENT = [0.5, 0.6, 0.8]
if args.material == "sand":
    SNOW = [float(c) for c in os.environ.get("SAND", "0.6 0.47 0.31").split()]
    DENT = [0.5 * c for c in SNOW]  # disturbed sand: darker, damper sand underneath
if args.material == "gravel":  # crushed granite: each grain grey, pink feldspar, black biotite or white quartz, sizes varied
    g = np.random.default_rng(1)
    GRANITE = np.array([[0.3, 0.29, 0.28], [0.42, 0.4, 0.38], [0.48, 0.34, 0.3], [0.09, 0.09, 0.09], [0.62, 0.6, 0.57]])
    SNOW = GRANITE[g.choice(5, model.particle_count, p=[0.35, 0.25, 0.2, 0.1, 0.1])] * g.uniform(0.85, 1.1, (model.particle_count, 1))
    DENT = 0.8 * SNOW
    draw_r = wp.array((model.particle_radius.numpy() * 1.6 * g.uniform(0.7, 1.25, model.particle_count)).astype(np.float32))
snow_colors = wp.array(np.broadcast_to(np.asarray(SNOW, dtype=np.float32), (model.particle_count, 3)).copy(), dtype=wp.vec3)

rng = np.random.default_rng(0)
box = np.array([6.0, 6.0, 4.0])
flakes = rng.uniform(-0.5, 0.5, (args.flakes, 3)) * box
flake_v = rng.uniform(0.6, 1.2, args.flakes)
flake_phase = rng.uniform(0, 2 * np.pi, args.flakes)
flake_r = wp.array(rng.uniform(0.0015, 0.0035, args.flakes).astype(np.float32))
flake_c = wp.full(args.flakes, wp.vec3(1.0, 1.0, 1.0), dtype=wp.vec3)

command = np.array([args.vx, 0.0, 0.0])
q0 = state.particle_q.numpy().copy()
action = np.zeros(12)
frames, t = [], 0.0
for step in range(int(args.seconds / 0.02)):
    if args.replay:  # the recorded physics instead of stepping it
        t += 0.02
        if step % 2:
            continue
        f = os.path.join(args.replay, f"{step:05d}.npz")
        if not os.path.exists(f):
            break
        rec = np.load(f)
        state.body_q.assign(wp.array(rec["body_q"], dtype=wp.transform)); state.particle_q.assign(wp.array(rec["particle_q"], dtype=wp.vec3))
        data.qpos[:3] = rec["base"]
    else:
        rot = data.xmat[1].reshape(3, 3)
        ang = data.qvel[3:6]
        lin = rot.T @ data.qvel[:3] + np.cross(ang, mj.body_ipos[1])
        obs = np.concatenate([lin, ang, rot.T @ np.array([0, 0, -1.0]), command, data.qpos[qadr] - default, data.qvel[vadr], action])
        if n_obs == 50:  # the Gait task's clock (1 s period)
            obs = np.append(obs, [np.sin(2 * np.pi * (step * 0.02 % 1.0)), np.cos(2 * np.pi * (step * 0.02 % 1.0))])
        with torch.no_grad():
            action = policy(torch.from_numpy(obs.astype(np.float32))[None]).numpy()[0].astype(float)
        data.ctrl[act_id] = default + (0.25 if n_obs == 50 else 0.5) * action
        if args.whole_body:  # arm_swing.py: each shoulder swings opposite its leg on the gait clock, elbows bent 0.3 rad
            swing = 0.35 * min(abs(command[0]) / 0.8, 1.0) * np.sin(2 * np.pi * (step * 0.02 % 1.0 - 0.55 / 2 + 0.25))
            for name, target in (("l_shoulder_pitch", -swing), ("r_shoulder_pitch", swing), ("l_elbow", -0.3), ("r_elbow", -0.3)):
                data.ctrl[mj.actuator(name).id] = target
        for _ in range(SUB):
            mujoco.mj_step(mj, data)
        sync_bodies()
        mpm.step(state, state, contacts=None, control=None, dt=0.02)
        t += 0.02
        if data.qpos[2] < 0.4:
            print(f"fell at {t:.2f} s")
            break
    if step % 2:
        continue
    base = data.qpos[:3].copy()
    sway = 0.15 * np.sin(1.3 * t + flake_phase)
    flakes[:, 2] -= flake_v * 0.04
    flakes[:, 0] += 0.04 * (0.3 + sway)
    flakes[:, 1] += 0.04 * sway
    rel = flakes - np.array([base[0], base[1], box[2] / 2])
    flakes -= np.round(rel / box) * box  # wrap the flake box around the robot
    if args.view == "tour" and t < args.seconds / 2:  # low, ahead and to the right, looking back at the robot
        cam = np.array([base[0] + 2.5, base[1] - 2.0, 1.1]); viewer.set_camera(wp.vec3(*cam), pitch=-5, yaw=141)
    elif args.view == "tour":  # low side shot travelling with it: the feet and the trail behind
        cam = np.array([base[0] + 0.5, base[1] - 3.0, 1.0]); viewer.set_camera(wp.vec3(*cam), pitch=-5, yaw=99)
    elif args.view == "front":  # ahead of the robot and to its right, looking back at it
        cam = np.array([base[0] + 2.6, base[1] - 1.7, 1.35]); viewer.set_camera(wp.vec3(*cam), pitch=-14, yaw=147)
    else:
        cam = np.array([base[0] - 2.5, base[1] - 2.0, 1.55]); viewer.set_camera(wp.vec3(*cam), pitch=-18, yaw=38)
    near = np.linalg.norm(flakes - cam, axis=1) < 0.9  # no big blurry flakes right in front of the lens
    if args.record:
        np.savez(os.path.join(args.record, f"{step:05d}.npz"), body_q=state.body_q.numpy(), particle_q=state.particle_q.numpy(), base=base)
        if step % 50 == 0:
            print(f"t {t:.1f} s, base x {base[0]:.2f} m", flush=True)
        continue
    viewer.begin_frame(t)
    viewer.log_state(state)
    k = np.clip((q0[:, 2] - state.particle_q.numpy()[:, 2]) / 0.02, 0, 1)[:, None]  # how far the snow sank
    snow_colors.assign(((1 - k) * SNOW + k * np.array(DENT)).astype(np.float32))
    viewer.log_points("/model/particles", state.particle_q, draw_r, snow_colors)
    viewer.log_points("/flakes", wp.array(np.where(near[:, None], -10.0, flakes).astype(np.float32), dtype=wp.vec3), flake_r, flake_c)
    viewer.end_frame()
    frames.append(viewer.get_frame().numpy())
    if step % 50 == 0:
        print(f"t {t:.1f} s, base x {base[0]:.2f} m", flush=True)

if args.record:
    print(f"recorded {args.record}, walked {data.qpos[0]:.2f} m")
    sys.exit()
imageio.mimsave(args.out, frames, fps=25, macro_block_size=8)
for k in (len(frames) // 3, 2 * len(frames) // 3, -1):
    imageio.imwrite(args.out.replace(".mp4", f"_{k % len(frames)}.png"), frames[k])
q1 = state.particle_q.numpy(); d = np.linalg.norm(q1 - q0, axis=1); print('moved >1cm', (d > 0.01).sum(), 'max', d.max().round(3), 'top z before/after', q0[:, 2].max().round(3), np.percentile(q1[:, 2], 99).round(3), 'feet z', data.xpos[[mj.body('l_foot').id, mj.body('r_foot').id], 2].round(3))
m = d > 0.01
print('base xy', data.qpos[:2].round(2), 'moved particles x', np.percentile(q1[m, 0], [1, 50, 99]).round(2), 'y', np.percentile(q1[m, 1], [1, 50, 99]).round(2), 'slab x', q0[:, 0].min().round(2), q0[:, 0].max().round(2))
sunk = q1[:, 2] < q0[:, 2] - 0.02
print('sunk >2cm', sunk.sum(), 'x', np.percentile(q1[sunk, 0], [1, 99]).round(2) if sunk.any() else '')
viewer.set_camera(wp.vec3(data.qpos[0] - 0.6, data.qpos[1], 2.2), pitch=-89, yaw=0)
viewer.begin_frame(t); viewer.log_state(state); viewer.log_points("/model/particles", state.particle_q, draw_r, snow_colors); viewer.end_frame()
imageio.imwrite(args.out.replace(".mp4", "_top.png"), viewer.get_frame().numpy())
jp = state.mpm.particle_Jp.numpy(); print('Jp percentiles 1/10/50/90/99', np.percentile(jp, [1, 10, 50, 90, 99]).round(3))
print(f"wrote {args.out}: {len(frames)} frames, walked {data.qpos[0]:.2f} m")
