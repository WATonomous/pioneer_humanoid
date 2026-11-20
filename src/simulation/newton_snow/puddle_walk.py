"""Pioneer whole-body humanoid wading along a flooded trail: MPM muddy water (Newton implicit MPM, the dam-break
water plus a 200 Pa yield stress), its surface extracted as a mesh every frame and path traced (OVRTX) over a background USD (puddle_scene.py).

Walking runs in MuJoCo as snow_walk.py; the feet are kinematic colliders for the water (one-way coupling). The water is
held in a box (x0..x1, y0..y1); the background's banks cover the box walls, so the visible shoreline is where they cross.

  python puddle_walk.py <policy.pt> <out.mp4> --bg out/puddle.usda [--seconds 12] [--vx 0.4] [--depth 0.07]
"""
import argparse
import sys

import imageio
import mujoco
import numpy as np
import torch
import warp as wp

import newton
import newton.viewer
from newton.solvers import SolverImplicitMPM

sys.argv, ARGV = sys.argv[:2] + ["_", "--whole_body"], sys.argv  # reuse snow_walk.py's robot builders without its CLI
import snow_walk_lib as lib  # noqa: E402

sys.argv = ARGV
p = argparse.ArgumentParser()
p.add_argument("policy")
p.add_argument("out")
p.add_argument("--bg", required=True)
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--vx", type=float, default=0.4)
p.add_argument("--depth", type=float, default=0.07)
p.add_argument("--dx", type=float, default=0.03)
p.add_argument("--ppc", type=float, default=2.0)
p.add_argument("--box", type=float, nargs=4, default=[-1.0, 5.8, -1.2, 1.2], help="water box x0 x1 y0 y1 (m)")
p.add_argument("--spp", type=int, default=16)
p.add_argument("--size", type=int, nargs=2, default=[1280, 720])
p.add_argument("--color", type=float, nargs=3, default=[0.14, 0.09, 0.05])
p.add_argument("--opacity", type=float, default=1.0)
p.add_argument("--yield_stress", type=float, default=200.0, help="Pa; pure water (0) slowly loses depth in this thin layer, mud holds it")
p.add_argument("--viscosity", type=float, default=1.0)
args = p.parse_args()

mj, data = lib.mj, lib.data
builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
mj_of_newton = lib.robot_from_mujoco(builder)
builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0))
SolverImplicitMPM.register_custom_attributes(builder)
x0, x1, y0, y1 = args.box
lo, hi = np.array([x0, y0, 0.0]), np.array([x1, y1, args.depth])
res = np.ceil(args.ppc * (hi - lo) / args.dx).astype(int)
cell = (hi - lo) / res
builder.add_particle_grid(pos=wp.vec3(lo + 0.5 * cell), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=int(res[0]), dim_y=int(res[1]),
                          dim_z=int(res[2]), cell_x=float(cell[0]), cell_y=float(cell[1]), cell_z=float(cell[2]),
                          mass=float(np.prod(cell) * 1000.0), jitter=float(cell.max()), radius_mean=float(0.5 * cell.max()),
                          flags=newton.ParticleFlags.ACTIVE)
model = builder.finalize()
print(f"water particles {model.particle_count}")
model.mpm.friction.fill_(0.0)
model.mpm.tensile_yield_ratio.fill_(1.0)
model.mpm.viscosity.fill_(args.viscosity)
model.mpm.yield_stress.fill_(args.yield_stress)
cfg = SolverImplicitMPM.Config()
cfg.voxel_size = args.dx
cfg.max_iterations = 50
cfg.tolerance = 1.0e-3
cfg.warmstart_mode = "particles"
cfg.strain_basis = "P0"
cfg.collider_basis = "pic"
cfg.velocity_basis = "Q1"
cfg.collider_velocity_mode = "backward"
mpm = SolverImplicitMPM(model, config=cfg)
state = model.state()
state.mpm.particle_Jp.fill_(1.0)


@wp.kernel
def keep_in_box(q: wp.array[wp.vec3], v: wp.array[wp.vec3], lo: wp.vec3, hi: wp.vec3):
    i = wp.tid()
    p = wp.vec3(q[i])
    u = wp.vec3(v[i])
    for k in range(2):
        if p[k] < lo[k]:
            p[k] = lo[k]
            u[k] = wp.max(u[k], 0.0)
        elif p[k] > hi[k]:
            p[k] = hi[k]
            u[k] = wp.min(u[k], 0.0)
    if p[2] < 0.0:
        p[2] = 0.0
        u[2] = wp.max(u[2], 0.0)
    q[i] = p
    v[i] = u


def sync_bodies():
    pose = np.concatenate([data.xpos[mj_of_newton], data.xquat[mj_of_newton][:, [1, 2, 3, 0]]], 1).astype(np.float32)
    state.body_q.assign(wp.array(pose, dtype=wp.transform))


sync_bodies()
mpm.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=state.body_q)
mpm.project_outside(state, state, 0.005)
spacing = float(cell.max())
surface = mpm.create_particle_surface(voxel_size=0.6 * args.dx, max_grid_cells=3_000_000, kernel_radius=max(3.0 * spacing, 0.45 * args.dx),
                                      threshold=0.25, smooth_lambda=0.0, anisotropic=True, kernel_scale=0.6, anisotropy_ratio=16.0,
                                      anisotropy_scale=2.0, anisotropy_min_neighbors=4, anisotropy_binning=True, anisotropy_strength=0.95,
                                      field_smooth_iterations=1, field_smooth_radius=2, field_mode="density", redistance_iterations=0,
                                      mesh_smooth_iterations=1)

viewer = newton.viewer.ViewerRTX(width=args.size[0], height=args.size[1], headless=True, async_rendering=False, environment="none",
                                 render_settings={"omni:rtx:pt:samplesPerPixel": ("uint", args.spp),
                                                  "omni:rtx:pt:denoising:enabled": ("bool", True)})
viewer.add_background_usd(args.bg)
viewer.set_model(model)
viewer.show_particles = False
bound = False


def bind_water():
    """Wet mud: opaque brown with a soft sheen (UsdPreviewSurface with ior and clearcoat)."""
    from pxr import Sdf, UsdShade  # noqa: PLC0415
    prim = viewer.stage.GetPrimAtPath("/root/model/water")
    if not prim:
        return False
    mat = UsdShade.Material.Define(viewer.stage, "/root/Materials/Water")
    sh = UsdShade.Shader.Define(viewer.stage, "/root/Materials/Water/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    for k, t, v in (("diffuseColor", Sdf.ValueTypeNames.Color3f, tuple(args.color)), ("roughness", Sdf.ValueTypeNames.Float, 0.35),
                    ("opacity", Sdf.ValueTypeNames.Float, args.opacity), ("opacityThreshold", Sdf.ValueTypeNames.Float, 0.0),
                    ("ior", Sdf.ValueTypeNames.Float, 1.333), ("clearcoat", Sdf.ValueTypeNames.Float, 0.35),
                    ("clearcoatRoughness", Sdf.ValueTypeNames.Float, 0.2)):
        sh.CreateInput(k, t).Set(v)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    return True


command = np.array([args.vx, 0.0, 0.0])
qadr, vadr, act_id, default = lib.qadr, lib.vadr, lib.act_id, lib.default
action = np.zeros(12)
frames, t, peak = [], 0.0, 0.0
for step in range(int(args.seconds / 0.02)):
    rot = data.xmat[1].reshape(3, 3)
    ang = data.qvel[3:6]
    lin = rot.T @ data.qvel[:3] + np.cross(ang, mj.body_ipos[1])
    phase = step * 0.02 % 1.0
    obs = np.concatenate([lin, ang, rot.T @ np.array([0, 0, -1.0]), command, data.qpos[qadr] - default, data.qvel[vadr], action,
                          [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]])
    with torch.no_grad():
        action = lib.policy(torch.from_numpy(obs.astype(np.float32))[None]).numpy()[0].astype(float)
    data.ctrl[act_id] = default + 0.25 * action
    swing = 0.35 * min(abs(command[0]) / 0.8, 1.0) * np.sin(2 * np.pi * (phase - 0.55 / 2 + 0.25))
    for name, target in (("l_shoulder_pitch", -swing), ("r_shoulder_pitch", swing), ("l_elbow", -0.3), ("r_elbow", -0.3)):
        data.ctrl[mj.actuator(name).id] = target
    for _ in range(4):  # water in 5 ms steps, the feet posed from MuJoCo at each
        for _ in range(lib.SUB // 4):
            mujoco.mj_step(mj, data)
        sync_bodies()
        mpm.step(state, state, contacts=None, control=None, dt=0.005)
        mpm.project_outside(state, state, 0.005)
        wp.launch(keep_in_box, dim=model.particle_count, inputs=[state.particle_q, state.particle_qd, wp.vec3(lo), wp.vec3(hi)])
    t += 0.02
    if data.qpos[2] < 0.4:
        print(f"fell at {t:.2f} s")
        break
    if step % 2:
        continue
    base = data.qpos[:3].copy()
    if t < args.seconds / 2:  # low, ahead and to the right, looking back at the robot
        viewer.set_camera(wp.vec3(base[0] + 2.5, base[1] - 2.0, 0.9), pitch=-6, yaw=141)
    else:  # low side shot travelling with it
        viewer.set_camera(wp.vec3(base[0] + 0.5, base[1] - 3.0, 0.8), pitch=-5, yaw=99)
    viewer.begin_frame(t)
    viewer.log_state(state)
    verts, idx, normals = mpm.extract_particle_surface(state, surface).to_arrays()
    viewer.log_mesh("/model/water", verts, idx, normals, dynamic=True, color=tuple(args.color), roughness=0.35, metallic=0.0)
    bound = bound or bind_water()
    viewer.end_frame()
    frames.append(viewer.get_frame().numpy())
    z = state.particle_q.numpy()[:, 2]
    peak = max(peak, float(z.max()))
    if step % 50 == 0:
        print(f"t {t:.1f} s, base x {base[0]:.2f} m, water top {np.percentile(z, 99.9):.3f} max {z.max():.3f}", flush=True)

imageio.mimsave(args.out, frames, fps=25, macro_block_size=8)
for k in (len(frames) // 3, 2 * len(frames) // 3, -1):
    imageio.imwrite(args.out.replace(".mp4", f"_{k % len(frames)}.png"), frames[k])
print(f"wrote {args.out}: {len(frames)} frames, walked {data.qpos[0]:.2f} m, highest splash {peak:.3f} m")
