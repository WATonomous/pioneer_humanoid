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
p.add_argument("--material", choices=["snow", "sand"], default="snow")
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


