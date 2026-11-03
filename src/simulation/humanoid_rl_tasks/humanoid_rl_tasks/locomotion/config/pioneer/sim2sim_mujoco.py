"""Run a walking policy trained in Isaac Lab on the Pioneer humanoid in MuJoCo (sim2sim).

    python sim2sim_mujoco.py models/pioneer_leg_only_flat/model_2999.pt --vx 0.5 --video walk.mp4
    python sim2sim_mujoco.py models/pioneer_whole_body_flat/model_2299.pt --whole_body --vx 0.5

The same robot as Isaac Lab trains (pioneer_humanoid/leg_only.py, whole_body.py): the URDF re-rooted at the hips with +X
forward, the same PD gains and torque limits, no self-collision, a policy step every 20 ms. The observation is
flat_env_cfg.py's; target = default + 0.25 * action. With --whole_body, the arms swing as in arm_swing.py and the hands
are fists. Takes an rsl_rl checkpoint (model_<n>.pt) directly.
"""
import argparse
import math
import os
import re

import mujoco
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))
HIP_HEIGHT = 0.847917
JOINTS = ["Hip_F_L", "Hip_F_R", "Hip_A_L", "Hip_A_R", "Hip_R_L", "Hip_R_R", "Knee_L", "Knee_R", "Ankle_P_L", "Ankle_P_R", "Ankle_R_L", "Ankle_R_R"]
DEFAULT = {"Hip_F": 0.25, "Knee": -0.5, "Ankle_P": -0.25}
KP = {"Hip_F": 200.0, "Hip_A": 150.0, "Hip_R": 150.0, "Knee": 200.0, "Ankle_P": 40.0, "Ankle_R": 40.0}
KD = {"Hip_F": 5.0, "Hip_A": 5.0, "Hip_R": 5.0, "Knee": 5.0, "Ankle_P": 2.0, "Ankle_R": 2.0}
EFFORT = {"Hip_F": 222.0, "Hip_A": 120.0, "Hip_R": 60.0, "Knee": 222.0, "Ankle_P": 60.0, "Ankle_R": 60.0}
ARMS = {"l_shoulder_pitch": 53.0, "r_shoulder_pitch": 53.0, "l_elbow": 22.0, "r_elbow": 22.0}  # effort; PD 40 / 2
FIST = {
    f"{side}_{finger}{joint}": angle
    for side in ("l", "r")
    for finger in ("", "i_", "r_", "l_")
    for joint, angle in (("mcp", 1.4), ("pip", 1.5), ("dip", 1.2))
} | {f"{side}_th_{joint}": angle for side in ("l", "r") for joint, angle in (("c", 0.6), ("f", 0.9), ("mcp", 0.8), ("ip", 0.9))}
PERIOD, STANCE = 1.0, 0.55  # gait.py
SUBSTEPS = 16  # physics steps per policy step; at Isaac's 5 ms MuJoCo's stiff PD + contacts mis-integrate


def _bake(src: str, joint: str, angle: float) -> str:
    """Fix `joint` at `angle`: rotate its origin about its axis (as whole_body.py does for the fists)."""
    block = re.search(rf'<joint name="{joint}" type="[^"]*">.*?</joint>', src, flags=re.S).group(0)
    x, y, z = (float(v) for v in re.search(r'<axis xyz="([^"]*)"', block).group(1).split())
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([x, y, z]) / math.sqrt(x * x + y * y + z * z), angle)
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, q)
    r = r.reshape(3, 3)
    rpy = (math.atan2(r[2, 1], r[2, 2]), math.asin(-np.clip(r[2, 0], -1, 1)), math.atan2(r[1, 0], r[0, 0]))
    baked = re.sub(r'(<origin xyz="[^"]*")', rf'\1 rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"', block, count=1)
    return src.replace(block, baked)


def build(whole_body: bool) -> mujoco.MjModel:
    robot = "pioneer_whole_body_humanoid" if whole_body else "pioneer_leg_only_humanoid"
    asset = os.path.join(REPO, "assets", robot)
    src = open(os.path.join(asset, f"{robot}.urdf")).read()
    src = re.sub(r'filename="([^"/][^"]*)"', lambda m: f'filename="{os.path.join(asset, m.group(1))}"', src)
    if whole_body:  # everything above the hips rigid, but the shoulders and elbows; fingers fixed curled
        for joint, angle in FIST.items():
            src = _bake(src, joint, angle)
        free = "|".join(f'{j}"' for j in JOINTS + list(ARMS))
        src = re.sub(rf'(<joint name="(?!{free})[^"]*" type=)"(revolute|prismatic|continuous)"', r'\1"fixed"', src)
    src = src.replace('<link name="base_link">', '<link name="base"/><joint name="base_to_base_link" type="fixed"><parent link="base"/>'
                      f'<child link="base_link"/><origin xyz="0 0 {-HIP_HEIGHT}" rpy="0 0 1.5707963"/></joint><link name="base_link">', 1)
    src = re.sub(r"<mujoco>.*?</mujoco>", '<mujoco><compiler balanceinertia="true" discardvisual="false"/></mujoco>', src, flags=re.S)
    spec = mujoco.MjSpec.from_string(src)
    for geom in spec.geoms:  # as Isaac Lab trains: no self-collision; the CAD meshes are only drawn
        mesh = geom.type == mujoco.mjtGeom.mjGEOM_MESH
        geom.contype, geom.conaffinity, geom.group = 0, 0 if mesh else 1, 1 if mesh else 3
    spec.option.timestep = 0.02 / SUBSTEPS
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.first_body().add_freejoint()
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.47, 0.47, 0.48], rgb2=[0.40, 0.40, 0.41], width=512, height=512)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.75, 0.78, 0.82], rgb2=[0.45, 0.48, 0.52], width=512, height=512)
    spec.add_material(name="grid", texrepeat=[20, 20], textures=["", "grid"])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.4, 0.4, 0.4], [0.5, 0.5, 0.5]
    for name in JOINTS + (list(ARMS) if whole_body else []):
        kp, kd, effort = (KP[name[:-2]], KD[name[:-2]], EFFORT[name[:-2]]) if name in JOINTS else (40.0, 2.0, ARMS[name])
        joint = spec.joint(name)
        joint.armature, joint.damping = 0.01, [0.0, 0.0, 0.0]
        spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=[kp] + [0] * 9, biasprm=[0, -kp, -kd] + [0] * 7,
                          forcelimited=True, forcerange=[-effort, effort])
    return spec.compile()


def load_actor(path: str) -> torch.nn.Module:
    """The actor MLP (ELU, no observation normalisation) of an rsl_rl checkpoint."""
    state = torch.load(path, map_location="cpu")["model_state_dict"]
    weights = sorted((k for k in state if k.startswith("actor.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    layers = []
    for i, k in enumerate(weights):
        linear = torch.nn.Linear(*reversed(state[k].shape))
        linear.weight.data, linear.bias.data = state[k], state[k.replace("weight", "bias")]
        layers += [linear] + ([torch.nn.ELU()] if i < len(weights) - 1 else [])
    return torch.nn.Sequential(*layers).eval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("--whole_body", action="store_true")
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--push", type=float, default=0.0, help="every 3 s, kick the base by up to this horizontal speed (m/s)")
    p.add_argument("--seed", type=int, default=None, help="perturb the start: joints +-0.05 rad, base velocity +-0.2 m/s")
    p.add_argument("--video", default=None, help="mp4 path")
    args = p.parse_args()

    model = build(args.whole_body)
    data = mujoco.MjData(model)
    policy = load_actor(args.checkpoint)
    qadr = np.array([model.joint(j).qposadr[0] for j in JOINTS])
    vadr = np.array([model.joint(j).dofadr[0] for j in JOINTS])
    act = np.array([model.actuator(j).id for j in JOINTS])
    default = np.array([DEFAULT.get(j[:-2], 0.0) for j in JOINTS])
    data.qpos[2] = 0.84
    data.qpos[qadr] = default
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        data.qpos[qadr] += rng.uniform(-0.05, 0.05, 12)
        data.qvel[:2] = rng.uniform(-0.2, 0.2, 2)
    mujoco.mj_forward(model, data)
    command = np.array([args.vx, args.vy, args.wz])
    action = np.zeros(12)
    renderer, frames = (mujoco.Renderer(model, 480, 640), []) if args.video else (None, None)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 2.6, -15, 120
    start, forward, lateral = data.qpos[:2].copy(), 0.0, 0.0
    for step in range(int(args.seconds / 0.02)):
        rot = data.xmat[1].reshape(3, 3)
        ang = data.qvel[3:6]  # the free joint's angular velocity is in the body frame
        lin = rot.T @ data.qvel[:3] + np.cross(ang, model.body_ipos[1])  # Isaac Lab reports the base's centre-of-mass velocity
        forward += lin[0] * 0.02
        lateral += lin[1] * 0.02
        phase = step * 0.02 % PERIOD / PERIOD  # the left leg's
        obs = np.concatenate([lin, ang, rot.T @ np.array([0, 0, -1.0]), command, data.qpos[qadr] - default, data.qvel[vadr], action,
                              [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]])
        with torch.no_grad():
            action = policy(torch.from_numpy(obs.astype(np.float32))[None]).numpy()[0].astype(float)
        data.ctrl[act] = default + 0.25 * action
        if args.whole_body:  # arm_swing.py
            swing = 0.35 * min(abs(command[0]) / 0.8, 1.0) * math.sin(2 * math.pi * (phase - STANCE / 2 + 0.25))
            for name, target in (("l_shoulder_pitch", -swing), ("r_shoulder_pitch", swing), ("l_elbow", -0.3), ("r_elbow", -0.3)):
                data.ctrl[model.actuator(name).id] = target
        if args.push and step > 0 and step % 150 == 0:
            data.qvel[:2] += rng.uniform(-args.push, args.push, 2)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        if renderer is not None and step % 2 == 0:
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera)
            frames.append(renderer.render())
        if data.qpos[2] < 0.4 or not np.isfinite(data.qpos).all():
            print(f"fell at {data.time:.2f} s")
            break
    moved = data.qpos[:2] - start
    print(f"{data.time:.1f} s, moved {moved.round(2)} m (asked {command[0] * data.time:.2f} m forward), "
          f"mean speed forward {forward / data.time:.2f} m/s, lateral {lateral / data.time:.2f} m/s")
    if renderer is not None:
        import imageio

        imageio.mimwrite(args.video, frames, fps=25)


if __name__ == "__main__":
    main()
