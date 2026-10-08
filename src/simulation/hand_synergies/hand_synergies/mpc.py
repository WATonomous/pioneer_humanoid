"""Spin a cube in the palm-up hand with sampling MPC (predictive sampling), in joint or synergy space.

No training: every CONTROL_DT the planner perturbs its nominal plan (a few control knots over a short
horizon) N times, rolls each out on a copy of the simulation (mujoco.rollout, multithreaded), and
keeps the cheapest. The cost asks the cube to spin about the palm normal (world +Z) at TARGET_SPIN
while staying in the palm.

The only difference between the action spaces is where the noise lives:
  joint       every one of the 20 joint targets perturbed independently
  synergy-k   perturbations along the first k grasp PCs only (dq = a @ PC[:k]), same expected size

    python -m hand_synergies.mpc --space joint --seconds 10 --gif out/mpc_joint.gif
    python -m hand_synergies.mpc --space synergy-3 --seconds 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout
from pioneer_humanoid.mujoco_hand import JOINT_NAMES, PALM_BODY, hand_spec, joint_ranges

TIMESTEP = 0.002
CONTROL_DT = 0.02          # replan period
HORIZON_S = 0.3
N_KNOTS = 3                # control knots over the horizon, linearly interpolated
N_SAMPLES = 32             # rollouts per plan (incl. the unperturbed nominal)
NOISE_NORM = 0.5           # rad, expected norm of a 20-D perturbation in every space
TARGET_SPIN = 1.0          # rad/s about world +Z
CUBE_HALF = 0.02
CUBE_HOME = np.array([-0.012, 0.10, 0.03])  # resting spot in the palm (world)

# cost weights
W_SPIN = 1.0
W_POS = 400.0              # per m^2 off CUBE_HOME (xy) -- keeps it in the palm
W_DROP = 50.0              # cube below DROP_Z
DROP_Z = 0.0
W_TILT = 1.0               # cube's own axes leaving vertical: spin about Z, not tumble
W_CTRL = 0.01

# Palm-up resting grasp (the Isaac in-hand task's INHAND_GRASP_JOINT_POS with the fingers unspread).
HOME_POSE = {
    "circumduction": 0.0, "MCP_A_thumb": 0.5, "PIP_thumb": -0.8, "DIP_thumb": -0.8,
    **{f"MCP_A_{i}": 0.0 for i in range(1, 5)},
    **{f"MCP_{i}": 0.8 for i in (1, 2, 4)}, "MCP_3": -0.8,
    **{f"PIP_{i}": -0.8 for i in range(1, 5)},
    **{f"DIP_{i}": 0.8 for i in range(1, 5)},
}


def make_model(width: int = 480, height: int = 360) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.modelname = "hand_cube_spin"
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.worldbody.add_light(pos=[0.2, 0.1, 0.6], dir=[-0.3, 0, -1], diffuse=[0.8, 0.8, 0.8])
    # Palm up: the hand frame's palm faces -Z, so turn it 180 deg about Y.
    spec.worldbody.add_frame(quat=[0, 0, 1, 0]).attach_body(hand_spec().body(PALM_BODY), "", "")
    cube = spec.worldbody.add_body(name="cube", pos=CUBE_HOME)
    cube.add_freejoint(name="cube")
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, density=400,
                  friction=[1.0, 0.01, 0.001], rgba=[0.9, 0.55, 0.2, 1])
    # A stripe on one face so the spin reads in renders.
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF * 1.01, 0.004, CUBE_HALF * 1.01],
                  contype=0, conaffinity=0, density=0, rgba=[0.15, 0.15, 0.15, 1])
    spec.add_sensor(name="cube_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_BODY,
                    objname="cube")
    spec.add_sensor(name="cube_angvel", type=mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="cube")
    spec.add_sensor(name="cube_zaxis", type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="cube")
    return spec.compile()


def stage_cost(sens: np.ndarray, ctrl_delta: np.ndarray) -> np.ndarray:
    """Per-step cost from sensordata [..., 9] (pos 3, angvel 3, z-axis 3)."""
    pos, angvel, zaxis = sens[..., 0:3], sens[..., 3:6], sens[..., 6:9]
    # Tilt: the cube's own Z axis (vertical at the start) should stay vertical -- spin, don't tumble.
    tilt = 1.0 - np.abs(zaxis[..., 2])
    return (
        W_SPIN * (angvel[..., 2] - TARGET_SPIN) ** 2
        + W_POS * np.sum((pos[..., :2] - CUBE_HOME[:2]) ** 2, axis=-1)
        + W_DROP * (pos[..., 2] < DROP_Z)
        + W_TILT * tilt
        + W_CTRL * np.sum(ctrl_delta**2, axis=-1)
    )


class Planner:
    def __init__(self, model: mujoco.MjModel, space: str, pca: dict | None, nthread: int, seed: int = 0):
        self.model = model
        self.nstep = int(round(HORIZON_S / TIMESTEP))
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = joint_ranges(model).T
        self.knot_t = np.linspace(0.0, HORIZON_S, N_KNOTS)
        self.step_t = np.arange(self.nstep) * TIMESTEP
        self.datas = [mujoco.MjData(model) for _ in range(nthread)]
        if space == "joint":
            self.basis = np.eye(len(JOINT_NAMES))
        else:
            k = int(space.split("-")[1])
            self.basis = pca["components"][:k]  # (k, 20), orthonormal rows
        # isotropic within the subspace, scaled so E|dq| = NOISE_NORM whatever its dimension
        self.sigma = NOISE_NORM / np.sqrt(len(self.basis))
        self.nominal = None

    def _interp(self, knots: np.ndarray) -> np.ndarray:
        """(..., N_KNOTS, 20) -> (..., nstep, 20) piecewise-linear."""
        idx = np.clip(np.searchsorted(self.knot_t, self.step_t, side="right") - 1, 0, N_KNOTS - 2)
        w = ((self.step_t - self.knot_t[idx]) / (self.knot_t[idx + 1] - self.knot_t[idx]))[:, None]
        return knots[..., idx, :] * (1 - w) + knots[..., idx + 1, :] * w

    def plan(self, data: mujoco.MjData) -> np.ndarray:
        if self.nominal is None:
            self.nominal = np.tile(data.ctrl, (N_KNOTS, 1))
        else:  # shift the previous plan forward by one control period
            t = self.knot_t + CONTROL_DT
            self.nominal = np.stack([np.interp(np.minimum(t, HORIZON_S), self.knot_t, self.nominal[:, j])
                                     for j in range(self.nominal.shape[1])], axis=1)
        noise = self.rng.normal(size=(N_SAMPLES, N_KNOTS, len(self.basis))) * self.sigma @ self.basis
        noise[0] = 0.0
        knots = np.clip(self.nominal + noise, self.lo, self.hi)
        ctrl = self._interp(knots)  # (N, nstep, 20)

        state = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(self.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        _, sens = rollout.rollout(self.model, self.datas, state[None], ctrl, persistent_pool=True)
        dctrl = np.diff(ctrl, axis=1, prepend=ctrl[:, :1]) / TIMESTEP * CONTROL_DT
        cost = stage_cost(sens, dctrl).sum(axis=1)
        best = int(np.argmin(cost))
        self.nominal = knots[best]
        return ctrl[best, 0]


def run(space: str, seconds: float, pca: dict | None, nthread: int = 4, seed: int = 0, gif: str | None = None):
    model = make_model()
    data = mujoco.MjData(model)
    home = np.array([HOME_POSE[n] for n in JOINT_NAMES])
    data.qpos[:20] = home
    data.ctrl[:] = home
    mujoco.mj_step(model, data, nstep=int(0.5 / TIMESTEP))  # let the cube settle in the palm
    planner = Planner(model, space, pca, nthread, seed)
    sub = int(round(CONTROL_DT / TIMESTEP))
    renderer = mujoco.Renderer(model, 360, 480) if gif else None
    frames, yaw, spin_log, dropped_at = [], 0.0, [], None
    t0 = time.time()
    for i in range(int(seconds / CONTROL_DT)):
        data.ctrl[:] = planner.plan(data)
        for _ in range(sub):
            mujoco.mj_step(model, data)
            yaw += data.sensordata[5] * TIMESTEP
        spin_log.append(data.sensordata[5])
        if dropped_at is None and data.sensordata[2] < DROP_Z:
            dropped_at = data.time
        if renderer is not None and i % 3 == 0:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [-0.012, 0.10, 0.02]
            cam.distance, cam.azimuth, cam.elevation = 0.32, 90, -50
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    wall = time.time() - t0
    if renderer is not None:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(CONTROL_DT * 3 * 1000), loop=0)
        renderer.close()
    return dict(space=space, yaw=yaw, mean_spin=float(np.mean(spin_log)), dropped_at=dropped_at,
                realtime=seconds / wall)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--space", default="joint", help="joint | synergy-<k>")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--synergies", default="out/synergies.npz")
    p.add_argument("--gif", default=None)
    p.add_argument("--noise", type=float, default=NOISE_NORM, help="rad, expected perturbation norm")
    p.add_argument("--horizon", type=float, default=HORIZON_S, help="s")
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    args = p.parse_args()
    globals().update(NOISE_NORM=args.noise, HORIZON_S=args.horizon, N_SAMPLES=args.samples)
    pca = dict(np.load(args.synergies)) if args.space != "joint" else None
    r = run(args.space, args.seconds, pca, args.threads, args.seed, args.gif)
    drop = f"dropped at {r['dropped_at']:.1f} s" if r["dropped_at"] is not None else "never dropped"
    print(f"{r['space']} noise={NOISE_NORM} horizon={HORIZON_S} samples={N_SAMPLES}: turned {np.degrees(r['yaw']):.0f} deg in {args.seconds:.0f} s "
          f"(mean {r['mean_spin']:.2f} rad/s, target {TARGET_SPIN}), {drop}, {r['realtime']:.2f}x real time")


if __name__ == "__main__":
    main()
