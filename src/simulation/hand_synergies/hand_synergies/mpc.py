"""Manipulate a cube in the palm-up hand with sampling MPC (predictive sampling), in joint or synergy space.

No training: every CONTROL_DT the planner perturbs its nominal plan (a few control knots over a short
horizon) N times, rolls each out on a copy of the simulation (mujoco.rollout, multithreaded), and
keeps the cheapest. Two tasks (--task):
  spin   spin the cube about the palm normal (world +Z) at TARGET_SPIN while it stays in the palm
  yaw    the Isaac in-hand task: turn the cube to a random goal yaw (about +Z), success when the
         orientation error is < YAW_SUCCESS rad, then a new goal (Isaac resamples on success too)
  roll   a ball instead of the cube: roll it to random target spots on the palm, success within
         ROLL_SUCCESS m, then a new target

The only difference between the action spaces is where the noise lives:
  joint       every one of the 20 joint targets perturbed independently
  synergy-k   perturbations along the first k grasp PCs only (dq = a @ PC[:k]), same expected size

    python -m hand_synergies.mpc --space joint --seconds 10 --gif out/mpc_joint.gif
    python -m hand_synergies.mpc --space synergy-3 --seconds 10
    python -m hand_synergies.mpc --task yaw --seconds 30 --gif out/mpc_yaw.gif
    python -m hand_synergies.mpc --task roll --seconds 30 --gif out/mpc_roll.gif
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout
from pioneer_humanoid.mujoco_hand import JOINT_NAMES, PALM_BODY, hand_spec, joint_ranges

from .scene import stiffen_contacts

TIMESTEP = 0.002
CONTROL_DT = 0.02          # replan period
HORIZON_S = 0.3
N_KNOTS = 3                # control knots over the horizon, linearly interpolated
N_SAMPLES = 32             # rollouts per plan (incl. the unperturbed nominal)
NOISE_NORM = 0.5           # rad, expected norm of a 20-D perturbation in every space
TARGET_SPIN = 1.0          # rad/s about world +Z
CUBE_HALF = 0.02
CUBE_HOME = np.array([-0.012, 0.10, 0.03])  # resting spot in the palm (world)

TASK = "spin"
YAW_SUCCESS = 0.4          # rad, the Isaac task's success threshold
GOAL_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
GOAL_MARKER_OFFSET = np.array([0.11, 0.0, 0.01])  # the goal-orientation ghost cube, beside the hand
BALL_RADIUS = 0.015
ROLL_SUCCESS = 0.01        # m
ROLL_MIN_MOVE = 0.015      # m, a new target is at least this far from the ball
ROLL_TARGET_BOX = 0.02     # targets are uniform within +-this (m, xy) of where the ball settled
TARGET_XY = np.zeros(2)
W_ROLL = 2000.0            # roll task: per m^2 from the target (1 cm -> 0.2)
W_ROLL_VEL = 0.5           # roll task: per (m/s)^2 of ball speed -- arrive and stop
W_ROLL_LEAVE = 20.0        # roll task: ball more than ROLL_LEAVE_R from where it settled (about to fall off)
ROLL_LEAVE_R = 0.035

# cost weights
W_ORI = 1.0                # yaw task: per rad of orientation error
# yaw task: track a spin toward the goal, YAW_GAIN x the signed yaw error capped at TARGET_SPIN. The
# orientation error alone barely changes over one 0.3 s horizon, so the planner just held still.
YAW_GAIN = 2.0
W_SPIN = 1.0
W_POS = 400.0              # per m^2 off CUBE_HOME (xy) -- keeps it in the palm
W_DROP = 50.0              # cube below DROP_Z
DROP_Z = 0.0
W_TILT = 1.0               # cube's own axes leaving vertical: spin about Z, not tumble
W_CTRL = 0.01

# --smooth: MPPI (cost-weighted average of all samples, not the single best) with temperature
# MPPI_TEMP x the cost spread, the jump from the command being executed is penalized too, and
# W_CTRL_SMOOTH replaces W_CTRL. Picking the single best sample every 20 ms makes the fingers jerk.
SMOOTH = False
MPPI_TEMP = 0.1
W_CTRL_SMOOTH = 0.5
# Low-pass on the executed command (any mode): ctrl = (1 - a) * previous + a * planned; 1.0 = off.
FILTER_ALPHA = 1.0

# Palm-up resting grasp (the Isaac in-hand task's INHAND_GRASP_JOINT_POS with the fingers unspread).
HOME_POSE = {
    "circumduction": 0.0, "MCP_A_thumb": 0.5, "PIP_thumb": -0.8, "DIP_thumb": -0.8,
    **{f"MCP_A_{i}": 0.0 for i in range(1, 5)},
    **{f"MCP_{i}": 0.8 for i in (1, 2, 4)}, "MCP_3": -0.8,
    **{f"PIP_{i}": -0.8 for i in range(1, 5)},
    **{f"DIP_{i}": 0.8 for i in range(1, 5)},
}


def make_model(width: int = 480, height: int = 360, thumb: str = "stock",
               finger_collisions: bool = False, ball: bool = False) -> mujoco.MjModel:
    """Palm-up hand with a free object named "cube" (a ball of BALL_RADIUS if ``ball``)."""
    from .thumb import thumb_kwargs

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
    spec.worldbody.add_frame(quat=[0, 0, 1, 0]).attach_body(
        hand_spec(**thumb_kwargs(thumb), finger_collisions=finger_collisions).body(PALM_BODY), "", "")
    cube = spec.worldbody.add_body(name="cube", pos=CUBE_HOME)
    cube.add_freejoint(name="cube")
    if ball:
        cube.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], density=400,
                      friction=[1.0, 0.01, 0.001], rgba=[0.2, 0.5, 0.9, 1])
    else:
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
    spec.add_sensor(name="cube_quat", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="cube")
    spec.add_sensor(name="cube_linvel", type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="cube")
    # Goal ghost for renders: a mocap cube that never collides (hidden in the spin task).
    goal = spec.worldbody.add_body(name="goal", mocap=True, pos=CUBE_HOME + GOAL_MARKER_OFFSET)
    if ball:  # target spot: a flat green disc
        goal.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ROLL_SUCCESS, 0.001, 0], contype=0,
                      conaffinity=0, rgba=[0.3, 0.9, 0.4, 0.0])
    else:
        goal.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, contype=0, conaffinity=0,
                      rgba=[0.3, 0.8, 0.4, 0.0])
        goal.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF * 1.01, 0.004, CUBE_HALF * 1.01],
                      contype=0, conaffinity=0, rgba=[0.15, 0.15, 0.15, 0.0])
    stiffen_contacts(spec)
    return spec.compile()


def orientation_error(quat: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) between unit quaternions [..., 4] and ``goal``."""
    return 2.0 * np.arccos(np.clip(np.abs(quat @ goal), 0.0, 1.0))


def yaw_error(quat: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Signed rotation about world +Z (rad, in [-pi, pi]) that takes ``quat`` [..., 4] toward ``goal``:
    the Z part of goal * conj(quat)."""
    w1, x1, y1, z1 = goal
    w2, x2, y2, z2 = quat[..., 0], -quat[..., 1], -quat[..., 2], -quat[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.angle(np.exp(2j * np.arctan2(z, w)))


def stage_cost(sens: np.ndarray, ctrl_delta: np.ndarray) -> np.ndarray:
    """Per-step cost from sensordata [..., 16] (pos 3, angvel 3, z-axis 3, quat 4, linvel 3)."""
    pos, angvel, zaxis, quat = sens[..., 0:3], sens[..., 3:6], sens[..., 6:9], sens[..., 9:13]
    if TASK == "roll":
        return (
            W_ROLL * np.sum((pos[..., :2] - TARGET_XY) ** 2, axis=-1)
            + W_ROLL_VEL * np.sum(sens[..., 13:16] ** 2, axis=-1)
            + W_ROLL_LEAVE * (np.linalg.norm(pos[..., :2] - CUBE_HOME[:2], axis=-1) > ROLL_LEAVE_R)
            + W_DROP * (pos[..., 2] < DROP_Z)
            + (W_CTRL_SMOOTH if SMOOTH else W_CTRL) * np.sum(ctrl_delta**2, axis=-1)
        )
    common = (
        W_POS * np.sum((pos[..., :2] - CUBE_HOME[:2]) ** 2, axis=-1)
        + W_DROP * (pos[..., 2] < DROP_Z)
        + (W_CTRL_SMOOTH if SMOOTH else W_CTRL) * np.sum(ctrl_delta**2, axis=-1)
    )
    # Tilt: the cube's own Z axis (vertical at the start) should stay vertical -- spin, don't tumble.
    tilt = 1.0 - np.abs(zaxis[..., 2])
    if TASK == "yaw":
        spin_ref = np.clip(YAW_GAIN * yaw_error(quat, GOAL_QUAT), -TARGET_SPIN, TARGET_SPIN)
        return (W_SPIN * (angvel[..., 2] - spin_ref) ** 2 + W_ORI * orientation_error(quat, GOAL_QUAT)
                + W_TILT * tilt + common)
    return W_SPIN * (angvel[..., 2] - TARGET_SPIN) ** 2 + W_TILT * tilt + common


# Thumb poses (circumduction, MCP_A, PIP, DIP) tried in order for the start: the first that leaves the cube
# resting on the hand wins. A moved thumb mount can make HOME_POSE's thumb knock the cube off.
THUMB_HOME_CANDIDATES = (
    (0.0, 0.5, -0.8, -0.8),
    (0.0, 0.0, -0.8, -0.8),
    (0.0, 1.0, -0.8, -0.8),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.5, 0.0, 0.0),
    (0.0, 2.0, 0.0, 0.0),   # thumb swung fully over: needed when the base sits near the knuckles ("fwd")
)
DROP_HEIGHT = 0.03  # the cube starts this far above CUBE_HOME and settles wherever the hand lets it
SETTLE_S = 0.8


def _settled_home(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Reset to a home pose with the cube dropped onto the palm and settled.

    Returns (home joint targets, settled cube position). The settled position becomes the "stay here"
    target of the cost, so a thumb mount that takes up part of the palm isn't charged for it.
    """
    cube_q = model.joint("cube").qposadr[0]
    for thumb in THUMB_HOME_CANDIDATES:
        mujoco.mj_resetData(model, data)
        home = np.array([HOME_POSE[n] for n in JOINT_NAMES])
        home[:4] = thumb
        data.qpos[:20] = home
        data.ctrl[:] = home
        data.qpos[cube_q:cube_q + 3] = CUBE_HOME + [0, 0, DROP_HEIGHT]
        mujoco.mj_forward(model, data)
        if any(c.dist < 0 for c in data.contact[: data.ncon]):
            continue
        mujoco.mj_step(model, data, nstep=int(SETTLE_S / TIMESTEP))
        pos = data.sensordata[:3].copy()
        speed = np.linalg.norm(data.qvel[model.joint("cube").dofadr[0]:][:6])
        if pos[2] > DROP_Z + 0.01 and np.linalg.norm(pos[:2] - CUBE_HOME[:2]) < 0.04 and speed < 0.05:
            return home, pos
    raise RuntimeError("no thumb home pose keeps the cube on the hand")


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
        # change per control period; with SMOOTH the first step counts the jump from the current command
        first = np.broadcast_to(data.ctrl, ctrl[:, :1].shape) if SMOOTH else ctrl[:, :1]
        dctrl = np.diff(ctrl, axis=1, prepend=first) / TIMESTEP * CONTROL_DT
        cost = stage_cost(sens, dctrl).sum(axis=1)
        if not SMOOTH:
            best = int(np.argmin(cost))
            self.nominal = knots[best]
            return ctrl[best, 0]
        spread = max(float(np.std(cost)), 1e-9)
        w = np.exp(-(cost - cost.min()) / (MPPI_TEMP * spread))
        self.nominal = np.einsum("n,nkj->kj", w / w.sum(), knots)
        return self.nominal[0].copy()


def run(space: str, seconds: float, pca: dict | None, nthread: int = 4, seed: int = 0, gif: str | None = None,
        thumb: str = "stock", finger_collisions: bool = False):
    model = make_model(thumb=thumb, finger_collisions=finger_collisions, ball=TASK == "roll")
    if TASK == "roll":
        globals()["DROP_Z"] = -0.02  # the ball rests in the cup of the palm, lower than the cube
    data = mujoco.MjData(model)
    home, cube_rest = _settled_home(model, data)
    globals()["CUBE_HOME"] = cube_rest  # stage_cost keeps the cube near where it settled
    planner = Planner(model, space, pca, nthread, seed)
    sub = int(round(CONTROL_DT / TIMESTEP))
    renderer = mujoco.Renderer(model, 360, 480) if gif else None
    frames, yaw, spin_log, dropped_at, ctrl_log = [], 0.0, [], None, []
    goal_rng = np.random.default_rng([seed, 7])
    cube_q0 = data.sensordata[9:13].copy()  # settled orientation: goals are yaws of it
    goal_log, err_log, success_times = [], [], []

    def new_goal():
        while True:  # at least 2 x YAW_SUCCESS from the cube, so a new goal is never already reached
            half = goal_rng.uniform(-np.pi, np.pi) / 2
            qz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            g = np.empty(4)
            mujoco.mju_mulQuat(g, qz, cube_q0)
            if orientation_error(data.sensordata[9:13], g) >= 2 * YAW_SUCCESS:
                break
        globals()["GOAL_QUAT"] = g
        mid = model.body("goal").mocapid[0]
        data.mocap_pos[mid] = cube_rest + GOAL_MARKER_OFFSET
        data.mocap_quat[mid] = g
        goal_log.append(g)

    def new_target():
        while True:  # at least ROLL_MIN_MOVE from the ball, so a new target is never already reached
            xy = cube_rest[:2] + goal_rng.uniform(-ROLL_TARGET_BOX, ROLL_TARGET_BOX, size=2)
            if np.linalg.norm(xy - data.sensordata[:2]) >= ROLL_MIN_MOVE:
                break
        globals()["TARGET_XY"] = xy
        mid = model.body("goal").mocapid[0]
        data.mocap_pos[mid] = [xy[0], xy[1], cube_rest[2] - BALL_RADIUS + 0.001]
        data.mocap_quat[mid] = [1, 0, 0, 0]
        goal_log.append(xy)

    if TASK in ("yaw", "roll"):
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == model.body("goal").id:
                model.geom_rgba[gid, 3] = 0.45 if TASK == "yaw" else 0.8
        new_goal() if TASK == "yaw" else new_target()
    t0 = time.time()
    for i in range(int(seconds / CONTROL_DT)):
        data.ctrl[:] = (1 - FILTER_ALPHA) * data.ctrl + FILTER_ALPHA * planner.plan(data)
        ctrl_log.append(data.ctrl.copy())
        for _ in range(sub):
            mujoco.mj_step(model, data)
            yaw += data.sensordata[5] * TIMESTEP
        spin_log.append(data.sensordata[5])
        if data.sensordata[2] < DROP_Z:  # off the hand: the episode is over (a falling cube still "spins")
            dropped_at = data.time
            break
        if TASK == "yaw":
            err = float(orientation_error(data.sensordata[9:13], GOAL_QUAT))
            err_log.append(err)
            if err < YAW_SUCCESS:
                success_times.append(len(spin_log) * CONTROL_DT)
                new_goal()
        elif TASK == "roll":
            err = float(np.linalg.norm(data.sensordata[:2] - TARGET_XY))
            err_log.append(err)
            if err < ROLL_SUCCESS:
                success_times.append(len(spin_log) * CONTROL_DT)
                new_target()
        if renderer is not None and i % 3 == 0:
            cam = mujoco.MjvCamera()
            if TASK == "roll":
                cam.lookat[:] = cube_rest
                cam.distance, cam.azimuth, cam.elevation = 0.2, 90, -65
            else:
                cam.lookat[:] = [0.03, 0.10, 0.02] if TASK == "yaw" else [-0.012, 0.10, 0.02]
                cam.distance, cam.azimuth, cam.elevation = (0.38 if TASK == "yaw" else 0.32), 90, -50
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    wall = time.time() - t0
    if renderer is not None:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(CONTROL_DT * 3 * 1000), loop=0)
        renderer.close()
    c = np.array(ctrl_log)
    jerk = float(np.mean(np.linalg.norm(np.diff(c, axis=0), axis=1))) if len(c) > 1 else 0.0
    return dict(space=space, ctrl=c, jerk=jerk, yaw=yaw, mean_spin=float(np.mean(spin_log)),
                successes=len(success_times), success_times=success_times,
                mean_err=float(np.mean(err_log)) if err_log else float("nan"),
                dropped_at=dropped_at, sim_seconds=len(spin_log) * CONTROL_DT,
                realtime=len(spin_log) * CONTROL_DT / wall)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="spin", choices=("spin", "yaw", "roll"))
    p.add_argument("--space", default="joint", help="joint | synergy-<k>")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--synergies", default="out/synergies.npz")
    p.add_argument("--gif", default=None)
    p.add_argument("--log", default=None, help="save the commanded joint targets (npz)")
    p.add_argument("--noise", type=float, default=NOISE_NORM, help="rad, expected perturbation norm")
    p.add_argument("--horizon", type=float, default=HORIZON_S, help="s")
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--target-spin", type=float, default=TARGET_SPIN, help="rad/s about +Z (sign = direction)")
    p.add_argument("--thumb", default="stock", help="thumb mount (thumb.THUMB_MOUNTS)")
    p.add_argument("--finger-collisions", action="store_true", help="digits collide with each other")
    p.add_argument("--smooth", action="store_true", help="MPPI averaging + control-change penalty (see SMOOTH)")
    p.add_argument("--w-ctrl", type=float, default=None, help="control-change weight (overrides the mode's)")
    p.add_argument("--temp", type=float, default=MPPI_TEMP, help="MPPI temperature (x cost std)")
    p.add_argument("--filter", type=float, default=FILTER_ALPHA, help="low-pass alpha on the executed command")
    args = p.parse_args()
    globals().update(TASK=args.task, NOISE_NORM=args.noise, HORIZON_S=args.horizon, N_SAMPLES=args.samples,
                     TARGET_SPIN=args.target_spin, SMOOTH=args.smooth, MPPI_TEMP=args.temp, FILTER_ALPHA=args.filter)
    if args.w_ctrl is not None:
        globals().update(W_CTRL=args.w_ctrl, W_CTRL_SMOOTH=args.w_ctrl)
    pca = dict(np.load(args.synergies)) if args.space != "joint" else None
    r = run(args.space, args.seconds, pca, args.threads, args.seed, args.gif, args.thumb, args.finger_collisions)
    if args.log:
        np.savez(args.log, ctrl=r["ctrl"], joint_names=np.array(JOINT_NAMES), control_dt=CONTROL_DT)
    drop = f"dropped at {r['dropped_at']:.1f} s" if r["dropped_at"] is not None else "never dropped"
    flags = (f"{' fingercoll' if args.finger_collisions else ''}{' smooth' if args.smooth else ''}"
             f" w_ctrl={args.w_ctrl} temp={args.temp} filter={args.filter}")
    head = f"thumb={args.thumb}{flags} {r['space']} noise={NOISE_NORM} horizon={HORIZON_S} samples={N_SAMPLES}: "
    if args.task in ("yaw", "roll"):
        times = ", ".join(f"{t:.1f}" for t in r["success_times"])
        unit = "rad orientation" if args.task == "yaw" else "m distance"
        print(head + f"{args.task}: {r['successes']} goals reached in {r['sim_seconds']:.1f} s (at {times} s), "
              f"mean {unit} error {r['mean_err']:.3f}, {drop}, jerk {r['jerk']:.3f} rad/step, "
              f"{r['realtime']:.2f}x real time")
        return
    print(head + f"turned {np.degrees(r['yaw']):.0f} deg in {r['sim_seconds']:.1f} s "
          f"(mean {r['mean_spin']:.2f} rad/s, target {TARGET_SPIN}), {drop}, "
          f"jerk {r['jerk']:.3f} rad/step, {r['realtime']:.2f}x real time")


if __name__ == "__main__":
    main()
