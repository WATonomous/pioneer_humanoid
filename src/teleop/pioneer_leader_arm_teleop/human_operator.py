"""A simulated person moving the leader arm: human-like hand motion for synthetic teleop demos.

The operator thinks in terms of where the follower's gripper should go (what they see on screen) and
moves the leader, a 1:1 kinematic copy of the arm, so the leader's joint angles are IK of the hand pose
they are aiming for. The hand pose is built the way human reaching is modelled:

- Submovements: every move is one or more minimum-jerk strokes (bell-shaped speed) that overlap and
  add up (Flash & Hogan; Morasso & Mussa-Ivaldi). A move is split into two overlapping strokes through
  a slightly offset midpoint, so paths bow a little instead of being ruler-straight.
- Leader-arm curvature: moving a leader arm, people partly move joints rather than the hand in a
  straight line, so the hand bends towards the joint-space path (a per-person share of it).
- Carrying over things (``arc``): lift, across and down overlap into one swoop instead of
  up-stop-over-stop-down.
- Timing from Fitts' law: duration grows with log2(distance / tolerance); per-person speed.
- Aiming error: the first stroke lands off target (a little short, scattered ~5% of the distance);
  after a visual reaction delay the operator sees the follower's error and adds corrective strokes
  until it is within tolerance. Vision is noisy too: objects are seen a few mm off.
- A trace of tremor (8-12 Hz, < 0.1 mm: the leader's mass and geared servos filter the hand's) and slow
  postural drift (Ornstein-Uhlenbeck, ~0.5 mm) on the hand, a little wobble in its orientation and grip;
  hesitation pauses before grasping and releasing.

Every parameter that differs between people is drawn per episode (``Style.sample``).
No simulator stepping here: ``tick(q_leader, grip)`` is the caller's (one control step per call).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Fitts' law for leader-arm teleop (s, s/bit): slower than free reaching, you watch the follower.
FITTS_A = 0.25
FITTS_B = 0.25
# rad/s: fastest a hand turns a leader joint (a brisk human reach peaks around 2 rad/s at the elbow).
LEADER_MAX_SPEED = 3.0
# The leader's down-facing gripper, from its home orientation: fingers toward -Z, jaws closing along Y.
R_DOWN_FROM_HOME = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def minimum_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


@dataclass
class Style:
    """How one person teleoperates (drawn once per episode)."""

    speed: float          # movement time multiplier (>1: slower)
    aim: float            # endpoint scatter of a first stroke, fraction of its distance
    react: float          # s, visual reaction time before a correction
    tremor: float         # m, tremor amplitude
    drift: float          # m, postural drift (std)
    wobble_deg: float     # orientation drift (std)
    hesitate: float       # pause scale before grasps/releases
    see: float            # m, error in judging where an object is
    joint_blend: float    # share of the joint-space path in a reach's shape (leader-arm curvature)
    overlap: float        # how early the next part of an arc starts (0: in sequence .. 1: all at once)

    @classmethod
    def sample(cls, rng: np.random.Generator, pace: float = 1.0) -> "Style":
        """A practised operator at ``pace`` 1; 0.5 is a slow, careful beginner, 1.5 brisk."""
        speed = rng.uniform(0.65, 1.0) / pace
        return cls(
            speed=speed,
            aim=min(0.1, rng.uniform(0.03, 0.06) * 0.8 / speed),   # speed-accuracy trade-off: hurried people aim worse
            react=rng.uniform(0.15, 0.25) / math.sqrt(pace),
            # Small: the leader arm's own mass and geared servos filter most of a hand's tremor.
            tremor=rng.uniform(0.02e-3, 0.08e-3),
            drift=rng.uniform(0.3e-3, 0.8e-3),
            wobble_deg=rng.uniform(0.3, 0.8),
            hesitate=rng.uniform(0.35, 0.8) / pace,
            see=rng.uniform(1.0e-3, 2.5e-3),
            joint_blend=rng.uniform(0.25, 0.7),
            overlap=rng.uniform(0.35, 0.65),
        )


class _Channel:
    """A vector built from overlapping minimum-jerk strokes on top of a base value."""

    def __init__(self, base):
        self.base = np.array(base, float)
        self.strokes: list[tuple[float, float, np.ndarray]] = []   # (start, duration, displacement)

    def add(self, t0: float, dur: float, delta) -> None:
        self.strokes.append((t0, max(dur, 1e-3), np.asarray(delta, float)))

    def goal(self) -> np.ndarray:
        return self.base + sum((d for _, _, d in self.strokes), np.zeros_like(self.base))

    def at(self, t: float) -> np.ndarray:
        done = [s for s in self.strokes if t >= s[0] + s[1]]
        for s in done:                        # fold finished strokes into the base
            self.base = self.base + s[2]
            self.strokes.remove(s)
        return self.base + sum((d * minimum_jerk((t - t0) / dur) for t0, dur, d in self.strokes), np.zeros_like(self.base))

    def end_time(self) -> float:
        return max((t0 + dur for t0, dur, _ in self.strokes), default=0.0)


class _Noise:
    """Tremor (a few 8-12 Hz sines) plus an Ornstein-Uhlenbeck drift, per axis."""

    def __init__(self, rng: np.random.Generator, dim: int, tremor: float, drift: float, tau: float = 0.8):
        self.rng, self.drift, self.tau = rng, drift, tau
        self.freq = rng.uniform(8.0, 12.0, (3, dim))
        self.phase = rng.uniform(0, 2 * math.pi, (3, dim))
        self.amp = tremor / math.sqrt(3) * rng.uniform(0.6, 1.4, (3, dim))
        self.ou = rng.normal(0.0, drift, dim)

    def __call__(self, t: float, dt: float) -> np.ndarray:
        a = math.exp(-dt / self.tau)
        self.ou = a * self.ou + math.sqrt(1 - a * a) * self.drift * self.rng.normal(size=self.ou.shape)
        return self.ou + (self.amp * np.sin(2 * math.pi * self.freq * t + self.phase)).sum(0)


class HumanOperator:
    """Drives ``tick(q_leader, grip)`` once per control step with a human-like leader trajectory.

    ``model`` is the arm's (scene's) MjModel; ``observe()`` returns the follower's current MjData
    (what the operator sees). The arm joints are ``joints``, the hand body ``hand``; the tool centre
    point is the midpoint of the finger collision boxes.
    """

    def __init__(self, model: mujoco.MjModel, observe: Callable[[], mujoco.MjData], tick: Callable[[np.ndarray, float], None],
                 joints: list[str], q_home, dt: float, rng: np.random.Generator, style: Style | None = None,
                 fingers: tuple[str, str] = ("link7l", "link8l"), hand: str = "link6l"):
        self.m, self.observe, self._tick, self.dt, self.rng = model, observe, tick, dt, rng
        self.style = style or Style.sample(rng)
        self.hand = model.body(hand).id
        self.qadr = [model.joint(j).qposadr[0] for j in joints]
        self.vadr = [model.joint(j).dofadr[0] for j in joints]
        self.lim = np.array([model.jnt_range[model.joint(j).id] for j in joints])
        self.kin = mujoco.MjData(model)
        self.q = np.array(q_home, float)                 # leader joint angles (= follower targets)
        self.q_home = self.q.copy()
        self.kin.qpos[self.qadr] = self.q
        mujoco.mj_kinematics(model, self.kin)
        jaws = [next(g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(b).id and model.geom_contype[g])
                for b in fingers]
        R = self.kin.xmat[self.hand].reshape(3, 3).copy()
        self.tcp_off = R.T @ (self.kin.geom_xpos[jaws].mean(0) - self.kin.xpos[self.hand])
        self.R_home = R
        self.R_down = R_DOWN_FROM_HOME @ R
        self.t = 0.0
        self.pos = _Channel(self._tcp(self.kin))
        self.rot = _Channel(np.zeros(3))                 # rotation vector, world frame, applied to R_home
        self.grip = _Channel([0.0])
        s = self.style
        self.pos_noise = _Noise(rng, 3, s.tremor, s.drift, tau=1.5)
        self.rot_noise = _Noise(rng, 3, math.radians(0.02), math.radians(s.wobble_deg), tau=2.0)
        self.grip_noise = _Noise(rng, 1, 0.001, 0.005, tau=1.0)
        self.bends: list[tuple[float, float, np.ndarray]] = []   # (start, duration, path offsets at s = 0..1)
        self.trace: list[str] = []                       # one line per move, for debugging plans

    # ------------------------------------------------------------------ what the operator sees
    def _tcp(self, d: mujoco.MjData) -> np.ndarray:
        return d.xpos[self.hand] + d.xmat[self.hand].reshape(3, 3) @ self.tcp_off

    def follower_tcp(self) -> np.ndarray:
        return self._tcp(self.observe())

    def see(self, body: str) -> np.ndarray:
        """Where the operator judges ``body`` to be (its frame origin, a few mm off)."""
        d = self.observe()
        return d.xpos[self.m.body(body).id] + self.rng.normal(0.0, self.style.see, 3)

    def see_offset(self, a: str, b: str) -> np.ndarray:
        """Where ``b`` is relative to ``a`` when both are close together in view: judged better than
        either one's absolute position (a relative, side-by-side comparison)."""
        d = self.observe()
        rel = d.xpos[self.m.body(b).id] - d.xpos[self.m.body(a).id]
        return rel + self.rng.normal(0.0, 0.4 * self.style.see, 3)

    def see_yaw(self, body: str, symmetry: float = math.pi / 2) -> float:
        """Body's yaw (rad) folded into +-symmetry/2, judged within a few degrees."""
        R = self.observe().xmat[self.m.body(body).id].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        yaw = (yaw + symmetry / 2) % symmetry - symmetry / 2
        return yaw + math.radians(self.rng.normal(0.0, 2.0))

    def down(self, yaw: float = 0.0, tilt_y_deg: float = 0.0) -> np.ndarray:
        """Gripper pointing down, turned ``yaw`` about Z and tilted about world Y."""
        return (Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("y", tilt_y_deg, degrees=True).as_matrix()
                @ self.R_down)

    # ------------------------------------------------------------------ primitives
    def fitts(self, dist: float, tol: float) -> float:
        bits = math.log2(1.0 + max(dist, 0.0) / max(tol, 1e-3))
        return self.style.speed * (FITTS_A + FITTS_B * bits) * self.rng.lognormal(0.0, 0.12)

    def move(self, target, R=None, grip: float | None = None, tol: float = 0.01, via: bool = False,
             corrections: int = 3, axes=(1, 1, 1), speed: float = 1.0, bow: float = 0.1) -> float:
        """Bring the follower's tool point to ``target`` (orientation ``R``, gripper closure ``grip``).

        ``via``: pass through on the way to the next move (returns ~2/3 through, no corrections).
        ``axes``: which world axes the operator corrects (e.g. not Z while pushing down on something).
        Returns the follower's remaining distance to the target.
        """
        target = np.asarray(target, float)
        start = self.pos.goal()
        delta = target - start
        dist = float(np.linalg.norm(delta))
        dur = self.fitts(dist, tol) / speed
        if R is not None:   # turning the wrist takes time too: ~0.5 s per 45 deg on top of the reach
            ang = np.linalg.norm(self._rotvec(R) - self.rot.goal())
            dur = max(dur, self.style.speed * (0.4 + 0.6 * ang / (math.pi / 4)))
        if dist > 1e-3:
            self._reach(self.t, dur, start, self._aim(start, target, axes), bow, R)
        if R is not None:
            self.rot.add(self.t + 0.05 * dur, 0.9 * dur, self._rotvec(R) - self.rot.goal())
        if grip is not None:
            self.grip.add(self.t + 0.2 * dur, 0.6 * dur, [grip - self.grip.goal()[0]])
        if via:
            self.run(0.65 * dur)
            self._log("via", target, dur)
            return float(np.linalg.norm(target - self.follower_tcp()))
        self.run(dur)
        return self._correct(target, tol, corrections, axes, dur)

    def arc(self, target, clear_z: float, R=None, grip: float | None = None, tol: float = 0.01, corrections: int = 3,
            axes=(1, 1, 1), via: bool = False, speed: float = 1.0, land: float = 0.45) -> float:
        """Carry/reach to ``target`` over things up to height ``clear_z``: up, across and down as one swoop
        (overlapping strokes), not up-stop-over-stop-down. ``land``: how far through the move across the descent
        starts (late, ~0.75, to clear something just before the target). Corrections as in ``move``."""
        target = np.asarray(target, float)
        start = self.pos.goal()
        up = max(0.0, clear_z - start[2])
        down = target[2] - max(clear_z, start[2])          # negative: lowering at the end
        flat = np.array([target[0] - start[0], target[1] - start[1], 0.0])
        t_up = self.fitts(up, 0.02) / speed if up > 0.005 else 0.0
        t_across = self.fitts(float(np.linalg.norm(flat)), tol) / speed
        t_down = self.fitts(abs(down), tol) / speed if abs(down) > 0.005 else 0.0
        if R is not None:
            ang = np.linalg.norm(self._rotvec(R) - self.rot.goal())
            t_across = max(t_across, self.style.speed * (0.4 + 0.6 * ang / (math.pi / 4)))
        ov = float(np.clip(self.style.overlap * self.rng.uniform(0.8, 1.2), 0.0, 1.0))
        t0 = self.t
        if t_up:
            t_up = max(t_up, 0.35 * t_across)                # the lift spreads over the first part of the swoop
            self.pos.add(t0, t_up, [0.0, 0.0, up])
        t1 = t0 + (1 - ov) * 0.5 * t_up                      # across starts soon after the lift...
        top = start + (0.0, 0.0, up)
        over = np.array([target[0], target[1], top[2]])
        self._reach(t1, t_across, top, self._aim(top, over, (axes[0], axes[1], 0)), 0.05, R)
        t2 = t1 + float(np.clip(land + 0.2 * (0.5 - ov), 0.2, 0.9)) * t_across   # ...and the descent part-way across
        if t_down:
            self.pos.add(t2, t_down, [0.0, 0.0, down])
        if R is not None:
            self.rot.add(t1, t_across, self._rotvec(R) - self.rot.goal())
        if grip is not None:
            self.grip.add(t1, t_across, [grip - self.grip.goal()[0]])
        end = max(t1 + t_across, t2 + t_down)
        if via:
            self.run(t0 + 0.8 * (end - t0) - self.t)
            self._log("arcv", target, end - t0)
            return float(np.linalg.norm(target - self.follower_tcp()))
        self.run(end - self.t)
        return self._correct(target, tol, corrections, axes, end - t0)

    def _aim(self, start, target, axes) -> np.ndarray:
        """Where a first stroke actually goes: a little short, scattered ~aim x distance (only along ``axes``)."""
        delta = target - start
        dist = float(np.linalg.norm(delta))
        if dist < 1e-3:
            return target.copy()
        u = delta / dist
        side = np.cross(u, self.rng.normal(size=3))
        side /= np.linalg.norm(side) + 1e-9
        err = u * self.rng.normal(-0.02, self.style.aim) * dist + side * self.rng.normal(0.0, 0.5 * self.style.aim) * dist
        return target + np.clip(err, -0.25 * dist, 0.25 * dist) * np.asarray(axes)

    def _reach(self, t0: float, dur: float, start, aim, bow: float, R=None) -> None:
        """Two overlapping strokes through a bowed midpoint (a slightly curved, single-hump reach), bent towards
        the leader's joint-space path."""
        start, aim = np.asarray(start, float), np.asarray(aim, float)
        dist = float(np.linalg.norm(aim - start))
        u = (aim - start) / max(dist, 1e-9)
        perp = np.cross(u, self.rng.normal(size=3))
        perp /= np.linalg.norm(perp) + 1e-9
        mid = start + 0.5 * (aim - start) + perp * self.rng.normal(0.0, bow) * dist
        self.pos.add(t0, 0.6 * dur, mid - start)
        self.pos.add(t0 + 0.3 * dur, 0.7 * dur, aim - mid)
        if dist > 0.03:
            self._bend(t0, dur, start, aim, R)

    def _bend(self, t0: float, dur: float, start, aim, R=None) -> None:
        """Offsets that pull this reach towards the path a straight line in the leader's joints would trace."""
        if R is None:
            R = Rotation.from_rotvec(self.rot.goal()).as_matrix() @ self.R_home
        q0, e0 = self._solve(self.q, start, R, 60)
        q1, e1 = self._solve(q0, aim, R, 60)
        if max(e0, e1) > 2e-3:
            return
        s = np.linspace(0.0, 1.0, 17)
        path = []
        for si in s:
            self.kin.qpos[self.qadr] = q0 + si * (q1 - q0)
            mujoco.mj_kinematics(self.m, self.kin)
            path.append(self._tcp(self.kin))
        path = np.array(path)
        dev = path - (path[0] + s[:, None] * (path[-1] - path[0]))
        dist = float(np.linalg.norm(aim - start))
        scale = min(1.0, 0.3 * dist / (np.abs(dev).max() + 1e-9))      # never more than 30% of the reach
        self.bends.append((t0, dur, self.style.joint_blend * scale * dev))
        self.kin.qpos[self.qadr] = self.q

    def _bend_at(self, t: float) -> np.ndarray:
        out = np.zeros(3)
        for b in [b for b in self.bends if t >= b[0] + b[1]]:
            self.bends.remove(b)
        s = np.linspace(0.0, 1.0, 17)
        for t0, dur, dev in self.bends:
            if t > t0:
                ph = float(minimum_jerk((t - t0) / dur))
                out += np.array([np.interp(ph, s, dev[:, k]) for k in range(3)])
        return out

    def _correct(self, target, tol: float, corrections: int, axes, dur: float) -> float:
        """Nudge the follower onto the target as it comes to rest (what a person does after a reach): the next
        correction starts while the arm is still slowing down, not after it has stopped."""
        self.glimpse()
        mask = np.asarray(axes, float)
        last = None
        for _ in range(corrections):
            seen = self.follower_tcp() + self.rng.normal(0.0, 0.3 * self.style.see, 3)
            err = (target - seen) * mask
            e = float(np.linalg.norm(err))
            if e < tol or (last is not None and e > 0.7 * last):
                break               # there, or the last nudge didn't help (blocked): stop pushing
            if np.linalg.norm(self.follower_tcp() - self._tcp(self.kin)) > max(0.015, 2 * tol) and not np.any(mask == 0):
                break               # the follower isn't where the leader is (caught on something): don't push harder
            last = e
            d = self.fitts(e, tol) * 0.8
            self.pos.add(self.t, d, err * self.rng.uniform(0.75, 1.0))
            self.run(d)
            self.glimpse()
        self._log("move", target, dur)
        return float(np.linalg.norm((target - self.follower_tcp()) * mask))

    def _log(self, kind: str, target, dur: float) -> None:
        f = self.follower_tcp()
        self.trace.append(f"t={self.t:6.2f} {kind:4s} target={np.round(target, 3)} follower={np.round(f, 3)} "
                          f"err={np.linalg.norm(target - f) * 1000:5.1f}mm leader={np.round(self._tcp(self.kin), 3)} stroke={dur:.2f}s grip={self.grip.goal()[0]:.2f}")

    def nudge(self, delta, size: float = 0.003) -> None:
        """A small corrective push of the hand (fine alignment), then watch what it did."""
        dur = self.fitts(float(np.linalg.norm(delta)), size) * 0.8
        self.pos.add(self.t, dur, np.asarray(delta, float))
        self.run(dur)
        self.glimpse()

    def set_grip(self, closure: float, hesitate: bool = True) -> None:
        """Squeeze or open the leader's claw (0 open .. 1 closed) at a hand's pace."""
        if hesitate and self.rng.random() < 0.3:      # now and then a moment's hesitation before committing
            self.run(self.style.hesitate * self.rng.uniform(0.1, 0.4))
        dur = self.style.speed * self.rng.uniform(0.25, 0.4)
        self.grip.add(self.t, dur, [closure - self.grip.goal()[0]])
        self.run(dur + self.rng.uniform(0.03, 0.1))

    def glimpse(self) -> None:
        """Look again once the arm has nearly stopped (< 5 cm/s, at most 0.3 s); half a reaction time to judge it."""
        p, waited = self.follower_tcp(), 0.0
        while waited < 0.3:
            self.run(0.03)
            waited += 0.03
            q = self.follower_tcp()
            if np.linalg.norm(q - p) / 0.03 < 0.05:
                break
            p = q
        self.run(0.5 * self.style.react)

    def settle(self, speed: float = 0.03, limit: float = 0.5) -> None:
        """Wait until the follower has (nearly) stopped, then a reaction time: judging the error before that is guessing."""
        p, waited = self.follower_tcp(), 0.0
        while waited < limit:
            self.run(0.05)
            waited += 0.05
            q = self.follower_tcp()
            if np.linalg.norm(q - p) / 0.05 < speed:
                break
            p = q
        self.run(self.style.react)

    def glance(self) -> None:
        """Between steps: look at what's next before moving (a person doesn't chain steps instantly)."""
        self.pause(0.2, 0.7)

    def pause(self, lo: float, hi: float) -> None:
        self.run(self.style.hesitate * self.rng.uniform(lo, hi))

    def run(self, sec: float) -> None:
        for _ in range(max(1, int(round(sec / self.dt)))):
            self.t += self.dt
            p = self.pos.at(self.t) + self._bend_at(self.t) + self.pos_noise(self.t, self.dt)
            rv = self.rot.at(self.t) + self.rot_noise(self.t, self.dt)
            R = Rotation.from_rotvec(rv).as_matrix() @ self.R_home
            g = float(np.clip(self.grip.at(self.t)[0] + self.grip_noise(self.t, self.dt)[0], 0.0, 1.0))
            self._tick(self._ik(p, R), g)

    # ------------------------------------------------------------------ leader kinematics
    def _rotvec(self, R) -> np.ndarray:
        return Rotation.from_matrix(np.asarray(R) @ self.R_home.T).as_rotvec()

    def _ik(self, target: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Leader joint angles for the hand pose: warm-started; reseeded from home if stuck in a bad branch.
        A hand moves the leader at a finite speed, so the result is rate-limited."""
        q, err = self._solve(self.q, target, R, 30)
        if err > 2e-3:
            q2, err2 = self._solve(self.q_home, target, R, 80)
            if err2 < err:
                q = q2
        self.q = self.q + np.clip(q - self.q, -LEADER_MAX_SPEED * self.dt, LEADER_MAX_SPEED * self.dt)
        return self.q

    def _solve(self, q0, target, R, iters):
        m, kin = self.m, self.kin
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q = np.array(q0, float)
        err = np.inf
        for _ in range(iters):
            kin.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, kin)
            mujoco.mj_comPos(m, kin)
            Rc = kin.xmat[self.hand].reshape(3, 3)
            tcp = self._tcp(kin)
            e = np.concatenate([target - tcp, 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))])
            err = float(np.linalg.norm(e))
            if err < 1e-5:
                break
            mujoco.mj_jac(m, kin, jp, jr, tcp, self.hand)
            J = np.vstack([jp[:, self.vadr], jr[:, self.vadr]])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), self.lim[:, 0], self.lim[:, 1])
        return q, err
