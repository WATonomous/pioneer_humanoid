"""Physics-only privileged peg expert, and the CLI that records its demonstrations.

The expert sees exact peg/block poses and drives the left arm through the position actuators with
a damped-least-squares Jacobian servo at 25 Hz (40 physics steps of 1 ms per command). Only
``reset`` writes simulator state; every motion goes through ``ctrl`` and ``mj_step``.

Case table — ``Phase`` | trigger | behaviour:

| Phase              | Trigger                                   | Behaviour                                                   |
|--------------------|-------------------------------------------|-------------------------------------------------------------|
| APPROACH           | start, or grasp retry                     | fingers 18 cm above the measured peg, gripper open          |
| DESCEND            | approach reached                          | descend at 2.5 cm/s to grasp the upper 3 cm of the peg      |
| CLOSE              | descent reached                           | close 1.5 s; no two-finger contact -> reopen, retry (3 max) |
| LIFT               | both fingers touch                        | lift 12 cm; peg not 8 cm up -> reopen, retry                |
| (grasp takeover)   | policy near an upright, ungrasped peg     | rise clear of the peg top, then APPROACH (lower) -> LIFT    |
| TAKEOVER_HOLD      | correction handoff from a policy          | keep the inherited grasp still for ``hold_seconds``         |
| TAKEOVER_RETRACT   | touching the block, or too low to rotate  | lift in 5 mm steps at the current orientation until clear   |
| ALIGN              | first attempt                             | servo the peg upright to just above the hole (3 cm/s)       |
| RECOVER_REALIGN    | insertion stalled/jammed                  | retract (as above), then align again                        |
| INSERT             | aligned                                   | descend at 6 mm/s with peg-pose feedback (3 attempts)       |
| VERIFY_SEATING     | ``guarded`` and seated                    | keep holding 0.4 s; must stay seated before opening          |
| RELEASE            | seated                                    | open the gripper for 1 s                                    |
| RETREAT            | released                                  | lift 12 cm, wait 0.5 s, peg must still be inserted          |

``guarded=True`` is for corrections that start from policy states, where a wedged peg is likely: an
early rim jam (footprint does not fit, or the bottom stalls 0.36 s near the rim) aborts the descent
into RECOVER_REALIGN, one extra attempt is allowed, and the gripper only opens after
VERIFY_SEATING. On full expert runs it changes nothing (N=40: 35/40 either way) and costs ~5 s,
so demos use the plain path. Lost grasps, unreachable targets, unstable physics and failed final
checks raise ``ExpertFailure`` and the episode is discarded.

  python -m humanoid_mujoco_scenes.peg_insert.expert --episodes 20 --seed 0 --verbose
  python -m humanoid_mujoco_scenes.peg_insert.expert --episodes 100 --successes 50 --output datasets/peg_insert_raw
"""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from enum import Enum
from pathlib import Path

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model
from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, LEFT_GRIPPER_JOINTS
from pioneer_humanoid.mujoco_bimanual_arm import set_home
from humanoid_mujoco_scenes import TABLE_TOP_Z
from .scene import BLOCK_HEIGHT, BLOCK_POS, CLEARANCE, PEG_HEIGHT, PEG_POS, PEG_SIZE

FPS = 25
DT = 1 / FPS
JOINT_NAMES = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
               "left_elbow_pitch", "left_elbow_roll", "left_wrist_pitch", "left_gripper"]
TASK = "pick up the square peg and insert it into the square hole"
GRIPPER_SPAN = np.array([-0.05, 0.05])  # finger joint targets (m) at closure 1
GRASP_HEIGHT = 0.060                    # m above peg centre: upper 3 cm, fingers clear the block
APPROACH_HEIGHT = 0.18                  # m above peg centre before descending to grasp
GRASP_CLEARANCE = 0.03                  # m, fingers above the peg top before moving sideways in a grasp takeover
LIFT_HEIGHT = 0.12                      # m
LIFTED = TABLE_TOP_Z + PEG_HEIGHT / 2 + 0.08  # peg centre z that counts as lifted
SEATED_Z = TABLE_TOP_Z + PEG_HEIGHT / 2
MAX_JITTER = 0.02                       # m, largest validated reset jitter


class Phase(str, Enum):
    RESET = "reset"
    APPROACH = "approach"
    DESCEND = "descend_to_grasp"
    CLOSE = "close"
    LIFT = "lift"
    TAKEOVER_HOLD = "takeover_hold"
    TAKEOVER_RETRACT = "takeover_retract"
    ALIGN = "align"
    RECOVER_REALIGN = "recover_realign"
    INSERT = "insert"
    VERIFY_SEATING = "verify_seating"
    RELEASE = "release"
    RETREAT = "retreat"
    SETUP = "setup"  # preparing a diagnostic/correction start (handoffs.py); never a training label

    def __str__(self):
        return self.value


class ExpertFailure(RuntimeError):
    pass


class PegExpert:
    """Measured-pose DLS servo with integral actuator targets and bounded motion."""

    def __init__(self, model, data, on_frame=None, verbose=False, guarded=False):
        self.model, self.data = model, data
        self.on_frame, self.verbose, self.guarded = on_frame, verbose, guarded
        self.arm_q = np.array([model.joint(n).qposadr[0] for n in LEFT_ARM_JOINTS])
        self.arm_v = np.array([model.joint(n).dofadr[0] for n in LEFT_ARM_JOINTS])
        self.arm_a = np.array([model.actuator(n).id for n in LEFT_ARM_JOINTS])
        self.grip_q = np.array([model.joint(n).qposadr[0] for n in LEFT_GRIPPER_JOINTS])
        self.grip_a = np.array([model.actuator(n).id for n in LEFT_GRIPPER_JOINTS])
        self.limits = model.jnt_range[[model.joint(n).id for n in LEFT_ARM_JOINTS]]
        self.wrist = model.body("link6l").id
        self.fingers = [model.body(n).id for n in ("link7l", "link8l")]
        self.finger_geoms = [next(g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_contype[g])
                             for b in self.fingers]
        self.peg = model.body("peg").id
        self.peg_geom = model.geom("peg").id
        self.peg_dof = model.joint("peg").dofadr[0]
        self.block = model.body("hole_block").id
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        self.phase = Phase.RESET
        self.closure = 0.0
        self.takeover_strategy = None
        self.steps = round(DT / model.opt.timestep)
        if not np.isclose(self.steps * model.opt.timestep, DT):
            raise ValueError("Physics timestep must divide the 25 Hz control period")

    # ---- episode setup (the only simulator state writes) ----

    def reset(self, seed, randomize=0.01, peg_xy=None, hole_xy=None):
        if (peg_xy is None) != (hole_xy is None):
            raise ValueError("Explicit layouts require both peg and hole positions")
        if peg_xy is not None:
            from .workspace import validate_layout
            validate_layout(peg_xy, hole_xy)
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        set_home(self.model, self.data)
        adr = self.model.joint("peg").qposadr[0]
        self.data.qpos[adr:adr + 2] = np.array(PEG_POS) + rng.uniform(-randomize, randomize, 2) if peg_xy is None else peg_xy
        # The block is static: its placement is model data, changed only here.
        self.model.body_pos[self.block, :2] = np.array(BLOCK_POS) + rng.uniform(-randomize, randomize, 2) if hole_xy is None else hole_xy
        self.closure = 0.0
        self.phase = Phase.RESET
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=500)

    # ---- measured state ----

    @property
    def center(self):
        return self.data.geom_xpos[self.finger_geoms].mean(axis=0)

    @property
    def peg_pos(self):
        return self.data.xpos[self.peg].copy()

    @property
    def hole_pos(self):
        return self.data.xpos[self.block].copy()

    def peg_rotation(self):
        return self.data.xmat[self.peg].reshape(3, 3)

    def state(self):
        """Policy observation: six measured arm angles (rad) + measured gripper closure (0-1)."""
        grip = np.mean(self.data.qpos[self.grip_q] / GRIPPER_SPAN)
        return np.r_[self.data.qpos[self.arm_q], np.clip(grip, 0, 1)].astype(np.float32)

    def contacts(self):
        """Both fingers touch the peg."""
        touched = set()
        for c in self.data.contact:
            if c.geom1 == self.peg_geom:
                touched.add(int(self.model.geom_bodyid[c.geom2]))
            if c.geom2 == self.peg_geom:
                touched.add(int(self.model.geom_bodyid[c.geom1]))
        return all(b in touched for b in self.fingers)

    def released(self):
        """No peg/finger contact, so holding a seated peg cannot count as success."""
        fingers = set(self.finger_geoms)
        return not any(self.peg_geom in (c.geom1, c.geom2) and fingers & {int(c.geom1), int(c.geom2)} for c in self.data.contact)

    def block_contacts(self):
        """Peg/finger contact with the insertion block, excluding the grasp itself."""
        held = {self.peg_geom, *self.finger_geoms}
        for c in self.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            if (a in held and self.model.geom_bodyid[b] == self.block) or (b in held and self.model.geom_bodyid[a] == self.block):
                return True
        return False

    def inserted(self):
        """All eight corners inside the aperture, bottom on the table, top above the rim, at rest."""
        half = np.array([PEG_SIZE / 2, PEG_SIZE / 2, PEG_HEIGHT / 2])
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
        corners = corners @ self.peg_rotation().T + self.peg_pos
        xy = np.max(np.abs(corners[:, :2] - self.hole_pos[:2]), axis=0)
        return bool(np.all(xy < (PEG_SIZE + CLEARANCE) / 2 + 0.0002)
                    and abs(corners[:, 2].min() - TABLE_TOP_Z) < 0.002
                    and self.peg_pos[2] + PEG_HEIGHT / 2 > TABLE_TOP_Z + BLOCK_HEIGHT
                    and np.linalg.norm(self.data.qvel[self.peg_dof:self.peg_dof + 6]) < 0.02)

    def lifted(self):
        return bool(self.contacts() and self.peg_pos[2] > LIFTED)

    def alignment_ready(self, has_lifted, radius=0.03):
        """Held, lifted peg within ``radius`` XY of the hole and above the seated height."""
        return bool(has_lifted and self.contacts() and not self.inserted()
                    and np.linalg.norm(self.peg_pos[:2] - self.hole_pos[:2]) <= radius
                    and self.peg_pos[2] > SEATED_Z + 0.015)

    def footprint_fits(self, margin=0.0001):
        half = np.abs(self.peg_rotation()[:2]) @ np.array([PEG_SIZE / 2, PEG_SIZE / 2, PEG_HEIGHT / 2])
        return bool(np.all(np.abs(self.peg_pos[:2] - self.hole_pos[:2]) + half < (PEG_SIZE + CLEARANCE) / 2 - margin))

    def alignment_clearance_height(self):
        """Peg-centre z at which it can translate/rotate over the rim (tilt sweeps a larger extent)."""
        rotation = self.peg_rotation()
        half = np.array([PEG_SIZE / 2, PEG_SIZE / 2, PEG_HEIGHT / 2])
        extent = np.abs(rotation[2]) @ half
        if rotation[2, 2] < np.cos(np.radians(1)):
            extent = max(extent, np.linalg.norm(half))
        return TABLE_TOP_Z + BLOCK_HEIGHT + max(extent, half[2]) + 0.003

    # ---- actuation ----

    def apply(self, action):
        """Policy action -> actuators with the recording's units; returns (applied, clipped)."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ExpertFailure("policy action must contain seven finite values")
        bounded = action.copy()
        bounded[:6] = np.clip(bounded[:6], self.limits[:, 0], self.limits[:, 1])
        bounded[6] = np.clip(bounded[6], 0, 1)
        self.data.ctrl[self.arm_a] = bounded[:6]
        self.data.ctrl[self.grip_a] = bounded[6] * GRIPPER_SPAN
        return bounded, bool(np.any(np.abs(action - bounded) > 1e-6))

    def step(self):
        mujoco.mj_step(self.model, self.data, nstep=self.steps)
        if not np.isfinite(self.data.qpos).all() or np.linalg.norm(self.data.qvel) > 100:
            raise ExpertFailure("unstable physics")

    def tick(self, target, rotation=np.eye(3), closure=None, speed=0.06):
        """One 25 Hz servo command toward a finger-centre target and wrist rotation."""
        d = self.data
        if closure is not None:
            self.closure += np.clip(closure - self.closure, -0.04, 0.04)
        pos_error = np.asarray(target) - self.center
        rot_error = self._rotation_error(rotation)
        mujoco.mj_jac(self.model, d, self.jp, self.jr, self.center, self.wrist)
        jac = np.vstack([self.jp[:, self.arm_v], self.jr[:, self.arm_v]])
        dp = pos_error * 0.35
        dp *= min(1.0, speed * DT / max(np.linalg.norm(dp), 1e-9))
        dr = rot_error * 0.2
        dr *= min(1.0, 0.5 * DT / max(np.linalg.norm(dr), 1e-9))
        dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, np.r_[dp, dr])
        dq = np.clip(dq, -0.6 * DT, 0.6 * DT)
        # Accumulating on ctrl, rather than qpos, corrects gravity-induced PD error.
        d.ctrl[self.arm_a] = np.clip(d.ctrl[self.arm_a] + dq, self.limits[:, 0], self.limits[:, 1])
        d.ctrl[self.grip_a] = self.closure * GRIPPER_SPAN
        if self.on_frame:
            self.on_frame(self, np.r_[d.ctrl[self.arm_a], self.closure].astype(np.float32))  # obs at t, action over [t, t+DT)
        self.step()
        return np.linalg.norm(pos_error), np.linalg.norm(rot_error)

    def _rotation_error(self, desired):
        # Quaternion error also handles the 90-degree initial approach reliably.
        q, qd, qr = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.xmat[self.wrist])
        mujoco.mju_mat2Quat(qd, np.asarray(desired, dtype=np.float64).ravel())
        mujoco.mju_negQuat(qr, q)
        mujoco.mju_mulQuat(qr, qd, qr.copy())
        error = np.zeros(3)
        mujoco.mju_quat2Vel(error, qr, 1.0)
        return error

    def settled(self):
        return np.linalg.norm(self.data.qvel[self.arm_v]) < 0.01

    def log(self):
        if self.verbose:
            print(self.phase, "t=", round(self.data.time, 2), "peg=", self.peg_pos, flush=True)

    def move(self, phase, target, closure, speed=0.06, timeout=15, tolerance=0.0003):
        """Finger-centre servo until within ``tolerance`` and at rest."""
        self.phase = phase
        for _ in range(round(timeout * FPS)):
            ep, er = self.tick(target, closure=closure, speed=speed)
            if ep < tolerance and er < 0.005 and self.settled():
                return self.log()
        raise ExpertFailure(f"{phase} timeout: position error={ep:.4f}, rotation error={er:.4f}")

    def hold(self, seconds, target, closure, rotation=np.eye(3)):
        for _ in range(round(seconds * FPS)):
            self.tick(target, rotation, closure=closure, speed=0.01)

    def peg_servo(self, desired, speed, timeout, rotation=None):
        """Servo the measured peg (not the fingers) to a pose; corrects in-hand slip and tilt."""
        desired_rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        for _ in range(round(timeout * FPS)):
            if not self.contacts():
                raise ExpertFailure(f"lost grasp during {self.phase}")
            if self.guarded and self.phase == Phase.INSERT and self._early_jam():
                raise ExpertFailure("early jam: relift before further descent")
            correction = desired_rotation @ self.peg_rotation().T
            target = desired + correction @ (self.center - self.peg_pos)
            self.tick(target, correction @ self.data.xmat[self.wrist].reshape(3, 3), speed=speed)
            ep = np.linalg.norm(desired - self.peg_pos)
            er = np.linalg.norm(self.peg_rotation() - desired_rotation)
            if ep < 0.00025 and er < 0.005 and self.settled():
                return self.log()
        raise ExpertFailure(f"{self.phase} timeout: peg position error={ep:.4f}, rotation error={er:.4f}")

    def _early_jam(self):
        """Rim contact with a footprint that cannot fit, or a 0.36 s stall within 8 mm of the rim."""
        bottom = self.peg_pos[2] - np.abs(self.peg_rotation()[2]) @ np.array([PEG_SIZE / 2, PEG_SIZE / 2, PEG_HEIGHT / 2])
        if not self.block_contacts() or self.inserted() or bottom <= TABLE_TOP_Z + 0.003:
            self._jam_history = []
            return False
        self._jam_history = (self._jam_history + [bottom])[-9:]
        stalled = len(self._jam_history) == 9 and abs(bottom - self._jam_history[0]) < 0.0005
        return not self.footprint_fits(margin=0) or (stalled and bottom >= TABLE_TOP_Z + BLOCK_HEIGHT - 0.008)

    # ---- task ----

    def run(self):
        self.grasp_and_lift()
        self.insert_and_release()

    def grasp_and_lift(self, approach_height=APPROACH_HEIGHT):
        for _ in range(3):
            self.move(Phase.APPROACH, self.peg_pos + [0, 0, approach_height], 0)
            grasp = self.peg_pos + [0, 0, GRASP_HEIGHT]
            self.move(Phase.DESCEND, grasp, 0, speed=0.025)
            self.phase = Phase.CLOSE
            self.hold(1.5, grasp, 1)
            if not self.contacts():
                self.hold(0.6, grasp, 0)
                continue
            self.move(Phase.LIFT, grasp + [0, 0, LIFT_HEIGHT], 1, speed=0.03)
            if self.lifted():
                return
            self.hold(0.6, self.center.copy(), 0)
        raise ExpertFailure("grasp failed after three attempts")

    def takeover_grasp(self):
        """Grasp and lift an upright, ungrasped peg from wherever a policy left the hand.

        Rises vertically first if the fingers are below the peg top (a sideways move there would
        push the peg), then approaches only as high as needed instead of the full 18 cm.
        """
        if self.contacts() or self.lifted():
            raise ExpertFailure("grasp takeover needs an ungrasped peg")
        self.closure = float(np.clip(np.mean(self.data.ctrl[self.grip_a] / GRIPPER_SPAN), 0, 1))
        clear = self.peg_pos[2] + PEG_HEIGHT / 2 + GRASP_CLEARANCE
        if self.center[2] < clear:
            self.move(Phase.APPROACH, np.r_[self.center[:2], clear], 0)
        self.grasp_and_lift(approach_height=max(self.center[2] - self.peg_pos[2], GRASP_HEIGHT + GRASP_CLEARANCE))

    def takeover(self, hold_seconds=0.5):
        """Continue from a held peg (policy handoff) without resetting, reopening, or moving objects."""
        if not self.contacts():
            raise ExpertFailure("takeover requires a verified two-finger grasp")
        self.closure = 1.0 if self.guarded else float(np.clip(np.mean(self.data.ctrl[self.grip_a] / GRIPPER_SPAN), 0, 1))
        self.phase = Phase.TAKEOVER_HOLD
        target, rotation = self.center.copy(), self.data.xmat[self.wrist].reshape(3, 3).copy()
        for _ in range(round(hold_seconds * FPS)):
            if not self.contacts():
                raise ExpertFailure("lost grasp while securing takeover")
            self.tick(target, rotation, closure=1, speed=0.01)
        self.takeover_strategy = self.clear_for_alignment()
        self.insert_and_release(approach_height=self.peg_pos[2])

    def clear_for_alignment(self):
        """Lift (5 mm steps, orientation kept) until clear of the block and high enough to rotate."""
        if not self.block_contacts() and self.peg_pos[2] >= self.alignment_clearance_height():
            return "direct_align"
        self.phase = Phase.TAKEOVER_RETRACT
        rotation = self.peg_rotation().copy()
        for _ in range(6):
            retract = self.peg_pos
            retract[2] = max(self.alignment_clearance_height(), retract[2] + 0.005)
            self.peg_servo(retract, speed=0.015, timeout=15, rotation=rotation)
            if not self.block_contacts() and self.peg_pos[2] >= self.alignment_clearance_height() - 0.00025:
                return "retract_then_align"
        raise ExpertFailure("could not clear block contact before alignment")

    def insert_and_release(self, approach_height=None):
        attempts = 4 if self.guarded else 3
        for attempt in range(attempts):
            if attempt > 0 or self.guarded:
                self.clear_for_alignment()
            self.align(attempt, approach_height)
            if self.insert(last=attempt == attempts - 1):
                break
        else:
            raise ExpertFailure("insertion depth/fit verification failed")
        self.release_and_retreat()

    def align(self, attempt, approach_height):
        self.phase = Phase.ALIGN if attempt == 0 else Phase.RECOVER_REALIGN
        goal = self.hole_pos
        clear = self.guarded or approach_height is not None
        goal[2] = max(self.peg_pos[2], self.alignment_clearance_height()) if clear else SEATED_Z + 0.10
        self.peg_servo(goal, speed=0.03, timeout=15)
        if self.guarded and not self.footprint_fits():
            raise ExpertFailure("alignment footprint gate rejected descent")

    def insert(self, last):
        """Descend into the hole; False means retract and realign."""
        self.phase = Phase.INSERT
        self._jam_history = []
        goal = self.hole_pos
        goal[2] = SEATED_Z
        try:
            self.peg_servo(goal, speed=0.006, timeout=25)
        except ExpertFailure:
            if last:
                raise
            return False
        if self.guarded:
            self.verify_seating()
        return self.inserted()

    def verify_seating(self):
        self.phase = Phase.VERIFY_SEATING
        target, rotation = self.center.copy(), self.data.xmat[self.wrist].reshape(3, 3).copy()
        for _ in range(10):
            if not self.inserted() or not self.contacts():
                raise ExpertFailure("seating verification failed before release")
            self.tick(target, rotation, closure=1, speed=0.01)

    def release_and_retreat(self):
        target, rotation = self.center.copy(), self.data.xmat[self.wrist].reshape(3, 3).copy()
        self.phase = Phase.RELEASE
        for _ in range(FPS):
            if self.guarded and not self.inserted():
                raise ExpertFailure("seating lost during release")
            self.tick(target, rotation, closure=0, speed=0.01)
        self.move(Phase.RETREAT, target + [0, 0, LIFT_HEIGHT], 0, speed=0.03)
        self.hold(0.5, self.center.copy(), 0)
        if not self.inserted() or not self.released():
            raise ExpertFailure("peg did not remain inserted after release")


def collect(args):
    """Record expert demos; ``--output`` reopens an existing collection and continues its seeds."""
    from .recording import CAMERAS, RawDataset

    raw = RawDataset(args.output, randomize=args.randomize) if args.output else None
    if raw and raw.manifest.get("randomize") != args.randomize:
        raise SystemExit("existing dataset used a different --randomize")
    model = make_model("peg_insert", cameras=CAMERAS if raw else {})
    data = mujoco.MjData(model)
    expert = PegExpert(model, data, verbose=args.verbose)
    first = max([r["seed"] for r in raw.results], default=args.seed - 1) + 1 if raw else args.seed
    results = []
    with ExitStack() as stack:
        renderers = {n: stack.enter_context(mujoco.Renderer(model, h, w)) for n, (h, w) in CAMERAS.items()} if raw else {}
        for seed in range(first, first + args.episodes):
            if raw and args.successes is not None and raw.successes() >= args.successes:
                break
            expert.reset(seed, args.randomize)
            results.append(record_episode(expert, raw, seed, renderers))
    successes = raw.successes() if raw else sum(r["success"] for r in results)
    print(f"Success: {sum(r['success'] for r in results)}/{len(results)}")
    if (successes < args.successes) if args.successes is not None else not all(r["success"] for r in results):
        raise SystemExit(1)


def record_episode(expert, raw, seed, renderers):
    result = {"seed": seed, "success": False}
    try:
        if raw is None:
            expert.run()
        else:
            with raw.episode(seed, renderers) as writer:
                expert.on_frame = writer
                expert.run()
                result.update(writer.commit())
        result["success"] = True
    except ExpertFailure as exc:
        result["failure"] = str(exc)
    finally:
        expert.on_frame = None
    result["sim_seconds"] = round(expert.data.time, 3)
    if raw is not None:
        raw.add(result)
    else:
        print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=10, help="attempts; failures are logged, never saved")
    p.add_argument("--seed", type=int, default=0, help="first layout seed of a new collection")
    p.add_argument("--randomize", type=float, default=0.01, help=f"reset xy jitter in metres (max {MAX_JITTER})")
    p.add_argument("--output", type=Path, help="raw dataset directory (resumed if it exists); omit for physics-only checks")
    p.add_argument("--successes", type=int, help="stop once the collection holds this many successful demos")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.episodes < 1 or not 0 <= args.randomize <= MAX_JITTER:
        p.error(f"episodes must be positive; randomize must be in [0, {MAX_JITTER}]")
    if args.successes is not None and (args.output is None or args.successes < 1):
        p.error("--successes needs --output and a positive count")
    collect(args)


if __name__ == "__main__":
    main()
