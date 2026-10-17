"""iPhone teleoperation of the Pioneer LEFT arm in plain MuJoCo (CPU), recording like the other MuJoCo teleops.

    pip install hebi-py --no-deps        # --no-deps: it pins numpy<2, which would downgrade the repo's numpy
    python src/teleop/phone_teleop/phone_teleop.py --scene tidy_table [--record --cameras top,wrist_left]

On the phone: the free "HEBI Mobile I/O" app, open and in the foreground, on this computer's Wi-Fi (see
phone_stream_test.py, which only prints what the phone sends -- try that first). Sim only.

The first run, and any run with --calibrate, starts in this terminal with the gripper-button check and the
four-move calibration of phone_stream_test.py; it is saved (phone_calibration.json here) and later runs use
it. The app's level axes point wherever the phone faced when its tracking began, so the saved calibration only
holds while the app has stayed open and tracking: if the gripper goes the wrong way, run with --calibrate. Sitting
in front of the screen, show +X to your RIGHT and +Y AWAY from you. Then the scene window opens, and the
gripper moves on the screen the way your hand does: right is right, away from you is into the picture, up is
up. That follows the view: orbit it with the mouse and the directions turn with it (taken at each B1 press).
Tilting the phone sideways or turning it flat turns the gripper the same way; tilting its top away from you
is reversed (TILT_SIGNS), which is how it reads with the gripper facing you.

Grabbing: the fingers are 87 mm long and the objects lie on the table, so the finger TIPS have to come down
to it. A plumb line hangs from between the tips to a disc on the table: put the disc on the object, lower
until line and disc turn green (the corner of the window says how high the tips are), then close.

Phone:
  B1   hold: the gripper follows the phone -- its position (1:1, or --scale) and its rotation -- from wherever
       both are when you press. Release, move your hand back, press again: the arm picks up where it stopped.
  B2   open / close the gripper
Scene window keys (click it once):
  B discard the take   H glide home   R reset arm and scene (new layout)   Esc quit
  F    follow on / off, if B1 does not reach this computer

Recording (--record) runs by itself. A take starts at the first B1 press of each layout, with the arm still
at home. When the scene's last step is done the arm glides home, and the take is saved if the scene scores it
a success (tidy_table: every object binned in order, nothing dropped or toppled) -- else it is thrown away.
Either way a new layout comes up, ready for the next take. So every saved take has the same shape: from
home, the whole task, back home. (S / B3 start a take by hand after a B; N ends one in a scene with no steps.)
If an object falls off the table, the arm and scene reset by themselves (a take in progress is discarded).

The gripper is not kept pointing down: it turns with the phone, so the arm reaches as far as it does with the
leader arm. It can't reach every rotation (no wrist roll); it then keeps the position and turns as far as it can.

Recording (--record, src/robot_learning/config/dataset_schema_pioneer_v1.yaml): the keyboard teleop's features.
leader_angles / leader_counts are NaN (no leader arm), so phone, keyboard, leader and scripted takes can share
one dataset.
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2]
# pioneer_humanoid, humanoid_mujoco_scenes, humanoid_robot_learning; this fallback keeps an
# uninstalled checkout working.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "simulation" / "mujoco_scenes"))
sys.path.insert(0, str(_SRC / "robot_learning"))

from humanoid_robot_learning.sim_teleop_record import add_record_args, load_record_schema, make_sim_recorder  # noqa: E402
from pioneer_humanoid.mujoco_scene_window import SceneWindow  # noqa: E402

from phone_stream_test import (  # noqa: E402
    JUMP, Calibration, enter_pressed, load_calibration, quat_matrix, read_inputs, save_calibration,
)

CONTROL_DT = 0.01        # as the other teleops: one target update per 10 ms of sim time
WRIST_DAMPING = 2.5      # leader teleop's joint6l kv (the stock 18 makes the wrist lag)
TABLE_Z = 0.705          # table top of every humanoid_mujoco_scenes table scene
JOINT_RATE = 2.0         # rad/s, cap on every arm joint target
GRIP_RATE = 2.5          # gripper command per s (open -> closed in 0.4 s)
NARROW_OPEN = 0.3        # gripper command when open while working: jaws ~66 mm apart, sparing the neighbours
LEAD = 0.06              # m: the commanded grasp point never gets further than this ahead of the real one, so
                         # pushing on the table or a bin can't wind up force, and the arm never lunges
REACHED = 0.004          # m: an IK answer further than this from the point asked for is not used
MAX_TILT = math.radians(90)  # the gripper never leans past level (home points it straight ahead, level) ...
YAW_LIMIT = math.radians(75)  # ... and, pointing mostly down, turns at most this far either way (see limit_turn)
YAW_FROM_TILT = math.radians(60)   # the turn is only defined, and limited, while it leans less than this
TILT_SLACK = math.radians(10)      # an IK answer may lean this much past MAX_TILT on the way, no further
ROT_WEIGHT = 0.3       # the IK gives up rotation before position (leader IK default: 0.6)
WORKSPACE = ((0.05, 0.56), (-0.05, 0.75), (TABLE_Z + 0.01, 1.15))   # grasp point x, y, z box
STALE_S = 0.3            # no phone message for this long: stop following
GLIDE_S = 1.2            # glide home
HOME_HOLD_S = 0.5        # a finished take records this long at home before it is saved
B_FOLLOW, B_GRIP, B_TAKE = 1, 2, 3
GRASP_HEIGHT = 0.015     # m: finger tips this close to the table are low enough to close on any object
# The gripper's turn per turn of the phone, about the hand's x (right: tilting the phone's top away / toward
# you), y (away: tilting it sideways) and z (up: turning it flat). The forward tilt is reversed: the gripper
# faces you on the screen, and it reads right that way round.
TILT_SIGNS = (-1.0, 1.0, 1.0)
LEADER_SERVOS = ["A", "B", "C", "D", "E", "F", "G"]   # pioneer_leader_arm_teleop/servo_leader.SERVO_IDS


class PhoneReader:
    """The app's stream on a thread. latest(): position (jumps of the tracking taken out), rotation matrix, the
    buttons held, how often each was pressed so far, and the age of all that in seconds."""

    def __init__(self, family: str, name: str, wait: float):
        try:
            import hebi
        except ImportError:
            raise SystemExit("hebi-py is not installed:  pip install hebi-py --no-deps")
        print(f"[PHONE] Looking for the app (Family {family!r}, Name {name!r}) for up to {wait:.0f} s ...", flush=True)
        lookup = hebi.Lookup()
        self.group, deadline = None, time.monotonic() + wait
        while self.group is None and time.monotonic() < deadline:
            time.sleep(1.0)
            self.group = lookup.get_group_from_names([family], [name])
        if self.group is None:
            raise SystemExit("[PHONE] Not found. Run phone_stream_test.py: it says what is on the network.")
        self._lock = threading.Lock()
        self._pos, self._raw, self._R = np.zeros(3), None, np.eye(3)
        self._held, self._presses, self._stamp = [0] * 8, [0] * 8, 0.0
        self.jumps = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            feedback = self.group.get_next_feedback(timeout_ms=200)
            if feedback is None:
                continue
            module = feedback[0]
            position, quat = getattr(module, "ar_position", None), getattr(module, "ar_orientation", None)
            held, _ = read_inputs(getattr(module, "io", None))
            tracked = position is not None and quat is not None and bool(np.all(np.isfinite([*position, *quat])))
            with self._lock:
                self._presses = [n + (h and not was) for n, h, was in zip(self._presses, held, self._held)]
                self._held = held
                if not tracked:
                    continue                      # buttons still count; the pose (and its age) stay as they were
                raw = np.array([float(v) for v in position])
                if self._raw is not None:
                    step = raw - self._raw
                    if np.linalg.norm(step) > JUMP:   # the tracking corrected itself: the hand did not move
                        self.jumps += 1
                    else:
                        self._pos = self._pos + step
                self._raw = raw
                self._R = quat_matrix(*(float(v) for v in quat))      # the app sends w, x, y, z
                self._stamp = time.monotonic()

    def latest(self):
        with self._lock:
            return self._pos.copy(), self._R.copy(), list(self._held), list(self._presses), time.monotonic() - self._stamp

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def calibrate_phone(reader) -> tuple[np.ndarray, np.ndarray]:
    """In the terminal: check the gripper button, then the four moves. (position axes, rotation axes)."""
    print(f"[GRIPPER] First the gripper button: press B{B_GRIP} in the app. (Enter here skips this.)", flush=True)
    seen = reader.latest()[3][B_GRIP - 1]
    while True:
        _, _, held, presses, age = reader.latest()
        if presses[B_GRIP - 1] != seen:
            print(f"\n[GRIPPER] B{B_GRIP} works.", flush=True)
            break
        if enter_pressed():
            print(f"\n[GRIPPER] Skipped: B{B_GRIP} was NOT seen. K in the scene window also opens / closes.", flush=True)
            break
        print(f"\r[PHONE] waiting for B{B_GRIP} | buttons {''.join(map(str, held))} | "
              f"{'no tracked pose yet' if age > STALE_S else 'tracking'}".ljust(100), end="", flush=True)
        time.sleep(0.05)

    calibration = Calibration()
    print("[CALIBRATE] Sitting in front of the screen: +X is to your RIGHT, +Y is AWAY from you.", flush=True)
    print(f"[CALIBRATE] Enter here starts and ends each move (so does B{B_FOLLOW} in the app).", flush=True)
    print(calibration.prompt(), flush=True)
    taps = reader.latest()[3][B_FOLLOW - 1]
    while calibration.result is None:
        p, _, held, presses, age = reader.latest()
        pressed = enter_pressed() or presses[B_FOLLOW - 1] != taps
        taps = presses[B_FOLLOW - 1]
        if age > STALE_S:
            if pressed:
                print("\n[CALIBRATE] No tracked pose right now, so that press was ignored.", flush=True)
            print("\r[PHONE] no tracked pose: point the camera at a detailed, well-lit scene".ljust(100), end="", flush=True)
        else:
            calibration.add(p)
            if pressed:
                print(f"\n[CALIBRATE] {calibration.press(p)}", flush=True)
                if calibration.result is None and calibration.samples is None:
                    print(calibration.prompt(), flush=True)
            elif calibration.samples is None:
                print(f"\r[PHONE] waiting for Enter to start {calibration.step}".ljust(100), end="", flush=True)
            else:
                print(f"\r[PHONE] RECORDING {calibration.step}: {np.linalg.norm(p - calibration.samples[0]) * 100:3.0f} cm "
                      "from its start, Enter to end".ljust(100), end="", flush=True)
        time.sleep(CONTROL_DT)
    print("[PHONE] Calibrated.\n", flush=True)
    return calibration.result


def finger_tips(model, data) -> np.ndarray:
    """The point midway between the two finger tips, from the arm as it is now. Objects are 25-75 mm tall and
    the fingers 87 mm long, so it is the tips, not the middle of the fingers, that have to come down to them."""
    from pioneer_humanoid.mujoco_left_arm_ik import FINGERS_BELOW_GRASP, GRASP_POINT

    body = model.body("link6l").id
    R = data.xmat[body].reshape(3, 3)
    return data.xpos[body] + R @ GRASP_POINT - FINGERS_BELOW_GRASP * R[:, 2]      # the gripper points along its -z


def height_guide(model, data):
    """A decorate() for the scene window: a plumb line from between the finger tips to the table and a disc
    where it lands, green once the tips are low enough to grasp. On a flat screen the gripper's height over
    the table is otherwise a guess."""
    import mujoco

    def add(scn, kind, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom:
            return None
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                            np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        scn.ngeom += 1
        return geom

    def decorate(scn):
        tip = finger_tips(model, data)
        height = tip[2] - TABLE_Z
        if not 0.0 <= height < 0.6:
            return
        rgba = [0.1, 0.9, 0.2, 0.9] if height <= GRASP_HEIGHT else [1.0, 0.85, 0.1, 0.9]
        add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.0008, 0], [tip[0], tip[1], TABLE_Z + 0.001], rgba)
        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0], tip, rgba)
        line = add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0.001, 0.001, 0.001], tip, rgba)
        if line is not None and height > 0.002:
            mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, tip, np.array([tip[0], tip[1], TABLE_Z]))

    return decorate


def view_axes(azimuth_deg: float) -> np.ndarray:
    """Hand axes (x right, y away, z up, as calibrated) -> the robot's, for a scene seen from `azimuth_deg`
    (MuJoCo free camera): right on the screen, into the picture, up."""
    a = math.radians(azimuth_deg)
    return np.array([[math.sin(a), math.cos(a), 0.0], [-math.cos(a), math.sin(a), 0.0], [0.0, 0.0, 1.0]])


def rotation_vector(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis x angle (rad)."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    norm = float(np.linalg.norm(axis))
    return axis / norm * angle if norm > 1e-9 else np.zeros(3)


def rotation_matrix(v: np.ndarray) -> np.ndarray:
    """Axis x angle (rad) -> rotation matrix."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def gripper_tilt(R: np.ndarray) -> float:
    """How far (rad) the gripper leans from pointing straight down: 0 down, pi/2 level, pi fingers up."""
    return math.acos(max(-1.0, min(1.0, float(R[2, 2]))))


def limit_turn(R: np.ndarray) -> np.ndarray:
    """A gripper rotation (link6l's frame; pointing down is its z axis up) brought inside what the arm can hold
    without folding up: never leaning past level (MAX_TILT), and while it points mostly down, turned at most
    YAW_LIMIT about vertical. Past those the joints wind into a pose they can't get out of (the fingers pointing
    up, or the forearm twisted round)."""
    R = np.asarray(R, dtype=float)
    up = R[:, 2]
    tilt = math.acos(max(-1.0, min(1.0, float(up[2]))))
    if tilt > MAX_TILT:
        axis = np.cross(up, [0.0, 0.0, 1.0])
        norm = float(np.linalg.norm(axis))
        axis = axis / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        R = rotation_matrix(axis * (tilt - MAX_TILT)) @ R
    yaw = math.atan2(R[1, 0], R[0, 0])
    if min(tilt, MAX_TILT) < YAW_FROM_TILT and abs(yaw) > YAW_LIMIT:
        R = rotation_matrix(np.array([0.0, 0.0, math.copysign(YAW_LIMIT, yaw) - yaw])) @ R
    return R


class PhoneArm:
    """Phone pose (in the hand's axes: x right, y away, z up) -> smooth left-arm joint targets. No I/O: step()
    once per CONTROL_DT."""

    def __init__(self, model, data, scale: float = 1.0):
        from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS, LEFT_ARM_JOINTS
        from pioneer_humanoid.mujoco_left_arm_ik import LeftArmIK

        self.m, self.d, self.scale = model, data, scale
        self.ik = LeftArmIK(model)
        self.qadr = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
        self.home = np.array([DEFAULT_JOINT_POS[j] for j in LEFT_ARM_JOINTS])
        self.reset()

    def reset(self) -> None:
        """Arm at home, gripper wide open, following nothing."""
        self.q = self.home.copy()
        self.grip, self.grip_goal = 0.0, 0.0
        self.following = False
        self._anchor = None          # phone position / rotation and grasp point / gripper rotation when B1 went down
        self._glide = None           # (from, steps done) while gliding home

    @property
    def at_home(self) -> bool:
        return self._glide is None and np.allclose(self.q, self.home) and self.grip == self.grip_goal == 0.0

    def toggle_grip(self) -> None:
        self.grip_goal = NARROW_OPEN if self.grip_goal > 0.5 else 1.0

    def go_home(self) -> None:
        if self._glide is None and not np.allclose(self.q, self.home):
            self._glide = (self.q.copy(), 0)
        self.grip_goal = 0.0             # home is wide open, as the leader teleop's home

    def step(self, phone_pos, phone_R, follow: bool, view=None) -> tuple[np.ndarray, float]:
        """Advance one CONTROL_DT; returns (6 joint targets, gripper command 0..1). ``view``: hand axes -> the
        robot's (view_axes), taken when following starts; None: they are the same."""
        self.following = False
        if self._glide is not None:
            q0, i = self._glide
            s = min((i + 1) * CONTROL_DT / GLIDE_S, 1.0)
            self.q = q0 + (s * s * (3 - 2 * s)) * (self.home - q0)
            self._glide, self._anchor = (None if s >= 1.0 else (q0, i + 1)), None
        elif not follow:
            self._anchor = None
        else:
            self._follow(np.asarray(phone_pos, dtype=float), np.asarray(phone_R, dtype=float),
                         np.eye(3) if view is None else np.asarray(view, dtype=float))
        g_step = GRIP_RATE * CONTROL_DT
        self.grip += float(np.clip(self.grip_goal - self.grip, -g_step, g_step))
        return self.q.copy(), self.grip

    def _follow(self, phone_pos, phone_R, view) -> None:
        self.following = True
        if self._anchor is None:         # B1 just went down: from here on the gripper copies the phone's motion
            grasp, R = self.ik.fk(self.d.qpos, self.q)
            self._anchor = [phone_pos, phone_R, grasp, R, view]
            if self.grip_goal == 0.0:
                self.grip_goal = NARROW_OPEN     # home is wide open; work narrow
        phone_pos0, phone_R0, grasp0, R0, view = self._anchor     # the view as it was then: orbiting can't jerk the arm
        moved = self.scale * (view @ (phone_pos - phone_pos0))
        turned = view @ rotation_matrix(np.array(TILT_SIGNS) * rotation_vector(phone_R @ phone_R0.T)) @ view.T
        want = grasp0 + moved
        lo, hi = np.array([w[0] for w in WORKSPACE]), np.array([w[1] for w in WORKSPACE])
        want = np.clip(want, lo, hi)
        actual, _ = self.ik.fk(self.d.qpos, self.d.qpos[self.qadr])
        lead = want - actual
        if np.linalg.norm(lead) > LEAD:
            want = actual + lead / np.linalg.norm(lead) * LEAD
        # At the edge of the arm's reach the IK has no answer and its best guess can swing the arm away: take
        # the furthest point toward the phone that it does reach (within REACHED), else stay put.
        now, _ = self.ik.fk(self.d.qpos, self.q)
        R_want = limit_turn(turned @ R0)
        q = self.q
        # Never accept an answer that leans the gripper further past level than it already is: answers that let
        # the turn go (rot_weight 0, below) could otherwise drift it over, fingers up, where it gets stuck.
        tilt_now = gripper_tilt(self.ik.fk(self.d.qpos, self.q)[1])
        # With a joint against its limit, holding the rotation can block every move: then the gripper's turn
        # gives way to the hand's position (rot_weight 0), so the arm can always be moved out again.
        for part, weight in ((1.0, ROT_WEIGHT), (1.0, 0.0), (0.5, ROT_WEIGHT), (0.5, 0.0), (0.2, 0.0), (0.0, ROT_WEIGHT)):
            target = now + part * (want - now)
            q_try, err = self.ik.solve_pose(self.d.qpos, self.q, target, R_want, iters=15, rot_weight=weight)
            tilt = gripper_tilt(self.ik.fk(self.d.qpos, q_try)[1])
            if err < REACHED and tilt <= max(MAX_TILT + TILT_SLACK, tilt_now):
                want, q = target, q_try
                break
        else:
            want = now
        # Whatever the box, the lead or the arm's reach took off is dropped for good, not paid back later: the
        # arm then follows the hand's next motion straight away instead of first catching up with the old one.
        self._anchor[2] = want - moved
        # The same for a turn past the limits (limit_turn): turning the phone further adds nothing, and turning
        # it back turns the gripper back straight away -- strokes of B1 can't wind it past where it can go.
        self._anchor[3] = turned.T @ R_want
        dq = q - self.q
        # A step some joint can't make in time is scaled on every joint alike (see the keyboard teleop).
        self.q = self.q + dq * min(1.0, JOINT_RATE * CONTROL_DT / max(float(np.abs(dq).max()), 1e-9))


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="tidy_table", help="scene in humanoid_mujoco_scenes (an unknown name lists them)")
    parser.add_argument("--family", default="HEBI", help="the app's Family setting (default: HEBI)")
    parser.add_argument("--name", default="mobileIO", help="the app's Name setting (default: mobileIO)")
    parser.add_argument("--wait", type=float, default=20.0, help="seconds to look for the phone before giving up")
    parser.add_argument("--calibrate", action="store_true",
                        help="calibrate the phone's axes again (default: use the saved calibration, if there is one)")
    parser.add_argument("--scale", type=float, default=1.0, help="gripper travel per phone travel (default 1: the same)")
    add_record_args(parser, task_description="sim phone teleop demonstration")
    args = parser.parse_args()
    record = load_record_schema(parser, args)

    import mujoco
    from humanoid_mujoco_scenes import (
        list_scenes, make_model, scene_camera, scene_condition, scene_progress, scene_reset, scene_step,
    )
    from humanoid_mujoco_scenes.fallen import off_table
    from pioneer_humanoid.arm_params import (
        CAMERA_NAMES, LEFT_ARM_JOINTS, LEFT_GRIPPER_CLOSED, LEFT_GRIPPER_JOINTS, LEFT_GRIPPER_OPEN,
    )
    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    cameras = {name: (int(spec["height"]), int(spec["width"])) for name, spec in record.images.items()}
    model = make_model(args.scene, cameras={n: hw for n, hw in cameras.items() if n in CAMERA_NAMES})
    unknown = sorted(n for n in cameras
                     if n not in CAMERA_NAMES and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) < 0)
    if unknown:
        raise SystemExit(f"{record.path}: images {unknown} are neither arm cameras {list(CAMERA_NAMES)} "
                         f"nor cameras of scene {args.scene!r}")

    reader = PhoneReader(args.family, args.name, args.wait)
    saved = None if args.calibrate else load_calibration()
    if saved is None:
        saved = calibrate_phone(reader)
        save_calibration(*saved)
    pos_axes, rot_axes = saved

    model.actuator_biasprm[model.actuator("joint6l").id, 2] = -WRIST_DAMPING
    data = mujoco.MjData(model)
    hook, randomise = scene_step(args.scene), scene_reset(args.scene)
    steps_of, cond = scene_progress(args.scene), scene_condition(args.scene)
    rng = np.random.default_rng()
    arm_acts = [model.actuator(j).id for j in LEFT_ARM_JOINTS]
    grip_acts = [model.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
    arm_qpos = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
    grip_qpos = [model.joint(j).qposadr[0] for j in LEFT_GRIPPER_JOINTS]
    g_open = np.array([LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS])
    g_closed = np.array([LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS])
    substeps = max(1, round(CONTROL_DT / model.opt.timestep))

    def reset_scene():
        mujoco.mj_resetData(model, data)
        set_home(model, data)
        if randomise is not None:
            randomise(model, data, rng)
        mujoco.mj_forward(model, data)
        grounded.clear()
        grounded.update(off_table(model, data))

    grounded: set[str] = set()             # loose bodies that start below the table: not "fallen"
    reset_scene()
    arm = PhoneArm(model, data, args.scale)

    extra = {"leader_angles": LEADER_SERVOS, "leader_counts": LEADER_SERVOS}
    if steps_of is not None:
        extra["subtask_index"] = ["subtask_index"]
    if cond is not None:
        extra["observation.environment_state"] = cond[0]
    recorder, record_every = make_sim_recorder(args, record, device="cpu", sim_dt=CONTROL_DT, extra_features=extra)
    renderers = {name: mujoco.Renderer(model, h, w) for name, (h, w) in cameras.items()}
    nan7 = np.full(len(LEADER_SERVOS), np.nan, dtype=np.float32)

    def read_images():
        out = {}
        for name, r in renderers.items():
            r.update_scene(data, name)
            out[name] = r.render()
        return out

    window = SceneWindow(model, data, f"Pioneer phone teleop: {args.scene}", scene_camera(args.scene), one_shot="sknbhrf")
    window.decorate = height_guide(model, data)
    print(f"[INFO] Hold B{B_FOLLOW} in the app to move the arm; B{B_GRIP} gripper. Scene window: H home, R reset, Esc quit"
          + (", B discard the take." if recorder is not None else "."), flush=True)
    if recorder is not None:
        print(f"[RECORD] Each take starts by itself when you first press B{B_FOLLOW}, and saves itself when the task is "
              "done.", flush=True)
    recording, saved, finishing, home_steps = False, 0, False, 0
    armed = recorder is not None           # the next B1 press, from home, starts a take
    wrap_up = False                        # N in a scene with no steps: go home, then save
    try:                                   # the scene's own verdict on an episode, if it gives one (tidy_table does)
        status_of = getattr(importlib.import_module(f"humanoid_mujoco_scenes.{args.scene}.scene"), "episode_status", None)
    except ImportError:
        status_of = None
    follow_key = False                     # F: follow without B1
    step, shown, msg = 0, None, ""
    seen = reader.latest()[3]
    azimuth = float((scene_camera(args.scene) or {}).get("azimuth", 90.0))
    view = view_axes(azimuth)

    def new_layout():
        nonlocal recording, shown, finishing, home_steps, follow_key, armed, wrap_up
        recording, shown, finishing, home_steps, follow_key = False, None, False, 0, False
        armed, wrap_up = recorder is not None, False
        with window.lock:
            reset_scene()
        arm.reset()

    def save_take():
        """The take has ended at home. Only a success is kept, so every saved take has the same shape: from
        home, the whole task in order, back home."""
        nonlocal saved
        status = status_of(model, data) if status_of is not None and not wrap_up else None
        if status is not None and not status["success"]:
            why = ", ".join(f"{k} {status[k]}" for k in ("dropped", "toppled", "early") if status.get(k)) or "task not done"
            recorder.cancel_recording()
            print(f"\n[RECORD] Take NOT saved ({why}); new layout.", flush=True)
        else:
            recorder.save_episode()
            saved += 1
            print(f"\n[RECORD] Take {saved} saved; new layout.", flush=True)
        new_layout()

    def start_take():
        nonlocal recording, armed
        if recorder is not None and not recording:
            recording, armed = True, False
            print("\n[RECORD] Recording a take (B discards it).", flush=True)

    try:
        next_t = time.monotonic()
        while window.is_running():
            phone_pos, phone_R, held, presses, age = reader.latest()
            pressed = [n != was for n, was in zip(presses, seen)]
            seen = presses
            events = list(window.events)
            window.events.clear()
            if pressed[B_GRIP - 1]:
                events.append("k")
            if pressed[B_TAKE - 1]:
                events.append("s")
            for ev in events:
                if ev == "esc":
                    raise KeyboardInterrupt
                if ev in ("k", "h", "f") and finishing:
                    pass                               # the arm is going home by itself
                elif ev == "k":
                    arm.toggle_grip()
                elif ev == "h":
                    arm.go_home()
                elif ev == "f":
                    follow_key = not follow_key
                elif ev == "s":
                    start_take()
                elif ev == "r":
                    if recording:
                        recorder.cancel_recording()
                        print("\n[RECORD] Reset mid-take: take discarded.", flush=True)
                    new_layout()
                elif ev == "n" and recording and steps_of is None:
                    wrap_up = True                     # no steps to tell when it is done: home, then save
                elif ev == "n" and recorder is not None:
                    print("\n[RECORD] Takes save themselves when the last step is done; nothing to do." if steps_of is not None
                          else "\n[RECORD] Nothing to save: no take is recording.", flush=True)
                elif ev == "b" and recording:
                    recorder.cancel_recording()
                    recording, finishing, home_steps, wrap_up = False, False, 0, False
                    print("\n[RECORD] Take discarded. R for a new layout (S records from here).", flush=True)

            fresh = age < STALE_S
            follow = fresh and not finishing and (bool(held[B_FOLLOW - 1]) or follow_key)
            if follow and armed and arm.at_home:
                start_take()                           # before the arm moves: the take's first frame is at home
            view = view_axes(float(getattr(getattr(window, "cam", None), "azimuth", azimuth)))
            # App frame -> hand axes (the calibration: x right, y away, z up); the arm turns those into the
            # robot's, as the view shows them.
            q, grip = arm.step(pos_axes @ phone_pos, rot_axes @ phone_R @ rot_axes.T, follow, view)
            if recording and step % record_every == 0:
                closure = np.mean((data.qpos[grip_qpos] - g_open) / (g_closed - g_open))
                state = np.append(data.qpos[arm_qpos], np.clip(closure, 0.0, 1.0)).astype(np.float32)
                extras = {"leader_angles": nan7, "leader_counts": nan7}
                task = None
                if steps_of is not None:
                    index, _, task = steps_of(model, data)
                    extras["subtask_index"] = np.array([index], dtype=np.float32)
                if cond is not None:
                    extras["observation.environment_state"] = cond[1](model, data)
                recorder.push_frame_to_buffer(np.append(q, grip).astype(np.float32), state, read_images(),
                                              extras=extras, task=task)
            with window.lock:
                data.ctrl[arm_acts] = q
                data.ctrl[grip_acts] = g_open + grip * (g_closed - g_open)
                if hook is not None:
                    hook(model, data)
                mujoco.mj_step(model, data, nstep=substeps)
            step += 1
            fell = off_table(model, data) - grounded
            if fell:
                # Nothing can pick it up from the floor: this layout is over.
                print(f"\n[RESET] {', '.join(sorted(fell))} fell off the table: new layout"
                      + (" (take discarded)." if recording else "."), flush=True)
                if recording:
                    recorder.cancel_recording()
                new_layout()
            done = False
            if steps_of is not None:
                index, total, instruction = steps_of(model, data)
                done = index >= total
                if (index, instruction) != shown:
                    shown = (index, instruction)
                    done_msg = "going home, then saving" if recording else "R for a new layout"
                    msg = f"all steps done -- {done_msg}" if done else f"step {index + 1}/{total}: {instruction}"
                    print(f"\n[TASK] {msg}", flush=True)
            # The task is done: glide home, hold there a moment and save, so every take ends the same way.
            finishing = recording and (done or wrap_up)
            if finishing:
                arm.go_home()                          # does nothing once gliding or home
                home_steps = home_steps + 1 if arm.at_home else 0
                if home_steps >= int(HOME_HOLD_S / CONTROL_DT):
                    save_take()
            height = float(finger_tips(model, data)[2]) - TABLE_Z
            lines = ["" if recorder is None else "RECORDING" if recording
                     else f"ready: recording starts when you press B{B_FOLLOW}" if armed else "not recording (S starts)",
                     "PHONE: no tracked pose" if not fresh else "FOLLOWING the phone" if arm.following
                     else f"hold B{B_FOLLOW} to move the arm",
                     "finger tips at the table: low enough to grab" if height <= GRASP_HEIGHT
                     else f"finger tips {height * 100:.0f} cm above the table: lower them to grab", msg]
            window.sync("\n".join(s for s in lines if s))
            next_t += CONTROL_DT
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        window.close()
        reader.close()
        if reader.jumps:
            print(f"\n[PHONE] {reader.jumps} tracking jump(s) were left out of the motion.", flush=True)
        if recorder is not None:
            if recording:
                recorder.cancel_recording()
                print("\n[RECORD] The take in progress was not saved (N saves).", flush=True)
            # Interrupted here, the dataset folder is left half-written and can't be re-opened.
            if saved:
                print("\n[RECORD] Finishing the saved takes -- wait, don't press Ctrl+C.", flush=True)
            recorder.finalize()
            print(f"\n[RECORD] {saved} take(s) saved this session under {recorder.dataset_root}")


if __name__ == "__main__":
    run()
