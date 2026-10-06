"""Keyboard teleoperation of the Pioneer LEFT arm in plain MuJoCo (CPU), recording like the leader teleop.

    python src/teleop/keyboard_teleop/mujoco_keyboard_teleop.py --scene tidy_table [--record --cameras top,wrist_left]

Keep the TERMINAL focused, not the viewer: keys are read globally (pynput), and the MuJoCo viewer maps most
letters to display toggles (W wireframe, R reflections, C contact points, ...). On macOS run with ``mjpython``
and allow the terminal under Privacy > Accessibility (pynput needs it to see key releases).

The gripper always points straight down (top-down grasps; this arm has no wrist roll) and keys move its grasp
point, the spot between the finger pads:

  W / S   away from / toward the robot (x)        A / D   left / right (y)        Q / E   up / down
  C / V   turn the gripper left / right           K       open / close the gripper (eases over 0.4 s)
  J       open narrow (default, ~66 mm: fits every object in these scenes and spares its neighbours) / wide
  Shift   hold for fine control (quarter speed)    H       glide back home (end every take with it)
  R       reset the arm and the scene (new layout); drops a take in progress
  --record:  P start a take    N save it (then reset)    B discard it    Esc quit

The first move key glides the arm from home (gripper pointing forward) to gripper-down above the table
before you take over: asking for "down" straight from home would flip the wrist in one step and sweep the
fingers through whatever is on the table.

Smooth, and contacts stay clean: speeds ramp up and down, joint targets are rate-limited, the commanded pose
never leads the real grasp point by more than LEAD (so pushing on a bin or an object can't wind up force),
the finger tips can't be commanded into the table, and a pose the arm can't reach just stops the motion.

Recording (--record, src/robot_learning/config/dataset_schema_pioneer_v1.yaml), the leader teleop's features:
observation.state, action (6 joint targets + gripper 0..1), observation.images.*, task / subtask_index and
observation.environment_state for multi-step scenes. leader_angles / leader_counts are NaN (no leader arm),
so keyboard, leader and scripted takes can share one dataset.
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2]
# pioneer_humanoid, humanoid_mujoco_scenes, humanoid_robot_learning; this fallback keeps an
# uninstalled checkout working.
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "simulation" / "mujoco_scenes"))
sys.path.insert(0, str(_SRC / "robot_learning"))

from humanoid_robot_learning.sim_teleop_record import add_record_args, load_record_schema, make_sim_recorder  # noqa: E402

CONTROL_DT = 0.01        # same as the leader teleop: one key read and target update per 10 ms of sim time
WRIST_DAMPING = 2.5      # leader teleop's joint6l kv (the stock 18 makes the wrist lag)
TABLE_Z = 0.705          # table top of every humanoid_mujoco_scenes table scene
SPEED = 0.12             # m/s, grasp point, keys held
YAW_RATE = 0.8           # rad/s
FINE = 0.25              # Shift
ACCEL = 1.0              # m/s^2 (and YAW_ACCEL rad/s^2): reaches full speed in ~0.1 s, stops as fast
YAW_ACCEL = 6.0
JOINT_RATE = 2.0         # rad/s, cap on every arm joint target
STALL_STEPS = 5          # control steps (50 ms) pushing while the arm doesn't follow = blocked by something
GRIP_RATE = 2.5          # gripper command per s (open -> closed in 0.4 s)
NARROW_OPEN = 0.3        # gripper command for the narrow opening: jaws ~66 mm apart (wide open = 0: ~96 mm)
LEAD = 0.002             # m: how far the commanded grasp point may get ahead of the real one when still ...
LEAD_LAG = 0.08          # s: ... plus speed x this, the arm's normal lag behind a moving command. Blocked, the
                         # speed drops to 0 and so does the lead: the stiff joints press lightly, not ~120 N.
TIP_CLEARANCE = 0.003    # m: finger tips stay this far above the table
WORKSPACE = ((0.10, 0.50), (0.12, 0.58), (None, 1.10))   # grasp point x, y, z box (z floor from the table)
YAW_LIMIT = math.radians(60)
READY = (0.27, 0.30, 0.96)   # grasp point, gripper down: high over the table, near the home pose's height
GLIDE_S = 1.2            # home <-> ready
LEADER_SERVOS = ["A", "B", "C", "D", "E", "F", "G"]   # pioneer_leader_arm_teleop/servo_leader.SERVO_IDS

KEY_HELP = """[KEYS] W/S x  A/D y  Q/E up/down  C/V turn  K gripper  J narrow/wide  Shift fine  H home  R reset  Esc quit
[KEYS] recording: P start take  N save  B discard"""
MOVE_KEYS = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1), "q": (2, +1), "e": (2, -1)}
TURN_KEYS = {"c": +1, "v": -1}


class KeyboardArm:
    """Held keys -> smooth, collision-safe left-arm joint targets. No I/O: step() once per CONTROL_DT."""

    def __init__(self, model, data):
        import mujoco  # noqa: F401
        from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS, LEFT_ARM_JOINTS
        from pioneer_humanoid.mujoco_left_arm_ik import FINGERS_BELOW_GRASP, LeftArmIK

        self.m, self.d = model, data
        self.ik = LeftArmIK(model)
        self.qadr = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
        self.home = np.array([DEFAULT_JOINT_POS[j] for j in LEFT_ARM_JOINTS])
        self.z_floor = TABLE_Z + FINGERS_BELOW_GRASP + TIP_CLEARANCE
        q, err = self.ik.solve_any(data.qpos, [self.home], READY, 0.0)
        if err > 0.003:
            raise RuntimeError(f"ready pose {READY} out of reach ({err * 1000:.0f} mm)")
        self.q_ready = q
        self.reset()

    def reset(self) -> None:
        """Arm at home, gripper open, waiting for the first move key."""
        self.mode = "home"           # home -> glide_ready -> teleop -> glide_home -> home
        self.q = self.home.copy()
        self.open_cmd = NARROW_OPEN
        self.grip, self.grip_goal = 0.0, 0.0
        self.vel, self.yaw_vel = np.zeros(3), 0.0
        self.pos, self.yaw = np.array(READY, dtype=float), 0.0
        self.blocked = None          # unit direction the arm was pushed into something along, while still pushed
        self._stall, self._last_actual = 0, None
        self._glide = None

    # -- one-shot commands
    def toggle_grip(self) -> None:
        self.grip_goal = self.open_cmd if self.grip_goal > 0.5 else 1.0

    def toggle_wide(self) -> None:
        """Switch the open width between narrow and wide (applies now if the gripper is open)."""
        self.open_cmd = 0.0 if self.open_cmd > 0 else NARROW_OPEN
        if self.grip_goal < 0.5:
            self.grip_goal = self.open_cmd

    def go_home(self) -> None:
        if self.mode in ("teleop", "glide_ready"):
            self.vel[:], self.yaw_vel = 0.0, 0.0
            self.grip_goal = 0.0                     # home is wide open, as the leader teleop's home
            self._start_glide(self.home, "glide_home")

    def _start_glide(self, q_to, mode) -> None:
        self._glide = (self.q.copy(), np.asarray(q_to, dtype=float), 0)
        self.mode = mode

    # -- per control step
    def step(self, held: set[str], fine: bool = False) -> tuple[np.ndarray, float]:
        """Advance one CONTROL_DT with these keys held; returns (6 joint targets, gripper command 0..1)."""
        moving = any(k in held for k in MOVE_KEYS) or any(k in held for k in TURN_KEYS)
        if self.mode == "home" and moving:
            self._start_glide(self.q_ready, "glide_ready")
            self.grip_goal = self.open_cmd        # home is wide open; work narrow (unless J chose wide)
        if self.mode in ("glide_ready", "glide_home"):
            q0, q1, i = self._glide
            n = int(GLIDE_S / CONTROL_DT)
            i += 1
            s = min(i / n, 1.0)
            self.q = q0 + (s * s * (3 - 2 * s)) * (q1 - q0)
            self._glide = (q0, q1, i)
            if s >= 1.0:
                if self.mode == "glide_ready":
                    self.mode = "teleop"
                    self._last_actual, self._stall, self.blocked = None, 0, None
                    self.pos, R = self.ik.fk(self.d.qpos, self.q)
                    self.yaw = self.ik.yaw_of(R)
                else:
                    self.mode = "home"
        elif self.mode == "teleop":
            self._teleop(held, fine)
        g_step = GRIP_RATE * CONTROL_DT
        self.grip += float(np.clip(self.grip_goal - self.grip, -g_step, g_step))
        return self.q.copy(), self.grip

    def _teleop(self, held, fine) -> None:
        scale = FINE if fine else 1.0
        want = np.zeros(3)
        for k, (axis, sign) in MOVE_KEYS.items():
            if k in held:
                want[axis] += sign
        if np.linalg.norm(want) > 0:
            want = want / np.linalg.norm(want) * SPEED * scale
        actual, _ = self.ik.fk(self.d.qpos, self.d.qpos[self.qadr])
        arm_vel = (actual - self._last_actual) / CONTROL_DT if self._last_actual is not None else np.zeros(3)
        self._last_actual = actual
        if self.blocked is not None:
            if float(want @ self.blocked) > 0:   # still pushing into it: keep holding where the arm is
                self.vel[:] = 0.0
                self._hold()
                return
            self.blocked = None
        dv = ACCEL * CONTROL_DT
        self.vel += np.clip(want - self.vel, -dv, dv)
        want_yaw = sum(s for k, s in TURN_KEYS.items() if k in held) * YAW_RATE * scale
        dw = YAW_ACCEL * CONTROL_DT
        self.yaw_vel += float(np.clip(want_yaw - self.yaw_vel, -dw, dw))
        if not np.any(self.vel) and self.yaw_vel == 0.0:
            self._stall = 0
            return

        pos = self.pos + self.vel * CONTROL_DT
        (x0, x1), (y0, y1), (_, z1) = WORKSPACE
        lo, hi = np.array([x0, y0, self.z_floor]), np.array([x1, y1, z1])
        edge = ((pos <= lo) & (self.vel < 0)) | ((pos >= hi) & (self.vel > 0))
        self.vel[edge] = 0.0                     # at the workspace edge (finger tips just over the table): stop
        pos = np.clip(pos, lo, hi)
        # Don't lead the real grasp point by more than its normal lag behind a moving command: blocked by a
        # contact, the command waits for the arm instead of winding up force.
        lead, max_lead = pos - actual, LEAD + LEAD_LAG * float(np.linalg.norm(self.vel))
        if np.linalg.norm(lead) > max_lead:
            pos = np.clip(actual + lead / np.linalg.norm(lead) * max_lead, lo, hi)
        # Blocked: pushed along an axis for STALL_STEPS while the arm itself doesn't follow.
        moving = self.vel != 0
        stalled = moving & (arm_vel * np.sign(self.vel) < 0.2 * np.abs(self.vel)) \
            & ((pos - actual) * np.sign(self.vel) > LEAD)
        self._stall = self._stall + 1 if np.any(stalled) else 0
        if self._stall >= STALL_STEPS:
            # Stop pushing and hold the arm where it actually is, joint for joint, until the push stops or turns
            # away. These stiff joints barely sag (~0.6 mrad hovering), so that presses with ~4 N; re-solving
            # for "pointing down" would instead fight the tilt the contact gives the gripper -- through the
            # contact, ~80-110 N.
            d = np.where(stalled, np.sign(self.vel), 0.0)
            self.blocked = d / np.linalg.norm(d) if np.any(d) else None
            self.vel[:], self._stall = 0.0, 0
            self._hold()
            return
        yaw = float(np.clip(self.yaw + self.yaw_vel * CONTROL_DT, -YAW_LIMIT, YAW_LIMIT))

        # Seeded from the arm's actual joints, not the last target: pressed against something, a target that
        # wanders along a direction the contact pins (elbow up, wrist down) makes the joints fight each other
        # through the fingers while the grasp point barely moves.
        q, err = self.ik.solve(self.d.qpos, self.d.qpos[self.qadr], pos, yaw, iters=15)
        if err > 0.004:                  # out of reach that way: stop there, smoothly
            self.vel[:], self.yaw_vel = 0.0, 0.0
            return
        dq = np.clip(q - self.q, -JOINT_RATE * CONTROL_DT, JOINT_RATE * CONTROL_DT)
        self.q = self.q + dq
        self.pos, self.yaw = pos, yaw

    def _hold(self) -> None:
        q_actual = self.d.qpos[self.qadr]
        self.q = self.q + np.clip(q_actual - self.q, -JOINT_RATE * CONTROL_DT, JOINT_RATE * CONTROL_DT)
        self.pos, R = self.ik.fk(self.d.qpos, q_actual)
        self.yaw = self.ik.yaw_of(R)


class KeyReader:
    """Global key state from pynput: held letters + Shift, and press events for the one-shot keys."""

    ONE_SHOT = set("kjhrpnb")

    def __init__(self):
        from pynput import keyboard as kb

        self.kb = kb
        self.held: set[str] = set()
        self.shift = False
        self.events: queue.Queue = queue.Queue()
        self.listener = kb.Listener(on_press=self._press, on_release=self._release)
        self.listener.daemon = True
        self.listener.start()

    def _char(self, key):
        c = getattr(key, "char", None)
        return c.lower() if c else None

    def _press(self, key):
        if key in (self.kb.Key.shift, self.kb.Key.shift_l, self.kb.Key.shift_r):
            self.shift = True
        elif key == self.kb.Key.esc:
            self.events.put("esc")
        else:
            c = self._char(key)
            if c and c not in self.held:           # key repeat sends presses while held: act on the first
                self.held.add(c)
                if c in self.ONE_SHOT:
                    self.events.put(c)

    def _release(self, key):
        if key in (self.kb.Key.shift, self.kb.Key.shift_l, self.kb.Key.shift_r):
            self.shift = False
        else:
            self.held.discard(self._char(key))

    def stop(self):
        self.listener.stop()


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="tidy_table", help="scene in humanoid_mujoco_scenes (an unknown name lists them)")
    add_record_args(parser, task_description="sim keyboard teleop demonstration")
    args = parser.parse_args()
    record = load_record_schema(parser, args)

    import mujoco
    import mujoco.viewer
    from humanoid_mujoco_scenes import (
        list_scenes, make_model, scene_camera, scene_condition, scene_progress, scene_reset, scene_step,
    )
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

    reset_scene()
    arm = KeyboardArm(model, data)

    extra = {"leader_angles": LEADER_SERVOS, "leader_counts": LEADER_SERVOS}
    if steps_of is not None:
        extra["subtask_index"] = ["subtask_index"]
    if cond is not None:
        extra["observation.environment_state"] = cond[0]
    recorder, record_every = make_sim_recorder(args, record, device="cpu", sim_dt=CONTROL_DT, extra_features=extra)
    renderers = {name: mujoco.Renderer(model, h, w) for name, (h, w) in cameras.items()}
    nan7 = np.full(len(LEADER_SERVOS), np.nan, dtype=np.float32)
    recording = False

    def read_images():
        out = {}
        for name, r in renderers.items():
            r.update_scene(data, name)
            out[name] = r.render()
        return out

    try:
        keys = KeyReader()
    except ImportError:
        raise SystemExit("keyboard teleop needs pynput: pip install pynput")
    print(KEY_HELP, flush=True)
    print("[INFO] Keep this terminal focused. Press a move key to start.", flush=True)
    step, shown = 0, None
    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            for k, v in (scene_camera(args.scene) or {}).items():
                setattr(viewer.cam, k, v)
            next_t = time.monotonic()
            while viewer.is_running():
                while not keys.events.empty():
                    ev = keys.events.get()
                    if ev == "esc":
                        raise KeyboardInterrupt
                    if ev == "k":
                        arm.toggle_grip()
                    elif ev == "j":
                        arm.toggle_wide()
                    elif ev == "h":
                        arm.go_home()
                    elif ev == "r" or (ev == "n" and recording):
                        if ev == "n":
                            recorder.save_episode()
                            print("\n[RECORD] Take saved; new layout.", flush=True)
                        elif recording:
                            recorder.cancel_recording()
                            print("\n[RECORD] Reset mid-take: take discarded.", flush=True)
                        recording = False
                        with viewer.lock():
                            reset_scene()
                        arm.reset()
                        shown = None
                    elif ev == "p" and recorder is not None and not recording:
                        recording = True
                        print("\n[RECORD] Recording a take (N save, B discard).", flush=True)
                    elif ev == "b" and recording:
                        recorder.cancel_recording()
                        recording = False
                        print("\n[RECORD] Take discarded (P to start again).", flush=True)

                q, grip = arm.step(keys.held, keys.shift)
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
                    with viewer.lock():
                        recorder.push_frame_to_buffer(np.append(q, grip).astype(np.float32), state, read_images(),
                                                      extras=extras, task=task)
                with viewer.lock():
                    data.ctrl[arm_acts] = q
                    data.ctrl[grip_acts] = g_open + grip * (g_closed - g_open)
                    if hook is not None:
                        hook(model, data)
                    mujoco.mj_step(model, data, nstep=substeps)
                step += 1
                if steps_of is not None:
                    index, total, instruction = steps_of(model, data)
                    if (index, instruction) != shown:
                        shown = (index, instruction)
                        msg = "all steps done -- H, then N to save" if index >= total else \
                            f"step {index + 1}/{total}: {instruction}"
                        print(f"\n[TASK] {msg}", flush=True)
                viewer.sync()
                next_t += CONTROL_DT
                time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        if recorder is not None:
            if recording:
                recorder.cancel_recording()
            recorder.finalize()
            print(f"\n[RECORD] Saved under {recorder.dataset_root}")


if __name__ == "__main__":
    run()
