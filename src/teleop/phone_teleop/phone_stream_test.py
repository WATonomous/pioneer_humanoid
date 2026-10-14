"""Does an iPhone's tracked position and rotation reach this computer? Prints them; drives nothing.

    pip install hebi-py --no-deps        # --no-deps: it pins numpy<2, which would downgrade the repo's numpy
    python src/teleop/phone_teleop/phone_stream_test.py [--calibrate]

On the phone: the free "HEBI Mobile I/O" app (App Store), open and in the foreground, on the same Wi-Fi as
this computer. The app streams the phone's ARKit pose (camera-based tracking) plus its on-screen buttons
B1..B8 and sliders A1..A8. Ctrl+C stops.

B2 is the gripper: each press switches it between OPEN and CLOSED (shown on the line; nothing is driven).
With --calibrate, the first thing asked for is one press of B2, to see that the button arrives.

The position is shown from zero: relative to where the phone was when the display started.

--calibrate: the app's level axes point wherever the phone faced when tracking began (and which of its three
axes is vertical is worked out from the moves, not assumed). Show the
directions you want instead with four moves: toward +X, back toward -X, toward +Y, back toward -Y. Press
Enter in this terminal (or B1 in the app) at the start and at the end of each. A straight line is
fitted through every position recorded during a move, and each axis is taken from both its moves. x and y (and the rotation) are then shown in those axes,
zero where the first move began. They are saved (phone_calibration.json here) and used by later runs
without --calibrate -- right only while the app has stayed open and tracking; after that, calibrate again.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

CALIBRATION_FILE = Path(__file__).resolve().parent / "phone_calibration.json"
MIN_STROKE = 0.10        # m: a calibration move shorter than this is asked for again
GRIPPER_BUTTON = 2       # B2 in the app: each press opens / closes the gripper
JUMP = 0.03              # m between two messages (~10 ms): no hand moves 3 m/s, so the tracking corrected itself


def quat_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix -> roll, pitch, yaw in degrees (about x, y, z)."""
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def read_inputs(io) -> tuple[list[int], list[float]]:
    """The app's buttons B1..B8 (0/1) and sliders A1..A8, however this hebi-py version reports them."""
    def pin(bank, n, kinds):
        for kind in kinds:
            try:
                if getattr(bank, f"has_{kind}")(n):
                    return float(getattr(bank, f"get_{kind}")(n))
            except Exception:
                pass
        return 0.0

    if io is None:
        return [0] * 8, [0.0] * 8
    return ([int(pin(io.b, n, ("int", "float", "bool")) != 0) for n in range(1, 9)],
            [pin(io.a, n, ("float", "int")) for n in range(1, 9)])


def enter_pressed() -> bool:
    """True once per Enter / Space typed in this terminal (Windows; elsewhere the phone's buttons only)."""
    try:
        import msvcrt
    except ImportError:
        return False
    hit = False
    while msvcrt.kbhit():
        hit = msvcrt.getwch() in ("\r", "\n", " ") or hit
    return hit


def horizontal(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    return v - (v @ up) * up


def up_axis(move_x: np.ndarray, move_y: np.ndarray) -> tuple[np.ndarray, float]:
    """Which of the app's axes is vertical: the one most square to both moves (they are made level, across a
    table). Not assumed: ARKit itself has y up, other apps z. Also how level the moves were, 1 = exactly."""
    normal = np.cross(move_x, move_y)
    normal = normal / np.linalg.norm(normal)
    up = np.zeros(3)
    up[int(np.argmax(np.abs(normal)))] = 1.0
    return up, float(abs(normal @ up))


def calibrate_axes(move_x: np.ndarray, move_y: np.ndarray, UP: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """From the X and Y directions shown, (position axes, rotation axes, angle between the moves in deg, mirrored).

    Rows of each 3x3 are the new x, y, z in the app's frame. +X is the first direction, flattened to horizontal.
    +Y is square to it, on the side the second one points (they are never exactly 90 deg apart by hand).
    The third axis is UP, the app's vertical.
    If +Y was shown to the right of +X (seen from above) the axes are mirrored: positions follow
    what was shown, but a rotation can't be mirrored, so it uses +Y to the left.
    """
    x = horizontal(move_x, UP)
    x = x / np.linalg.norm(x)
    y_shown = horizontal(move_y, UP)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(x @ y_shown) / float(np.linalg.norm(y_shown))))))
    y_left = np.cross(UP, x)
    mirrored = float(y_shown @ y_left) < 0
    y = -y_left if mirrored else y_left
    return np.vstack([x, y, UP]), np.vstack([x, y_left, UP]), angle, mirrored


def save_calibration(pos_axes: np.ndarray, rot_axes: np.ndarray, path: Path = CALIBRATION_FILE) -> None:
    path.write_text(json.dumps({"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "position_axes": np.asarray(pos_axes).tolist(),
                                "rotation_axes": np.asarray(rot_axes).tolist()}, indent=2) + "\n")
    print(f"[CALIBRATE] Saved to {path}.", flush=True)


def load_calibration(path: Path = CALIBRATION_FILE):
    """(position axes, rotation axes) saved by an earlier run, or None. It is only right while the app has kept
    tracking since: the app's level axes are set anew whenever its tracking starts."""
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    print(f"[CALIBRATE] Using the calibration saved {saved.get('saved', '?')} ({path.name}). It only holds while the "
          "app has stayed open and tracking since then: if directions are off, run again with --calibrate.", flush=True)
    return np.array(saved["position_axes"], dtype=float), np.array(saved["rotation_axes"], dtype=float)


def fit_line(samples: list[np.ndarray]) -> tuple[np.ndarray, float, int]:
    """The straight line through every position recorded during one move: (direction x length travelled along
    it, RMS distance of the path from the line, tracking jumps left out).

    A jump (see JUMP) is the tracking correcting itself, not the hand: the path is rebuilt from the steps
    between samples without those. The line is the path's principal axis, pointing the way the move went.
    """
    steps = np.diff(np.array(samples), axis=0)
    real = np.linalg.norm(steps, axis=1) <= JUMP
    path = np.vstack([np.zeros(3), np.cumsum(steps[real], axis=0)])
    centred = path - path.mean(axis=0)
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    net = path[-1] - path[0]
    if direction @ net < 0:
        direction = -direction
    off_line = centred - np.outer(centred @ direction, direction)
    return direction * float(direction @ net), float(np.sqrt((off_line ** 2).sum(axis=1).mean())), int((~real).sum())


class Calibration:
    """Four recorded moves -> axes. press() is an Enter (starts, then ends, a move); add() is every position
    while one is being recorded."""

    STEPS = ("+X", "-X", "+Y", "-Y")

    def __init__(self):
        self.start = None                    # where the first move began: the zero
        self.samples = None                  # positions of the move being recorded, else None
        self.moves: list[np.ndarray] = []    # fitted moves accepted so far, in STEPS order
        self.result = None                   # (position axes, rotation axes) once all four are in

    @property
    def step(self) -> str:
        return self.STEPS[len(self.moves)]

    def prompt(self) -> str:
        n, name = len(self.moves), self.step
        where = "Hold the phone at your start position" if n == 0 else \
            "Stay there, or move anywhere first (nothing is recorded now)"
        way = f"straight back the other way, toward {name}" if n in (1, 3) else f"about 30 cm toward {name}"
        return f"[CALIBRATE] {n + 1}/4 {name}: {where}. Press Enter, move the phone {way}, press Enter again."

    def add(self, p: np.ndarray) -> None:
        if self.samples is not None:
            self.samples.append(p)

    def press(self, p: np.ndarray) -> str:
        n, name = len(self.moves), self.step
        if self.samples is None:
            self.samples = [p]
            return f"Recording {name}: move now, then press Enter."
        samples, self.samples = self.samples + [p], None
        move, rms, jumps = fit_line(samples)
        length = float(np.linalg.norm(move))
        if length < MIN_STROKE:
            return f"Only {length * 100:.0f} cm along the line (need {MIN_STROKE * 100:.0f}): {name} again."

        def cos_to(v):
            return float(move @ v) / (length * float(np.linalg.norm(v)))

        if n in (1, 3) and cos_to(self.moves[n - 1]) > -0.7:      # more than 45 deg off straight back
            return f"That was not back along the {self.STEPS[n - 1]} move, so it can't be {name}: again."
        if n == 2 and abs(cos_to(self.moves[0] - self.moves[1])) > 0.7:
            return "That was along the X line, not across it, so it can't be +Y: again."
        if n == 0:
            self.start = samples[0]
        self.moves.append(move)
        text = (f"{name} taken: {length * 100:.0f} cm, line fitted through {len(samples)} positions, path within "
                f"{rms * 1000:.0f} mm of it" + (f", {jumps} tracking jump(s) left out." if jumps else "."))
        if len(self.moves) < 4:
            return text
        mx, my = self.moves[0] - self.moves[1], self.moves[2] - self.moves[3]
        up, level = up_axis(mx, my)
        pos_axes, rot_axes, angle, mirrored = calibrate_axes(mx, my, up)
        self.result = (pos_axes, rot_axes)

        def bend(a, b):                     # how far from exactly opposite the two moves of an axis were
            return math.degrees(math.acos(max(-1.0, min(1.0, -float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b))))))

        text += (f"\n[CALIBRATE] Your +X and -X lines were {bend(*self.moves[:2]):.0f} deg apart, +Y and -Y "
                 f"{bend(*self.moves[2:]):.0f} deg (0 is the same line); the X and Y lines were {angle:.0f} deg apart "
                 "(90 is square; +Y is made square to +X).")
        text += f"\n[CALIBRATE] The app's vertical axis is its {'xyz'[int(np.argmax(up))]}; shown below as z."
        if level < 0.9:
            text += " Your moves were not level (tilted about %.0f deg), so x and y may be off: redo on a table." \
                % math.degrees(math.acos(level))
        if mirrored:
            text += ("\n[CALIBRATE] Note: +Y is to the RIGHT of +X seen from above. The robot's axes have +Y to the "
                     "left; positions follow what you showed, the rotation uses +Y left.")
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="HEBI", help="the app's Family setting (default: HEBI)")
    parser.add_argument("--name", default="mobileIO", help="the app's Name setting (default: mobileIO)")
    parser.add_argument("--wait", type=float, default=20.0, help="seconds to look for the phone before giving up")
    parser.add_argument("--calibrate", action="store_true",
                        help="record moves toward +X, -X, +Y, -Y (Enter at each end) and save the axes they give; "
                             "without it the saved ones are used, if there are any")
    args = parser.parse_args()

    try:
        import hebi
    except ImportError:
        raise SystemExit("hebi-py is not installed:  pip install hebi-py --no-deps")

    print(f"[PHONE] Looking for the app (Family {args.family!r}, Name {args.name!r}) for up to {args.wait:.0f} s ...", flush=True)
    lookup = hebi.Lookup()
    group, deadline, seen = None, time.monotonic() + args.wait, set()
    while group is None and time.monotonic() < deadline:
        time.sleep(1.0)
        for entry in lookup.entrylist:
            if (entry.family, entry.name) not in seen:
                seen.add((entry.family, entry.name))
                print(f"[PHONE] On the network: Family {entry.family!r}, Name {entry.name!r}", flush=True)
        group = lookup.get_group_from_names([args.family], [args.name])
    if group is None:
        hint = "Something answered, but not under that Family / Name: pass --family / --name, or change them in the app." \
            if seen else "Nothing answered. Is the app open and in the foreground, and the phone on this computer's Wi-Fi? " \
                         "Networks that keep devices apart (eduroam, guest Wi-Fi) block this: try a phone hotspot."
        raise SystemExit(f"[PHONE] Not found. {hint}")

    print("[PHONE] Connected. Ctrl+C stops.", flush=True)
    calibration = Calibration() if args.calibrate else None      # None once done (or not asked for)
    checking_button = calibration is not None                    # first of all: does the gripper button arrive?
    if checking_button:
        print(f"[GRIPPER] First the gripper button: press B{GRIPPER_BUTTON} in the app. "
              "(Enter here skips this if the button does nothing.)", flush=True)
    else:
        print("[PHONE] Move and turn the phone; press the buttons (B2 = gripper), drag A3.\n", flush=True)
    pos_axes, rot_axes = (None if args.calibrate else load_calibration()) or (np.eye(3), np.eye(3))
    origin = last_p = None
    was_b1 = was_grip = gripper_closed = False
    jumps = 0
    start, frames, with_pose, last_print = time.monotonic(), 0, 0, 0.0
    try:
        while True:
            feedback = group.get_next_feedback(timeout_ms=1000)
            if feedback is None:
                print("\r[PHONE] no data for 1 s (app in the background, or the phone locked?)".ljust(118), end="", flush=True)
                continue
            frames += 1
            module = feedback[0]
            position, quat = getattr(module, "ar_position", None), getattr(module, "ar_orientation", None)
            buttons, sliders = read_inputs(getattr(module, "io", None))
            pressed = "".join(str(b) for b in buttons)           # B1..B8, shown as e.g. 10000000
            ok = position is not None and quat is not None and all(math.isfinite(v) for v in (*position, *quat))
            now = time.monotonic()
            rate = frames / max(now - start, 1e-6)
            tapped = bool(buttons[0]) and not was_b1
            was_b1 = bool(buttons[0])
            marked = calibration is not None and (enter_pressed() or tapped)
            grip_now = bool(buttons[GRIPPER_BUTTON - 1])
            grip_pressed, was_grip = grip_now and not was_grip, grip_now       # on the press, not while held
            if checking_button:
                if grip_pressed or marked:
                    checking_button = False
                    print(f"\n[GRIPPER] B{GRIPPER_BUTTON} works: it will open / close the gripper." if grip_pressed else
                          f"\n[GRIPPER] Skipped: B{GRIPPER_BUTTON} was NOT seen, so the gripper button is unconfirmed.", flush=True)
                    print("[CALIBRATE] Enter in this terminal starts and ends each move (so does a tap on B1 in the app).", flush=True)
                    print(calibration.prompt(), flush=True)
                elif now - last_print >= 0.1:
                    last_print = now
                    print(f"\r[PHONE] waiting for B{GRIPPER_BUTTON} in the app (Enter skips) | buttons {pressed} | {rate:4.0f} Hz"
                          .ljust(126), end="", flush=True)
                continue
            if grip_pressed:
                gripper_closed = not gripper_closed
                print(f"\n[GRIPPER] B{GRIPPER_BUTTON} pressed: gripper {'CLOSED' if gripper_closed else 'OPEN'}", flush=True)
            if not ok:
                if marked:
                    print("\n[CALIBRATE] No tracked pose right now, so that press was ignored.", flush=True)
                if now - last_print >= 0.1:
                    last_print = now
                    print(f"\r[PHONE] connected, {rate:4.0f} Hz, but no tracked pose yet: point the camera at a textured, "
                          "well-lit scene".ljust(118), end="", flush=True)
                continue
            with_pose += 1
            p = np.array([float(v) for v in position])
            if last_p is not None and np.linalg.norm(p - last_p) > JUMP:
                jumps += 1
                print(f"\n[PHONE] Tracking jumped {np.linalg.norm(p - last_p) * 100:.0f} cm in one message: the phone lost its place "
                      "and corrected. Give the camera a detailed, well-lit view; move slower.", flush=True)
            last_p = p

            if calibration is not None:
                calibration.add(p)
            if marked:
                print(f"\n[CALIBRATE] {calibration.press(p)}", flush=True)
                if calibration.result is None:
                    if calibration.samples is None:
                        print(calibration.prompt(), flush=True)
                else:
                    (pos_axes, rot_axes), origin, calibration = calibration.result, calibration.start, None
                    save_calibration(pos_axes, rot_axes)
                    print("[PHONE] Calibrated; zero is where your first move began. Move and turn the phone.\n", flush=True)
            if now - last_print < 0.1:
                continue
            last_print = now
            if calibration is not None:
                moved = f"waiting for Enter to start {calibration.step} | " if calibration.samples is None else \
                    (f"RECORDING {calibration.step}: {np.linalg.norm(p - calibration.samples[0]) * 100:3.0f} cm from its "
                     "start, Enter to end | ")
                line = (f"{moved}raw {p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f} m | "
                        f"gripper {'CLOSED' if gripper_closed else 'OPEN'} | buttons {pressed} | {rate:4.0f} Hz")
            else:
                origin = p if origin is None else origin
                x, y, z = pos_axes @ (p - origin)
                roll, pitch, yaw = euler_deg(rot_axes @ quat_matrix(*(float(v) for v in quat)) @ rot_axes.T)   # w, x, y, z
                line = (f"pos x {x:+.3f} y {y:+.3f} z {z:+.3f} m | roll {roll:+6.1f} pitch {pitch:+6.1f} yaw {yaw:+6.1f} deg | "
                        f"gripper {'CLOSED' if gripper_closed else 'OPEN  '} | buttons {pressed} | {rate:4.0f} Hz")
            print(f"\r[PHONE] {line}".ljust(126), end="", flush=True)
    except KeyboardInterrupt:
        pass
    seconds = time.monotonic() - start
    print(f"\n[PHONE] {frames} messages in {seconds:.0f} s ({frames / max(seconds, 1e-6):.0f} Hz), {with_pose} with a tracked pose, "
          f"{jumps} tracking jump(s).")


if __name__ == "__main__":
    main()