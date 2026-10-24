"""Where a correction or insertion-only evaluation starts: prepared, detected or restored held states.

Every start is reached through actuator targets and physics after a normal ``reset``; restoring a
saved state (``restore``) copies a state that physics already produced. Nothing here teleports
the peg during a rollout.

| Start                      | How it is reached                                            | Used by                  |
|----------------------------|--------------------------------------------------------------|--------------------------|
| insertion case             | expert grasps, lifts, servos 10 cm above the hole (+offset/tilt) | eval ``--start-stage``, tasks |
| broad start                | as above, at an arbitrary mm offset/clearance/tilt            | ``tasks`` corrections     |
| paired anchor              | peg held 11 cm up at a fixed point; hole reset 5-20 mm either side | ``tasks`` corrections |
| approach (detector)        | policy holds the lifted peg within 3 cm XY of the hole        | ``takeover`` corrections  |
| targeted stall (detector)  | ... and stalls/tilts/descends misaligned at the rim           | ``takeover`` corrections  |
| rim jam (detector)         | held peg touches the rim tilted >=3 deg or stalls 0.36 s      | ``takeover`` corrections  |
| pre-grasp (detector)       | hand within 4 cm (XY), 5 cm above an upright peg, or 15 s pass | ``takeover --skill grasp`` |
| failed-trace state         | first rim contact / hover / pre-drop frame of an eval trace   | ``traces`` corrections    |
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from .expert import DT, ExpertFailure, Phase
from .scene import BLOCK_HEIGHT, PEG_HEIGHT, PEG_SIZE, TABLE_TOP_Z

INSERTION_CASES = ("aligned", "offset_positive", "offset_negative", "tilt_positive", "tilt_negative")
CASE_OFFSET = 0.003       # m, offset_* cases move the held peg (+x,-y) or (-x,+y)
CASE_TILT = 5.0           # deg, tilt_* cases roll the held peg about x
HANDOFF_HEIGHT = 0.10     # m above the seated peg centre for insertion cases
RIM_Z = TABLE_TOP_Z + BLOCK_HEIGHT
HALF_EXTENT = np.array([PEG_SIZE / 2, PEG_SIZE / 2, PEG_HEIGHT / 2])


def rotation_about(axis, degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def peg_geometry(expert):
    """(lowest peg corner z, tilt from vertical in degrees)."""
    rotation = expert.data.xmat[expert.peg].reshape(3, 3)
    bottom = expert.peg_pos[2] - np.abs(rotation[2]) @ HALF_EXTENT
    return float(bottom), float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))


def pick_up(expert):
    """Same grasp and lift motions as ``PegExpert.run``, without insertion."""
    grasp = expert.peg_pos + [0, 0, 0.060]
    expert.move(Phase.SETUP, expert.peg_pos + [0, 0, 0.18], 0)
    expert.move(Phase.SETUP, grasp, 0, speed=0.025)
    expert.hold(1.5, grasp, 1)
    if not expert.contacts():
        raise ExpertFailure("setup could not verify grasp")
    expert.move(Phase.SETUP, grasp + [0, 0, 0.12], 1, speed=0.03)


def hold_above_hole(expert, offset_m=(0, 0), height_m=None, rotation=np.eye(3)):
    target = expert.hole_pos.copy()
    target[:2] += np.asarray(offset_m)
    target[2] = TABLE_TOP_Z + PEG_HEIGHT / 2 + HANDOFF_HEIGHT if height_m is None else RIM_Z + PEG_HEIGHT / 2 + height_m
    expert.phase = Phase.SETUP
    expert.peg_servo(target, speed=0.03, timeout=25, rotation=rotation)
    if not expert.contacts() or expert.block_contacts() or expert.inserted():
        raise ExpertFailure("handoff must be held, clear of the block, and not seated")
    return target


def prepare_insertion(expert, case):
    """One of the five fixed insertion-only diagnostic starts."""
    if case not in INSERTION_CASES:
        raise ValueError(case)
    start = expert.data.time
    pick_up(expert)
    sign = -1 if case.endswith("negative") else 1
    offset = sign * CASE_OFFSET * np.array([1, -1]) if case.startswith("offset") else (0, 0)
    rotation = rotation_about("x", sign * CASE_TILT) if case.startswith("tilt") else np.eye(3)
    target = hold_above_hole(expert, offset, rotation=rotation)
    return {"case": case, "preparation_seconds": expert.data.time - start, "target_peg_position": target.tolist(),
            "initial_qpos": expert.data.qpos.tolist(), "initial_qvel": expert.data.qvel.tolist(),
            "initial_ctrl": expert.data.ctrl.tolist()}


def prepare_task(expert, task):
    """A task dict from the generators below, or a fixed insertion case (``{"condition": case}``)."""
    if "anchor_xy" in task:
        pick_up(expert)
        expert.phase = Phase.SETUP
        expert.peg_servo(np.r_[task["anchor_xy"], TABLE_TOP_Z + PEG_HEIGHT / 2 + 0.11], speed=0.03, timeout=20)
        if not expert.contacts() or np.linalg.norm(expert.peg_pos[:2] - task["anchor_xy"]) > 0.0004:
            raise ExpertFailure("paired anchor not reached while held")
        return {"held_peg_position": expert.peg_pos.tolist(), "held_state": expert.state().tolist()}
    if "offset_mm" not in task:
        return prepare_insertion(expert, task["condition"])
    pick_up(expert)
    rotation = rotation_about(task["tilt_axis"], task["tilt_degrees"])
    target = hold_above_hole(expert, np.array(task["offset_mm"]) / 1000, task["clearance_mm"] / 1000, rotation)
    return {"target_peg_position": target.tolist(), "actual_peg_position": expert.peg_pos.tolist()}


def broad_coverage_tasks():
    """80 held starts around the block; front means robot-facing (-X). Offsets in mm."""
    positions = [(0, 0)] + [(-x, 0) for x in [1, 3, 6, 12, 25, 45, 65, 80]]
    positions += [(-x, y) for x in [12, 30, 60] for y in [-30, 30]]
    positions += [(0, y) for y in [-60, -30, -12, -3, 3, 12, 30, 60]] + [(x, 0) for x in [3, 12, 30, 60, 75]]
    positions += [(x, y) for x in [25, 55] for y in [-30, 30]]
    tasks = [{"stage": "insertion", "condition": "aligned", "offset_mm": [x, y], "clearance_mm": c,
              "tilt_degrees": 0, "tilt_axis": "x", "coverage_region": _region(x)}
             for x, y in positions for c in [20, 60]]
    for x, y in [(0, 0), (-3, 0), (-25, 0), (30, 0)]:
        for axis in ["x", "y"]:
            for angle in ([-5, 5] if (x, y) == (0, 0) else [-2, 2]):
                tasks.append({"stage": "insertion", "condition": "tilt_positive" if angle > 0 else "tilt_negative",
                              "offset_mm": [x, y], "clearance_mm": 40, "tilt_degrees": angle, "tilt_axis": axis,
                              "coverage_region": _region(x)})
    for index, task in enumerate(tasks):
        task.update(layout_seed=10000 + index, broad_start=True)
    return tasks


def _region(x):
    return "front" if x < 0 else "back" if x > 0 else "center_or_side"


def large_tilt_tasks():
    """24 held starts tilted 6-12 deg; +X tilt is capped at 9 deg, the verified grasp limit."""
    tasks = []
    for axis in ["x", "y"]:
        for angle in [-12, -9, -6, 6, 9, 12]:
            for offset in [[0, 0], [-5, 3]]:
                tasks.append({"stage": "insertion", "condition": "tilt_positive" if angle > 0 else "tilt_negative",
                              "layout_seed": 13000 + len(tasks), "offset_mm": offset, "clearance_mm": 10,
                              "tilt_degrees": 9 if axis == "x" and angle == 12 else angle, "tilt_axis": axis,
                              "broad_start": True})
    return tasks


def paired_hole_tasks(cells, reserved=(), pairs=11, seed=400225, verify=None):
    """Opposite hole placements around one held-peg anchor: the robot state is the same, only the hole differs.

    Pair ``k`` moves the hole 5/8/12/16/20 mm (cycling) along angle ``k * 360/11`` deg either side of
    an anchor jittered 0.5 mm from a valid mapped hole cell; pairs 0-8 use the 3-7 cm band, 9-10 the
    inner band. Anchors are spread apart; cells in ``reserved`` (held-out) are never touched. A
    final aligned control holds the peg straight over the nominal hole. ``verify(task) -> bool``
    (a physical pilot) rejects anchors whose pair the expert cannot finish; the next-best is tried.
    """
    from .curate import eligible
    from .scene import BLOCK_POS, PEG_POS
    from .workspace import band, validate_layout

    rng, tasks, anchors = np.random.default_rng(seed), [], []
    for pair in range(pairs):
        angle, distance = pair * 2 * np.pi / pairs, [0.005, 0.008, 0.012, 0.016, 0.02][pair % 5]
        delta = distance * np.array([np.cos(angle), np.sin(angle)])
        options = []
        for cell in (c for c in cells if c["valid"] and c["group"] == "hole" and c["band"] == ("3_7" if pair < 9 else "0_3")):
            anchor = np.asarray(cell["hole_xy"]) + rng.uniform(-0.0005, 0.0005, 2)
            layouts = []
            for sign in (-1, 1):
                point = anchor + sign * delta
                offset = point - np.array(BLOCK_POS)
                layout = {"peg_xy": list(PEG_POS), "hole_xy": point.tolist(), "group": "hole", "band": band(np.linalg.norm(offset)),
                          "cell": np.rint(offset / 0.01).astype(int).tolist()}
                try:
                    validate_layout(layout["peg_xy"], layout["hole_xy"])
                except ValueError:
                    break
                if np.linalg.norm(offset) > 0.07 or (pair < 9 and layout["band"] != "3_7"):
                    break
                if not eligible({"placement": {"group": "hole", "xy": layout["hole_xy"], "cell": layout["cell"]}}, reserved):
                    break
                layouts.append((sign, layout))
            if len(layouts) == 2:
                spread = min((np.linalg.norm(anchor - a) for a in anchors), default=np.linalg.norm(anchor - np.array(BLOCK_POS)))
                options.append((spread, anchor, layouts))
        for _, anchor, layouts in sorted(options, key=lambda o: -o[0]):
            candidate = [{"layout_seed": seed + len(tasks) + i, "pair_id": pair, "sign": sign, "anchor_xy": anchor.tolist(),
                          "layout": layout, "condition": "paired_alignment"} for i, (sign, layout) in enumerate(layouts)]
            if verify is None or all(verify(task) for task in candidate):
                anchors.append(anchor)
                tasks += candidate
                break
        else:
            raise ValueError(f"no feasible anchor for pair {pair}")
    nominal = {"peg_xy": list(PEG_POS), "hole_xy": list(BLOCK_POS), "group": "hole", "band": "0_3", "cell": [0, 0]}
    return tasks + [{"layout_seed": seed + len(tasks), "pair_id": pairs, "sign": 0, "anchor_xy": list(BLOCK_POS),
                     "layout": nominal, "condition": "paired_alignment"}]


class ApproachDetector:
    """Hand off on entry to the hole approach region (not after a jam)."""

    def __init__(self, radius=0.03):
        self.radius = radius

    def __call__(self, expert, lifted, seconds):
        return "hole_approach" if expert.alignment_ready(lifted, self.radius) else None


class StallDetector:
    """Hand off when a held peg near the hole stops progressing, tilts, or descends misaligned."""

    def __init__(self, radius=0.03, window=3.0, progress=0.003, tilt_degrees=10.0):
        self.radius, self.window, self.progress, self.tilt_degrees = radius, window, progress, tilt_degrees
        self.history = deque()

    def __call__(self, expert, lifted, seconds):
        if not expert.alignment_ready(lifted, self.radius):
            self.history.clear()
            return None
        position = expert.peg_pos
        self.history.append((seconds, position))
        while len(self.history) > 1 and seconds - self.history[1][0] >= self.window:
            self.history.popleft()
        if peg_geometry(expert)[1] > self.tilt_degrees:
            return "tilted_approach"
        if seconds - self.history[0][0] < self.window - DT / 2:
            return None
        at_rim = abs(position[2] - (RIM_Z + PEG_HEIGHT / 2)) < 0.015
        # A stationary endpoint alone can conceal oscillation; use the whole window.
        if np.max(np.linalg.norm(np.array([p for _, p in self.history]) - self.history[0][1], axis=1)) <= self.progress:
            return "rim_stall" if at_rim else "alignment_stall"
        if at_rim and np.linalg.norm(position[:2] - expert.hole_pos[:2]) > 0.001:
            return "misaligned_rim_descent"
        return None


class RimJamDetector:
    """Hand off on the first tick of an early rim jam, before the policy wedges the peg further."""

    WINDOW = 0.40          # s of bottom-height history
    STALL = 0.0005         # m of bottom motion that counts as stalled over the window
    TILT = 3.0             # deg at rim contact that counts as a jam immediately

    def __init__(self):
        self.history = []

    def __call__(self, expert, lifted, seconds):
        bottom, tilt = peg_geometry(expert)
        self.history = [h for h in self.history + [(seconds, bottom)] if seconds - h[0] <= self.WINDOW]
        if not expert.contacts() or not expert.block_contacts() or expert.inserted():
            return None
        if np.linalg.norm(expert.peg_pos[:2] - expert.hole_pos[:2]) > 0.05 or not RIM_Z - 0.008 <= bottom <= RIM_Z + 0.003:
            return None
        if tilt >= self.TILT:
            return "tilted_rim_contact"
        if len(self.history) >= 9 and abs(bottom - self.history[0][1]) < self.STALL:
            return "vertical_stall"
        return None


class PreGraspDetector:
    """Hand off the *grasp* before the policy disturbs the peg: hand near an upright, ungrasped peg,
    or no grasp after ``timeout`` seconds. Used with ``collect_corrections --skill grasp``."""

    RADIUS = 0.04          # m, finger centre to peg axis (XY)
    ABOVE_TOP = 0.05       # m, finger centre at most this far above the peg top (home pose is ~8 cm above)
    UPRIGHT = 3.0          # deg; a tipped peg is a different (unhandled) recovery

    def __init__(self, timeout=15.0):
        self.timeout = timeout

    def __call__(self, expert, lifted, seconds):
        if lifted or expert.contacts() or peg_geometry(expert)[1] > self.UPRIGHT:
            return None
        near = np.linalg.norm(expert.center[:2] - expert.peg_pos[:2]) < self.RADIUS \
            and expert.center[2] < expert.peg_pos[2] + PEG_HEIGHT / 2 + self.ABOVE_TOP
        return "near_peg" if near else "no_grasp_timeout" if seconds >= self.timeout else None


DETECTORS = {"approach": ApproachDetector, "stall": StallDetector, "rim-jam": RimJamDetector, "pre-grasp": PreGraspDetector}


def restore(expert, state, index=None):
    """Load a saved physical state (eval trace frame or handoff.npz) into ``expert.data``."""
    def pick(key):
        return state[key] if index is None else state[key][index]
    mujoco.mj_resetData(expert.model, expert.data)
    expert.model.body_pos[expert.block] = pick("hole_body_pos")
    for key in ("qpos", "qvel", "ctrl"):
        getattr(expert.data, key)[:] = pick(key)
    expert.data.time = float(pick("sim_time"))
    expert.closure = float(np.clip(np.mean(expert.data.ctrl[expert.grip_a] / [-0.05, 0.05]), 0, 1))
    mujoco.mj_forward(expert.model, expert.data)


def trace_starts(trace, min_seconds=14):
    """Policy states worth correcting in a failed eval trace, before wedging or a drop.

    ``alignment``: first held, raised frame after 20 s; ``early_rim_contact``: first held rim
    contact, else ``hover_stall``: last raised frame before 40 s; ``pre_grasp_loss``: last held
    frame >=5 steps before the peg is dropped above the table.
    """
    pos, rot, seconds = trace["peg_position"], trace["peg_rotation"], trace["seconds"]
    held, inserted = trace["held"].astype(bool), trace["inserted"].astype(bool)
    bottom = pos[:, 2] - np.sum(np.abs(rot[:, 2, :]) * HALF_EXTENT, axis=1)
    raised = held & ~inserted & (bottom > RIM_Z + 0.004) & (seconds > min_seconds)
    starts = []
    if (index := np.flatnonzero(raised & (seconds >= 20))).size:
        starts.append(("alignment", int(index[0])))
    rim = held & ~inserted & trace["block_contact"].astype(bool) & (bottom >= RIM_Z - 0.008) & (bottom <= RIM_Z + 0.003)
    if (index := np.flatnonzero(rim & (seconds > min_seconds))).size:
        starts.append(("early_rim_contact", int(index[0])))
    elif (index := np.flatnonzero(raised & (seconds <= 40))).size:
        starts.append(("hover_stall", int(index[-1])))
    drop = held[:-1] & ~held[1:] & ~inserted[1:] & (bottom[1:] > TABLE_TOP_Z + 0.005) & (seconds[1:] > min_seconds)
    if (index := np.flatnonzero(drop)).size:
        i = max(0, int(index[0]) - 5)
        while i > 0 and not held[i]:
            i -= 1
        starts.append(("pre_grasp_loss", i))
    return list(dict.fromkeys(starts))
