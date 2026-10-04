"""Zip-tie tightening: a pre-threaded zip tie loops around a bundle of rods; grab the tail and pull it tight.

One continuous strap (a standard-duty 300 x 3.6 x 1.0 mm nylon tie): it starts under the head, wraps
around four rods, comes back over its own root through the head's slot and leaves it as the tail,
pointing at the robot. The strap
is a chain of short flat segments; each joint bends easily across the thickness, resists twist, and is
is rigid in-plane (it is ~20x stiffer that way), from nylon's modulus and the cross-section. Its rest shape is straight, so the loop
springs open against the slot like a real tie.

Pulling the tail draws strap through the slot and the loop shrinks onto the rods. The rods are loose
(spring-mounted at their base, a few mm apart), so the loop gathers them and then squeezes them: it
gets hard to pull exactly when the bundle is cinched. The head's pawl lets strap come out through the
slot but not go back in further than one tooth.

Success (``is_tight``): the loop is no longer than TIGHT_LENGTH, i.e. the rods are pulled together.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_mujoco_scenes import add_floor, scene

TABLE_TOP_Z = 0.705             # same table as peg_insert
TABLE_X = (0.15, 0.85)
TABLE_HALF_Y = 0.6

BUNDLE_POS = (0.45, 0.29)       # xy of the bundle centre, inside the left arm's reach
LOOP_Z = TABLE_TOP_Z + 0.06     # strap height on the bundle
ROD_RADIUS = 0.006              # four rods in a 2x2 cluster, ROD_GAP apart (loose) at rest
ROD_GAP = 0.005
ROD_HEIGHT = 0.10
ROD_STIFFNESS = 15.0            # N/m, each rod's spring back to its rest spot (the bundle's give)
ROD_FRICTION = 0.1              # nylon on steel

# Strap: standard-duty nylon tie (300 x 3.6 x 1.0 mm). Stiffness from E = 2.5 GPa, G = 0.9 GPa and the
# section; a heavy-duty 4.8 x 1.3 mm tie is ~3x stiffer and takes more pull than the sim gripper holds.
STRAP_LENGTH = 0.30
STRAP_WIDTH = 0.0036            # vertical (world Z)
STRAP_THICK = 0.0010            # along the jaw direction where the gripper pinches the tail
# Collision thickness: a 1 mm box lets the jaws or a rod push right through it under load (MuJoCo
# contacts are soft, and softer on light bodies); the strap collides as COLLIDE_THICK, drawn at STRAP_THICK.
COLLIDE_THICK = 0.003
SEGMENT_LEN = 0.008
_E, _G = 2.5e9, 0.9e9
_EI_EASY = _E * STRAP_WIDTH * STRAP_THICK ** 3 / 12            # bending across the thickness
_GJ = _G * STRAP_WIDTH * STRAP_THICK ** 3 / 3 * (1 - 0.63 * STRAP_THICK / STRAP_WIDTH)
# MuJoCo scales contact stiffness with the bodies' mass: at nylon's real ~0.06 g per segment the jaws
# sink through the 1.3 mm strap. Heavier segments keep a squeezed strap from being crushed.
MASS_SCALE = 5.0
SEGMENT_MASS = MASS_SCALE * 1140 * STRAP_WIDTH * STRAP_THICK * SEGMENT_LEN
# Armature keeps each joint's natural frequency well under 1/timestep (the real segments are ~0.04 g).
JOINT_ARMATURE = {"bend": 1e-6, "twist": 1e-6}

LOOP_RADIUS = 0.022             # as threaded: loop half-width around the bundle (loose)
SLOT_CLEARANCE = 0.0004         # slot wider than the strap, each side
HEAD_DEPTH = 0.009              # slot length (along the tail)
SLOT_WALL = 0.001               # wall between the strap's root and the strap passing through the slot
LAYER_GAP = STRAP_THICK + SLOT_WALL + 2 * SLOT_CLEARANCE   # root centreline to slot centreline (head is visual)
TOOTH_PITCH = 0.0015            # pawl engages every 1.5 mm of strap
TOOTH_DRAG = 0.3                # N to drag the strap through the head
MAX_PULL = 0.09                 # carriage travel: more than the loop can ever give

TIGHT_LENGTH = 0.107            # m of loop: at or below, all four rods are pulled together

# Strap collides with the world and the arm, not itself or the head (contype 4 / conaffinity 1|2).
STRAP_CONTYPE = 4
STRAP_CONAFFINITY = 3


def _strap_path():
    """Strap centreline as threaded, from the root: [(x, y, heading)] every SEGMENT_LEN, and the arc
    length at the slot centre.

    Like a real tie: the root runs under the head toward the robot (-X), the strap goes clockwise round
    the bundle (a stadium: west half-circle, top, east half-circle), comes back along the bottom just
    outside its own root, passes straight through the slot and carries on toward the robot as the tail.
    No bend near the slot, so the segment chain feeds through it freely.
    """
    bx, by = BUNDLE_POS
    r_west = LOOP_RADIUS
    r_east = LOOP_RADIUS + LAYER_GAP / 2        # the returning strap lands LAYER_GAP outside the root
    lead = 0.002                                 # root past the head before it curves
    entry = 0.004                                # straight run into the slot
    top = HEAD_DEPTH + lead + entry
    x_head = bx - top / 2 + lead + HEAD_DEPTH / 2
    pos = np.array([x_head + HEAD_DEPTH / 2, by - r_west])
    heading, step = math.pi, SEGMENT_LEN / 16
    pts, arc = [(*pos, heading)], [0.0]

    def run(length, curvature):
        nonlocal pos, heading
        for _ in range(max(1, round(length / step))):
            ds = length / max(1, round(length / step))
            heading += curvature * ds
            pos = pos + ds * np.array([math.cos(heading), math.sin(heading)])
            pts.append((*pos, heading))
            arc.append(arc[-1] + ds)

    run(HEAD_DEPTH + lead, 0.0)                  # root, under the head
    run(math.pi * r_west, -1 / r_west)           # west half-circle, clockwise
    run(top, 0.0)                                # across the top
    run(math.pi * r_east, -1 / r_east)           # east half-circle, back to heading -X
    run(entry + HEAD_DEPTH / 2, 0.0)             # into the slot, to its centre
    slot_arc = arc[-1]
    run(STRAP_LENGTH - arc[-1], 0.0)             # tail
    pts, arc = np.array(pts), np.array(arc)
    out = []
    for s in np.arange(0, STRAP_LENGTH - SEGMENT_LEN / 2, SEGMENT_LEN):
        x0, y0 = np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])
        x1, y1 = np.interp(s + SEGMENT_LEN, arc, pts[:, 0]), np.interp(s + SEGMENT_LEN, arc, pts[:, 1])
        out.append((x0, y0, math.atan2(y1 - y0, x1 - x0)))
    return out, slot_arc, (x_head, by - r_west - LAYER_GAP)


_PATH, SLOT_ARC, _SLOT_XY = _strap_path()
SLOT_CENTRE = (_SLOT_XY[0], _SLOT_XY[1], LOOP_Z)
SLOT_AXIS = np.array([-1.0, 0.0, 0.0])      # strap leaves the slot this way (toward the robot)
N_SEGMENTS = len(_PATH)


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Head mechanics, once per control step (all state lives in ``data``, so a reset clears it).

    The strap in the slot is welded to the carriage, whose travel is the strap pulled through; this
    moves the welds along the strap as it feeds, and the pawl's target up to the last tooth passed.
    """
    fed = data.qpos[model.joint("zip_carriage").qposadr[0]]
    data.userdata[0] = max(data.userdata[0], fed)
    data.ctrl[model.actuator("zip_pawl").id] = math.floor(data.userdata[0] / TOOTH_PITCH) * TOOTH_PITCH
    data.eq_active[_weld_ids(model)] = _in_slot(fed)


@scene("zip_tie", camera=dict(lookat=[0.40, 0.29, 0.77], distance=0.55, azimuth=215, elevation=-35), step=step)
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    x0, x1 = TABLE_X
    world.add_geom(
        name="table", type=box, size=[(x1 - x0) / 2, TABLE_HALF_Y, TABLE_TOP_Z / 2],
        pos=[(x0 + x1) / 2, 0.0, TABLE_TOP_Z / 2], rgba=[0.55, 0.42, 0.3, 1],
    )

    # Loose rods: each slides in XY on a spring to its rest spot, standing on the table (so a loose loop
    # that slides down them can't slip underneath).
    bx, by = BUNDLE_POS
    off = ROD_RADIUS + ROD_GAP / 2
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        rod = world.add_body(name=f"rod{i}", pos=[bx + sx * off, by + sy * off, TABLE_TOP_Z + ROD_HEIGHT / 2])
        spec.add_exclude(bodyname1="world", bodyname2=f"rod{i}")  # stands on the table without dragging on it
        for axis, name in (([1, 0, 0], "x"), ([0, 1, 0], "y")):
            rod.add_joint(name=f"rod{i}_{name}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis,
                          stiffness=ROD_STIFFNESS, damping=3.0, armature=0.01)
        rod.add_geom(type=cyl, size=[ROD_RADIUS, ROD_HEIGHT / 2, 0], mass=0.02, priority=1,
                     friction=[ROD_FRICTION, 0.005, 0.0001], rgba=[0.72, 0.72, 0.75, 1])

    # Head: the slot walls and block are visual. The slot itself is a carriage sliding along the slot
    # axis (its travel = strap pulled through), with the strap's segments in the slot welded to it
    # (welds added below): it holds the strap exactly in line, which wall contacts on a 1.3 mm strap
    # don't. The pawl is a one-way actuator on the carriage, the tooth drag its friction.
    head_rgba = [0.93, 0.92, 0.88, 1]  # natural nylon
    sx, sy, sz = SLOT_CENTRE
    gap_y, gap_z = STRAP_THICK / 2 + SLOT_CLEARANCE, STRAP_WIDTH / 2 + SLOT_CLEARANCE
    wall = SLOT_WALL
    for name, dy, dz, size in (
        ("slot_wall_out", -(gap_y + wall / 2), 0, [HEAD_DEPTH / 2, wall / 2, gap_z + wall]),
        ("slot_roof", 0, gap_z + wall / 2, [HEAD_DEPTH / 2, gap_y, wall / 2]),
        ("slot_floor", 0, -(gap_z + wall / 2), [HEAD_DEPTH / 2, gap_y, wall / 2]),
    ):
        world.add_geom(name=name, type=box, size=size, pos=[sx, sy + dy, sz + dz], rgba=head_rgba,
                       contype=0, conaffinity=0)
    root_y = _PATH[0][1]
    y_lo, y_hi = sy + gap_y, root_y + STRAP_THICK / 2 + 0.0005
    world.add_geom(name="head_body", type=box, contype=0, conaffinity=0, rgba=head_rgba,
                   size=[HEAD_DEPTH / 2, (y_hi - y_lo) / 2, gap_z + wall], pos=[sx, (y_hi + y_lo) / 2, sz])
    carriage = world.add_body(name="zip_carriage", pos=[sx, sy, sz])
    carriage.add_joint(name="zip_carriage", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(SLOT_AXIS),
                       range=[0, MAX_PULL], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                       frictionloss=TOOTH_DRAG, damping=1.0, armature=0.01)
    carriage.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], mass=0.002,
                      contype=0, conaffinity=0, group=3)
    pawl = spec.add_actuator(name="zip_pawl", target="zip_carriage", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    pawl.set_to_position(kp=3000.0, kv=30.0)
    pawl.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    pawl.forcerange = [0.0, 60.0]   # pushes strap back out to the last tooth, never pulls it in

    # Strap: root fixed to the head, then one body per segment with two hinges at its start (in-plane
    # bending, ~20x stiffer, is left rigid: a third hinge per joint makes the long chain slow to solve).
    # Fine stripes along the strap for its teeth (one per TOOTH_PITCH).
    teeth = spec.add_texture(name="strap_teeth", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                             rgb1=[0.95, 0.94, 0.9], rgb2=[0.78, 0.77, 0.72], width=64, height=64)
    mat = spec.add_material(name="strap", texrepeat=[1 / TOOTH_PITCH / 2, 0.001], texuniform=True, specular=0.3, shininess=0.4)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = teeth.name
    parent, prev_heading = world, None
    for i, (x, y, heading) in enumerate(_PATH):
        if prev_heading is None:
            seg = parent.add_body(name="strap0", pos=[x, y, LOOP_Z], quat=_zquat(heading))
        else:
            bend = _wrap(heading - prev_heading)
            seg = parent.add_body(name=f"strap{i}", pos=[SEGMENT_LEN, 0, 0], quat=_zquat(bend))
            # Rest shape straight: each easy-bend spring is relaxed at minus its threaded bend.
            for kind, axis, k, ref in (
                ("bend", [0, 0, 1], _EI_EASY / SEGMENT_LEN, -bend),
                ("twist", [1, 0, 0], _GJ / SEGMENT_LEN, 0.0),
            ):
                seg.add_joint(name=f"strap{i}_{kind}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis,
                              stiffness=k, springref=ref, damping=4 * k * 1e-3, armature=JOINT_ARMATURE[kind])
        tip = i == N_SEGMENTS - 1
        width = STRAP_WIDTH * (0.6 if tip else 1.0)  # tapered tip
        seg.add_geom(type=box, size=[SEGMENT_LEN / 2, COLLIDE_THICK / 2, width / 2], pos=[SEGMENT_LEN / 2, 0, 0],
                     mass=SEGMENT_MASS, contype=STRAP_CONTYPE, conaffinity=STRAP_CONAFFINITY,
                     friction=[1.0, 0.005, 0.0001], condim=4, group=3)
        seg.add_geom(type=box, size=[SEGMENT_LEN / 2, STRAP_THICK / 2, width / 2], pos=[SEGMENT_LEN / 2, 0, 0],
                     mass=0, material="strap", contype=0, conaffinity=0)
        parent, prev_heading = seg, heading

    # One weld per segment to the carriage, placing it where it sits when that much strap has fed
    # through: segment origin at arc i*L lies (SLOT_ARC - i*L) along -SLOT_AXIS from the carriage,
    # heading out of the slot. step() enables the ones in the slot.
    active = _in_slot(0.0)
    for i in range(N_SEGMENTS):
        weld = spec.add_equality(name=f"slot_weld{i}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                 name1="zip_carriage", name2=f"strap{i}", active=bool(active[i]))
        weld.data[:11] = [0, 0, 0, *(-(SLOT_ARC - i * SEGMENT_LEN) * SLOT_AXIS), 0, 0, 0, 1, 1]


def _zquat(angle: float) -> list[float]:
    return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _in_slot(fed: float) -> np.ndarray:
    """Which segments lie in the slot (centre within HEAD_DEPTH / 2 of the slot centre) at this feed."""
    along = (np.arange(N_SEGMENTS) + 0.5) * SEGMENT_LEN - (SLOT_ARC - fed)
    return np.abs(along) < HEAD_DEPTH / 2


def _weld_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.equality(f"slot_weld{i}").id for i in range(N_SEGMENTS)])


def loop_length(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Strap from the root round the bundle to the slot (m)."""
    return SLOT_ARC - float(data.qpos[model.joint("zip_carriage").qposadr[0]])


def is_tight(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return loop_length(model, data) <= TIGHT_LENGTH
