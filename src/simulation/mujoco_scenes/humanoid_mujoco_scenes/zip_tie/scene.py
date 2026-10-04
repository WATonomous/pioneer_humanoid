"""Zip-tie tightening: a pre-threaded zip tie loops around a bundle of rods; grab the tail and pull it tight.

The head is fixed against the bundle. The tail leaves the head toward the robot, on edge (flat faces
along world Y) so the gripper's jaws pinch it. Pulling the tail slides it out of the head on a slide
joint -- the ratchet -- against tooth drag (frictionloss), and the loop around the bundle shrinks until
it is tight on the rods. ``step`` makes the ratchet one-way: once pulled, the tail can't slip back.

The loop is drawn as a tendon wrapped around the bundle; its far end (the slack) is coupled to the
ratchet, so the loop visibly shrinks as the tail comes out. Success (``is_tight``): the loop is within
TIGHT_SLACK of its length snug on the rods.
"""
from __future__ import annotations

import math

import mujoco

from humanoid_mujoco_scenes import add_floor, scene

TABLE_TOP_Z = 0.705             # same table as peg_insert
TABLE_X = (0.15, 0.85)
TABLE_HALF_Y = 0.6

BUNDLE_POS = (0.45, 0.29)       # xy of the rod bundle, inside the left arm's reach
ROD_RADIUS = 0.009              # four vertical rods in a 2x2 cluster
ROD_HEIGHT = 0.12
LOOP_Z = TABLE_TOP_Z + 0.075    # height of the zip tie on the bundle
WRAP_RADIUS = ROD_RADIUS * (1 + math.sqrt(2)) + 0.001   # circle around the cluster the loop wraps

HEAD_HALF = (0.008, 0.009, 0.007)   # zip-tie head (m, half extents), on the bundle's -X side
STRAP_THICK = 0.003             # tail cross-section: thickness along Y (jaw direction), width along Z
STRAP_WIDTH = 0.008
TAIL_SEGMENTS = 6               # tail = chain of short segments, bending about Z (in the jaw direction)
SEGMENT_LEN = 0.017
SEGMENT_MASS = 0.002
TAIL_BEND_STIFFNESS = 0.02      # Nm/rad per joint: springy, like nylon
TAIL_BEND_DAMPING = 0.002

PULL_TRAVEL = 0.06              # tail travel (m) from as-threaded to snug on the rods
TOOTH_DRAG = 1.0                # N to drag the tail through the head (well under the gripper's hold)
TIGHT_SLACK = 0.004             # success: loop within this (m) of snug
SNUG_GAP = 0.006                # loop far end's clearance off the wrap circle when snug

# Tail collides with the arm only (arm geoms: contype 2 / conaffinity 1), not with the world or itself.
TAIL_CONTYPE = 4
TAIL_CONAFFINITY = 2


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Ratchet: the pawl's target follows the tail out and never back (ctrl resets with the data)."""
    pawl = model.actuator("zip_pawl").id
    data.ctrl[pawl] = max(data.ctrl[pawl], data.qpos[model.joint("zip_ratchet").qposadr[0]])


@scene("zip_tie", camera=dict(lookat=[0.40, 0.29, 0.78], distance=0.7, azimuth=210, elevation=-30),
       step=step)
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    x0, x1 = TABLE_X
    world.add_geom(
        name="table", type=box, size=[(x1 - x0) / 2, TABLE_HALF_Y, TABLE_TOP_Z / 2],
        pos=[(x0 + x1) / 2, 0.0, TABLE_TOP_Z / 2], rgba=[0.55, 0.42, 0.3, 1],
    )

    bx, by = BUNDLE_POS
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        world.add_geom(
            name=f"rod{i}", type=cyl, size=[ROD_RADIUS, ROD_HEIGHT / 2, 0],
            pos=[bx + sx * ROD_RADIUS, by + sy * ROD_RADIUS, TABLE_TOP_Z + ROD_HEIGHT / 2], rgba=[0.75, 0.75, 0.78, 1],
        )
    # Invisible, collision-free cylinder the loop tendon wraps around.
    world.add_geom(name="bundle_wrap", type=cyl, size=[WRAP_RADIUS, 0.01, 0], pos=[bx, by, LOOP_Z],
                   contype=0, conaffinity=0, group=3, rgba=[0, 0, 0, 0])
    world.add_site(name="wrap_side_pos", pos=[bx, by + WRAP_RADIUS + 0.01, LOOP_Z], group=3)
    world.add_site(name="wrap_side_neg", pos=[bx, by - WRAP_RADIUS - 0.01, LOOP_Z], group=3)

    hx = bx - WRAP_RADIUS - HEAD_HALF[0]
    world.add_geom(name="zip_head", type=box, size=list(HEAD_HALF), pos=[hx, by, LOOP_Z], rgba=[0.12, 0.12, 0.12, 1])
    for name, side in (("loop_a", 1), ("loop_b", -1)):
        world.add_site(name=name, pos=[hx + HEAD_HALF[0], by + side * (HEAD_HALF[1] - 0.002), LOOP_Z], group=3)

    # Loop slack: a massless-ish point beyond the bundle, coupled to the ratchet (below).
    slack = world.add_body(name="loop_slack", pos=[bx, by, LOOP_Z])
    slack.add_joint(name="loop_slack", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    slack.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.002, 0, 0], mass=1e-3,
                   contype=0, conaffinity=0, group=3)
    # q = 0 as threaded; the tip reaches SNUG_GAP off the rods at full travel (kept clear of the wrap
    # cylinder even when a hard pull overshoots the ratchet stop).
    slack.add_site(name="loop_tip", pos=[WRAP_RADIUS + SNUG_GAP + PULL_TRAVEL / 2, 0, 0], group=3)

    # Tail: root slides out of the head toward the robot (-X) on the ratchet joint.
    world.add_site(name="head_exit", pos=[hx - HEAD_HALF[0] + 0.002, by, LOOP_Z], group=3)
    root = world.add_body(name="tail_root", pos=[hx - HEAD_HALF[0], by, LOOP_Z])
    root.add_site(name="tail_start", pos=[0.001, 0, 0], group=3)
    root.add_joint(
        name="zip_ratchet", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, PULL_TRAVEL],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE, frictionloss=TOOTH_DRAG, damping=2.0, armature=0.05,
        solref_limit=[0.005, 1], solref_friction=[0.004, 1],
    )
    parent = root
    for i in range(TAIL_SEGMENTS):
        seg = parent.add_body(name=f"tail{i}", pos=[0 if i == 0 else -SEGMENT_LEN, 0, 0])
        if i > 0:
            seg.add_joint(name=f"tail_bend{i}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                          stiffness=TAIL_BEND_STIFFNESS, damping=TAIL_BEND_DAMPING, armature=1e-4)
        seg.add_geom(
            type=box, size=[SEGMENT_LEN / 2, STRAP_THICK / 2, STRAP_WIDTH / 2], pos=[-SEGMENT_LEN / 2, 0, 0],
            mass=SEGMENT_MASS, contype=TAIL_CONTYPE, conaffinity=TAIL_CONAFFINITY,
            friction=[1.0, 0.005, 0.0001], condim=4, rgba=[0.12, 0.12, 0.12, 1],
        )
        parent = seg

    # The loop shortens by about the tail pulled: its far end moves in by half of it.
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="loop_slack", name2="zip_ratchet")
    eq.data[:5] = [0, -0.5, 0, 0, 0]

    loop = spec.add_tendon(name="zip_loop", width=0.0035, rgba=[0.12, 0.12, 0.12, 1])
    loop.wrap_site("loop_a")
    loop.wrap_geom("bundle_wrap", "wrap_side_pos")
    loop.wrap_site("loop_tip")
    loop.wrap_geom("bundle_wrap", "wrap_side_neg")
    loop.wrap_site("loop_b")

    # Strap already pulled through the head: fills the gap between the head and the sliding tail.
    fed = spec.add_tendon(name="zip_fed", width=0.0035, rgba=[0.12, 0.12, 0.12, 1])
    fed.wrap_site("head_exit")
    fed.wrap_site("tail_start")

    # One-way pawl: pushes the tail back out to the furthest point it reached (ctrl, set by step())
    # and never pulls it in. Forcerange [0, F] clips the pulling direction to zero.
    pawl = spec.add_actuator(name="zip_pawl", target="zip_ratchet", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    pawl.set_to_position(kp=3000.0, kv=50.0)
    pawl.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    pawl.forcerange = [0.0, 200.0]


def loop_length(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(data.ten_length[model.tendon("zip_loop").id])


def snug_length(model: mujoco.MjModel) -> float:
    """Loop length with the tail pulled all the way (snug on the rods)."""
    d = mujoco.MjData(model)
    d.qpos[model.joint("zip_ratchet").qposadr[0]] = PULL_TRAVEL
    d.qpos[model.joint("loop_slack").qposadr[0]] = -PULL_TRAVEL / 2
    mujoco.mj_forward(model, d)
    return loop_length(model, d)


def is_tight(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return loop_length(model, data) <= snug_length(model) + TIGHT_SLACK
