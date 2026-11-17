"""Open a cabinet door by its handle, a standard articulated-object benchmark.

The cabinet is fixed during each take, but every reset randomises its reachable
table position and yaw.  Its door starts closed (with a small random crack), is
hinged on its outer edge, and swings toward the robot.  A chunky vertical pull
handle leaves enough clearance for the parallel gripper.  The task completes
once the door is at least ``OPEN_ANGLE`` from closed.

The one-step ``progress`` hook gives Quest recording the instruction
automatically, so a take can be collected without passing a separate task
description.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene

T = TABLE_TOP_Z

# Cabinet dimensions.  The front is far enough back that the open door and
# handle remain over the table and inside the left arm's workspace.
CABINET_FRONT_X = 0.56
CABINET_CENTER_Y = 0.35
CABINET_DEPTH = 0.18
CABINET_WIDTH = 0.20
CABINET_HEIGHT = 0.24
WALL = 0.012

# Domain randomisation for demonstrations.  These bounds keep the whole
# cabinet on the workcell and the handle inside the left arm's practical reach.
CABINET_FRONT_X_RANGE = (0.58, 0.66)
CABINET_CENTER_Y_RANGE = (0.24, 0.46)
CABINET_YAW_RANGE = tuple(map(math.radians, (-12, 12)))
# Consecutive demonstrations must not look like the same setup.  Sampling with
# this minimum translation makes every saved/reset episode visibly different,
# while the full pose remains continuously random inside the safe region.
MIN_RESET_TRANSLATION = 0.10

# The door is inset from the carcass edges so it never starts in penetration.
DOOR_WIDTH = 0.19
DOOR_HEIGHT = 0.21
DOOR_THICKNESS = 0.018
DOOR_CLEARANCE_Z = 0.015
HINGE_LIMIT = math.radians(100)
OPEN_ANGLE = math.radians(75)
RESET_CRACK = math.radians(4)
GRASP_RELEASE = 0.002
GRASP_CONNECT_NAMES = ("cabinet_grasp_low", "cabinet_grasp_high")
GRASP_CONNECT_OFFSET = 0.035
GRASP_CAPTURE_COMMAND = 0.020
GRASP_NONE, GRASP_PINCH = range(2)

# Handle: a vertical bar on two short standoffs, near the free edge.  It sits
# 41 mm in front of the panel, giving the 61 mm fingers room to pinch it.
HANDLE_FROM_FREE_EDGE = 0.025
HANDLE_PROJECTION = 0.041
HANDLE_HALF_WIDTH = 0.012
HANDLE_HALF_HEIGHT = 0.055

INSTRUCTION = "open the cabinet door"


def _box(body, name, pos, size, rgba, **kwargs):
    return body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=size,
        rgba=rgba,
        **kwargs,
    )


def door_angle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Door angle in radians: zero is closed and negative is open."""
    return float(data.qpos[model.joint("cabinet_door_hinge").qposadr[0]])


def is_open(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return door_angle(model, data) <= -OPEN_ANGLE


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng) -> None:
    """Move the cabinet to a visibly new reachable pose and add a small crack."""
    cabinet = model.body("cabinet").id
    previous = model.body_pos[cabinet, :2].copy()
    for _ in range(128):
        xy = rng.uniform(
            [CABINET_FRONT_X_RANGE[0], CABINET_CENTER_Y_RANGE[0]],
            [CABINET_FRONT_X_RANGE[1], CABINET_CENTER_Y_RANGE[1]],
        )
        if math.dist(previous, xy) >= MIN_RESET_TRANSLATION:
            break
    else:  # deterministic safe fallback; practically unreachable
        far_y = max(CABINET_CENTER_Y_RANGE, key=lambda y: abs(y - previous[1]))
        xy = [rng.uniform(*CABINET_FRONT_X_RANGE), far_y]
    model.body_pos[cabinet] = [xy[0], xy[1], T]
    yaw = rng.uniform(*CABINET_YAW_RANGE)
    model.body_quat[cabinet] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    adr = model.joint("cabinet_door_hinge").qposadr[0]
    data.qpos[adr] = -rng.uniform(0.0, RESET_CRACK)
    data.userdata[0] = 0.0
    data.userdata[1] = 0.0
    data.userdata[2] = GRASP_NONE
    for name in GRASP_CONNECT_NAMES:
        data.eq_active[model.equality(name).id] = False


def _set_grasp(model: mujoco.MjModel, data: mujoco.MjData, active: bool) -> None:
    """Attach/release two handle points, preserving the exact capture pose.

    Two point constraints prevent translation and tilt but still allow the bar
    to rotate about its own axis.  Anchors are computed from the current 3-D
    pose, so activation cannot teleport the door or claw.
    """
    ids = [model.equality(name).id for name in GRASP_CONNECT_NAMES]
    if not active:
        data.eq_active[ids] = False
        return
    link = model.body("left_link7").id
    door = model.body("cabinet_door").id
    handle = model.geom("cabinet_handle").id
    link_rotation = data.xmat[link].reshape(3, 3)
    door_rotation = data.xmat[door].reshape(3, 3)
    axis = data.geom_xmat[handle].reshape(3, 3)[:, 2]
    data.eq_active[ids] = False
    for eq_id, offset in zip(ids, (-GRASP_CONNECT_OFFSET, GRASP_CONNECT_OFFSET)):
        # The scene also compiles against the legacy armv2 robot, whose names
        # differ. Constraints are declared on base_link and retargeted here in
        # the 8-DoF VR scene where this hook is used.
        model.eq_obj1id[eq_id] = link
        model.eq_obj2id[eq_id] = door
        point = data.geom_xpos[handle] + offset * axis
        model.eq_data[eq_id, :3] = link_rotation.T @ (point - data.xpos[link])
        model.eq_data[eq_id, 3:6] = door_rotation.T @ (point - data.xpos[door])
        data.eq_active[eq_id] = True


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Latch completion and turn real handle contact into a physical grasp."""
    # The differential gripper lets the first jaw stop while the opposite jaw
    # continues closing. Attach only after both visible pads truly touch (never
    # from proximity), preserve that exact pose, and cap the average tendon
    # closure until the operator opens the trigger.
    cap = float(data.userdata[1])
    phase = int(data.userdata[2])
    actuator = model.actuator("left_gripper").id
    command = float(data.ctrl[actuator])
    touching = {"a": False, "b": False}
    for contact in data.contact[:data.ncon]:
        first, second = model.geom(contact.geom1).name, model.geom(contact.geom2).name
        if first == "cabinet_handle":
            other = second
        elif second == "cabinet_handle":
            other = first
        else:
            continue
        touching["a"] |= other.startswith("left_finger_a_contact_pad_")
        touching["b"] |= other.startswith("left_finger_b_contact_pad_")

    if phase == GRASP_NONE:
        if command >= GRASP_CAPTURE_COMMAND and all(touching.values()):
            cap = float(np.mean([
                data.qpos[model.joint("left_gripper").qposadr[0]],
                data.qpos[model.joint("left_gripper_mimic").qposadr[0]],
            ]))
            data.userdata[1] = cap
            data.userdata[2] = GRASP_PINCH
            _set_grasp(model, data, True)

    elif phase == GRASP_PINCH:
        if command < cap - GRASP_RELEASE:
            data.userdata[1] = 0.0
            data.userdata[2] = GRASP_NONE
            _set_grasp(model, data, False)
        elif command > cap:
            data.ctrl[actuator] = cap

    if data.userdata[0] == 0.0 and is_open(model, data):
        data.userdata[0] = 1.0


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    return int(data.userdata[0]), 1, INSTRUCTION


@scene(
    "door_open",
    camera=dict(
        lookat=[CABINET_FRONT_X - 0.06, CABINET_CENTER_Y, T + CABINET_HEIGHT / 2],
        distance=0.65,
        azimuth=340,
        elevation=-30,
    ),
    step=step,
    reset=reset,
    progress=progress,
)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    spec.nuserdata = 3  # latched completion + handle-grasp closure cap + grasp phase

    wood = [0.78, 0.68, 0.52, 1.0]
    door_colour = [0.36, 0.56, 0.72, 1.0]
    handle_colour = [0.18, 0.19, 0.21, 1.0]

    # Put the carcass in one fixed body.  Door coordinates below are relative
    # to its front-centre-bottom point, which keeps the geometry easy to audit.
    cabinet = spec.worldbody.add_body(
        name="cabinet",
        pos=[CABINET_FRONT_X, CABINET_CENTER_Y, T],
    )
    half_w = CABINET_WIDTH / 2
    half_d = CABINET_DEPTH / 2
    half_h = CABINET_HEIGHT / 2
    _box(cabinet, "cabinet_bottom", [half_d, 0, WALL / 2],
         [half_d, half_w, WALL / 2], wood)
    _box(cabinet, "cabinet_top", [half_d, 0, CABINET_HEIGHT - WALL / 2],
         [half_d, half_w, WALL / 2], wood)
    _box(cabinet, "cabinet_left_side", [half_d, half_w - WALL / 2, half_h],
         [half_d, WALL / 2, half_h - WALL], wood)
    _box(cabinet, "cabinet_right_side", [half_d, -half_w + WALL / 2, half_h],
         [half_d, WALL / 2, half_h - WALL], wood)
    _box(cabinet, "cabinet_back", [CABINET_DEPTH - WALL / 2, 0, half_h],
         [WALL / 2, half_w - WALL, half_h - WALL], wood)

    # Hinge at the high-Y edge.  A negative angle swings the handle toward the
    # robot (-X), matching the joint range and is_open() convention.
    hinge_y = DOOR_WIDTH / 2
    door = cabinet.add_body(
        name="cabinet_door",
        pos=[-DOOR_THICKNESS / 2 - 0.002, hinge_y, DOOR_CLEARANCE_Z],
    )
    door.add_joint(
        name="cabinet_door_hinge",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        range=[-HINGE_LIMIT, 0],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        frictionloss=0.25,
        damping=0.35,
        armature=0.01,
    )
    _box(
        door,
        "cabinet_door_panel",
        [0, -DOOR_WIDTH / 2, DOOR_HEIGHT / 2],
        [DOOR_THICKNESS / 2, DOOR_WIDTH / 2, DOOR_HEIGHT / 2],
        door_colour,
        mass=1.2,
        friction=[1.0, 0.02, 0.001],
        condim=4,
    )
    # Inactive until the trigger closes around the handle inside the measured
    # jaw volume. Runtime code replaces both bodies' anchors at capture time.
    for name in GRASP_CONNECT_NAMES:
        spec.add_equality(
            name=name,
            type=mujoco.mjtEq.mjEQ_CONNECT,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1="base_link",
            name2="cabinet_door",
            active=False,
            solref=[0.003, 1.0],
            solimp=[0.98, 0.999, 0.001, 0.5, 2.0],
        )

    handle_y = -DOOR_WIDTH + HANDLE_FROM_FREE_EDGE
    handle_x = -DOOR_THICKNESS / 2 - HANDLE_PROJECTION
    handle_z = DOOR_HEIGHT / 2
    for z in (handle_z - HANDLE_HALF_HEIGHT, handle_z + HANDLE_HALF_HEIGHT):
        _box(
            door,
            f"cabinet_handle_standoff_{'low' if z < handle_z else 'high'}",
            [(handle_x - DOOR_THICKNESS / 2) / 2, handle_y, z],
            [(-DOOR_THICKNESS / 2 - handle_x) / 2, HANDLE_HALF_WIDTH, 0.009],
            handle_colour,
            mass=0.02,
        )
    _box(
        door,
        "cabinet_handle",
        [handle_x, handle_y, handle_z],
        [HANDLE_HALF_WIDTH, HANDLE_HALF_WIDTH, HANDLE_HALF_HEIGHT],
        handle_colour,
        mass=0.08,
        friction=[1.2, 0.02, 0.001],
        condim=4,
    )
