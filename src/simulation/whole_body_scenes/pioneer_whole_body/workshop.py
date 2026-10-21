"""A workshop room for the scenes that should look like a real place (``add_workshop`` in place of ``add_studio`` +
``add_bench``): a concrete floor, painted walls behind and beside the robot, a pegboard with hanging tools over a
low cabinet, steel shelving with cardboard boxes, a butcher-block workbench on a grey steel frame, warm overhead
light. The props are CAD meshes and procedural textures in ``assets/workshop`` (``make_meshes.py``,
``make_textures.py``).

The bench keeps ``studio``'s size and height (``BENCH_TOP_Z``, ``BENCH_X``), so the robot works at it the same way.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco

from .studio import BENCH_HALF_Y, BENCH_TOP_Z, BENCH_X

ASSETS = Path(__file__).resolve().parents[4] / "assets" / "workshop"
BOX = mujoco.mjtGeom.mjGEOM_BOX
MESH = mujoco.mjtGeom.mjGEOM_MESH
BACK_X = -1.35          # the wall behind the robot
SIDE_Y = 2.0            # the wall on its left


def texture(spec, name: str, repeat=(1, 1), rgba=(1, 1, 1, 1), specular=0.1, shininess=0.2, reflectance=0.0,
            uniform=False, mat: str | None = None):
    """A material (named ``mat``, else ``name``) with a PNG texture from ``assets/workshop/textures``."""
    if spec.texture(name) is None:
        spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(ASSETS / "textures" / f"{name}.png"))
    m = spec.add_material(name=mat or name, texrepeat=list(repeat), rgba=list(rgba), specular=specular,
                          shininess=shininess, reflectance=reflectance, texuniform=uniform)
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    return m.name


def mesh(spec, name: str, file: str | None = None, scale=1.0) -> str:
    spec.add_mesh(name=name, file=str(ASSETS / "meshes" / f"{file or name}.stl"), scale=[scale] * 3)
    return name


def paint(spec, name: str, rgba, specular=0.3, shininess=0.4, reflectance=0.0) -> str:
    spec.add_material(name=name, rgba=list(rgba), specular=specular, shininess=shininess, reflectance=reflectance)
    return name


def add_workshop(spec: mujoco.MjSpec, bench: bool = True) -> None:
    v = spec.visual
    v.headlight.ambient = [0.20, 0.20, 0.20]
    v.headlight.diffuse = [0.18, 0.18, 0.18]
    v.headlight.specular = [0.03, 0.03, 0.03]
    v.quality.shadowsize = 4096
    v.map.znear = 0.003
    v.map.zfar = 20.0
    v.rgba.haze = [0.55, 0.55, 0.55, 1]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.55, 0.55, 0.55], rgb2=[0.55, 0.55, 0.55], width=32, height=32)
    world = spec.worldbody
    vis = dict(contype=0, conaffinity=0)

    # floor and walls
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.05],
                   material=texture(spec, "concrete", repeat=(3, 3), specular=0.12, shininess=0.15))
    wall = texture(spec, "wall", repeat=(2, 2), specular=0.02)
    world.add_geom(type=BOX, size=[0.05, 3.5, 1.6], pos=[BACK_X - 0.05, 0, 1.6], material=wall, **vis)
    world.add_geom(type=BOX, size=[4.0, 0.05, 1.6], pos=[0.5, SIDE_Y + 0.05, 1.6], material=wall, **vis)
    skirting = paint(spec, "skirting", [0.25, 0.26, 0.28, 1])
    world.add_geom(type=BOX, size=[0.01, 3.5, 0.05], pos=[BACK_X + 0.01, 0, 0.05], material=skirting, **vis)
    world.add_geom(type=BOX, size=[4.0, 0.01, 0.05], pos=[0.5, SIDE_Y - 0.01, 0.05], material=skirting, **vis)

    # the back wall: a low steel cabinet, a pegboard over it with tools hanging
    steel_grey = paint(spec, "cabinet_grey", [0.36, 0.38, 0.41, 1], specular=0.35, shininess=0.5)
    cab_x = BACK_X + 0.25
    world.add_geom(type=BOX, size=[0.25, 0.9, 0.42], pos=[cab_x, 0.2, 0.42], material=steel_grey, **vis)
    for i in range(3):                                 # its doors' seams and handles
        y = 0.2 - 0.9 + 0.3 + i * 0.6
        world.add_geom(type=BOX, size=[0.002, 0.004, 0.38], pos=[cab_x + 0.251, y + 0.3, 0.42],
                       rgba=[0.2, 0.21, 0.23, 1], **vis)
        world.add_geom(type=BOX, size=[0.012, 0.006, 0.07], pos=[cab_x + 0.262, y + 0.24, 0.62],
                       rgba=[0.75, 0.76, 0.78, 1], **vis)
    world.add_geom(type=BOX, size=[0.26, 0.92, 0.015], pos=[cab_x, 0.2, 0.855],
                   material=texture(spec, "butcher_block", repeat=(1, 3), specular=0.2, mat="cabinet_top"), **vis)
    peg = texture(spec, "pegboard", repeat=(3, 2), specular=0.05)
    world.add_geom(type=BOX, size=[0.008, 1.0, 0.45], pos=[BACK_X + 0.01, 0.2, 1.45], material=peg, **vis)
    chrome = paint(spec, "chrome", [0.78, 0.79, 0.82, 1], specular=0.9, shininess=0.9, reflectance=0.15)
    red_grip = paint(spec, "red_grip", [0.72, 0.10, 0.08, 1], specular=0.2, shininess=0.3)
    black_grip = paint(spec, "black_grip", [0.08, 0.08, 0.09, 1], specular=0.15, shininess=0.2)
    wood = paint(spec, "hammer_wood", [0.62, 0.45, 0.28, 1], specular=0.1)
    yellow = paint(spec, "tape_yellow", [0.92, 0.72, 0.08, 1], specular=0.3)
    px = BACK_X + 0.03
    # (each tool's mesh, material, position on the board, and how it's turned: lying in the board's plane)
    tools = [("wrench", chrome, (0.95, 1.55), 90), ("wrench_small", chrome, (0.95, 1.40), 90),
             ("hammer", wood, (0.65, 1.22), 0), ("pliers", red_grip, (0.35, 1.30), 0),
             ("tape_measure", yellow, (0.10, 1.62), 0), ("wrench", chrome, (-0.20, 1.50), 75),
             ("pliers", black_grip, (-0.45, 1.28), 10)]
    for k, (m, mat, (y, z), a) in enumerate(tools):
        name = f"peg_{m}_{k}"
        mesh(spec, name, m)
        # flat on the board: the tool's thin axis (z) turned to face out (+x), then turned in the board's plane
        q = _quat_mul(_quat_axis([1, 0, 0], math.radians(a)), _quat_axis([0, 1, 0], math.pi / 2))
        world.add_geom(type=MESH, meshname=name, pos=[px + 0.012, y, z], quat=q, material=mat, **vis)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.003, 0.02, 0], pos=[px + 0.01, y, z + 0.06],
                       quat=_quat_axis([0, 1, 0], math.pi / 2), material=chrome, **vis)    # its peg
    # shelving on the left wall: a steel frame, three shelves, cardboard boxes
    shelf_x, shelf_y = 0.2, SIDE_Y - 0.25
    frame = paint(spec, "shelf_steel", [0.20, 0.30, 0.52, 1], specular=0.3)
    board = paint(spec, "shelf_board", [0.70, 0.62, 0.50, 1], specular=0.05)
    for dx in (-0.6, 0.6):
        for dy in (-0.2, 0.2):
            world.add_geom(type=BOX, size=[0.018, 0.018, 0.9], pos=[shelf_x + dx, shelf_y + dy, 0.9], material=frame,
                           **vis)
    card = paint(spec, "cardboard", [0.62, 0.47, 0.30, 1], specular=0.05)
    tape = paint(spec, "tape", [0.80, 0.72, 0.55, 1], specular=0.2)
    blue_bin = paint(spec, "bin_blue", [0.15, 0.32, 0.62, 1], specular=0.2)
    for z in (0.35, 0.95, 1.55):
        world.add_geom(type=BOX, size=[0.62, 0.22, 0.012], pos=[shelf_x, shelf_y, z], material=board, **vis)
        world.add_geom(type=BOX, size=[0.62, 0.012, 0.02], pos=[shelf_x, shelf_y - 0.215, z], material=frame, **vis)
    boxes = [(0.35, -0.38, 0.17, 0.15, card), (0.35, 0.05, 0.13, 0.11, card), (0.35, 0.38, 0.15, 0.14, blue_bin),
             (0.95, -0.35, 0.16, 0.18, card), (0.95, 0.12, 0.10, 0.09, blue_bin), (0.95, 0.30, 0.10, 0.09, blue_bin),
             (1.55, -0.25, 0.20, 0.14, card), (1.55, 0.25, 0.18, 0.12, card)]
    for z, dx, hw, hh, mat in boxes:
        world.add_geom(type=BOX, size=[hw, 0.17, hh], pos=[shelf_x + dx, shelf_y + 0.02, z + 0.012 + hh],
                       material=mat, **vis)
        if mat == card:
            world.add_geom(type=BOX, size=[0.025, 0.171, hh * 0.98], pos=[shelf_x + dx, shelf_y + 0.02,
                           z + 0.012 + hh], material=tape, **vis)
    # light: a warm key over the bench, a cool fill from the left, overhead
    world.add_light(name="key", pos=[1.2, -1.6, 3.0], dir=[-0.3, 0.5, -0.8], diffuse=[0.55, 0.52, 0.48],
                    specular=[0.25, 0.25, 0.25], castshadow=True, cutoff=55, exponent=3)
    world.add_light(name="fill", pos=[0.5, 1.5, 2.4], dir=[0.1, -0.6, -0.8], diffuse=[0.18, 0.19, 0.21],
                    specular=[0, 0, 0], castshadow=False, cutoff=80)
    world.add_light(name="top", pos=[0.2, 0.2, 3.2], dir=[0, 0, -1], diffuse=[0.20, 0.20, 0.19],
                    specular=[0.05, 0.05, 0.05], castshadow=False, cutoff=75, exponent=1)
    if bench:
        add_workbench(spec)


def add_workbench(spec: mujoco.MjSpec, x=BENCH_X, half_y=BENCH_HALF_Y, top_z=BENCH_TOP_Z) -> None:
    """The workbench: a 40 mm butcher-block top (collides) on a grey steel frame with a lower shelf (visual)."""
    world = spec.worldbody
    x0, x1 = x
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    world.add_geom(name="bench_top", type=BOX, size=[hx, half_y, 0.02], pos=[cx, 0, top_z - 0.02],
                   material=texture(spec, "butcher_block", repeat=(1, 2), specular=0.25, shininess=0.35),
                   friction=[0.8, 0.01, 0.001])
    steel = paint(spec, "bench_steel", [0.30, 0.31, 0.33, 1], specular=0.4, shininess=0.55)
    vis = dict(contype=0, conaffinity=0, material=steel)
    for sx in (-1, 1):
        for sy in (-1, 1):
            world.add_geom(type=BOX, size=[0.022, 0.022, (top_z - 0.04) / 2],
                           pos=[cx + sx * (hx - 0.05), sy * (half_y - 0.05), (top_z - 0.04) / 2], **vis)
            world.add_geom(type=BOX, size=[0.04, 0.04, 0.006], pos=[cx + sx * (hx - 0.05), sy * (half_y - 0.05),
                           0.006], rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0)                 # (its feet)
        world.add_geom(type=BOX, size=[0.02, half_y - 0.05, 0.03], pos=[cx + sx * (hx - 0.05), 0, top_z - 0.07], **vis)
    world.add_geom(type=BOX, size=[hx - 0.05, 0.02, 0.03], pos=[cx, half_y - 0.05, top_z - 0.07], **vis)
    world.add_geom(type=BOX, size=[hx - 0.05, 0.02, 0.03], pos=[cx, -(half_y - 0.05), top_z - 0.07], **vis)
    world.add_geom(type=BOX, size=[hx - 0.06, half_y - 0.06, 0.01], pos=[cx, 0, 0.16], **vis)       # the shelf


def _quat_axis(axis, a):
    s = math.sin(a / 2)
    return [math.cos(a / 2), axis[0] * s, axis[1] * s, axis[2] * s]


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
