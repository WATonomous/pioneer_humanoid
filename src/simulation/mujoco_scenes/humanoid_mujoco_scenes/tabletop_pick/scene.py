"""Put the named object in the red bowl: two household objects and a bowl on the table, a different object asked for each reset.

The objects are Google Scanned Objects (CC-BY 4.0, Google LLC), as converted to MuJoCo with convex-decomposed collision
by github.com/kevinzakka/mujoco_scanned_objects (``assets/props/scanned_objects``). Every reset shuffles which two of
the pool are on the table, where they and the bowl stand and how they are turned, and which one the instruction names, so the policy has to find the
object from the words. The table top, the lightbox's walls and the room floor get one of a few textures each (Poly Haven, CC0;
``assets/props/table_textures``, ``wall_textures``, ``floor_textures``) at random each reset.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_HALF_Y, TABLE_TOP_Z, TABLE_X, add_workcell, scene
from humanoid_mujoco_scenes.workcell import TABLE_CENTER
from pioneer_humanoid.mujoco_bimanual_arm import ARM_CONTYPE

T = TABLE_TOP_Z
PROPS = Path(__file__).resolve().parents[5] / "assets" / "props"
ASSETS = PROPS / "scanned_objects"
TABLE_TEXTURES = ("wood_table_001", "oak_veneer_01", "plywood", "laminate_floor_02", "concrete_floor_02")
TABLE_TEXTURE_TILE = 0.4         # metres per texture repeat
FLOOR_TEXTURES = ("concrete_floor_worn_001", "grey_tiles", "dark_wooden_planks")
FLOOR_TEXTURE_TILE = 1.0
WALL_TEXTURES = ("painted_plaster_wall", "beige_wall_001", "white_plaster_02", "red_brick_03", "grey_plaster_02",
                 "blue_painted_planks")
WALL_TEXTURE_TILE = 0.6
# visual sheets just inside the lightbox's back panel (inner face at x = 0.82) and its sides, sharing one texture per reset
WALL_BACK_X, WALL_HALF_Y, WALL_TOP_Z = 0.818, 0.740, 1.44

# name in the instruction: (folder, mass kg, scale, base yaw). Scaled so the narrow side is at most ~6.5 cm (the jaws open
# 8.3 cm and close across world Y); the base yaw turns that side across the jaws.
OBJECTS = {
    "orange robot": ("Android_Figure_Orange", 0.08, 1.0, 0.0),            # 5.5 x 5.0 x 8.0 cm
    "fire truck": ("FIRE_TRUCK", 0.10, 0.85, np.pi / 2),                  # 10.5 x 5.8 x 8.7
    "stacking toy": ("STACKING_BEAR", 0.08, 0.55, 0.0),                   # 5.8 x 5.6 x 9.7
    "green lion": ("My_First_Rolling_Lion", 0.08, 0.55, 0.0),             # 8.5 x 5.8 x 8.2
    "green mug": ("Cole_Hardware_Mug_Classic_Blue", 0.12, 0.5, 0.0),      # 8.1 x 6.3 x 4.7 (handle along X)
}
ON_TABLE = 2
BOWL = ("Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl", 0.4, 0.7)   # 20 cm across, 11 cm deep at this scale
BOWL_INNER_R = 0.085            # an object counts as in the bowl when its centre is this close to the bowl's axis
# Where the objects' and the bowl's centres are drawn from. The left arm's gripper-down reach with the wrist square (IK,
# warm-started from home) is x 0.30..0.58, y 0.20..0.45, but a real reach stalls high past x ~0.53 at grasp height
# (turning the wrist 0.3 rad shrinks it to ~16 x 14 cm); the bowl only needs reaching at carrying height. The front is
# pulled in so nothing stands at the table's edge.
OBJECT_AREA = ((0.34, 0.51), (0.21, 0.42))   # y <= 0.42 keeps them >= 40 px in from the ego image's edge
BOWL_AREA = ((0.36, 0.58), (0.21, 0.44))
# The gripper takes things fingers-forward, tipped 20 deg down: the hand hangs 5-6 cm over the table for 18 cm behind the
# grasp point and the wrist ~10 cm for 12 cm more, +-9 cm wide. Every object is taller than that, so nothing may stand
# behind an object (robot side) within this many metres sideways: objects 9 + 4 cm, the bowl 9 + 10 cm.
CLEAR_Y_OBJECT, CLEAR_Y_BOWL = 0.13, 0.19
BOWL_AHEAD = 0.17               # or the bowl stands this far beyond the object (finger tips 4 cm + its radius 10 cm + 3)
YAW_RANGE = 0.15                # rad either way about each object's base yaw: the scripted grasp keeps the wrist near its home yaw
PARK_X, PARK_DY = -0.6, 0.2       # objects not used this reset wait on the floor behind the robot, side by side
SETTLE_SECONDS = 0.3
MAX_SPEED = 0.01
NAMES = tuple(OBJECTS)


def _body(name: str) -> str:
    return "obj_" + name.replace(" ", "_")


def _park(i: int) -> list[float]:
    return [PARK_X, (i - (len(NAMES) - 1) / 2) * PARK_DY, 0.002]


def _quat(yaw: float) -> list[float]:
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def _layout(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bowl centre and ON_TABLE object centres, each object pickable without the hand hitting the rest (rejection sampling)."""
    for _ in range(100000):
        objects = rng.uniform(*zip(*OBJECT_AREA), size=(ON_TABLE, 2))
        if any(abs(a[1] - b[1]) < CLEAR_Y_OBJECT for i, a in enumerate(objects) for b in objects[i + 1:]):
            continue
        bowl = rng.uniform(*zip(*BOWL_AREA))
        if all(abs(o[1] - bowl[1]) >= CLEAR_Y_BOWL or bowl[0] - o[0] >= BOWL_AHEAD for o in objects):
            return bowl, objects
    raise RuntimeError("tabletop_pick: no layout fits the area; loosen the areas or the clearances")


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """ON_TABLE of the pool and the bowl at random places in reach, turned at random; one of them asked for.
    userdata[0] = index of the asked-for object, userdata[1] = last time it was not settled in the bowl."""
    chosen = list(rng.permutation(len(NAMES))[:ON_TABLE])
    bowl, places = _layout(rng)
    for i, name in enumerate(NAMES):
        adr = model.joint(_body(name)).qposadr[0]
        if i in chosen:
            data.qpos[adr:adr + 3] = [*places[chosen.index(i)], T + 0.002]
            data.qpos[adr + 3:adr + 7] = _quat(OBJECTS[name][3] + rng.uniform(-YAW_RANGE, YAW_RANGE))
        else:
            data.qpos[adr:adr + 3] = _park(i)
            data.qpos[adr + 3:adr + 7] = _quat(0.0)
    adr = model.joint("bowl").qposadr[0]
    data.qpos[adr:adr + 3] = [*bowl, T + 0.002]
    data.qpos[adr + 3:adr + 7] = _quat(rng.uniform(-np.pi, np.pi))
    wall = model.material(f"wall_{rng.choice(WALL_TEXTURES)}").id
    for name in ("wall_back", "wall_left", "wall_right"):
        model.geom_matid[model.geom(name).id] = wall
    model.geom_matid[model.geom("floor").id] = model.material(f"floor_{rng.choice(FLOOR_TEXTURES)}").id
    model.geom_matid[model.geom("table_surface").id] = model.material(f"table_{rng.choice(TABLE_TEXTURES)}").id
    target = int(rng.choice(chosen))
    if _flip_target():  # same layout, the other object asked for: paired demos that differ only in the instruction
        target = next(i for i in chosen if i != target)
    data.userdata[0] = target
    data.userdata[1] = data.time


def _flip_target() -> bool:
    value = os.environ.get("TABLETOP_PICK_FLIP_TARGET", "0")
    if value not in ("0", "1"):
        raise ValueError(f"TABLETOP_PICK_FLIP_TARGET must be 0 or 1, got {value!r}")
    return value == "1"


def _in_bowl(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> bool:
    body = model.body(_body(name)).id
    bowl = data.xpos[model.body("bowl").id]
    pos = data.xpos[body]
    return bool(np.linalg.norm(pos[:2] - bowl[:2]) < BOWL_INNER_R and T < pos[2] < bowl[2] + 0.12)


def _settled(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> bool:
    """In the bowl, still and not held: no contact with the arm."""
    body = model.body(_body(name)).id
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    if not _in_bowl(model, data, name) or np.linalg.norm(velocity[3:]) > MAX_SPEED:
        return False
    for contact in data.contact:
        bodies = (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2])
        if body in bodies:
            other = contact.geom2 if bodies[0] == body else contact.geom1
            if model.geom_contype[other] & ARM_CONTYPE:
                return False
    return True


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if not _settled(model, data, NAMES[int(data.userdata[0])]):
        data.userdata[1] = data.time


def success(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """The asked-for object rests in the bowl, released, for SETTLE_SECONDS, and no other object is in the bowl."""
    target = NAMES[int(data.userdata[0])]
    others = any(_in_bowl(model, data, n) for n in NAMES if n != target)
    return bool(data.time - data.userdata[1] >= SETTLE_SECONDS and not others)


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    return int(success(model, data)), 1, f"put the {NAMES[int(data.userdata[0])]} in the red bowl"


def _add_object(spec: mujoco.MjSpec, body_name: str, folder: str, mass: float, scale: float, pos) -> None:
    """A scanned object as a free body: the textured mesh for looks, its convex pieces for contact."""
    src = ASSETS / folder
    if not (src / "model.obj").is_file():
        raise FileNotFoundError(f"scanned object not found: {src}")
    spec.add_texture(name=f"{body_name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(src / "texture.png"))
    material = spec.add_material(name=f"{body_name}_mat", specular=0.3, shininess=0.3)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{body_name}_tex"
    spec.add_mesh(name=f"{body_name}_visual", file=str(src / "model.obj"), scale=[scale] * 3)
    pieces = sorted(src.glob("model_collision_*.obj"))
    if not pieces:
        raise FileNotFoundError(f"no collision pieces in {src}")
    body = spec.worldbody.add_body(name=body_name, pos=list(pos))
    body.add_freejoint(name=body_name)
    body.add_geom(name=f"{body_name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{body_name}_visual",
                  material=f"{body_name}_mat", contype=0, conaffinity=0, group=2, mass=0)
    for k, piece in enumerate(pieces):
        spec.add_mesh(name=f"{body_name}_col{k}", file=str(piece), scale=[scale] * 3)
        body.add_geom(name=f"{body_name}_col{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{body_name}_col{k}",
                      group=3, mass=mass / len(pieces), friction=[1.0, 0.02, 0.001], condim=4, rgba=[0.5, 0.5, 0.5, 0])


def _add_textures(spec: mujoco.MjSpec, folder: str, names: tuple[str, ...], repeat: list[float], **material) -> None:
    for name in names:
        src = PROPS / folder / f"{name}.png"
        if not src.is_file():
            raise FileNotFoundError(f"texture not found: {src}")
        prefix = folder.split("_")[0]
        spec.add_texture(name=f"{prefix}_{name}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(src))
        mat = spec.add_material(name=f"{prefix}_{name}", texrepeat=repeat, **material)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{prefix}_{name}"


def _add_table_surface(spec: mujoco.MjSpec) -> None:
    """A thin visual sheet over the lightbox's white boards; reset picks its material."""
    (x0, x1), cy = TABLE_X, TABLE_CENTER[1]
    half = [(x1 - x0) / 2 - 0.004, TABLE_HALF_Y - 0.02]
    _add_textures(spec, "table_textures", TABLE_TEXTURES, [2 * half[0] / TABLE_TEXTURE_TILE, 2 * half[1] / TABLE_TEXTURE_TILE],
                  specular=0.15, shininess=0.2)
    spec.worldbody.add_geom(name="table_surface", type=mujoco.mjtGeom.mjGEOM_BOX, size=[*half, 0.0002],
                            pos=[(x0 + x1) / 2, cy, T + 0.0002], material=f"table_{TABLE_TEXTURES[0]}",
                            contype=0, conaffinity=0, group=1)


def _add_walls(spec: mujoco.MjSpec) -> None:
    _add_textures(spec, "wall_textures", WALL_TEXTURES, [1 / WALL_TEXTURE_TILE] * 2, specular=0.05, texuniform=True)
    # 2D textures project along the box's local z, so the sheet's thin axis is local z (facing the robot), with x along y and y up
    wall = spec.worldbody.add_geom(name="wall_back", type=mujoco.mjtGeom.mjGEOM_BOX, size=[WALL_HALF_Y, (WALL_TOP_Z - T) / 2, 0.0005],
                                   pos=[WALL_BACK_X, 0.0, (WALL_TOP_Z + T) / 2], material=f"wall_{WALL_TEXTURES[0]}",
                                   contype=0, conaffinity=0, group=1)
    wall.quat = [0.5, 0.5, 0.5, 0.5]  # local x -> world y, local y -> world z, local z -> world x
    # side sheets just inside the lightbox's sides (CAD: frame inner face at y = +0.743, side panel at y = -0.731), from the
    # table's front edge to the back wall
    x0 = TABLE_X[0]
    for name, y in (("wall_left", 0.741), ("wall_right", -0.729)):
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[(WALL_BACK_X - x0) / 2, (WALL_TOP_Z - T) / 2, 0.0005],
                                pos=[(WALL_BACK_X + x0) / 2, y, (WALL_TOP_Z + T) / 2], quat=[0.70710678, 0.70710678, 0, 0],
                                material=f"wall_{WALL_TEXTURES[0]}", contype=0, conaffinity=0, group=1)  # local y -> world z


@scene("tabletop_pick", camera=dict(lookat=[0.37, 0.12, T + 0.05], distance=1.1, azimuth=0, elevation=-70),
       step=step, reset=reset, progress=progress, success=success)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    spec.nuserdata = 2
    _add_table_surface(spec)
    _add_walls(spec)
    # the light tube's housing hangs in front of the ego camera and blocks the top of the view; keep only its light
    for g in [g for g in spec.geoms if g.name in ("tube_housing", "tube_diffuser") or g.material == "tube_housing"]:
        spec.delete(g)
    floor = spec.geom("floor")
    _add_textures(spec, "floor_textures", FLOOR_TEXTURES, [2 * floor.size[0] / FLOOR_TEXTURE_TILE] * 2, specular=0.1, reflectance=0.05)
    floor.material = f"floor_{FLOOR_TEXTURES[0]}"
    for i, (name, (folder, mass, scale, _)) in enumerate(OBJECTS.items()):
        _add_object(spec, _body(name), folder, mass, scale, _park(i))
    _add_object(spec, "bowl", BOWL[0], BOWL[1], BOWL[2], (*np.mean(BOWL_AREA, axis=1), T + 0.002))
