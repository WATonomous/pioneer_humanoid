"""Rubik's cube and two whiteboard markers in the workcell (as mujoco_scenes' rubiks_markers).

A 57 mm 3x3 cube (yellow on top) left of centre and an orange and a red whiteboard marker lying on their
sides to its right, all free bodies on the table top. The cube's stickers and the markers' caps, plugs and
labels are visual children of their rigid bodies.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from humanoid_isaac_scenes import scene
from humanoid_rl_tasks.workcell import ROBOT_BASE_POS, ROBOT_BASE_X, TABLE_TOP_Z, LightboxWorkcellCfg

T = TABLE_TOP_Z

CUBE = 0.057                     # 3x3 Rubik's cube edge
CUBE_POS = (ROBOT_BASE_X + 0.53, 0.06)
CUBE_YAW = math.radians(32)      # green face turned toward the open front
# Rubik's colours: yellow top, orange toward the open front (-x), green on the left seen from the front (+y).
FACE_RGB = {
    "+z": (0.98, 0.85, 0.05), "-z": (0.95, 0.95, 0.95),
    "-x": (1.0, 0.42, 0.0), "+x": (0.85, 0.05, 0.08),
    "+y": (0.0, 0.6, 0.25), "-y": (0.05, 0.25, 0.85),
}
MARKER_LEN, MARKER_R = 0.135, 0.009


def _yaw(rad: float) -> tuple:
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


def _visual_box(path: str, size, pos, rgb) -> None:
    cfg = sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.3))
    cfg.func(path, cfg, translation=pos)


def _visual_cylinder(path: str, radius: float, length: float, x: float, rgb) -> None:
    cfg = sim_utils.CylinderCfg(radius=radius, height=length, axis="X",
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.4))
    cfg.func(path, cfg, translation=(x, 0.0, 0.0))


@clone
def spawn_rubiks_cube(prim_path, cfg, translation=None, orientation=None):
    """The cube's black core (the rigid body) with 9 stickers on each face."""
    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation, orientation)
    h, cell, sticker = CUBE / 2, CUBE / 3, CUBE / 3 * 0.84
    for face, rgb in FACE_RGB.items():
        sign, axis = (1 if face[0] == "+" else -1), "xyz".index(face[1])
        others = [a for a in range(3) if a != axis]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                pos, size = [0.0] * 3, [0.0] * 3
                pos[axis], size[axis] = sign * (h + 0.0004), 0.001
                pos[others[0]], pos[others[1]] = i * cell, j * cell
                size[others[0]] = size[others[1]] = sticker
                _visual_box(f"{prim_path}/sticker_{'pm'[sign < 0]}{face[1]}_{i + 1}{j + 1}", size, pos, rgb)
    return prim


def _marker_spawner(rgb):
    @clone
    def spawn_marker(prim_path, cfg, translation=None, orientation=None):
        """Whiteboard marker lying on its side: white barrel (the rigid body), coloured cap, plug and label band."""
        prim = sim_utils.spawn_cylinder(prim_path, cfg, translation, orientation)
        half = MARKER_LEN / 2
        _visual_cylinder(f"{prim_path}/cap", MARKER_R * 1.01, 0.044, -half + 0.022, rgb)
        _visual_cylinder(f"{prim_path}/plug", MARKER_R * 0.75, 0.008, half + 0.002, rgb)
        _visual_cylinder(f"{prim_path}/label", MARKER_R * 1.01, 0.032, 0.0, rgb)
        return prim

    return spawn_marker


def _marker(name: str, xy, yaw_deg: float, rgb) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*xy, T + MARKER_R + 0.003), rot=_yaw(math.radians(yaw_deg))),
        spawn=sim_utils.CylinderCfg(
            func=_marker_spawner(rgb),
            radius=MARKER_R, height=MARKER_LEN, axis="X",
            # PhysX has no rolling friction: angular damping brings a nudged marker to rest instead
            rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=10.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.017),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.96, 0.96, 0.95), roughness=0.4),
        ),
    )


@scene("rubiks_markers", robot_pos=ROBOT_BASE_POS,
       camera=([ROBOT_BASE_X - 1.05, -0.56, T + 0.95], [ROBOT_BASE_X + 0.5, 0.0, T]))
@configclass
class RubiksMarkersSceneCfg(LightboxWorkcellCfg):
    rubiks_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RubiksCube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CUBE_POS, T + CUBE / 2 + 0.003), rot=_yaw(CUBE_YAW)),
        spawn=sim_utils.CuboidCfg(
            func=spawn_rubiks_cube,
            size=(CUBE, CUBE, CUBE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.09),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.03, 0.03), roughness=0.5),
        ),
    )
    marker_orange = _marker("MarkerOrange", (ROBOT_BASE_X + 0.565, -0.17), -120, (1.0, 0.38, 0.02))
    marker_red = _marker("MarkerRed", (ROBOT_BASE_X + 0.485, -0.165), -140, (0.88, 0.06, 0.08))
