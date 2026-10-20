"""The room every whole-body scene builds on: a grey studio floor, soft key / fill / rim lights and a workbench."""
from __future__ import annotations

import mujoco

BENCH_TOP_Z = 0.85          # work surface height (m); the robot's elbows are at ~0.97 m
BENCH_X = (0.30, 0.90)      # near edge (robot side), far edge, in front of the robot's pelvis at x = 0
BENCH_HALF_Y = 0.65


def add_studio(spec: mujoco.MjSpec) -> None:
    spec.visual.headlight.ambient = [0.12, 0.12, 0.13]
    spec.visual.headlight.diffuse = [0.22, 0.22, 0.22]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.visual.quality.shadowsize = 4096
    spec.visual.map.znear = 0.0005
    spec.visual.rgba.haze = [0.42, 0.43, 0.45, 1]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.63, 0.66], rgb2=[0.30, 0.31, 0.33], width=256, height=256)
    tex = spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.47, 0.47, 0.48], rgb2=[0.43, 0.43, 0.44], width=512, height=512)
    mat = spec.add_material(name="floor", texrepeat=[10, 10], reflectance=0.03)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[8, 8, 0.05], material="floor")
    world.add_light(name="key", pos=[1.8, -2.2, 3.4], dir=[-0.45, 0.55, -0.7], diffuse=[0.42, 0.42, 0.40],
                    specular=[0.2, 0.2, 0.2], castshadow=True, cutoff=50, exponent=2)
    world.add_light(name="fill", pos=[1.5, 2.5, 2.6], dir=[-0.4, -0.6, -0.6], diffuse=[0.16, 0.16, 0.18],
                    specular=[0, 0, 0], castshadow=False, cutoff=70)
    world.add_light(name="top", pos=[0.6, 0, 3.0], dir=[0, 0, -1], diffuse=[0.22, 0.22, 0.22],
                    specular=[0.05, 0.05, 0.05], castshadow=False, cutoff=60, exponent=1)   # one shadow pass: CPU


def add_bench(spec: mujoco.MjSpec, x=BENCH_X, half_y=BENCH_HALF_Y, top_z=BENCH_TOP_Z, name="bench") -> None:
    """A workbench: 30 mm light wood top (collides), dark steel frame (visual)."""
    world = spec.worldbody
    x0, x1 = x
    cx, hx = (x0 + x1) / 2, (x1 - x0) / 2
    spec.add_material(name=f"{name}_wood", rgba=[0.74, 0.62, 0.47, 1], specular=0.15, shininess=0.3)
    spec.add_material(name=f"{name}_steel", rgba=[0.16, 0.17, 0.19, 1], specular=0.4, shininess=0.6)
    world.add_geom(name=f"{name}_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, half_y, 0.015],
                   pos=[cx, 0, top_z - 0.015], material=f"{name}_wood", friction=[0.8, 0.01, 0.001])
    vis = dict(contype=0, conaffinity=0, material=f"{name}_steel")
    for sx in (-1, 1):
        for sy in (-1, 1):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (top_z - 0.03) / 2],
                           pos=[cx + sx * (hx - 0.04), sy * (half_y - 0.04), (top_z - 0.03) / 2], **vis)
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, half_y - 0.04, 0.02],
                       pos=[cx + sx * (hx - 0.04), 0, 0.15], **vis)
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx - 0.02, half_y - 0.02, 0.03],
                   pos=[cx, 0, top_z - 0.06], **vis)
