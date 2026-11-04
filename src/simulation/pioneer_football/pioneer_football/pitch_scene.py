"""Procedural training stadium with shared rendering and physical goal geometry."""

import math

import mujoco
import numpy as np


# Build renderable geometry independently of the robot's contact model.
def box(spec, name, position, size, color, **kwargs):
    return spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=position,
        size=size,
        rgba=color,
        contype=0,
        conaffinity=0,
        **kwargs,
    )


# Use capsules for thin painted lines, net cords, and tubular frame members.
def tube(spec, name, start, end, radius, color, collision=False):
    return spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[*start, *end],
        size=[radius, 0, 0],
        rgba=color,
        contype=2 if collision else 0,
        conaffinity=1 if collision else 0,
        friction=[0.5, 0.002, 0.0001],
        solref=[0.008, 0.5],
    )


# Generate deterministic grass and mowing bands without downloaded texture dependencies.
def add_turf(spec, cfg):
    rng = np.random.default_rng(612)
    h, w = 768, 1152
    yy, xx = np.mgrid[:h, :w]
    bands = ((xx * 12 // w) % 2) * 13
    grain = rng.normal(0, 5, (h, w))
    blades = 3 * np.sin(yy * 2.7 + rng.normal(0, 1, (h, w)))
    rgb = np.stack(
        [
            35 + bands * 0.45 + grain,
            91 + bands + grain + blades,
            38 + bands * 0.4 + grain * 0.7,
        ],
        axis=-1,
    )
    texture = spec.add_texture(
        name="stadium_grass",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=w,
        height=h,
        nchannel=3,
        data=np.clip(rgb, 0, 255).astype(np.uint8).ravel(),
    )
    textures = [""] * 10
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture.name
    spec.add_material(
        name="stadium_turf",
        textures=textures,
        rgba=[1, 1, 1, 1],
        specular=0.03,
        shininess=0.05,
        texrepeat=[1, 1],
    )
    # A UV-mapped thin mesh retains its texture in both MuJoCo and the Viser converter.
    x, y = cfg.length / 2, cfg.width / 2
    vertices = [
        [a * x, b * y, z]
        for z in (-0.02, 0.001)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    faces = [
        0,
        2,
        1,
        0,
        3,
        2,
        4,
        5,
        6,
        4,
        6,
        7,
        0,
        1,
        5,
        0,
        5,
        4,
        1,
        2,
        6,
        1,
        6,
        5,
        2,
        3,
        7,
        2,
        7,
        6,
        3,
        0,
        4,
        3,
        4,
        7,
    ]
    spec.add_mesh(
        name="stadium_turf_mesh",
        uservert=np.asarray(vertices).ravel(),
        userface=faces,
        usertexcoord=[0, 0, 1, 0, 1, 1, 0, 1] * 2,
        userfacetexcoord=faces,
    )
    spec.worldbody.add_geom(
        name="stadium_turf",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="stadium_turf_mesh",
        material="stadium_turf",
        contype=0,
        conaffinity=0,
    )


# Paint small-sided soccer markings to match the existing 12-by-8-metre training world.
def add_markings(spec, cfg):
    x, y = cfg.length / 2, cfg.width / 2
    segments = [
        (-x, -y, x, -y),
        (-x, y, x, y),
        (-x, -y, -x, y),
        (x, -y, x, y),
        (0, -y, 0, y),
    ]
    for sign in (-1, 1):
        for depth, half_width in ((1.8, 2.25), (0.65, 1.45)):
            half_width = min(half_width, y * 0.8)
            end = sign * (x - depth)
            segments.extend(
                [
                    (sign * x, -half_width, end, -half_width),
                    (end, -half_width, end, half_width),
                    (end, half_width, sign * x, half_width),
                ]
            )
        # Corner arcs and penalty spots provide recognizable field detail.
        for sy in (-1, 1):
            for k in range(12):
                a, b = k * math.pi / 24, (k + 1) * math.pi / 24
                segments.append(
                    (
                        sign * (x - 0.22 * math.cos(a)),
                        sy * (y - 0.22 * math.sin(a)),
                        sign * (x - 0.22 * math.cos(b)),
                        sy * (y - 0.22 * math.sin(b)),
                    )
                )
        spec.worldbody.add_geom(
            name=f"stadium_penalty_spot_{sign}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[sign * (x - 1.35), 0, 0.004],
            size=[0.04, 0.001, 0],
            rgba=[0.96, 0.97, 0.93, 1],
            contype=0,
            conaffinity=0,
        )
    for k in range(96):
        a, b = 2 * math.pi * k / 96, 2 * math.pi * (k + 1) / 96
        segments.append((math.cos(a), math.sin(a), math.cos(b), math.sin(b)))
    for i, (a, b, c, d) in enumerate(segments):
        tube(
            spec,
            f"pitch_line_{i}",
            [a, b, 0.004],
            [c, d, 0.004],
            0.018,
            [0.96, 0.97, 0.93, 1],
        )
    spec.worldbody.add_geom(
        name="stadium_center_spot",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0.004],
        size=[0.045, 0.001, 0],
        rgba=[0.96, 0.97, 0.93, 1],
        contype=0,
        conaffinity=0,
    )


# Place posts outside the scored opening and approximate net collisions with rigid panels.
def add_goals(spec, cfg):
    white, net = [0.94, 0.96, 0.98, 1], [0.76, 0.81, 0.86, 1]
    r, width, height = cfg.post_radius, cfg.goal_width / 2, cfg.goal_height
    for sign in (-1, 1):
        front, back = sign * cfg.length / 2, sign * (cfg.length / 2 + cfg.goal_depth)

        def cord(name, a, b, radius=0.004, color=net, collision=False, sign=sign):
            tube(spec, f"goal_{sign}_{name}", a, b, radius, color, collision)

        for sy in (-1, 1):
            side = sy * (width + r)
            cord(
                f"post_{sy}",
                [front, side, r],
                [front, side, height + r],
                r,
                white,
                True,
            )
            cord(f"base_{sy}", [front, side, r], [back, side, r], r * 0.65, white)
            cord(
                f"support_{sy}",
                [back, side, r],
                [back, side, height * 0.82],
                r * 0.65,
                white,
            )
            cord(
                f"roof_{sy}",
                [front, side, height + r],
                [back, side, height * 0.82],
                r * 0.65,
                white,
            )
        cord(
            "crossbar",
            [front, -width - r, height + r],
            [front, width + r, height + r],
            r,
            white,
            True,
        )
        # Net strands are visual; thin panels stop the ball without dozens of tiny contacts.
        for i, y in enumerate(np.linspace(-width - r, width + r, 23)):
            cord(f"back_vertical_{i}", [back, y, 0.025], [back, y, height * 0.82])
            cord(f"roof_long_{i}", [front, y, height + r], [back, y, height * 0.82])
        for i, z in enumerate(np.linspace(0.025, height * 0.82, 10)):
            cord(f"back_horizontal_{i}", [back, -width - r, z], [back, width + r, z])
            for sy in (-1, 1):
                cord(
                    f"side_horizontal_{sy}_{i}",
                    [front, sy * (width + r), z],
                    [back, sy * (width + r), z],
                )
        for i, t in enumerate(np.linspace(0, 1, 11)):
            at = front + t * (back - front)
            top = (1 - t) * (height + r) + t * height * 0.82
            cord(f"roof_cross_{i}", [at, -width - r, top], [at, width + r, top])
            for sy in (-1, 1):
                cord(
                    f"side_vertical_{sy}_{i}",
                    [at, sy * (width + r), 0.025],
                    [at, sy * (width + r), top],
                )
        # Collision panels sit on the exterior of the visible net and leave the mouth open.
        for name, pos, size in [
            (
                "back",
                [back + sign * 0.02, 0, height * 0.41],
                [0.015, width + r, height * 0.41],
            ),
            (
                "left",
                [(front + back) / 2, -width - r - 0.02, height * 0.41],
                [cfg.goal_depth / 2, 0.015, height * 0.41],
            ),
            (
                "right",
                [(front + back) / 2, width + r + 0.02, height * 0.41],
                [cfg.goal_depth / 2, 0.015, height * 0.41],
            ),
        ]:
            geom = box(
                spec, f"goal_{sign}_net_collision_{name}", pos, size, [0, 0, 0, 0]
            )
            geom.contype, geom.conaffinity = 2, 1
            geom.solref = [0.015, 1]
        # Close the sloping roof with one soft rigid panel rather than colliding net strands.
        drop = height + r - height * 0.82
        angle = sign * math.atan2(drop, cfg.goal_depth)
        roof = box(
            spec,
            f"goal_{sign}_net_collision_roof",
            [(front + back) / 2, 0, (height + r + height * 0.82) / 2 + 0.02],
            [math.hypot(cfg.goal_depth, drop) / 2, width + r, 0.015],
            [0, 0, 0, 0],
            quat=[math.cos(angle / 2), 0, math.sin(angle / 2), 0],
        )
        roof.contype, roof.conaffinity = 2, 1
        roof.solref = [0.015, 1]


# Surround the playing area with a runoff apron, seating tiers, boards, and floodlight towers.
def add_surroundings(spec, cfg):
    x, y = cfg.length / 2, cfg.width / 2
    box(
        spec,
        "stadium_apron",
        [0, 0, -0.035],
        [x + 4, y + 4, 0.02],
        [0.24, 0.27, 0.26, 1],
    )
    colors = ([0.07, 0.24, 0.32, 1], [0.09, 0.38, 0.43, 1], [0.72, 0.78, 0.78, 1])
    for sign in (-1, 1):
        for row in range(5):
            box(
                spec,
                f"stadium_terrace_{sign}_{row}",
                [0, sign * (y + 1.65 + row * 0.48), row * 0.28],
                [x + 1.4, 0.25, 0.14],
                [0.42, 0.45, 0.44, 1],
            )
            for block in range(8):
                box(
                    spec,
                    f"stadium_seats_{sign}_{row}_{block}",
                    [
                        (block - 3.5) * (x + 1.4) / 4,
                        sign * (y + 1.65 + row * 0.48),
                        row * 0.28 + 0.24,
                    ],
                    [(x + 1.4) / 8 - 0.06, 0.16, 0.12],
                    colors[(block + row // 3) % 3],
                )
        for block in range(12):
            box(
                spec,
                f"stadium_board_{sign}_{block}",
                [(block - 5.5) * (x + 1) / 6, sign * (y + 0.85), 0.23],
                [(x + 1) / 12 - 0.015, 0.045, 0.23],
                colors[block % 3],
            )
        for sy in (-1, 1):
            px, py = sign * (x + 2.7), sy * (y + 2.2)
            tube(
                spec,
                f"stadium_light_pole_{sign}_{sy}",
                [px, py, 0],
                [px, py, 5.6],
                0.055,
                [0.48, 0.52, 0.55, 1],
            )
            box(
                spec,
                f"stadium_light_head_{sign}_{sy}",
                [px, py, 5.6],
                [0.42, 0.16, 0.22],
                [0.88, 0.9, 0.84, 1],
            )
            tube(
                spec,
                f"stadium_flagpole_{sign}_{sy}",
                [sign * x, sy * y, 0],
                [sign * x, sy * y, 0.8],
                0.012,
                [0.95, 0.95, 0.9, 1],
            )
            box(
                spec,
                f"stadium_flag_{sign}_{sy}",
                [sign * x + 0.09, sy * y, 0.71],
                [0.09, 0.008, 0.07],
                [1, 0.74, 0.15, 1],
            )


# Keep stadium construction shared by interactive playback and offscreen MuJoCo rendering.
def add_stadium(spec, cfg):
    add_turf(spec, cfg)
    add_markings(spec, cfg)
    add_goals(spec, cfg)
    add_surroundings(spec, cfg)
    spec.add_texture(
        name="stadium_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.32, 0.53, 0.72],
        rgb2=[0.82, 0.9, 0.94],
        width=128,
        height=768,
    )
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.65, 0.65, 0.65]
    spec.visual.quality.shadowsize = 2048
    spec.visual.map.zfar = 100
