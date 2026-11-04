"""Shared ball/pitch assets for physics checks and future state-based skills.

Contact parameters are engineering assumptions, not experimental calibration.
Rolling/torsional coefficients are lengths (m), not sliding coefficients.
"""

import math
from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class BallCfg:
    mass: float = 0.43
    radius: float = 0.11
    sliding_friction: float = 0.5
    foot_friction: float = 0.7
    torsional_friction: float = 0.002
    rolling_friction: float = 0.004
    contact_time: float = 0.008
    damping_ratio: float = 0.15

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def inertia(self):
        # Thin spherical shell approximation to an inflated ball.
        return (2 / 3) * self.mass * self.radius**2


@dataclass(frozen=True)
class PitchCfg:
    length: float = 12.0
    width: float = 8.0
    goal_width: float = 2.0
    goal_height: float = 1.0
    goal_depth: float = 0.9
    post_radius: float = 0.04

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in vars(self).values()):
            raise ValueError("Pitch dimensions must be finite and positive")
        if self.goal_width >= self.width:
            raise ValueError("Goal must be narrower than pitch")
        if self.post_radius >= min(self.goal_width / 4, self.goal_height / 4):
            raise ValueError("Goal posts must leave an open goal mouth")


DEFAULT_BALL_CFG = BallCfg()
DEFAULT_PITCH_CFG = PitchCfg()


def add_ball(spec, cfg=DEFAULT_BALL_CFG, position=None):
    body = spec.worldbody.add_body(
        name="ball",
        pos=position or [1.0, 0.0, cfg.radius + 0.002],
        mass=cfg.mass,
        inertia=[cfg.inertia] * 3,
        ipos=[0, 0, 0],
        explicitinertial=True,
    )
    body.add_freejoint(name="ball_root")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[cfg.radius, 0, 0],
        rgba=[0.95, 0.93, 0.85, 1],
        condim=6,
        priority=2,
        friction=[cfg.sliding_friction, cfg.torsional_friction, cfg.rolling_friction],
        solref=[cfg.contact_time, cfg.damping_ratio],
    )
    # Non-colliding bands make spin visible without changing sphere contacts.
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        for k in range(24):
            a, b = 2 * math.pi * k / 24, 2 * math.pi * (k + 1) / 24
            points = []
            for t in (a, b):
                xyz = [
                    0.0,
                    cfg.radius * 1.005 * math.cos(t),
                    cfg.radius * 1.005 * math.sin(t),
                ]
                if axis == [0, 1, 0]:
                    xyz = [xyz[1], xyz[0], xyz[2]]
                elif axis == [0, 0, 1]:
                    xyz = [xyz[1], xyz[2], xyz[0]]
                points.extend(xyz)
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=points,
                size=[0.0015, 0, 0],
                rgba=[0.08, 0.08, 0.08, 1],
                contype=0,
                conaffinity=0,
                mass=0,
            )
    return body


def ball_spec(cfg=DEFAULT_BALL_CFG):
    spec = mujoco.MjSpec()
    add_ball(spec, cfg)
    return spec


def add_pitch(
    spec, cfg=DEFAULT_PITCH_CFG, floor_name="floor", add_floor=True, stadium=True
):
    if add_floor:
        floor = spec.worldbody.add_geom(
            name=floor_name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[cfg.length / 2, cfg.width / 2, 0.1],
        )
    else:
        floor = spec.geom(floor_name)
    floor.rgba = [0.10, 0.34, 0.12, 1]
    floor.friction = [0.8, 0.005, 0.0001]
    floor.priority = 0
    # Share the modeled stadium with all soccer playback and physical ball harnesses.
    from .pitch_scene import add_markings, add_stadium

    (add_stadium if stadium else add_markings)(spec, cfg)


def add_contact_pairs(
    spec, cfg=DEFAULT_BALL_CFG, ball="ball_geom", floor="floor", feet=()
):
    spec.add_pair(
        name="ball_ground",
        geomname1=ball,
        geomname2=floor,
        condim=6,
        friction=[cfg.sliding_friction] * 2
        + [cfg.torsional_friction]
        + [cfg.rolling_friction] * 2,
        solref=[cfg.contact_time, cfg.damping_ratio],
    )
    for i, foot in enumerate(feet):
        spec.add_pair(
            name=f"ball_foot_{i}",
            geomname1=ball,
            geomname2=foot,
            condim=3,
            friction=[cfg.foot_friction] * 2 + [0, 0, 0],
            solref=[cfg.contact_time, cfg.damping_ratio],
        )


def physics_spec(cfg=DEFAULT_BALL_CFG, timestep=0.0005, robot=False, stadium=True):
    """CPU physics harness, sharing assets and pairs with the mjlab scene."""
    from .model import robot_spec

    spec = robot_spec(visuals=True) if robot else mujoco.MjSpec()
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    add_pitch(spec, stadium=stadium)
    add_ball(spec, cfg)
    feet = ("l_foot_collision_0", "r_foot_collision_0") if robot else ()
    add_contact_pairs(spec, cfg, feet=feet)
    return spec


def configure_scene(
    spec, ball_cfg=DEFAULT_BALL_CFG, pitch_cfg=DEFAULT_PITCH_CFG, stadium=True
):
    add_pitch(spec, pitch_cfg, floor_name="terrain", add_floor=False, stadium=stadium)
    add_contact_pairs(
        spec,
        ball_cfg,
        ball="ball/ball_geom",
        floor="terrain",
        feet=("robot/l_foot_collision_0", "robot/r_foot_collision_0"),
    )
