"""The punch target: a ball the gloves can hit, and the gloves' collision bits.

Gloves (bit 2) and ball (bit 4) only touch each other, so adding them leaves
the rest of the physics (feet on the floor) unchanged. The ball's gravity is
cancelled, so it floats where it is put. Used by teleop (a free ball held to
the punch's line each step).
"""

import mujoco
from mjlab.utils.spec_config import CollisionCfg

GLOVE_RADIUS = 0.06  # humanoid_boxing.fighters.GLOVE_RADIUS (not imported: importing humanoid_boxing registers the arena)
TARGET_MASS = 2.0  # kg, about a head
TARGET_RADIUS = 0.08
TARGET_RGBA = (0.95, 0.75, 0.10, 1.0)

# the arena's glove spheres (humanoid_boxing.fighters.add_hitboxes), touching only the ball
GLOVE_HIT = CollisionCfg(
  geom_names_expr=(r"^glove_[lr]$",),
  contype=2,
  conaffinity=4,
  condim=3,
  priority=0,
  friction=(0.6,),
  disable_other_geoms=False,
)


def _add_ball(body: mujoco.MjsBody, mass: float = TARGET_MASS) -> None:
  body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(TARGET_RADIUS, 0, 0),
    mass=mass,
    rgba=TARGET_RGBA,
    contype=4,
    conaffinity=2,
    condim=3,
  )


def free_ball_spec() -> mujoco.MjSpec:
  """Ball on a free joint, to be placed (and held to a line) by the caller."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="target", gravcomp=1.0)
  body.add_freejoint()
  _add_ball(body)
  return spec
