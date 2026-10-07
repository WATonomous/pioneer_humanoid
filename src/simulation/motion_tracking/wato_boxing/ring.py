"""Boxing ring: canvas, corner posts and three ropes per side, as a fixed entity.

Default size is scaled to Wato (about 1.5 m tall): 4 m between the ropes
(a pro ring is 4.9-7.3 m for 1.8 m people), ropes at 0.35 / 0.65 / 0.95 m.
Posts and ropes collide with both fighters' hitboxes (bit 3), not with feet.
"""

import mujoco

from mjlab.entity import EntityCfg
from wato_boxing.fighters import ROPE_BIT, TEAM_BIT

CANVAS_RGBA = (0.82, 0.80, 0.74, 1.0)
ROPE_RGBA = (0.92, 0.92, 0.92, 1.0)
POST_RGBA = {"red": (0.75, 0.1, 0.1, 1.0), "blue": (0.1, 0.2, 0.75, 1.0), "neutral": (0.85, 0.85, 0.85, 1.0)}


def get_ring_spec(size: float = 4.0, rope_heights=(0.35, 0.65, 0.95), post_height: float = 1.1) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  ring = spec.worldbody.add_body(name="ring")
  half = size / 2
  hit = TEAM_BIT["red"] | TEAM_BIT["blue"]

  # canvas: visual only, a hair above the floor so it shows
  ring.add_geom(
    name="canvas",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half + 0.3, half + 0.3, 0.001),
    pos=(0, 0, 0.001),
    rgba=CANVAS_RGBA,
    contype=0,
    conaffinity=0,
  )
  corners = {"red": (-half, -half), "blue": (half, half), "neutral": (half, -half), "neutral2": (-half, half)}
  for name, (x, y) in corners.items():
    ring.add_geom(
      name=f"post_{name}",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(0.05, post_height / 2, 0),
      pos=(x, y, post_height / 2),
      rgba=POST_RGBA[name.rstrip("2")],
      contype=ROPE_BIT,
      conaffinity=hit,
    )
  sides = [((-half, -half), (half, -half)), ((half, -half), (half, half)), ((half, half), (-half, half)), ((-half, half), (-half, -half))]
  for i, ((x0, y0), (x1, y1)) in enumerate(sides):
    for j, h in enumerate(rope_heights):
      ring.add_geom(
        name=f"rope_{i}_{j}",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(0.015, 0, 0),
        fromto=(x0, y0, h, x1, y1, h),
        rgba=ROPE_RGBA,
        contype=ROPE_BIT,
        conaffinity=hit,
      )
  return spec


def get_ring_cfg(size: float = 4.0) -> EntityCfg:
  return EntityCfg(spec_fn=lambda: get_ring_spec(size))
