"""Bare scene: the empty lightbox workcell + arm on its stand. The default scene."""
from __future__ import annotations

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene


@scene("bare", camera=dict(lookat=[0.5, 0.0, TABLE_TOP_Z], distance=1.9, azimuth=20, elevation=-30))
def build(spec):
    add_workcell(spec)
