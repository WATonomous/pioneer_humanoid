"""Scene registry for the whole-body humanoid: ``@scene``-decorated builders, discovered from
``pioneer_whole_body/scenes/<name>.py``. Same hooks as humanoid_mujoco_scenes (step, reset, progress, success)."""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Optional

import mujoco
import numpy as np

TIMESTEP = 0.002
CONTACT_SOLREF = [0.004, 1.0]
CONTACT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
_DEFAULT_SOLREF = [0.02, 1.0]
_DEFAULT_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]


@dataclass
class _Entry:
    build: Callable[[mujoco.MjSpec], None]
    robot_pos: tuple
    step: Optional[Callable] = None
    reset: Optional[Callable] = None
    progress: Optional[Callable] = None
    success: Optional[Callable] = None
    plan: Optional[Callable] = None        # the scripted demo: plan(robot_controller, notes)
    title: Optional[str] = None            # the video's title
    camera: Optional[tuple] = None         # the video's fixed camera: (lookat xyz, distance, azimuth, elevation)


_REGISTRY: dict[str, _Entry] = {}


def scene(name, *, robot_pos=(0.0, 0.0, 0.0), step=None, reset=None, progress=None, success=None, plan=None,
          title=None, camera=None):
    def deco(build):
        _REGISTRY[name] = _Entry(build, tuple(robot_pos), step, reset, progress, success, plan, title, camera)
        return build
    return deco


def _discover():
    from . import scenes
    for m in pkgutil.iter_modules(scenes.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"pioneer_whole_body.scenes.{m.name}")


def entry(name: str) -> _Entry:
    _discover()
    return _REGISTRY[name]


def list_scenes() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def make_model(name: str) -> mujoco.MjModel:
    """Compile scene ``name`` with the humanoid's pelvis at ``robot_pos``, facing +X."""
    return make_spec(name).compile()


def make_spec(name: str) -> mujoco.MjSpec:
    """The scene's MjSpec (``make_model`` compiles it; ``view --export`` writes it out as MJCF)."""
    from .robot import robot_spec

    e = entry(name)
    spec = mujoco.MjSpec()
    spec.modelname = name
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    e.build(spec)
    frame = spec.worldbody.add_frame(pos=list(e.robot_pos))
    frame.attach_body(robot_spec().body("pelvis"), "", "")
    for geom in spec.geoms:
        if np.allclose(geom.solref, _DEFAULT_SOLREF) and np.allclose(geom.solimp, _DEFAULT_SOLIMP):
            geom.solref = CONTACT_SOLREF
            geom.solimp = CONTACT_SOLIMP
    return spec
