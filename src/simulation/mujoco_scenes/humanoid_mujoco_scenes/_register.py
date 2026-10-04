"""Scene registry: ``@scene``-decorated builders, auto-discovered from
``humanoid_mujoco_scenes/<name>/scene.py``.

A scene is one function that adds world geometry to an ``mujoco.MjSpec``; ``make_model``
attaches the Pioneer arm (pioneer_humanoid.mujoco_bimanual_arm) at ``robot_pos`` and compiles.
CPU only: no Isaac imports anywhere in this package.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Optional

import mujoco
import numpy as np

# base_link lift that puts the stand's feet on the floor (z=0); same value as Isaac's push scene.
ROBOT_STAND_LIFT_Z = 1.1997

# 1 ms: at 2 ms a carried object still shifts ~4 mm in the jaws (2 mm at 1 ms). CPU cost is ~20 ms per
# simulated second, far below the camera rendering when recording.
TIMESTEP = 0.001
# Contact time constant 4 ms (>= 2 x TIMESTEP), solimp near 1: ~0.5 mm penetration under a grasp.
CONTACT_SOLREF = [0.004, 1.0]
CONTACT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
_DEFAULT_SOLREF = [0.02, 1.0]
_DEFAULT_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]


@dataclass
class _Entry:
    build: Callable[[mujoco.MjSpec], None]
    robot_pos: tuple = (0.0, 0.0, ROBOT_STAND_LIFT_Z)
    camera: Optional[dict] = None  # mujoco free-camera fields: lookat, distance, azimuth, elevation
    step: Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]] = None


_REGISTRY: dict[str, _Entry] = {}
_DISCOVERED = False


def scene(name: str, *, robot_pos=(0.0, 0.0, ROBOT_STAND_LIFT_Z), camera=None, step=None):
    """Register ``build(spec)`` under ``name``. ``robot_pos``: arm base placement (default: on its stand).

    ``step(model, data)``: optional, called by the teleop once per control step before stepping the
    physics -- for scene mechanics a static model can't express (e.g. a one-way ratchet). It must keep
    its state in ``data`` so ``mj_resetData`` resets it.
    """
    def deco(build):
        _REGISTRY[name] = _Entry(build, tuple(robot_pos), camera, step)
        return build

    return deco


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    import warnings

    import humanoid_mujoco_scenes

    for m in pkgutil.iter_modules(humanoid_mujoco_scenes.__path__):
        if not m.ispkg or m.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"humanoid_mujoco_scenes.{m.name}.scene")
        except Exception as e:  # noqa: BLE001 -- one broken scene must not hide the rest
            warnings.warn(f"humanoid_mujoco_scenes: skipping {m.name!r} -- {type(e).__name__}: {e}", stacklevel=2)
    _DISCOVERED = True


def list_scenes() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def scene_camera(name: str) -> Optional[dict]:
    _discover()
    return _REGISTRY[name].camera


def scene_step(name: str) -> Optional[Callable[[mujoco.MjModel, mujoco.MjData], None]]:
    _discover()
    return _REGISTRY[name].step


def add_floor(spec: mujoco.MjSpec) -> None:
    """Checker floor at z=0, a light, and a skybox."""
    tex = spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], width=512, height=512,
    )
    spec.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.6, 0.7, 0.8], rgb2=[0.1, 0.1, 0.15], width=256, height=256,
    )
    mat = spec.add_material(name="grid", texrepeat=[8, 8], reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05], material="grid")
    spec.worldbody.add_light(pos=[0.5, 0, 3.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=True)


def make_model(name: str, cameras: dict[str, tuple[int, int]] | None = None) -> mujoco.MjModel:
    """Compile scene ``name`` with the arm attached; joint/actuator names are the URDF joint names.

    ``cameras``: robot cameras to mount, {name: (height, width)} (pioneer_humanoid.arm_params.CAMERAS).
    """
    from pioneer_humanoid.mujoco_bimanual_arm import arm_spec

    _discover()
    entry = _REGISTRY[name]
    spec = mujoco.MjSpec()
    spec.modelname = name
    spec.option.timestep = TIMESTEP
    # Implicit damping keeps the stiff arm PD (kp up to 2270) stable at this step.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Grasping: elliptic friction cones with impratio 10 hold a squeezed object instead of
    # letting it creep (MuJoCo's defaults, pyramidal + 1, are soft in the tangential direction).
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    entry.build(spec)
    frame = spec.worldbody.add_frame(pos=list(entry.robot_pos))
    frame.attach_body(arm_spec(cameras).body("base_link"), "", "")
    for geom in spec.geoms:
        # Stiffer than MuJoCo's default contact (0.02 s, 0.9-0.95), which lets a 50 g peg sink mm
        # into the fingers and eats a 1 mm insertion clearance. Geoms a scene tuned are left alone.
        if np.allclose(geom.solref, _DEFAULT_SOLREF) and np.allclose(geom.solimp, _DEFAULT_SOLIMP):
            geom.solref = CONTACT_SOLREF
            geom.solimp = CONTACT_SOLIMP
    return spec.compile()
