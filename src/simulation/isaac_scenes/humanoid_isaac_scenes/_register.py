"""Scene registry: ``@scene``-decorated ``InteractiveSceneCfg`` classes,
auto-discovered from ``humanoid_isaac_scenes/<name>/scene.py``.

Adding a scene is normally one folder: ``@scene("my_scene") class
MySceneCfg(InteractiveSceneCfg): ...``, with ``robot = MISSING`` -- a teleop
script plugs its own arm in via ``make_scene_cfg``.

For assets that can't be a declarative cfg field (e.g. a particle-cloth
object built with a live constructor call, not a spawn-able ``AssetBaseCfg``),
``@scene(...)`` also takes ``post_init(scene, sim)``, run once after
``InteractiveScene``/``sim.reset()`` exist. No-op if unset.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class _Entry:
    cfg_cls: type
    robot_pos: tuple = (0.0, 0.0, 0.0)
    robot_rot: tuple = (1.0, 0.0, 0.0, 0.0)  # wxyz
    camera: Optional[tuple] = None  # (eye_xyz, target_xyz) for the teleop initial view
    post_init: Optional[Callable] = None  # post_init(scene, sim) -- see module docstring
    robot_joint_pos: Optional[dict] = None


_REGISTRY: dict[str, _Entry] = {}
_DISCOVERED = False


def scene(name: str, *, robot_pos=(0.0, 0.0, 0.0), robot_rot=(1.0, 0.0, 0.0, 0.0), camera=None, post_init=None, robot_joint_pos=None):
    """Register an ``InteractiveSceneCfg`` subclass under ``name``.

    robot_pos: where to place the arm base. Scenes with a low table (arm reaching
               down from origin) use ``(0, 0, 0)``; scenes where the arm stands
               on its floor stand use roughly ``(0, 0, 1.2)``.
    robot_rot: base orientation (wxyz); identity unless the scene needs the arm
               turned to face its workspace (e.g. a scene built for a different
               front axis than the arm's default +X).
    camera:    optional ``(eye, target)`` for the teleop initial view; ``None``
               lets the teleop script fall back to its default framing.
    post_init: optional ``post_init(scene, sim)`` for assets that can't be
               built from a declarative cfg field -- see module docstring.
    robot_joint_pos: optional initial joint pose override for task clearance;
               applied to the caller's robot config without changing its source.
    """
    def deco(cfg_cls):
        _REGISTRY[name] = _Entry(cfg_cls, tuple(robot_pos), tuple(robot_rot), camera, post_init, robot_joint_pos)
        return cfg_cls

    return deco


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    import warnings

    import humanoid_isaac_scenes

    for m in pkgutil.iter_modules(humanoid_isaac_scenes.__path__):
        if m.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"humanoid_isaac_scenes.{m.name}.scene")
        except Exception as e:  # noqa: BLE001 -- one broken scene must not hide the rest
            warnings.warn(f"humanoid_isaac_scenes: skipping {m.name!r} -- {type(e).__name__}: {e}", stacklevel=2)
    _DISCOVERED = True


def list_scenes() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def scene_camera(name: str):
    _discover()
    return _REGISTRY[name].camera


def scene_post_init(name: str):
    _discover()
    return _REGISTRY[name].post_init


def make_scene_cfg(name, robot_cfg, *, num_envs=1, env_spacing=2.0, prim_path="{ENV_REGEX_NS}/Robot"):
    """Instantiate scene ``name`` with ``robot_cfg`` plugged into its ``MISSING`` robot."""
    _discover()
    entry = _REGISTRY[name]
    cfg = entry.cfg_cls(num_envs=num_envs, env_spacing=env_spacing)
    cfg.robot = robot_cfg.replace(
        prim_path=prim_path,
        init_state=robot_cfg.init_state.replace(pos=entry.robot_pos, rot=entry.robot_rot),
    )
    if entry.robot_joint_pos is not None:
        cfg.robot.init_state.joint_pos = dict(entry.robot_joint_pos)
    if hasattr(cfg, "ee_frame"):
        cfg.ee_frame = None
    return cfg
