"""Vision domain randomization for push-block distillation (sim2real template).

Code-only DR plus a tiny local texture set (3 PNGs). Scale ranges / add more
textures later without changing the event wiring.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.views import XformPrimView


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "TEXTURE_DIR",
    "list_push_textures",
    "randomize_robot_camera_pose",
    "randomize_back_wall",
    "randomize_lights",
    "randomize_visual_materials",
]

TEXTURE_DIR = Path(__file__).resolve().parents[1] / "assets" / "textures"

def list_push_textures() -> list[str]:
    """Absolute paths to the small template texture pack."""
    if not TEXTURE_DIR.is_dir():
        return []
    return sorted(str(p) for p in TEXTURE_DIR.glob("*.png"))


def randomize_robot_camera_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    camera_cfg: SceneEntityCfg,
    mount: str,
    pos_range: tuple[float, float] = (-0.01, 0.01),
    rot_deg_range: tuple[float, float] = (-2.0, 2.0),
):
    """Jitter a robot-mounted camera around its mount in ``arm_params.CAMERAS[mount]``, per reset env.

    Sets the camera's pose local to its parent link, so it keeps following the link afterwards."""
    from pioneer_humanoid.arm_params import CAMERAS

    camera = env.scene.sensors[camera_cfg.name]
    views = env.__dict__.setdefault("_robot_camera_views", {})
    if camera_cfg.name not in views:
        views[camera_cfg.name] = XformPrimView(camera.cfg.prim_path, device=env.device)

    n = len(env_ids)
    cam = CAMERAS[mount]
    pos = torch.tensor(cam.pos, device=env.device, dtype=torch.float32).expand(n, 3)
    pos = pos + math_utils.sample_uniform(pos_range[0], pos_range[1], (n, 3), device=env.device)
    rot = torch.tensor(cam.rot, device=env.device, dtype=torch.float32).expand(n, 4)
    rot_noise = torch.deg2rad(math_utils.sample_uniform(rot_deg_range[0], rot_deg_range[1], (n, 3), device=env.device))
    rot = math_utils.quat_mul(rot, math_utils.quat_from_euler_xyz(rot_noise[:, 0], rot_noise[:, 1], rot_noise[:, 2]))
    views[camera_cfg.name].set_local_poses(translations=pos, orientations=rot, indices=env_ids)


def randomize_back_wall(env: ManagerBasedEnv, env_ids: torch.Tensor, texture_paths: Sequence[str], shader_path: str):
    """Show one of ``texture_paths`` on each reset env's back wall (``shader_path`` is the shader's path under the env)."""
    del env
    try:
        import omni.usd
        from pxr import Sdf
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    for env_id in env_ids.tolist():
        shader = stage.GetPrimAtPath(f"/World/envs/env_{int(env_id)}{shader_path}")
        if shader.IsValid():
            shader.GetAttribute("inputs:file").Set(Sdf.AssetPath(random.choice(list(texture_paths))))


def _warmth_scale(w: float) -> tuple[float, float, float]:
    """RGB multiplier for a colour axis from cool (w < 0) to neutral (0) to warm (w = 1, about 2800 K against white)."""
    if w >= 0.0:
        return (1.0, 1.0 - 0.25 * w, 1.0 - 0.55 * w)
    return (1.0 + 0.2 * w, 1.0 + 0.1 * w, 1.0)


def randomize_lights(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    light_groups: Sequence[tuple[Sequence[str], tuple[float, float], tuple[float, float] | None]] = (
        (("/TubeLight",), (0.6, 1.0), (-0.25, 0.35)),
        (("/World/light",), (0.5, 1.5), (-0.5, 1.0)),
        (("/RoomLight",), (0.5, 1.5), (0.0, 0.5)),
    ),
):
    """Randomize the intensity and colour of the workcell's lights.

    A path under ``/World`` is a stage-global light (the dome: one draw, shared by all envs); any other path is under each
    env (``/World/envs/env_i`` + path) and is drawn separately for every env that resets.

    A group is ``(prim paths, intensity scale range, warmth range or None)``. The intensity is scaled from the nominal. The
    colour moves along one cool-to-warm axis (``_warmth_scale``: -1 slightly cool, 0 the nominal colour, 1 warm), so a draw is
    always a believable light colour and never a green or magenta cast; None leaves the colour alone. The lights of a group
    share one draw (per env); groups draw independently."""
    try:
        import omni.usd
        from pxr import Gf
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    nominal = env.__dict__.setdefault("_light_nominal", {})

    for paths, intensity_range, warmth_range in light_groups:
        global_paths = [p for p in paths if p.startswith("/World/") and not p.startswith("/World/envs/")]
        local_paths = [p for p in paths if p not in global_paths]
        draws = [(global_paths, None)] + [(local_paths, int(e)) for e in env_ids.tolist()]
        for group, env_id in draws:
            if not group:
                continue
            scale = float(np.random.uniform(*intensity_range))
            mult = _warmth_scale(float(np.random.uniform(*warmth_range))) if warmth_range else None
            for path in group:
                path = path if env_id is None else f"/World/envs/env_{env_id}{path}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    continue
                inten_attr, color_attr = prim.GetAttribute("inputs:intensity"), prim.GetAttribute("inputs:color")
                nom_inten, nom_color = nominal.setdefault(path, (inten_attr.Get(), color_attr.Get()))
                inten_attr.Set(nom_inten * scale)
                if mult is not None:
                    color_attr.Set(Gf.Vec3f(*(min(c * m, 1.25) for c, m in zip(nom_color, mult))))


def _iter_shader_prims(stage, root_path: str):
    """Yield UsdShade-like shader prims under root_path (best-effort)."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return
    prefix = root_path.rstrip("/")
    for prim in root.GetAllChildren():
        stack = [prim]
        while stack:
            p = stack.pop()
            stack.extend(list(p.GetAllChildren()))
            path = str(p.GetPath())
            if not path.startswith(prefix):
                continue
            name = p.GetName().lower()
            type_name = p.GetTypeName()
            if "Shader" in type_name or name == "shader" or "omnipbr" in path.lower():
                yield p

def randomize_visual_materials(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    targets: dict[str, dict],
    texture_paths: Sequence[str] | None = None,
    apply_textures: bool = False,
):
    """Randomize colour, roughness and metallic of the prims named in ``targets``, per reset env.

    ``targets`` maps a prim suffix under each env (for example ``"/Box"``) to
    ``dict(tint=((r, g, b) low, (r, g, b) high), roughness=(low, high), metallic=(low, high))``, plus an optional
    ``clearcoat_roughness=(low, high)`` for clear-coated materials (the floor boards). Textures are only
    swapped in when ``apply_textures`` is set.

    Best-effort: skips prims/attributes that do not exist so USDs with different material graphs still run.
    """
    try:
        import omni.usd
        from pxr import Gf, Sdf
    except ImportError:
        return

    if apply_textures and texture_paths is None:
        texture_paths = list_push_textures()
    stage = omni.usd.get_context().get_stage()
    shader_cache = env.__dict__.setdefault("_vision_shader_cache", {})  # USD tree walk is slow

    for env_id in env_ids.tolist():
        env_root = f"/World/envs/env_{int(env_id)}"
        for suffix, rng in targets.items():
            root = env_root + suffix
            lo, hi = rng["tint"]
            tint = tuple(float(np.random.uniform(a, b)) for a, b in zip(lo, hi))
            roughness = float(np.random.uniform(*rng["roughness"]))
            metallic = float(np.random.uniform(*rng["metallic"]))
            coat = float(np.random.uniform(*rng["clearcoat_roughness"])) if "clearcoat_roughness" in rng else None
            tex = random.choice(texture_paths) if (apply_textures and texture_paths) else None

            key = (int(env_id), suffix)
            if key not in shader_cache:
                shader_cache[key] = list(_iter_shader_prims(stage, root))
            for shader in shader_cache[key]:
                for attr_name, value in (
                    ("inputs:diffuse_tint", Gf.Vec3f(*tint)),
                    ("inputs:reflection_roughness_constant", roughness),
                    ("inputs:metallic_constant", metallic),
                    ("inputs:roughness", roughness),
                    ("inputs:metallic", metallic),
                    ("inputs:diffuse_color_constant", Gf.Vec3f(*tint)),
                    ("inputs:diffuseColor", Gf.Vec3f(*tint)),
                    ("inputs:clearcoat_reflection_roughness", coat),
                ):
                    if value is None:
                        continue
                    attr = shader.GetAttribute(attr_name)
                    if attr.IsValid():
                        try:
                            attr.Set(value)
                        except Exception:
                            pass
                if tex is not None:
                    for attr_name in ("inputs:diffuse_texture", "inputs:diffuse_color_texture"):
                        attr = shader.GetAttribute(attr_name)
                        if attr.IsValid():
                            try:
                                attr.Set(Sdf.AssetPath(tex))
                            except Exception:
                                pass
