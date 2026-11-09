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
    """Show one of ``texture_paths`` on the back wall (stage-global, shared by all envs)."""
    del env, env_ids
    try:
        import omni.usd
        from pxr import Sdf
    except ImportError:
        return

    shader = omni.usd.get_context().get_stage().GetPrimAtPath(shader_path)
    if shader.IsValid():
        shader.GetAttribute("inputs:file").Set(Sdf.AssetPath(random.choice(list(texture_paths))))


def randomize_lights(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    light_groups: Sequence[tuple[Sequence[str], tuple[float, float], tuple[float, float] | None]] = (
        (("/World/tube_light",), (0.6, 1.0), None),
        (("/World/light",), (0.5, 1.5), (0.85, 1.15)),
        (("/World/room_light",), (0.5, 1.5), (0.85, 1.15)),
    ),
):
    """Randomize the intensity and colour of the workcell's lights (stage-global, shared by all envs).

    A group is ``(prim paths, intensity scale range, colour scale range or None)``. The intensity is scaled from the
    nominal; the colour is scaled per channel from the nominal, or left alone when the range is None. The lights of a
    group share one draw; groups draw independently."""
    del env_ids
    try:
        import omni.usd
        from pxr import Gf
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    nominal = env.__dict__.setdefault("_light_nominal", {})

    for paths, intensity_range, color_range in light_groups:
        scale = float(np.random.uniform(*intensity_range))
        tint = [float(np.random.uniform(*color_range)) for _ in range(3)] if color_range else None
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            inten_attr, color_attr = prim.GetAttribute("inputs:intensity"), prim.GetAttribute("inputs:color")
            nom_inten, nom_color = nominal.setdefault(path, (inten_attr.Get(), color_attr.Get()))
            inten_attr.Set(nom_inten * scale)
            if tint is not None:
                color_attr.Set(Gf.Vec3f(*(min(c * t, 1.25) for c, t in zip(nom_color, tint))))


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
    ``dict(tint=((r, g, b) low, (r, g, b) high), roughness=(low, high), metallic=(low, high))``. Textures are only
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
                ):
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
