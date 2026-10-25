"""Task-local PhysX self-collision filtering for the Wato humanoid.

URDF defines link collision geometry, but it has no portable switch for Isaac's
articulation self-collision behavior.  The stock articulation-wide switch makes
every non-filtered rigid-body pair collide; that is too broad for this asset.
This spawner enables the articulation switch in the task configuration while
filtering every internal pair except the two observed foot/opposite-calf pairs.
"""

from __future__ import annotations

import os
from itertools import combinations

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from pxr import Usd, UsdPhysics


WATO_RIGID_BODY_NAMES = frozenset(
    {
        "base",
        "Hip_F_L",
        "Hip_A_L",
        "Thigh_L",
        "Calf_L",
        "Ankle_L",
        "Foot_L",
        "Hip_F_R",
        "Hip_A_R",
        "Thigh_R",
        "Calf_R",
        "Ankle_R",
        "Foot_R",
    }
)

# Keep this deliberately narrow.  These are the pairs implicated by the visual
# foot-through-opposite-leg failure.  Foot/foot and every other internal pair
# remain filtered until a separate test demonstrates that enabling them helps.
WATO_ALLOWED_SELF_COLLISION_PAIRS = frozenset(
    {
        frozenset(("Foot_L", "Calf_R")),
        frozenset(("Foot_R", "Calf_L")),
    }
)

_SELECTIVE_USD_CACHE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "outputs",
        "isaac",
        "selective_self_collision_usd_cache",
    )
)


def _apply_selective_self_collision_filters(root_prim: Usd.Prim) -> None:
    """Filter all internal rigid-body pairs except the allowlist above."""
    bodies = {}
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name = prim.GetName()
            if name in bodies:
                raise RuntimeError(f"Duplicate Wato rigid-body name under {root_prim.GetPath()}: {name}")
            bodies[name] = prim

    actual_names = frozenset(bodies)
    if actual_names != WATO_RIGID_BODY_NAMES:
        missing = sorted(WATO_RIGID_BODY_NAMES - actual_names)
        extra = sorted(actual_names - WATO_RIGID_BODY_NAMES)
        raise RuntimeError(f"Unexpected Wato rigid-body hierarchy; missing={missing}, extra={extra}")

    # A floating URDF places ArticulationRootAPI on the base rigid body. A
    # filtered-pair target naming that prim can exclude the entire articulation,
    # silently suppressing the two intended foot/calf pairs as well. Target
    # its collision shapes instead; a relationship can reference an instance
    # proxy without authoring any properties onto the proxy itself.
    articulation_body_colliders = {}
    for name, body in bodies.items():
        if body.HasAPI(UsdPhysics.ArticulationRootAPI):
            colliders = [
                prim.GetPath()
                for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ]
            if not colliders:
                raise RuntimeError(f"No distinct collision shapes found for articulation-root body {name}.")
            articulation_body_colliders[name] = colliders

    for first_name, second_name in combinations(sorted(bodies), 2):
        pair = frozenset((first_name, second_name))
        if pair in WATO_ALLOWED_SELF_COLLISION_PAIRS:
            continue
        if first_name in articulation_body_colliders:
            first_name, second_name = second_name, first_name
        api = UsdPhysics.FilteredPairsAPI.Apply(bodies[first_name])
        targets = articulation_body_colliders.get(second_name, [bodies[second_name].GetPath()])
        for target in targets:
            api.CreateFilteredPairsRel().AddTarget(target)


def _apply_runtime_articulation_properties(root_prim: Usd.Prim, cfg) -> None:
    if cfg.articulation_props is None or cfg.articulation_props.enabled_self_collisions is not True:
        raise RuntimeError("Selective collision requires enabled_self_collisions=True at runtime.")
    schemas.modify_articulation_root_properties(str(root_prim.GetPath()), cfg.articulation_props)


@sim_utils.clone
def spawn_from_urdf_with_selective_self_collision(
    prim_path: str,
    cfg: sim_utils.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn one source URDF, author pair filters, then clone the result."""
    # Converter hashing includes both ``func`` and articulation properties.  Use
    # the canonical settings for conversion so this experiment never rewrites
    # the shared cached USD; turn the runtime property on only in the stage.
    spawn_cfg = cfg.replace(
        func=sim_utils.spawn_from_urdf,
        articulation_props=cfg.articulation_props.replace(enabled_self_collisions=False),
    )
    prim = sim_utils.spawn_from_urdf(prim_path, spawn_cfg, translation, orientation, **kwargs)
    _apply_selective_self_collision_filters(prim)
    _apply_runtime_articulation_properties(prim, cfg)
    return prim


@sim_utils.clone
def spawn_from_usd_with_selective_self_collision(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Native-viewer equivalent using the exact cached USD asset."""
    spawn_cfg = cfg.replace(
        func=sim_utils.spawn_from_usd,
        articulation_props=cfg.articulation_props.replace(enabled_self_collisions=False),
    )
    prim = sim_utils.spawn_from_usd(prim_path, spawn_cfg, translation, orientation, **kwargs)
    _apply_selective_self_collision_filters(prim)
    _apply_runtime_articulation_properties(prim, cfg)
    return prim


def enable_selective_self_collision(robot_cfg) -> None:
    """Apply the isolated Wato collision mode to an ArticulationCfg instance."""
    spawn = robot_cfg.spawn
    if spawn.articulation_props is None:
        raise RuntimeError("Wato selective self-collision requires articulation properties.")
    # Keep UrdfConverterCfg.self_collision=False.  That preserves importer-level
    # filtering for joint-adjacent links; the explicit filters above narrow all
    # remaining articulation pairs to the two allowlisted encounters.
    spawn.articulation_props.enabled_self_collisions = True
    if isinstance(spawn, sim_utils.UrdfFileCfg):
        # Never let an experimental converter configuration overwrite the
        # canonical cached asset used by established checkpoints/viewers.
        spawn.usd_dir = _SELECTIVE_USD_CACHE
        spawn.func = spawn_from_urdf_with_selective_self_collision
    elif isinstance(spawn, sim_utils.UsdFileCfg):
        spawn.func = spawn_from_usd_with_selective_self_collision
    else:
        raise TypeError(f"Unsupported Wato spawn configuration: {type(spawn).__name__}")
