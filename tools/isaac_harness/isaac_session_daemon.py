"""Persistent Isaac Sim session daemon.

Launches Isaac Sim once (headless) and keeps it running, reading JSON
commands from a queue directory and writing JSON responses — instead of the
relaunch-per-shot model of isaac_harness.sh / isaac_multiview_screenshot.sh.
A fresh Isaac Sim launch costs ~15-40s; against an already-running session,
each command (spawn an object, move the camera, take a screenshot, query a
pose) costs a fraction of a second. Use this when iterating repeatedly on the
same scene; use the one-shot scripts when you just need a single check.

Run from inside the sim container (see isaac_session.sh for the host-side
launcher that does this for you):

    cd /workspace && PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p \
        tools/isaac_harness/isaac_session_daemon.py --queue-dir /workspace/.isaac_session

The queue directory (shared with the host via the repo's bind mount — no
docker cp needed) has two subdirs:
  cmds/      — host writes <id>.cmd.json here (atomically: write to .tmp, rename)
  responses/ — daemon writes <id>.response.json here

Command shapes (all fields besides "id" and "cmd" are command-specific):
  {"id": "...", "cmd": "ping"}
  {"id": "...", "cmd": "spawn_primitive", "name": "Obj1", "shape": "cuboid",
   "size": [0.1,0.1,0.1], "pos": [0,0,0.5], "color": [1,0,0], "dynamic": true}
  {"id": "...", "cmd": "spawn_usd", "name": "Robot", "usd_path": "...",
   "pos": [0,0,0], "rot": [1,0,0,0], "articulation": true}
  {"id": "...", "cmd": "spawn_usd", "name": "Box", "usd_path": "...", "pos": [0,0,0.3],
   "apply_physics": true, "mass": 0.2}  # for bare-mesh assets with no physics baked in (e.g. YCB)
  {"id": "...", "cmd": "spawn_visual", "name": "Box", "usd_path": "...", "pos": [0,0,0]}
  {"id": "...", "cmd": "list_dir", "path": "omniverse://.../some/dir"}
  {"id": "...", "cmd": "remove", "name": "Obj1"}
  {"id": "...", "cmd": "set_pose", "name": "Obj1", "pos": [0,0,1], "rot": [1,0,0,0]}
  {"id": "...", "cmd": "query", "name": "Obj1"}
  {"id": "...", "cmd": "list"}
  {"id": "...", "cmd": "step", "n": 30}
  {"id": "...", "cmd": "screenshot", "eye": [1,1,1], "target": [0,0,0],
   "out_path": "/workspace/.isaac_session/shots/foo.png"}
  {"id": "...", "cmd": "bbox", "name": "Obj1"}
  {"id": "...", "cmd": "overlap", "name_a": "Obj1", "name_b": "Table"}
  {"id": "...", "cmd": "distance", "name_a": "Obj1", "name_b": "Obj2"}
  {"id": "...", "cmd": "scene_tree"}
  {"id": "...", "cmd": "shutdown"}

Structured introspection (bbox/overlap/distance/scene_tree) exists so the
scene can be checked programmatically — "is the rack actually resting on the
table", "are these two objects interpenetrating" — instead of only being
checkable by eyeballing a screenshot. bbox/scene_tree work on any prim on the
stage (raw USD), not just objects this daemon spawned and is tracking.

Response shape: {"id": "...", "ok": true, ...} or {"id": "...", "ok": false, "error": "..."}
"""

import argparse
import json
import os
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--queue-dir", type=str, required=True)
parser.add_argument("--poll-interval", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext, schemas as isaaclab_schemas

CMDS_DIR = os.path.join(args_cli.queue_dir, "cmds")
RESP_DIR = os.path.join(args_cli.queue_dir, "responses")
os.makedirs(CMDS_DIR, exist_ok=True)
os.makedirs(RESP_DIR, exist_ok=True)

sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0.0, 0.0, -1.2))
cfg_light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
cfg_light.func("/World/Light", cfg_light)

sim.reset()

# name -> RigidObject | Articulation
objects: dict = {}
_object_is_dynamic: dict = {}  # name -> bool, for the bbox staleness warning
_home_arms: set = set()  # spawn_bimanual_arm names


def _reset_sim():
    """sim.reset(), then the arms back to their home pose: every reset (PLAY) restarts them at the URDF zero."""
    sim.reset()
    for name in _home_arms & objects.keys():
        robot = objects[name]
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()

_PRIMITIVE_SPAWNERS = {
    "cuboid": sim_utils.CuboidCfg,
    "sphere": sim_utils.SphereCfg,
    "cylinder": sim_utils.CylinderCfg,
    "cone": sim_utils.ConeCfg,
}

_camera = None


def _get_camera() -> Camera:
    global _camera
    if _camera is None:
        cam_cfg = CameraCfg(
            prim_path="/World/SessionCamera",
            update_period=0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        )
        _camera = Camera(cfg=cam_cfg)
        _reset_sim()
        for _ in range(5):
            sim.step()
            simulation_app.update()
    return _camera


def _shape_kwargs(shape: str, size) -> dict:
    if shape == "sphere":
        return {"radius": size[0]}
    if shape == "cylinder" or shape == "cone":
        return {"radius": size[0], "height": size[1]}
    return {"size": tuple(size)}


def handle_spawn_primitive(cmd: dict) -> dict:
    name = cmd["name"]
    if name in objects:
        return {"ok": False, "error": f"name '{name}' already exists"}
    shape = cmd.get("shape", "cuboid")
    size = cmd.get("size", [0.1, 0.1, 0.1])
    color = cmd.get("color", [0.8, 0.8, 0.8])
    dynamic = cmd.get("dynamic", True)
    spawn_kwargs = _shape_kwargs(shape, size)
    spawn_kwargs["visual_material"] = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color))
    # RigidObject requires USD RigidBodyAPI on the prim regardless of whether
    # it's meant to move — "static" means kinematic (still a rigid body, just
    # not driven by physics forces), not "skip rigid_props entirely". Skipping
    # rigid_props leaves no RigidBodyAPI applied and RigidObject's
    # _initialize_impl() throws "Failed to find a rigid body".
    spawn_kwargs["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=not dynamic,
        kinematic_enabled=not dynamic,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
    )
    spawn_kwargs["mass_props"] = sim_utils.MassPropertiesCfg(mass=cmd.get("mass", 0.1))
    spawn_kwargs["collision_props"] = sim_utils.CollisionPropertiesCfg()
    spawn_cfg = _PRIMITIVE_SPAWNERS[shape](**spawn_kwargs)
    obj_cfg = RigidObjectCfg(
        prim_path=f"/World/{name}",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(cmd.get("pos", [0, 0, 0]))),
    )
    try:
        objects[name] = RigidObject(cfg=obj_cfg)
        _object_is_dynamic[name] = dynamic
        _reset_sim()
    except Exception:
        # don't leave a half-initialized object in the registry — a broken
        # entry there crashes every later command that touches it (query,
        # step, distance, ...) with an unrelated-looking AttributeError.
        objects.pop(name, None)
        _object_is_dynamic.pop(name, None)
        raise
    return {"ok": True}


def handle_spawn_usd(cmd: dict) -> dict:
    name = cmd["name"]
    if name in objects:
        return {"ok": False, "error": f"name '{name}' already exists"}
    pos = tuple(cmd.get("pos", [0, 0, 0]))
    rot = tuple(cmd.get("rot", [1, 0, 0, 0]))
    scale = tuple(cmd.get("scale", [1, 1, 1]))
    try:
        if cmd.get("articulation", False):
            # ArticulationCfg requires actuators for every DOF or it refuses
            # to validate at all (TypeError: Missing values detected...).
            # This repo's own robot has a real actuator config to use instead
            # (spawn_bimanual_arm) — for an arbitrary articulated asset (a
            # cabinet's doors, a generic prop) there's no such config to
            # reach for, so fall back to a permissive default: every joint
            # gets a plain ImplicitActuatorCfg with light damping and enough
            # effort to hold position, matching how IsaacLab's own official
            # cabinet examples (e.g. Isaac-Open-Drawer-Franka) actuate this
            # exact asset's joints. Good enough to make the object exist and
            # be interactable/steppable; not tuned for any specific task.
            actuators = {
                "default": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=cmd.get("actuator_stiffness", 10.0),
                    damping=cmd.get("actuator_damping", 1.0),
                    effort_limit_sim=cmd.get("actuator_effort_limit", 100.0),
                ),
            }
            obj_cfg = ArticulationCfg(
                prim_path=f"/World/{name}",
                spawn=sim_utils.UsdFileCfg(usd_path=cmd["usd_path"], scale=scale),
                init_state=ArticulationCfg.InitialStateCfg(pos=pos, rot=rot),
                actuators=actuators,
            )
            objects[name] = Articulation(cfg=obj_cfg)
        else:
            obj_cfg = RigidObjectCfg(
                prim_path=f"/World/{name}",
                spawn=sim_utils.UsdFileCfg(usd_path=cmd["usd_path"], scale=scale),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            )
            objects[name] = RigidObject(cfg=obj_cfg)
            if cmd.get("apply_physics", False):
                # Root cause (found, not guessed): UsdFileCfg.rigid_props/
                # collision_props/mass_props route through
                # isaaclab.sim.schemas.modify_rigid_body_properties() et al,
                # which only *modifies* an ALREADY-APPLIED RigidBodyAPI/
                # CollisionAPI/MassAPI — it returns False (no exception) if
                # the schema isn't there, so passing rigid_props to a
                # bare-mesh asset (no physics baked in, e.g. YCB) is a silent
                # no-op, not a failure at spawn time. The actual failure
                # surfaces later and confusingly, in
                # RigidObject._initialize_impl() at sim.reset() time:
                # "Failed to find a rigid body... Please ensure RigidBodyAPI
                # is applied." IsaacLab's own define_*_properties()
                # functions (used above by RigidObjectCfg for tables/etc
                # that already carry the API) apply the schema first, then
                # delegate to modify_*() — that's the missing step, done
                # here explicitly instead of through UsdFileCfg's cfg fields.
                prim_path = f"/World/{name}"
                isaaclab_schemas.define_rigid_body_properties(
                    prim_path,
                    sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                    ),
                )
                isaaclab_schemas.define_collision_properties(
                    prim_path, sim_utils.CollisionPropertiesCfg()
                )
                isaaclab_schemas.define_mass_properties(
                    prim_path, sim_utils.MassPropertiesCfg(mass=cmd.get("mass", 0.2))
                )
        # Whether a USD-file rigid body is actually kinematic depends on
        # what's baked into the asset itself, which we don't inspect here —
        # conservatively assume dynamic (i.e. bbox may go stale) unless
        # proven otherwise.
        _object_is_dynamic[name] = True
        _reset_sim()
    except Exception:
        objects.pop(name, None)
        _object_is_dynamic.pop(name, None)
        raise
    return {"ok": True}


def handle_spawn_bimanual_arm(cmd: dict) -> dict:
    """Spawn the repo's actual pioneer_bimanual_arm_8dof robot with its real,
    pre-built actuator config (BIMANUAL_ARM_8DOF_CFG from
    the pioneer_humanoid package) — generic
    spawn_usd --articulation builds a bare ArticulationCfg with no actuators
    and fails outright for this robot:
        TypeError: Missing values detected in object ArticulationCfg for
        the following fields:
          - actuators
    This command exists because that's a real, common case (spawning THE
    robot this repo cares about, not an arbitrary articulation), not a
    generic articulation-with-actuators spawner.
    """
    name = cmd["name"]
    if name in objects:
        return {"ok": False, "error": f"name '{name}' already exists"}

    import sys

    pioneer_humanoid_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/pioneer_humanoid"))
    if pioneer_humanoid_dir not in sys.path:
        sys.path.insert(0, pioneer_humanoid_dir)
    from pioneer_humanoid.bimanual_arm_8dof import BIMANUAL_ARM_8DOF_CFG

    pos = tuple(cmd.get("pos", [0, 0, 0]))
    rot = tuple(cmd.get("rot", [1, 0, 0, 0]))
    obj_cfg = BIMANUAL_ARM_8DOF_CFG.replace(prim_path=f"/World/{name}")
    obj_cfg.init_state.pos = pos
    obj_cfg.init_state.rot = rot
    try:
        objects[name] = Articulation(cfg=obj_cfg)
        _object_is_dynamic[name] = True
        _home_arms.add(name)
        _reset_sim()
    except Exception:
        objects.pop(name, None)
        _object_is_dynamic.pop(name, None)
        _home_arms.discard(name)
        raise
    return {"ok": True}


def handle_list_dir(cmd: dict) -> dict:
    """List a Nucleus (or local) directory via omni.client — for finding an
    asset's exact path/filename before spawning it, e.g. browsing
    ISAAC_NUCLEUS_DIR/Props/YCB/... to see what's actually there rather than
    guessing a filename.
    """
    import omni.client

    path = cmd["path"]
    result, entries = omni.client.list(path)
    if result != omni.client.Result.OK:
        return {"ok": False, "error": f"omni.client.list('{path}') failed: {result}"}
    return {
        "ok": True,
        "entries": [
            {"name": e.relative_path, "is_folder": bool(e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)}
            for e in entries
        ],
    }


def handle_spawn_visual(cmd: dict) -> dict:
    """Spawn a USD asset as a pure visual reference — no RigidObject/
    Articulation wrapper, no RigidBodyAPI requirement. For assets that don't
    need physics (a bare-mesh prop just being looked at/screenshotted, e.g.
    YCB) and would otherwise fail with "Failed to find a rigid body... please
    ensure RigidBodyAPI is applied" (a real open bug for those assets via
    spawn_usd/RigidObject, see #209) — sidesteps that requirement entirely
    since a purely visual placement was never a physics object to begin with.
    Not tracked in `objects` (no physics data to query/step/set_pose on);
    still visible to scene_tree/bbox/screenshot since those work off raw USD.
    """
    name = cmd["name"]
    prim_path = f"/World/{name}"
    import omni.usd

    if omni.usd.get_context().get_stage().GetPrimAtPath(prim_path).IsValid():
        return {"ok": False, "error": f"prim already exists at {prim_path}"}
    pos = tuple(cmd.get("pos", [0, 0, 0]))
    rot = tuple(cmd.get("rot", [1, 0, 0, 0]))
    scale = tuple(cmd.get("scale", [1, 1, 1]))
    cfg = sim_utils.UsdFileCfg(usd_path=cmd["usd_path"], scale=scale)
    cfg.func(prim_path, cfg, translation=pos, orientation=rot)
    return {"ok": True}


def handle_remove(cmd: dict) -> dict:
    """Remove a tracked object from the scene.

    Root cause (found, not guessed) of the old "remove corrupts every OTHER
    tracked Articulation's physics view" bug (ReferenceError: weakly-
    referenced object no longer exists / Failed to get DOF velocities from
    backend, on a completely untouched object): `del objects[name]` alone
    does not immediately garbage-collect the removed RigidObject/
    Articulation. `_create_buffers()` builds a `WrenchComposer(self)` for
    each asset — a reference cycle (asset -> composer -> asset) — so
    CPython's refcounting can't free it right away; only a cyclic-GC pass
    would, on no particular schedule. Until that happens, the removed
    object's PLAY/STOP timeline callbacks (subscribed in AssetBase.__init__,
    only unsubscribed in __del__ -> _clear_callbacks()) are STILL LIVE. The
    very next `sim.reset()` (stop() then play()) fires those callbacks
    against a prim `stage.RemovePrim()` already deleted — `_initialize_impl()`
    for a since-deleted prim path throws mid-callback, which is what was
    actually corrupting shared PhysX/tensor-view state for every other
    tracked Articulation, not anything about the reset cycle itself (plain
    resets from ordinary spawns never showed this — only removes did).
    Fixed by forcing a `gc.collect()` between removing the prim and
    resetting, so the removed object's `__del__`/`_clear_callbacks()` runs
    and unsubscribes it BEFORE the reset cycle can fire its now-orphaned
    callbacks. An earlier, different fix attempt here
    (`_reinitialize_tracked_objects()`, forcing every OTHER object to
    manually reinitialize) caused a genuine CUDA crash and was reverted —
    this fix doesn't touch other objects at all, it just stops the removed
    one from firing.
    """
    name = cmd["name"]
    if name not in objects:
        return {"ok": False, "error": f"no such object '{name}'"}
    import gc

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage.RemovePrim(f"/World/{name}")
    del objects[name]
    _object_is_dynamic.pop(name, None)
    gc.collect()
    _reset_sim()
    return {"ok": True}


def handle_set_pose(cmd: dict) -> dict:
    """Move an already-spawned object to a new pose.

    Two things had to be fixed here after they cost real debugging time:

    1. write_root_pose_to_sim() on a KINEMATIC object (spawn_primitive
       --static) crashes the whole process — a real PhysX/CUDA "illegal
       memory access" abort, not a Python exception, confirmed reproducible
       on a fresh daemon with nothing else going on. Kinematic bodies aren't
       driven through the normal dynamics tensor write path; move them by
       writing the USD prim's transform directly instead.
    2. A dynamic object's write_root_pose_to_sim() change used to get
       silently wiped by the next spawn/remove. Root cause (actually found,
       not guessed at): sim.reset() calls stop() then play() — the STOP
       event resets every tracked asset's `_is_initialized` flag to False
       (asset_base.py's _invalidate_initialize_callback), so the next PLAY
       event runs _initialize_impl() again IN FULL, including
       _process_cfg(), which re-derives `data.default_root_state` from
       `self.cfg.init_state` from scratch. Two earlier fix attempts both
       missed this: writing directly into `data.default_root_state` doesn't
       survive because that tensor gets fully REPLACED (not updated) on the
       next reinit. A second attempt (confirmed wrong THIS TIME, not just
       assumed): mutating `obj.cfg.init_state.pos`/`.rot` also didn't
       survive, despite `self.cfg` being a stored, never-reassigned
       reference on the live object (asset_base.py: `self.cfg = cfg.copy()`
       once, in __init__) that a mutation through `objects[name].cfg`
       genuinely does reach. The reason it still didn't survive: a real
       (non-soft) `sim.reset()` calls PhysX `stop()` then `play()`, and on
       PLAY, PhysX re-initializes a rigid body's pose from the USD prim's
       *authored xformOp transform on the stage* — not from
       `cfg.init_state`, not from `data.default_root_state`.
       `write_root_pose_to_sim()` only writes the live physics TENSOR
       (confirmed: `query()` reflected the new pose instantly, but a
       `bbox()` read of the same prim right after showed the transform
       still at its original spawn value) — it never touches the prim's
       actual USD attributes, so PLAY reads the stale, original transform
       every time regardless of any of the above. Fixed by writing the new
       pose directly to the prim's xformOps (the same thing the kinematic
       branch below was already doing) for EVERY object, dynamic or
       kinematic — `write_root_pose_to_sim()` stays too, so a dynamic
       object's `query()` reflects the move immediately without waiting for
       a `step()`/reset first.
    """
    name = cmd["name"]
    if name not in objects:
        return {"ok": False, "error": f"no such object '{name}'"}
    obj = objects[name]
    pos = cmd.get("pos")
    rot = cmd.get("rot")
    cur_pos = obj.data.root_pos_w.clone()
    cur_rot = obj.data.root_quat_w.clone()
    if pos is not None:
        cur_pos[:] = torch.tensor(pos, device=cur_pos.device)
    if rot is not None:
        cur_rot[:] = torch.tensor(rot, device=cur_rot.device)

    if _object_is_dynamic.get(name, True):
        obj.write_root_pose_to_sim(torch.cat([cur_pos, cur_rot], dim=-1))

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(f"/World/{name}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    p = cur_pos[0].tolist()
    q = cur_rot[0].tolist()  # (w, x, y, z)
    xform.AddTranslateOp().Set(Gf.Vec3d(*p))
    # ClearXformOpOrder() clears the op ORDER, not the underlying
    # attribute — a prim spawned via CuboidCfg/etc. already has an
    # orient xformOp authored at double precision, and AddOrientOp()'s
    # own default precision doesn't match it, raising a Tf error.
    # Match explicitly instead of guessing.
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3]))
    )

    # data.default_root_state and cfg.init_state are NOT what a real
    # sim.reset() reads on PLAY (see docstring above) — the USD prim xform
    # written above is the actual fix. Still update these two so query()
    # and anything reading cfg directly stay consistent before any reset.
    obj.data.default_root_state[:, 0:3] = cur_pos
    obj.data.default_root_state[:, 3:7] = cur_rot
    obj.cfg.init_state.pos = tuple(cur_pos[0].tolist())
    obj.cfg.init_state.rot = tuple(cur_rot[0].tolist())
    return {"ok": True}


def handle_query(cmd: dict) -> dict:
    name = cmd["name"]
    if name not in objects:
        return {"ok": False, "error": f"no such object '{name}'"}
    obj = objects[name]
    return {
        "ok": True,
        "pos": obj.data.root_pos_w[0].tolist(),
        "rot": obj.data.root_quat_w[0].tolist(),
    }


def handle_joint_state(cmd: dict) -> dict:
    """Per-joint state for an articulated object — query()/bbox() only report
    a single root-body world pose, which says nothing about a cabinet door's
    open/closed angle or any other articulated object's actual configuration.
    Only meaningful for Articulation objects (spawn_bimanual_arm, or
    spawn_usd --articulation); RigidObject has no joints.
    """
    name = cmd["name"]
    if name not in objects:
        return {"ok": False, "error": f"no such object '{name}'"}
    obj = objects[name]
    if not hasattr(obj, "joint_names"):
        return {"ok": False, "error": f"'{name}' is not an articulation (no joints)"}
    return {
        "ok": True,
        "joint_names": obj.joint_names,
        "joint_pos": obj.data.joint_pos[0].tolist(),
        "joint_vel": obj.data.joint_vel[0].tolist(),
    }


def handle_list(cmd: dict) -> dict:
    return {"ok": True, "names": list(objects.keys())}


def _prim_path_for(name: str) -> str:
    """Resolve a name to a /World prim path — works for tracked objects
    (spawned via spawn_primitive/spawn_usd) and untracked static prims
    (ground, light, or anything spawned outside the daemon's own registry)
    alike, since bbox uses raw USD, not the RigidObject/Articulation wrapper.
    """
    return f"/World/{name}"


def handle_bbox(cmd: dict) -> dict:
    """World-space AABB of a prim via raw USD (UsdGeom.BBoxCache).

    KNOWN LIMITATION: for a dynamic (non-kinematic) tracked object that has
    moved under physics since spawn, this can return a STALE, spawn-time
    bounding box. Physics for RigidObject/Articulation runs on a tensor/
    Fabric layer that does not reliably sync back to the USD stage's xform
    attributes every step, so a raw-USD bbox read can lag behind reality —
    confirmed by comparing against query()'s tensor-sourced pose, which
    stayed correct while this drifted. An attempted fix (reconstruct the
    world bbox from a cached local bbox + the live tracked pose) produced
    numbers that didn't match query() either and was reverted rather than
    ship something subtly wrong — needs a real fix, not a guess.
    For a moving tracked object, treat query()'s pos/rot as authoritative
    and use bbox mainly for STATIC prims (ground, table, anything kinematic
    or not yet stepped) until this is properly fixed.
    """
    from pxr import UsdGeom

    import omni.usd

    name = cmd["name"]
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(_prim_path_for(name))
    if not prim.IsValid():
        return {"ok": False, "error": f"no prim at /World/{name}"}

    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=True)
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    result = {"ok": True, "min": list(rng.GetMin()), "max": list(rng.GetMax())}
    if _object_is_dynamic.get(name):
        result["warning"] = (
            "bbox may be stale for a moved dynamic object — see query() for the authoritative live pose"
        )
    return result


def handle_overlap(cmd: dict) -> dict:
    """AABB overlap check between two named prims — a cheap proxy for
    'are these two things colliding/touching', not exact mesh-level
    collision, but usually enough to answer 'is the rack actually on the
    table' / 'are these two vials interpenetrating' without eyeballing a
    screenshot.
    """
    a = handle_bbox({"name": cmd["name_a"]})
    if not a["ok"]:
        return a
    b = handle_bbox({"name": cmd["name_b"]})
    if not b["ok"]:
        return b
    overlap = all(a["min"][i] <= b["max"][i] and b["min"][i] <= a["max"][i] for i in range(3))
    return {"ok": True, "overlap": overlap, "bbox_a": a, "bbox_b": b}


def handle_distance(cmd: dict) -> dict:
    """Center-to-center distance between two named objects. Uses tracked
    pose data when available (cheaper, already up to date after step()),
    falls back to bbox center for untracked/static prims.
    """
    def _center(name: str):
        if name in objects:
            return objects[name].data.root_pos_w[0].tolist()
        bbox = handle_bbox({"name": name})
        if not bbox["ok"]:
            return None
        return [(bbox["min"][i] + bbox["max"][i]) / 2 for i in range(3)]

    pos_a = _center(cmd["name_a"])
    if pos_a is None:
        return {"ok": False, "error": f"no such prim '{cmd['name_a']}'"}
    pos_b = _center(cmd["name_b"])
    if pos_b is None:
        return {"ok": False, "error": f"no such prim '{cmd['name_b']}'"}

    dist = sum((pos_a[i] - pos_b[i]) ** 2 for i in range(3)) ** 0.5
    return {"ok": True, "distance": dist, "pos_a": pos_a, "pos_b": pos_b}


def handle_scene_tree(cmd: dict) -> dict:
    """Full /World prim hierarchy with types — includes static prims (ground,
    light, anything spawned outside the daemon's own tracked-object registry)
    that `list` doesn't see, since `list` only reports objects this daemon
    instance spawned and is tracking Python-side.
    """
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        return {"ok": False, "error": "no /World prim on stage"}

    tree = []
    for prim in Usd.PrimRange(world):
        tree.append({
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "tracked": prim.GetPath().name in objects,
        })
    return {"ok": True, "tree": tree}


def handle_step(cmd: dict) -> dict:
    n = cmd.get("n", 1)
    dt = sim.get_physics_dt()
    for _ in range(n):
        sim.step()
        simulation_app.update()
        # RigidObject/Articulation cache their tensors — like the Camera
        # sensor, they need an explicit update() each step to stay in sync
        # with the physics view, or query() silently returns stale data.
        for obj in objects.values():
            obj.update(dt)
    return {"ok": True}


def handle_screenshot(cmd: dict) -> dict:
    camera = _get_camera()
    eye = cmd.get("eye", [2.0, 2.0, 2.0])
    target = cmd.get("target", [0.0, 0.0, 0.0])
    eyes = torch.tensor([eye], dtype=torch.float32, device=camera.device)
    targets = torch.tensor([target], dtype=torch.float32, device=camera.device)
    camera.set_world_poses_from_view(eyes, targets)
    for _ in range(10):
        sim.step()
        simulation_app.update()
    camera.update(dt=0.0, force_recompute=True)

    import numpy as np
    from PIL import Image

    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    out_path = cmd["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(rgb).save(out_path)
    return {"ok": True, "out_path": out_path}


def handle_ping(cmd: dict) -> dict:
    return {"ok": True, "pong": True}


HANDLERS = {
    "ping": handle_ping,
    "spawn_primitive": handle_spawn_primitive,
    "spawn_usd": handle_spawn_usd,
    "spawn_bimanual_arm": handle_spawn_bimanual_arm,
    "spawn_visual": handle_spawn_visual,
    "list_dir": handle_list_dir,
    "remove": handle_remove,
    "set_pose": handle_set_pose,
    "query": handle_query,
    "joint_state": handle_joint_state,
    "list": handle_list,
    "step": handle_step,
    "screenshot": handle_screenshot,
    "bbox": handle_bbox,
    "overlap": handle_overlap,
    "distance": handle_distance,
    "scene_tree": handle_scene_tree,
}


def process_command(cmd: dict) -> dict:
    cmd_type = cmd.get("cmd")
    handler = HANDLERS.get(cmd_type)
    if handler is None:
        return {"ok": False, "error": f"unknown command '{cmd_type}'"}
    try:
        return handler(cmd)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{exc}\n{traceback.format_exc()}"}
    finally:
        # A failed asset _initialize_impl() (e.g. a spawn_usd with no
        # RigidBodyAPI) stores its exception in this global, and
        # SimulationContext.step()/render() re-raises it on EVERY later call
        # until it's cleared — otherwise one failed spawn poisons every
        # command after it with the SAME stale error, including ones
        # touching a completely different, perfectly fine object. Confirmed:
        # a failed CrackerBox spawn made the next command's error message
        # still say "CrackerBox" even though it was spawning "Mustard".
        import builtins

        if getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None) is not None:
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None


def write_response(cmd_id: str, response: dict) -> None:
    response["id"] = cmd_id
    tmp_path = os.path.join(RESP_DIR, f"{cmd_id}.response.json.tmp")
    final_path = os.path.join(RESP_DIR, f"{cmd_id}.response.json")
    with open(tmp_path, "w") as f:
        json.dump(response, f)
    os.rename(tmp_path, final_path)


print("DAEMON_READY", flush=True)

running = True
while running and simulation_app.is_running():
    cmd_files = sorted(f for f in os.listdir(CMDS_DIR) if f.endswith(".cmd.json"))
    if not cmd_files:
        simulation_app.update()
        time.sleep(args_cli.poll_interval)
        continue

    for fname in cmd_files:
        path = os.path.join(CMDS_DIR, fname)
        try:
            with open(path) as f:
                cmd = json.load(f)
        except (json.JSONDecodeError, OSError):
            # still being written — try again next loop
            continue
        os.remove(path)

        cmd_id = cmd.get("id", fname)
        if cmd.get("cmd") == "shutdown":
            write_response(cmd_id, {"ok": True})
            running = False
            break

        response = process_command(cmd)
        write_response(cmd_id, response)

simulation_app.close()
