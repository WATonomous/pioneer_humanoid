"""Entity and scene specs for the mjlab task, split out of scene/badminton.xml.

The gate-validated scene file stays the single source of geometry and
calibration. Three views of it are produced here:

  robot_spec()    both arms + stand + two rackets, XML actuators kept
  shuttle_spec()  free-floating shuttle (3 geoms + visual mesh)
  court_fn(spec)  scene hook: attaches court/net/floor and re-adds every
                  explicit contact pair of the scene file (shuttle vs faces,
                  floor, net; inter-arm clashes) with entity prefixes

Splitting is by deletion, not reconstruction, so params.yaml edits and
build_scene.py reruns flow through unchanged.
"""

from __future__ import annotations

import os

import mujoco

import aero

_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_HERE, os.pardir, "scene", "badminton.xml")
SCENE_DIR = os.path.dirname(os.path.abspath(SCENE_XML))

ROBOT_ROOT = "arm_base_link"
SHUTTLE_ROOT = "shuttle"
_P = aero.load_params()["arm"]
ARM_JOINTS_R = tuple("arm_" + j for j in _P["joints"])
ARM_JOINTS_L = tuple("arm_" + j for j in _P["joints_left"])
ARM_JOINTS = ARM_JOINTS_R + ARM_JOINTS_L     # actuator / action order
FACE_SITES = ("face_center", "face_center_l")


def _load() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(os.path.abspath(SCENE_XML))
    # Attaching into the scene spec re-resolves asset paths relative to the
    # parent spec, so make every mesh path absolute first.
    for mesh in spec.meshes:
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = os.path.join(SCENE_DIR, mesh.file)
    return spec


def _delete_worldbody_furniture(spec: mujoco.MjSpec) -> None:
    """Drop floor/lines/lights/sites that sit directly under worldbody."""
    for geom in [g for g in spec.worldbody.geoms]:
        spec.delete(geom)
    for site in [s for s in spec.worldbody.sites]:
        spec.delete(site)
    for light in [li for li in spec.worldbody.lights]:
        spec.delete(light)


def _delete_keys_and_pairs(spec: mujoco.MjSpec) -> None:
    for key in [k for k in spec.keys]:
        spec.delete(key)
    for pair in [p for p in spec.pairs]:
        spec.delete(pair)


def robot_spec() -> mujoco.MjSpec:
    spec = _load()
    _delete_worldbody_furniture(spec)
    _delete_keys_and_pairs(spec)
    for name in (SHUTTLE_ROOT, "net_body"):
        spec.delete(spec.body(name))
    return spec


def shuttle_spec() -> mujoco.MjSpec:
    spec = _load()
    _delete_worldbody_furniture(spec)
    _delete_keys_and_pairs(spec)
    for name in (ROBOT_ROOT, "net_body"):
        spec.delete(spec.body(name))
    for act in [a for a in spec.actuators]:
        spec.delete(act)
    return spec


def court_fn(spec: mujoco.MjSpec) -> None:
    """SceneCfg.spec_fn: attach the court and declare the contact pairs.

    Entity geoms all carry contype=0/conaffinity=0, so these explicit pairs
    are the only source of contacts — same contract as the CPU scene.
    """
    court = _load()
    _delete_keys_and_pairs(court)
    for name in (ROBOT_ROOT, SHUTTLE_ROOT):
        court.delete(court.body(name))
    for act in [a for a in court.actuators]:
        court.delete(act)
    frame = spec.worldbody.add_frame()
    spec.attach(court, prefix="court/", frame=frame)

    # the scene file's pairs are the contact contract; re-add each one with
    # the prefix of the entity that now owns its geoms
    src = _load()
    shuttle_geoms = {g.name for g in src.body(SHUTTLE_ROOT).find_all("geom")}
    court_geoms = {g.name for g in src.worldbody.geoms} | {
        g.name for g in src.body("net_body").find_all("geom")}

    def owner(name: str) -> str:
        if name in shuttle_geoms:
            return "shuttle/" + name
        if name in court_geoms:
            return "court/" + name
        return "robot/" + name

    for pr in src.pairs:
        new = spec.add_pair(geomname1=owner(pr.geomname1),
                            geomname2=owner(pr.geomname2))
        new.solref = pr.solref
        new.solimp = pr.solimp


# Ready pose (the policy's action offset). joint5 1.5708 sits outside its
# 1.31 rad range, so the action clip holds it at 1.31; the trained policies
# carry that offset, so it stays. The left arm holds the mirrored pose.
_READY_R = (-1.5708, 0.0, 0.0, 0.0, 1.5708, -1.5708)
READY_JOINT_POS = {
    **dict(zip(ARM_JOINTS_R, _READY_R)),
    **{j: s * q for j, s, q in zip(ARM_JOINTS_L, _P["left_mirror"], _READY_R)},
}


def joint_ranges() -> dict[str, tuple[float, float]]:
    """Arm joint ranges from the built scene (params clipped to the URDF)."""
    m = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE_XML))
    return {j: tuple(float(v) for v in m.joint(j).range) for j in ARM_JOINTS}
