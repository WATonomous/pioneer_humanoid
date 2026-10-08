"""Grasp scene: the pioneer hand fixed palm-down (-Z) at the origin plus one free object.

The object body carries a sphere, a cylinder and a box geom; ``set_object`` enables one, sizes it
and sets the body's mass and inertia, so a whole run reuses one compiled model (recompiling the
hand meshes costs ~0.4 s, ~10x a grasp trial).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from pioneer_humanoid.mujoco_hand import JOINT_NAMES, PALM_BODY, hand_spec

TIMESTEP = 0.001
OBJECT_DENSITY = 400.0  # kg/m^3, wood-ish: a 3 cm radius sphere weighs 45 g
OBJECT_TYPES = ("sphere", "cylinder", "box")
_GEOM_TYPES = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
}
# Palm surface (the -Z face of the palm hull) in the hand frame.
PALM_SURFACE_Z = -0.008
# Same contact as humanoid_mujoco_scenes: time constant 4 ms (>= 2 x timestep) and solimp near 1. MuJoCo's
# default (0.02 s, 0.9-0.95) lets the fingers sink 3-12 mm into a squeezed object.
CONTACT_SOLREF = [0.004, 1.0]
CONTACT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]


def stiffen_contacts(spec: mujoco.MjSpec) -> None:
    for geom in spec.geoms:
        geom.solref = CONTACT_SOLREF
        geom.solimp = CONTACT_SOLIMP


def make_model(width: int = 640, height: int = 480, thumb: str = "stock") -> mujoco.MjModel:
    """``thumb``: a mount from thumb.THUMB_MOUNTS."""
    from .thumb import thumb_kwargs

    spec = mujoco.MjSpec()
    spec.modelname = "hand_grasp"
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Same grasp settings as humanoid_mujoco_scenes: elliptic cones + impratio 10 stop a squeezed
    # object creeping out of the fingers.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.worldbody.add_light(pos=[0.3, 0.3, 0.6], dir=[-0.5, -0.4, -1], diffuse=[0.7, 0.7, 0.7])
    spec.worldbody.add_light(pos=[-0.3, 0.0, -0.6], dir=[0.5, 0.2, 1], diffuse=[0.5, 0.5, 0.5])

    spec.worldbody.add_frame().attach_body(hand_spec(**thumb_kwargs(thumb)).body(PALM_BODY), "", "")

    obj = spec.worldbody.add_body(name="object", pos=[0, 0.1, -0.05])
    obj.add_freejoint(name="object")
    for name, gtype in _GEOM_TYPES.items():
        obj.add_geom(
            name=name, type=gtype, size=[0.02, 0.02, 0.02], rgba=[0.9, 0.55, 0.2, 1],
            friction=[1.0, 0.01, 0.001], density=0,
        )
    stiffen_contacts(spec)
    # Placeholder inertia; set_object writes the real one for each trial.
    obj.mass = 0.05
    obj.inertia = [1e-5, 1e-5, 1e-5]
    obj.explicitinertial = True
    return spec.compile()


@dataclass
class HandIndex:
    """Address lookups into a compiled grasp model."""

    qpos: np.ndarray       # (20,) hand joint qpos addresses, JOINT_NAMES order
    dof: np.ndarray        # (20,) hand joint dof addresses
    act: np.ndarray        # (20,) actuator ids
    obj_body: int
    obj_qpos: int          # first qpos of the object's freejoint (pos xyz, quat wxyz)
    obj_dof: int
    obj_geoms: dict        # type name -> geom id

    @classmethod
    def of(cls, model: mujoco.MjModel) -> "HandIndex":
        joints = [model.joint(n) for n in JOINT_NAMES]
        obj_joint = model.joint("object")
        return cls(
            qpos=np.array([j.qposadr[0] for j in joints]),
            dof=np.array([j.dofadr[0] for j in joints]),
            act=np.array([model.actuator(n).id for n in JOINT_NAMES]),
            obj_body=model.body("object").id,
            obj_qpos=obj_joint.qposadr[0],
            obj_dof=obj_joint.dofadr[0],
            obj_geoms={t: model.geom(t).id for t in OBJECT_TYPES},
        )


def set_object(model: mujoco.MjModel, idx: HandIndex, kind: str, size: np.ndarray) -> None:
    """Enable geom ``kind`` with MuJoCo ``size`` (sphere [r], cylinder [r, half_h], box [hx, hy, hz])
    and give the object body the matching mass and inertia at OBJECT_DENSITY."""
    for name, gid in idx.obj_geoms.items():
        on = name == kind
        model.geom_contype[gid] = 1 if on else 0
        model.geom_conaffinity[gid] = 1 if on else 0
        model.geom_rgba[gid, 3] = 1.0 if on else 0.0
    gid = idx.obj_geoms[kind]
    rho = OBJECT_DENSITY
    if kind == "sphere":
        r = size[0]
        mass = rho * 4.0 / 3.0 * np.pi * r**3
        inertia = np.full(3, 0.4 * mass * r**2)
        half = np.array([r, r, r])
        model.geom_size[gid] = [r, 0, 0]
    elif kind == "cylinder":
        r, h = size[0], size[1]
        mass = rho * np.pi * r**2 * 2 * h
        ixx = mass * (3 * r**2 + (2 * h) ** 2) / 12.0
        inertia = np.array([ixx, ixx, 0.5 * mass * r**2])
        half = np.array([r, r, h])
        model.geom_size[gid] = [r, h, 0]
    else:
        hx, hy, hz = size
        mass = rho * 8 * hx * hy * hz
        inertia = mass / 3.0 * np.array([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
        half = np.array([hx, hy, hz])
        model.geom_size[gid] = [hx, hy, hz]
    # Broadphase bounds follow the new size (MuJoCo computes them only at compile time).
    model.geom_aabb[gid] = np.concatenate([np.zeros(3), half])
    model.geom_rbound[gid] = np.linalg.norm(half) if kind != "sphere" else half[0]
    model.body_mass[idx.obj_body] = mass
    model.body_inertia[idx.obj_body] = inertia


def half_height(kind: str, size: np.ndarray, quat: np.ndarray) -> float:
    """World-Z half extent of the object at orientation ``quat`` (wxyz)."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rz = rot.reshape(3, 3)[2]  # world-Z row: z components of the body axes
    if kind == "sphere":
        return float(size[0])
    if kind == "cylinder":
        r, h = size[0], size[1]
        return float(abs(rz[2]) * h + r * np.sqrt(max(0.0, 1.0 - rz[2] ** 2)))
    return float(np.abs(rz) @ np.asarray(size))
