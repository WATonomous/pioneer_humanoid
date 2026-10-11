"""The Pioneer humanoid URDF (from Prometheus) in MuJoCo, standing at a scene, with position actuators.

The URDF lives in the Prometheus repo (Backend/cad_bench/projects/humanoid_urdf/, 206 MB of meshes), not
here: set PIONEER_HUMANOID_URDF_DIR to that folder, or check Prometheus out next to this repo.

Changed here, not in the URDF (all sim-side assumptions, see README.md):
  - collision: the URDF's per-link bounding boxes are dropped; the hands collide with their own visual meshes
    (MuJoCo uses each mesh's convex hull). The arms, torso and legs do not collide at all.
  - elbow and forearm-roll ranges widened (the URDF's +-90 deg arm limits are placeholders).
  - the arms do not collide with the body: Arm.ik keeps them out of it (BODY_BOXES).
  - base fixed, gravity compensated on every robot body; PD position actuators with guessed gains.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np

from humanoid_mujoco_scenes import _register as R

from . import matcha_hands

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_URDF_DIR = _REPO_ROOT.parent / "Prometheus" / "Backend" / "cad_bench" / "projects" / "humanoid_urdf"
URDF_NAME = "pioneer_humanoid_mujoco.urdf"   # the URDF with the MuJoCo compiler line already in it

ROBOT_POS = (-0.10, 0.0, 0.02)               # feet on the floor, standing behind the table edge, facing +X
# The URDF's arm limits are placeholders (+-90 deg); a handshake grip at table height needs ~115 deg of
# elbow bend. Assumed here: 0 to 145 deg of flexion (negative is flexion). Not from the CAD.
ELBOW_RANGE = [-2.53, 0.0]
# Same for the forearm roll: pouring palm-down needs ~140 deg. Assumed +-150 deg.
FOREARM_ROLL_RANGE = [-2.6, 2.6]
# The arms do not collide with the body in the sim (the URDF's boxes are useless for it), so the IK keeps them out
# of it: the torso as three boxes fitted to base_link's mesh (base frame: x to the robot's left, y back, z up),
# each arm as capsules of ARM_RADIUS along the upper arm and the forearm. (center, half-size) in metres.
BODY_BOXES = (
    ((0.0, 0.0325, 0.760), (0.222, 0.1075, 0.110)),   # pelvis with the hip drums, z 0.65-0.87
    ((0.0, -0.0025, 0.965), (0.112, 0.0775, 0.095)),  # waist, z 0.87-1.06
    ((0.0, -0.0125, 1.195), (0.165, 0.1125, 0.135)),  # chest, z 1.06-1.33
)
# Capsule radius per arm (the links are ~50 mm in radius). The right arm reaches across the body for the cup and
# came within 2 mm of it at 50 mm, so it gets 10 mm of margin; the left arm's poses clear the body by >15 mm.
ARM_RADIUS = {"l": 0.05, "r": 0.06}
FINGER_TORQUE = 1.5                          # N m per finger joint: the real tendon hand's is unknown
TOOLS = ("ladle", "pitcher", "whisk", "cup")

ARM = ("shoulder_pitch", "shoulder_roll", "upper_arm_yaw", "elbow", "forearm_roll", "wrist_dev", "wrist_flex")
# Grasp point in the palm frame (hand_H): a 25 mm post beside the palm, just short of the knuckles, so the
# fingers wrap round it (found by sweeping offsets on the whisk and the ladle; grasp 30-40 mm under the post top).
GRASP = {"l": (-0.036, 0.0, -0.112), "r": (0.036, 0.0, -0.112)}
# Handshake: fingers forward (+X), thumb up, palm facing +-Y. Columns are the palm frame's axes in world.
HANDSHAKE = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)


def urdf_dir() -> Path:
    d = Path(os.environ.get("PIONEER_HUMANOID_URDF_DIR", _DEFAULT_URDF_DIR))
    if not (d / URDF_NAME).exists():
        raise FileNotFoundError(
            f"{d / URDF_NAME} not found: set PIONEER_HUMANOID_URDF_DIR to Prometheus's "
            "Backend/cad_bench/projects/humanoid_urdf folder")
    return d


def humanoid_spec() -> mujoco.MjSpec:
    """MjSpec of the humanoid: hand-mesh colliders, sim joint ranges, one position actuator per joint."""
    d = urdf_dir()
    # The URDF's mesh filenames already start with meshes/, so the mesh dir is the URDF's own folder.
    s = (d / URDF_NAME).read_text().replace('<compiler meshdir="meshes"', f'<compiler meshdir="{d}/" fusestatic="false"')
    spec = mujoco.MjSpec.from_string(s)
    for b in spec.bodies:
        hand = "_hand_" in b.name and not b.name.endswith("_W")
        for g in list(b.geoms):
            if g.contype or g.conaffinity:
                # The URDF's collision is each link's bounding box: too coarse to grasp with (hands) and
                # far bigger than the link (arms: the forearm's box drags on the table). Only the hands collide.
                spec.delete(g)
                continue
            elif hand:                                  # visual mesh -> also a (convex hull) collider
                g.contype, g.conaffinity = 2, 1
                g.friction = [1.2, 0.02, 0.001]
                g.condim = 4
                g.density = 0
    for j in spec.joints:
        if j.name.endswith("_elbow"):
            j.range = ELBOW_RANGE
        if j.name.endswith("_forearm_roll"):
            j.range = FOREARM_ROLL_RANGE
    for j in spec.joints:
        n = j.name
        a = spec.add_actuator()
        a.name, a.target, a.trntype = n, n, mujoco.mjtTrn.mjTRN_JOINT
        if any(k in n for k in ("shoulder", "upper_arm", "elbow", "forearm", "hip", "knee")):
            kp, kv, f, arm = 400, 40, 60, 0.02
        elif "wrist" in n:
            kp, kv, f, arm = 60, 4, 10, 0.005
        else:                                            # fingers
            kp, kv, f, arm = 10, 0.4, FINGER_TORQUE, 0.0005
        a.set_to_position(kp=kp, kv=kv)
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-f, f]
        j.armature = arm
    return spec


def add_sites(spec: mujoco.MjSpec) -> None:
    """Grasp sites in the palms; gravity compensation on every robot body (the base is fixed)."""
    for s, p in GRASP.items():
        spec.body(f"{s}_hand_H").add_site(name=f"{s}_grasp", pos=list(p), size=[0.006, 0, 0], rgba=[1, 0, 0, 0.8], group=4)
    for b in spec.bodies:
        if b.name.startswith(("l_", "r_")) or b.name == "base_link":
            b.gravcomp = 1.0


def make_spec() -> mujoco.MjSpec:
    """matcha_hands scene + the humanoid, with the same contact settings as humanoid_mujoco_scenes."""
    spec = mujoco.MjSpec()
    spec.modelname = "matcha_hands"
    o = spec.option
    o.timestep = R.TIMESTEP
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    o.impratio = 10
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    matcha_hands.build(spec)
    fr = spec.worldbody.add_frame(pos=list(ROBOT_POS), quat=[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    hs = humanoid_spec()
    add_sites(hs)
    fr.attach_body(hs.body("base_link"), "", "")
    # Grasp assist: one inactive weld per hand and tool, switched on by Sim.attach once the fingers have
    # physically closed on the tool (the hull-collided fingers alone hold a light tool only now and then).
    for side in "lr":
        for tool in TOOLS:
            eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=f"{side}_hand_H", name2=tool,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)
            eq.name = f"{side}_grasp_{tool}"
            eq.solref = [0.003, 1.0]
    for g in spec.geoms:
        if np.allclose(g.solref, R._DEFAULT_SOLREF) and np.allclose(g.solimp, R._DEFAULT_SOLIMP):
            g.solref, g.solimp = R.CONTACT_SOLREF, R.CONTACT_SOLIMP
    return spec


def make_model() -> mujoco.MjModel:
    return make_spec().compile()


class Arm:
    """Damped-least-squares IK for one arm's 7 joints on the grasp site (position + orientation)."""

    def __init__(self, m: mujoco.MjModel, side: str):
        self.m, self.s = m, side
        self.jn = [f"{side}_{j}" for j in ARM]
        self.qadr = np.array([m.joint(j).qposadr[0] for j in self.jn])
        self.dof = np.array([m.joint(j).dofadr[0] for j in self.jn])
        self.act = np.array([m.actuator(j).id for j in self.jn])
        self.lo = np.array([m.joint(j).range[0] for j in self.jn])
        self.hi = np.array([m.joint(j).range[1] for j in self.jn])
        self.site = m.site(f"{side}_grasp").id
        self.kd = mujoco.MjData(m)          # kinematics scratch: IK never touches the simulated state
        self.rest = np.zeros(7)             # null-space posture
        # Arm capsule axis: (body, from-point body, to-point body); points sampled along each segment.
        b = lambda n: m.body(f"{side}_{n}").id
        self.segments = ((b("arm_link3"), b("arm_link3"), b("arm_link4")), (b("arm_link4"), b("arm_link4"), b("forearm")),
                         (b("forearm"), b("forearm"), b("hand_W")))
        self.base = m.body("base_link").id
        mujoco.mj_kinematics(m, self.kd)
        self.base_p, self.base_R = self.kd.xpos[self.base].copy(), self.kd.xmat[self.base].reshape(3, 3).copy()

    def fk(self, q):
        self.kd.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.kd)
        mujoco.mj_comPos(self.m, self.kd)
        return self.kd.site_xpos[self.site].copy(), self.kd.site_xmat[self.site].reshape(3, 3).copy()

    def body_clearance(self):
        """Rows (Jacobian 1x7, needed outward motion m) for arm-capsule points closer than ARM_RADIUS to the body."""
        rows, jp = [], np.zeros((3, self.m.nv))
        for body, b0, b1 in self.segments:
            p0, p1 = self.kd.xpos[b0], self.kd.xpos[b1]
            for u in (0.0, 0.25, 0.5, 0.75, 1.0):
                p = p0 + (p1 - p0) * u
                local = self.base_R.T @ (p - self.base_p)
                for c, h in BODY_BOXES:
                    rel = local - np.array(c)
                    out = np.abs(rel) - np.array(h)                 # >0: outside the box along that axis
                    if (out > 0).any():
                        v = np.where(out > 0, out, 0.0) * np.sign(rel)
                        dist = np.linalg.norm(v); n = v / dist
                    else:                                            # inside: leave by the nearest face
                        k = int(np.argmax(out)); dist = out[k]; n = np.zeros(3); n[k] = np.sign(rel[k]) or 1.0
                    if dist < ARM_RADIUS[self.s]:
                        mujoco.mj_jac(self.m, self.kd, jp, None, p, body)
                        rows.append(((self.base_R @ n) @ jp[:, self.dof], ARM_RADIUS[self.s] - dist))
        return rows

    def ik(self, pos, R, q0, iters=60, wrot=0.3):
        """Joint angles putting the site at pos with rotation R, keeping the arm off the body; returns
        (q, position error m, angle error deg)."""
        q = q0.copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, Rc = self.fk(q)
            ep = pos - p
            er = 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
            clear = self.body_clearance()
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3 and not clear:
                break
            mujoco.mj_jacSite(self.m, self.kd, jp, jr, self.site)
            J = np.vstack([jp[:, self.dof], wrot * jr[:, self.dof]] + [3.0 * r[0][None] for r in clear])
            e = np.concatenate([ep, wrot * er, [3.0 * r[1] for r in clear]])
            JJ = J @ J.T + 1e-3 * np.eye(len(e))
            dq = J.T @ np.linalg.solve(JJ, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJ, J)
            dq += N @ (0.05 * (self.rest - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo + 1e-3, self.hi - 1e-3)
        p, Rc = self.fk(q)
        return q, np.linalg.norm(pos - p), np.degrees(np.arccos(np.clip((np.trace(Rc.T @ R) - 1) / 2, -1, 1)))
