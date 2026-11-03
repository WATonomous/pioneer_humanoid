"""Checks the new forearm + wrist gripper (step_to_links.py) left the arm working as before:

  1. joints 1-5 and the inertias of links base..link4 are exactly as before the swap (the URDF at REF), and their
     meshes (now from the CAD, in colour) lie on the old ones; joint6 is the old axis
     turned 90 deg about joint5 with the new zero, through the joint5 axis;
  2. MuJoCo loads the URDF;
  3. the zero pose and the default pose (arm_params.DEFAULT_JOINT_POS) are inside every limit and close the
     fingers side to side (along base Y, joint7 the outer finger), as the old gripper did;
  4. each gripper opens 82 mm and closes to touching, never overlapping on the way;
  5. at the default pose, open and closed, no link touches another it is not jointed to;
  6. the left arm mirrors the right (link poses across the XZ plane);
and reports how far joint6 turns either way from the default pose before the gripper meets the arm.

    python check_new_gripper.py [REF]          # REF: a git commit with the old gripper, default 5244df7

Needs numpy, trimesh, python-fcl and mujoco.
"""
import os
import re
import subprocess
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "urdf", "pioneer_bimanual_arm.urdf")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "pioneer_humanoid"))
from pioneer_humanoid.arm_params import (DEFAULT_JOINT_POS, LEFT_GRIPPER_CLOSED,  # noqa: E402
                                         RIGHT_GRIPPER_CLOSED)

SIDES = {"right": dict(j=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"],
                       closed=RIGHT_GRIPPER_CLOSED),
         "left": dict(j=["joint1L", "joint2l", "joint3l", "joint4l", "joint5l", "joint6l", "joint7l", "joint8l"],
                      closed=LEFT_GRIPPER_CLOSED)}
failures = []


def check(ok, what):
    print(("  ok    " if ok else "  FAIL  ") + what)
    if not ok:
        failures.append(what)


def blocks(text, tag):
    return {m.group(1): m.group(0) for m in re.finditer(r"<%s\s+name=\"([^\"]+)\"[\s\S]*?</%s>" % (tag, tag), text)}


def parse(text):
    joints = {}
    for name, b in blocks(text, "joint").items():
        def g(pat, b=b):
            return re.search(pat, b).group(1)
        lim = re.search(r"lower=\"([^\"]+)\"\s+upper=\"([^\"]+)\"", b)
        joints[name] = dict(type=g(r"type=\"([^\"]+)\""), parent=g(r"<parent\s+link=\"([^\"]+)\""),
                            child=g(r"<child\s+link=\"([^\"]+)\""),
                            xyz=np.array([float(v) for v in g(r"<origin\s+xyz=\"([^\"]+)\"").split()]),
                            axis=np.array([float(v) for v in g(r"<axis\s+xyz=\"([^\"]+)\"").split()]),
                            lim=(float(lim.group(1)), float(lim.group(2))))
    return joints


def rot(axis, a):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def fk(joints, q):
    """{link: 4x4 in base_link} (every joint frame of this URDF is axis-aligned with its parent)"""
    by_child = {j["child"]: (n, j) for n, j in joints.items()}
    out = {}

    def T(link):
        if link in out:
            return out[link]
        if link not in by_child:
            out[link] = np.eye(4)
            return out[link]
        n, j = by_child[link]
        M = np.eye(4)
        M[:3, 3] = j["xyz"]
        D = np.eye(4)
        if j["type"] == "revolute":
            D[:3, :3] = rot(j["axis"], q.get(n, 0.0))
        elif j["type"] == "prismatic":
            D[:3, 3] = j["axis"] * q.get(n, 0.0)
        out[link] = T(j["parent"]) @ M @ D
        return out[link]
    for j in joints.values():
        T(j["child"])
    return out


def meshes(text, meshdir=os.path.join(HERE, "meshes"), only=None):
    """each link's collision mesh in its frame"""
    out = {}
    for name, b in blocks(text, "link").items():
        if only and name not in only:
            continue
        c = re.search(r"<collision>[\s\S]*?</collision>", b).group(0)
        f = re.search(r"filename=\"[^\"]*/meshes/([^\"]+)\"", c).group(1)
        o = np.array([float(v) for v in re.search(r"<origin\s+xyz=\"([^\"]+)\"", c).group(1).split()])
        m = trimesh.load(os.path.join(meshdir, f), force="mesh")
        m.apply_translation(o)
        out[name] = m
    return out


def touching(mesh, joints, q, skip_adjacent=True):
    """pairs of links in contact at pose q, links jointed to each other left out"""
    poses = fk(joints, q)
    adj = {frozenset((j["parent"], j["child"])) for j in joints.values()}
    adj |= {frozenset((a["child"], b["child"])) for a in joints.values() for b in joints.values()
            if a["parent"] == b["parent"] and a is not b}       # siblings sharing a parent (the fingers, the arms)
    cm = trimesh.collision.CollisionManager()
    for n, m in mesh.items():
        cm.add_object(n, m, poses.get(n, np.eye(4)))
    _, pairs = cm.in_collision_internal(return_names=True)
    return sorted(tuple(sorted(p)) for p in pairs if not (skip_adjacent and frozenset(p) in adj))


def main(ref="5244df7"):
    text = open(URDF).read()
    joints = parse(text)
    print("1. joints 1-5 and links base..link4 unchanged (vs %s)" % ref)
    old = subprocess.run(["git", "show", f"{ref}:assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm.urdf"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    for side in SIDES.values():
        for j in side["j"][:5]:
            check(blocks(old, "joint")[j] == blocks(text, "joint")[j], f"{j} identical")
    oj, z0 = parse(old), fk(joints, {})
    for side in SIDES.values():
        j5, j6 = side["j"][4], side["j"][5]
        a5 = z0[joints[j5]["child"]][:3, :3] @ joints[j5]["axis"]
        a6 = z0[joints[j6]["child"]][:3, :3] @ joints[j6]["axis"]
        p5, p6 = z0[joints[j5]["child"]][:3, 3], z0[joints[j6]["child"]][:3, 3]
        turn = -90.0 if j5 == "joint5" else 90.0
        old_a6 = oj[j6]["axis"]
        c, s_ = np.cos(np.radians(turn)), np.sin(np.radians(turn))
        want = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ old_a6
        check(np.allclose(a6, want), f"{j6} axis {a6} = the old {old_a6} turned {turn:+.0f} deg about {j5}")
        check(np.allclose(p6[:2], p5[:2], atol=1e-4) and abs(np.dot(a5, a6)) < 1e-9,
              f"{j6} crosses the {j5} axis at right angles")
    # base_link..link4: their meshes are rebuilt from the CAD (in colour), the same parts: inertias unchanged, and
    # the old mesh's surface on the new one
    import tempfile
    oldlinks = ["base_link", "link1", "link2", "link3", "link4", "link1L", "link2l", "link3l", "link4l"]
    with tempfile.TemporaryDirectory() as tmp:
        for link in oldlinks:
            f = re.search(r"filename=\"[^\"]*/meshes/([^\"]+)\"", blocks(old, "link")[link]).group(1)
            data = subprocess.run(["git", "show", f"{ref}:assets/pioneer_bimanual_arm/meshes/{f}"], cwd=REPO,
                                  capture_output=True, check=True).stdout
            open(os.path.join(tmp, f), "wb").write(data)
        old_mesh = meshes(old, tmp, oldlinks)
    new_mesh = {}                                                # the detailed (visual) meshes, all colours
    for link in oldlinks:
        fs = re.findall(r"<visual>[\s\S]*?filename=\"[^\"]*/meshes/([^\"]+)\"", blocks(text, "link")[link])
        new_mesh[link] = trimesh.util.concatenate([trimesh.load(os.path.join(HERE, "meshes", f), force="mesh") for f in fs])
    for link in oldlinks:
        inert = re.compile(r"<inertial>[\s\S]*?</inertial>")
        check(inert.search(blocks(old, "link")[link]).group(0) == inert.search(blocks(text, "link")[link]).group(0),
              f"{link} inertia identical")
        p, _ = trimesh.sample.sample_surface(old_mesh[link], 2000, seed=0)
        d = np.abs(trimesh.proximity.ProximityQuery(new_mesh[link]).signed_distance(p)) * 1000
        check(np.percentile(d, 95) < 1.5, f"{link} old surface on the new mesh: median {np.median(d):.2f}, "
                                          f"95% {np.percentile(d, 95):.2f} mm")

    print("2. MuJoCo loads it (mujoco_bimanual_arm.arm_spec, as the sim does)")
    try:
        from pioneer_humanoid.mujoco_bimanual_arm import arm_spec
        m = arm_spec().compile()
        check(m.njnt == 16, f"MuJoCo model: {m.njnt} joints, {m.nbody} bodies, {m.nu} actuators")
    except Exception as e:                                     # noqa: BLE001
        check(False, f"MuJoCo load: {e}")

    print("3. the zero pose and the default pose")
    q0 = dict(DEFAULT_JOINT_POS)
    for n, v in q0.items():
        lo, hi = joints[n]["lim"]
        check(lo <= v <= hi, f"{n} = {np.degrees(v) if joints[n]['type'] == 'revolute' else v:+.1f} in [{lo}, {hi}]")
    for label, q in (("zero", {}), ("default", q0)):
        poses = fk(joints, q)
        for side, s in SIDES.items():
            j7 = joints[s["j"][6]]
            d = poses[j7["parent"]][:3, :3] @ j7["axis"]
            check(abs(abs(d[1]) - 1) < 1e-6, f"{label}: {side} fingers travel along base {np.round(d, 3)} (side to side)")
            check(np.allclose(d, oj[s["j"][6]]["axis"]), f"{label}: {side} joint7 axis = the old gripper's")
            a7, a8 = poses[j7["child"]][:3, 3], poses[joints[s["j"][7]]["child"]][:3, 3]
            check(abs(a7[1]) > abs(a8[1]), f"{label}: {side} joint7's finger is the outer one, as before")
    poses = fk(joints, q0)

    print("4. the grippers open and close")
    mesh = meshes(text)
    for side, s in SIDES.items():
        j7, j8 = s["j"][6], s["j"][7]
        f7, f8 = joints[j7]["child"], joints[j8]["child"]
        gaps = []
        for t in np.linspace(0, 1, 11):
            q = dict(q0)
            q[j7], q[j8] = t * s["closed"][j7], t * s["closed"][j8]
            p = fk(joints, q)
            cm = trimesh.collision.CollisionManager()
            cm.add_object(f7, mesh[f7], p[f7])
            gaps.append(cm.min_distance_single(mesh[f8], p[f8]))
        check(abs(gaps[0] - 0.0821) < 1e-3, f"{side} open: {gaps[0] * 1000:.1f} mm between the fingers")
        check(all(g > 0 for g in gaps) and gaps[-1] < 1e-3,
              f"{side} closing: never overlapping, closed {gaps[-1] * 1000:.2f} mm apart")
        check(all(a > b for a, b in zip(gaps, gaps[1:])), f"{side} closing: the gap shrinks all the way")

    print("5. no link touches another at the default pose")
    for label, grip in (("open", 0.0), ("closed", 1.0)):
        q = dict(q0)
        for s in SIDES.values():
            for j in s["j"][6:]:
                q[j] = grip * s["closed"][j]
        pairs = touching(mesh, joints, q)
        check(not pairs, f"{label}: {pairs or 'no contacts'}")

    print("6. left mirrors right")
    M = np.diag([1.0, -1.0, 1.0])
    for r, l in zip(SIDES["right"]["j"], SIDES["left"]["j"]):
        cr, cl = joints[r]["child"], joints[l]["child"]
        pr, pl = poses[cr][:3, 3], poses[cl][:3, 3]
        if r in ("joint5",):                                    # joint5l's frame sits 61.7 mm lower on the same axis:
            z0 = fk(joints, {})                                 # compare the vertical axes at zero, by X and Y
            pr, pl = z0[cr][:3, 3][:2], z0[cl][:3, 3][:2]
            check(np.allclose(pr * [1, -1], pl, atol=2e-3), f"{cr} / {cl} on mirrored axes")
            continue
        if r in ("joint1", "joint2"):                           # the exporter's frames differ along their own axes
            continue
        check(np.allclose(M @ pr, pl, atol=2e-3), f"{cr} / {cl} mirrored ({np.round(M @ pr - pl, 4)} m)")
    for k in ("link5", "link6", "link7", "link8"):
        kl = k + "l"
        a = trimesh.Trimesh(mesh[k].vertices @ M, mesh[k].faces)
        b = mesh[kl]
        d = max(abs(a.bounds - (b.bounds - b.bounds.mean(0) + a.bounds.mean(0))).max(), 0)
        check(np.allclose(a.extents, b.extents, atol=1e-3), f"{k} / {kl} meshes mirror (extents within {d * 1000:.2f} mm)")

    print("joint6 from the default pose (fingers open): free turn before the gripper meets the arm")
    for side, s in SIDES.items():
        j6, j7, j8 = s["j"][5], s["j"][6], s["j"][7]
        lo, hi = joints[j6]["lim"]
        moving = [joints[j]["child"] for j in (j6, j7, j8)]
        base = fk(joints, q0)
        cm = trimesh.collision.CollisionManager()                # what joint6 does not move, its own link5 aside
        for n, m in mesh.items():
            if n not in moving and n != joints[j6]["parent"]:
                cm.add_object(n, m, base.get(n, np.eye(4)))
        free = {}
        for sign in (1, -1):
            a = 0.0
            for step in np.radians(np.arange(5, 361, 5)):
                q = dict(q0)
                q[j6] = q0[j6] + sign * step
                p = fk(joints, q)
                if not lo <= q[j6] <= hi or any(cm.in_collision_single(mesh[n], p[n]) for n in moving):
                    break
                a = step
            free[sign] = np.degrees(a)
        print(f"  {j6}: {-free[-1]:+.0f} .. {free[1]:+.0f} deg (5 deg steps; link5 itself not tested)")

    print("\nALL PASS" if not failures else f"\n{len(failures)} FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
