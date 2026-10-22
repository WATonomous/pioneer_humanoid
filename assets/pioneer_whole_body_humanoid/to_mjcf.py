"""A converted URDF here -> a MuJoCo model to open in the viewer: the robot standing with its soles on a floor, in its
CAD colours (MuJoCo's URDF reader drops them), the collision boxes hidden (group 3) and one position actuator a joint
(a joint that follows another, as a finger's DIP its PIP, has none), so each joint can be moved from the viewer's
control panel. The base is fixed: nothing here walks or balances.

    python to_mjcf.py pioneer_whole_body_humanoid.urdf             # -> pioneer_whole_body_humanoid.xml (hands)
    python to_mjcf.py pioneer_whole_body_humanoid_gripper.urdf     # -> pioneer_whole_body_humanoid_gripper.xml
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))


def main(urdf):
    path = os.path.join(HERE, urdf)
    text = open(path).read()
    colours = {}                                   # each visual mesh's colour, as the URDF gives it
    for vis in ET.fromstring(text).iter("visual"):
        mesh, col = vis.find("geometry/mesh"), vis.find("material/color")
        if mesh is not None and col is not None:
            colours[os.path.splitext(os.path.basename(mesh.get("filename")))[0]] = [float(x) for x in col.get("rgba").split()]
    if "<mujoco>" not in text:                     # (convert.py's output already carries it)
        text = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler meshdir="." balanceinertia="true" discardvisual="false"/></mujoco>',
                      text, count=1)
    spec = mujoco.MjSpec.from_string(text)
    spec.modelname = os.path.splitext(urdf)[0]
    spec.meshdir = HERE
    spec.compiler.degree = False
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.ambient = [0.12, 0.12, 0.13]
    spec.visual.headlight.diffuse = [0.22, 0.22, 0.22]
    mats = {}
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:   # the CAD's visuals: seen, not touched
            rgba = colours.get(g.meshname, [0.8, 0.8, 0.82, 1.0])
            key = tuple(round(c, 3) for c in rgba)
            if key not in mats:
                mats[key] = f"cad{len(mats)}"
                spec.add_material(name=mats[key], rgba=list(key), specular=0.25, shininess=0.35)
            g.material, g.group, g.contype, g.conaffinity, g.density = mats[key], 1, 0, 0, 0
        else:                                      # the collision boxes: hidden
            g.group, g.rgba = 3, [0.9, 0.3, 0.3, 0.3]
    world = spec.worldbody
    lift = -_lowest(spec)                          # the soles on the floor (z = 0), not 2 cm through it
    for b in world.bodies:
        b.pos = [b.pos[0], b.pos[1], b.pos[2] + lift]
    for g in world.geoms:                          # (the base link's own geoms sit on the world body: a fixed base)
        g.pos = [g.pos[0], g.pos[1], g.pos[2] + lift]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.63, 0.66], rgb2=[0.3, 0.31, 0.33], width=256, height=1536)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.47, 0.47, 0.48], rgb2=[0.43, 0.43, 0.44], width=512, height=512)
    floor = spec.add_material(name="floor", texrepeat=[10, 10], reflectance=0.03)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="floor")
    world.add_light(pos=[1.5, -2.0, 3.0], dir=[-0.4, 0.5, -1.0], diffuse=[0.6, 0.6, 0.6], castshadow=True)
    world.add_light(pos=[-1.5, -1.0, 2.5], dir=[0.5, 0.3, -1.0], diffuse=[0.3, 0.3, 0.3])
    for b in spec.bodies:                          # gravity compensated: the joints hold where they're put
        b.gravcomp = 1.0
    mimics = {j.get("name") for j in ET.fromstring(open(path).read()).iter("joint") if j.find("mimic") is not None}
    for j in spec.joints:                          # one position actuator a joint, over its range (not a joint that
        if j.name in mimics:                       # follows another, as a finger's DIP its PIP: it has none)
            continue
        slide = j.type == mujoco.mjtJoint.mjJNT_SLIDE
        kp = 200.0 if slide else 40.0
        a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=kp, kv=0.1 * kp)
        a.ctrllimited, a.ctrlrange = True, list(j.range)
        j.damping = [2.0 if slide else 1.0, 0.0, 0.0]
        j.armature = 0.01
    spec.compile()
    spec.meshdir = "."                             # (the file sits beside meshes/: no absolute path in it)
    out = os.path.splitext(path)[0] + ".xml"
    open(out, "w").write(spec.to_xml())
    print("wrote", out)


def _lowest(spec):
    """the lowest point of the robot's meshes in its zero pose (m)"""
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    low = 0.0
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            low = min(low, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
    return low


if __name__ == "__main__":
    main(sys.argv[1])
