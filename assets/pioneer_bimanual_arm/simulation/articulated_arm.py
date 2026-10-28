"""CAD-derived articulation and exact planar gripper linkage kinematics.

Angles are relative to the delivered CAD pose. Limits are configurable preview
ranges, not measured hardware stops. Motor/gear masses are not calibrated.
"""
from pathlib import Path
import json
import math
import re
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT/'arm_v2_updated_description'
PKG = ROOT
# Preserve the repository's forward-X, left-Y, shoulder-height base frame.
Q = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
BASE_OFFSET = np.array([.0215734774382594, 0., -1.179518070794978])
Z = .007020842383436
PIVOTS = [[.163325, 1.24, 0.], [.279325, 1.24, Z], [.279325, 1.1237000008028, Z],
          [.2815750008028, .9391000008028, Z], [.2815750008028, .8316000016056, Z],
          [.2885750008028, .7036000016056, Z]]
AXES = [[1, 0, 0], [0, 0, 1], [0, -1, 0], [1, 0, 0], [0, -1, 0], [1, 0, 0]]
LABELS = ['Shoulder pitch', 'Shoulder roll', 'Upper-arm twist', 'Elbow flexion', 'Forearm roll', 'Wrist pitch', 'Gripper crank']
LIMITS = [(-100, 100), (-75, 75), (-120, 120), (-110, 110), (-180, 180), (-90, 90), (-10, 90)]
# New-wrist STEP coordinates -> original right-arm CAD coordinates, in meters.
A = np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
AT = np.array([.288575000802814, .608288613304749, .00702084238342])
CRANK_PINS = [np.array([-.0259779008051355, .0125, .015004954840272]),
              np.array([.0259779008051344, .0125, -.0150049548402725])]
JAW_PINS = [np.array([-.05521109706815028, .0125, .021744408545733483]),
            np.array([.05521008397965892, .0125, -.021748801431873317])]


def fmt(v): return ' '.join(f'{float(x):.12g}' for x in v)
def mirrored(side, p): return np.diag([-1, 1, 1])@p if side == 'left' else np.array(p)
def new_point(side, p): return Q@mirrored(side, A@p+AT)+BASE_OFFSET


def new_axis(side, a):
    # Rotation axes are axial vectors: reflection contributes determinant -1.
    return (-1 if side == 'left' else 1)*Q@mirrored(side, A@np.array(a))


def linkage(theta):
    R = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    results = []
    for index, (pin, jaw) in enumerate(zip(CRANK_PINS, JAW_PINS)):
        p = R@pin
        v0 = jaw-pin
        length = np.linalg.norm(v0[[0, 2]])
        dz = jaw[2]-p[2]
        square = length*length-dz*dz
        if square < 0:
            raise ValueError('Gripper target outside linkage workspace')
        slider = p[0]+(-1 if index == 0 else 1)*math.sqrt(square)
        s = jaw.copy()
        s[0] = slider
        v = s-p
        angle = math.atan2(v[0]*v0[2]-v[2]*v0[0], v[0]*v0[0]+v[2]*v0[2])
        results.append((slider-jaw[0], angle-theta, float(np.linalg.norm(v)-length)))
    return results


def groups_and_joints():
    groups = {'stand': np.zeros(3)}
    joints = []
    for side in ['right', 'left']:
        previous = 'stand'
        for i, (p, axis) in enumerate(zip(PIVOTS, AXES), 1):
            key = f'{side}_link{i}'
            groups[key] = Q@mirrored(side, np.array(p))+BASE_OFFSET
            direction = Q@mirrored(side, np.array(axis))*(-1 if side == 'left' else 1)
            joints.append(dict(name=f'{side}_joint{i}', parent=previous, child=key, type='revolute',
                               axis=direction, limit=np.deg2rad(LIMITS[i-1])))
            previous = key
        crank = f'{side}_crank'
        groups[crank] = new_point(side, np.zeros(3))
        joints.append(dict(name=f'{side}_gripper', parent=previous, child=crank, type='revolute',
                           axis=new_axis(side, [0, 1, 0]), limit=np.deg2rad(LIMITS[6])))
        for k in range(2):
            jaw = f'{side}_jaw{k}'
            bar = f'{side}_bar{k}'
            groups[jaw] = new_point(side, JAW_PINS[k])
            groups[bar] = new_point(side, CRANK_PINS[k])
            # Slider direction is a polar vector, with no determinant factor.
            slide = Q@mirrored(side, A@np.array([1., 0, 0]))
            joints.append(dict(name=f'{side}_jaw_slide{k}', parent=previous, child=jaw, type='prismatic',
                               axis=slide, limit=[-.09, .09]))
            joints.append(dict(name=f'{side}_linkage_pin{k}', parent=crank, child=bar, type='revolute',
                               axis=new_axis(side, [0, 1, 0]), limit=[-2*math.pi, 2*math.pi]))
    return groups, joints


def assign(c):
    if c['role'] == 'retained':
        group = c['source_hierarchy'][1]
        match = re.match(r'(Mirror)?link([1-6])\b', group)
        return ('left' if match[1] else 'right')+'_link'+match[2] if match else 'stand'
    side = c['name'].split('_')[0]
    if c['role'] in ['new_forearm', 'new_cover']:
        return side+'_link5'
    index = int(re.search(r'_wrist_(\d+)', c['name'])[1])
    if index in (1, 2):
        return side+f'_bar{index-1}'
    if index == 3:
        return side+'_crank'
    if index in (4, 5, 6):
        return side+'_jaw0'
    if index in (15, 16, 17):
        return side+'_jaw1'
    return side+'_link6'


def compat_name(name):
    if name == 'stand':
        return 'base_link'
    for side in ['right', 'left']:
        suffix = '' if side == 'right' else 'l'
        if re.fullmatch(side+r'_link[1-6]', name):
            n = name.removeprefix(side+'_link')
            return 'link'+n+('L' if side == 'left' and n == '1' else suffix)
        if name.startswith(side+'_joint'):
            n = name.removeprefix(side+'_joint')
            return 'joint'+n+('L' if side == 'left' and n == '1' else suffix)
        for k in range(2):
            if name == f'{side}_jaw{k}':
                return f'link{k+7}'+suffix
            if name == f'{side}_jaw_slide{k}':
                return f'joint{k+7}'+suffix
    return name


def add_pin_constraints(spec):
    groups, _ = groups_and_joints()
    for side in ['right', 'left']:
        for k in range(2):
            eq = spec.add_equality()
            eq.name = f'{side}_jaw_pin{k}'
            eq.type = mujoco.mjtEq.mjEQ_CONNECT
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1 = compat_name(f'{side}_bar{k}')
            eq.name2 = compat_name(f'{side}_jaw{k}')
            eq.data[:3] = new_point(side, JAW_PINS[k])-groups[f'{side}_bar{k}']


def build(write_assets=False):
    manifest = json.loads((ROOT/'build_manifest.json').read_text())
    groups, joints = groups_and_joints()
    root = ET.Element('robot', name='arm_v2_articulated')
    links = {key: ET.SubElement(root, 'link', name=key) for key in groups}
    counts = {key: 0 for key in groups}
    for c in manifest['components']:
        key = assign(c)
        counts[key] += 1
        vis = ET.SubElement(links[key], 'visual', name=c['name'])
        ET.SubElement(vis, 'origin', xyz=fmt(BASE_OFFSET-groups[key]), rpy=fmt([math.pi/2, 0, -math.pi/2]))
        geom = ET.SubElement(vis, 'geometry')
        ET.SubElement(geom, 'mesh', filename='package://armv2URDF/meshes/'+c['mesh'])
        color = {'new_forearm': '0.85 0.55 0.12 1', 'new_cover': '0.75 0.6 0.3 1', 'new_wrist': '0.26 0.53 0.7 1'}.get(c['role'], '0.64 0.67 0.71 1')
        mat = ET.SubElement(vis, 'material', name='material_'+c['role'])
        ET.SubElement(mat, 'color', rgba=color)
    for j in joints:
        node = ET.SubElement(root, 'joint', name=j['name'], type=j['type'])
        ET.SubElement(node, 'parent', link=j['parent'])
        ET.SubElement(node, 'child', link=j['child'])
        ET.SubElement(node, 'origin', xyz=fmt(groups[j['child']]-groups[j['parent']]), rpy='0 0 0')
        # Preserve legacy jaw command signs: joint7 closes negative, joint8 positive.
        ET.SubElement(node, 'axis', xyz=fmt(-j['axis'] if j['type'] == 'prismatic' else j['axis']))
        ET.SubElement(node, 'limit', lower=str(j['limit'][0]), upper=str(j['limit'][1]), effort='100', velocity='1')
    ET.indent(root, space='  ')
    for node in root.findall('link'):
        node.set('name', compat_name(node.get('name')))
    for node in root.findall('joint'):
        original_name = node.get('name')
        node.set('name', compat_name(node.get('name')))
        match = re.fullmatch(r'(right|left)_joint([1-6])', original_name)
        if match:
            axes_right = [[0, 1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0], [0, 0, 1], [0, -1, 0]]
            axes_left = [[0, -1, 0], [1, 0, 0], [0, 0, -1], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
            node.find('axis').set('xyz', fmt((axes_right if match[1] == 'right' else axes_left)[int(match[2])-1]))
        for tag in ['parent', 'child']:
            node.find(tag).set('link', compat_name(node.find(tag).get('link')))
        if node.get('type') == 'prismatic':
            k = 0 if 'joint7' in node.get('name') else 1
            samples = [-linkage(math.radians(a))[k][0] for a in np.linspace(-10, 90, 201)]
            node.find('limit').set('lower', str(min(samples)-1e-6))
            node.find('limit').set('upper', str(max(samples)+1e-6))
    path = PKG/'urdf'/'pioneer_bimanual_arm.urdf'
    if write_assets:
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', '../meshes/'+Path(mesh.get('filename')).name)
    if write_assets:
        ET.ElementTree(root).write(PKG/'urdf'/'pioneer_bimanual_arm_portable.urdf', encoding='utf-8', xml_declaration=True)
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', Path(mesh.get('filename')).name)
    ext = ET.SubElement(root, 'mujoco')
    ET.SubElement(ext, 'compiler', meshdir=str(PKG/'meshes'), discardvisual='false',
                  fusestatic='false', balanceinertia='true', inertiafromgeom='true')
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
    spec.modelname = 'Articulated Arm V2 - keyboard joint control'
    for mesh in spec.meshes:
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 0
        geom.density = 1000
    spec.option.gravity[:] = 0
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            pos=[0, 0, -1.200518070795], size=[3, 3, .1], rgba=[.21, .23, .27, 1], contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[1, -2, 3], dir=[-.2, .5, -1], diffuse=[.9, .9, .9])
    spec.worldbody.add_light(pos=[-1, 1, 2], dir=[.2, -.3, -1], diffuse=[.5, .5, .5])
    # Close each real pin-to-slider connection in MJCF as well as in the
    # analytic keyboard controller. Anchor is local to the linkage body.
    add_pin_constraints(spec)
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200
    model = spec.compile()
    data = mujoco.MjData(model)
    set_controls(model, data, np.zeros(14))
    scene = ET.fromstring(spec.to_xml())
    scene.find('compiler').set('meshdir', '../meshes')
    ET.indent(scene, space='  ')
    if write_assets:
        ET.ElementTree(scene).write(ROOT/'simulation'/'articulated_arm_scene.xml', encoding='utf-8', xml_declaration=True)
    metadata = {'independent_controls': 14, 'movable_joints': 22, 'rigid_links': len(groups),
                'visual_components': 474, 'group_component_counts': counts,
                'pivot_source': 'cylindrical motor/bearing surfaces measured from original Arm V2 CAD',
                'gripper_source': '30 mm crank and 30 mm linkage geometry from dated wrist STEP',
                'limits': 'configurable preview ranges; hardware hard stops not supplied',
                'mode': 'kinematic joint control; mass, torque and contact dynamics not calibrated'}
    if write_assets:
        (ROOT/'simulation'/'articulation_manifest.json').write_text(json.dumps(metadata, indent=2))
    if write_assets:
        tips = {}
        for side in ['right', 'left']:
            for k in range(2):
                body = compat_name(f'{side}_jaw{k}')
                vertices = []
                for gid in range(model.ngeom):
                    if model.geom_bodyid[gid] != model.body(body).id:
                        continue
                    mid = model.geom_dataid[gid]
                    if mid < 0:
                        continue
                    verts = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
                    rot = np.zeros(9)
                    mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
                    vertices.append(verts@rot.reshape(3, 3).T+model.geom_pos[gid])
                points = np.concatenate(vertices)
                lowest = points[:, 2].min()
                distal = points[points[:, 2] < lowest+.002].mean(axis=0)
                distal[2] = lowest
                tips[body] = distal.tolist()
        pins = [{'side': side, 'bar': f'{side}_bar{k}', 'jaw': compat_name(f'{side}_jaw{k}'),
                 'bar_anchor': (new_point(side, JAW_PINS[k])-groups[f'{side}_bar{k}']).tolist()}
                for side in ['right', 'left'] for k in range(2)]
        config = {'finger_tip_local': tips, 'pin_connections': pins,
                  'gripper_closed': {compat_name(f'{side}_jaw_slide{k}'): -linkage(math.pi/2)[k][0]
                                     for side in ['right', 'left'] for k in range(2)}}
        (ROOT/'config'/'arm_v2_mechanism.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
        (ROOT/'config'/'joint_names_armv2URDF.yaml').write_text('controller_joint_names: ' +
                                                                json.dumps([model.joint(i).name for i in range(model.njnt)])+'\n', encoding='utf-8')
    return model, data


def set_controls(model, data, values):
    for s, side in enumerate(['right', 'left']):
        for i in range(6):
            j = model.joint(compat_name(f'{side}_joint{i+1}')).qposadr[0]
            data.qpos[j] = float(values[7*s+i])
        theta = float(values[7*s+6])
        data.qpos[model.joint(f'{side}_gripper').qposadr[0]] = theta
        for k, (shift, rotation, error) in enumerate(linkage(theta)):
            data.qpos[model.joint(compat_name(f'{side}_jaw_slide{k}')).qposadr[0]] = -shift
            data.qpos[model.joint(f'{side}_linkage_pin{k}').qposadr[0]] = rotation
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


if __name__ == '__main__':
    m, d = build(write_assets=True)
    print('Built articulated arm:', m.njnt, 'joints,', m.nq, 'coordinates,', m.ngeom, 'geometries')
