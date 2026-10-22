"""Convert the Pioneer Whole Body Humanoid URDF (Prometheus repo, humanoid_urdf/pioneer_v2) for MuJoCo here: every visual mesh
decimated to a binary STL (the CAD export is ~270 MB of OBJ; this is ~32 MB), the URDF rewritten to point at them.

    python convert.py <Prometheus>/Backend/cad_bench/projects/humanoid_urdf/pioneer_v2/pioneer_humanoid_v2_mujoco.urdf
    python convert.py <Prometheus>/.../pioneer_v2_gripper/pioneer_humanoid_v2_gripper_mujoco.urdf pioneer_whole_body_humanoid_gripper

The second form writes another model beside the first (here: the arm's own gripper in place of the hands). A mesh
already in meshes/ is kept, not decimated again: the two models share the head, torso and legs, and those come out of
make_urdf.py the same for both, so only the new links' meshes are added.

Collisions, joints, limits and inertias are copied unchanged.
"""
import os
import re
import sys

import fast_simplification
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_BUDGET = 400_000          # whole robot
MIN_FACES, MAX_FACES = 300, 60_000
KEEP_AREA = 0.98            # a decimated mesh must keep this share of the surface (less: it has holes)


def main(src, name="pioneer_whole_body_humanoid"):
    root = os.path.dirname(src)
    text = open(src).read()
    files = sorted(set(re.findall(r'filename="(meshes/[^"]+\.obj)"', text)))
    meshes = {f: trimesh.load(os.path.join(root, f), force="mesh", process=False) for f in files}
    total = sum(len(m.faces) for m in meshes.values())
    os.makedirs(os.path.join(HERE, "meshes"), exist_ok=True)
    for f, m in meshes.items():
        out = f.replace(".obj", ".stl")
        if name != "pioneer_whole_body_humanoid" and os.path.exists(os.path.join(HERE, out)):
            text = text.replace(f'"{f}"', f'"{out}"')       # (shared with the first model: kept as it is)
            continue
        # budget by area share and face share: big smooth shells need fewer triangles than small detailed parts
        target = int(np.clip(FACE_BUDGET * len(m.faces) / total, MIN_FACES, MAX_FACES))
        m.merge_vertices()             # the OBJ splits vertices at every normal seam: unmerged, the decimator
        v, fc = np.asarray(m.vertices, np.float32), np.asarray(m.faces)   # opens the seams (see-through legs)
        while len(fc) > target:
            dv, df = fast_simplification.simplify(v, fc, target_count=target)
            if trimesh.Trimesh(dv, df, process=False).area > KEEP_AREA * m.area:
                v, fc = dv, df
                break
            if target >= MAX_FACES:           # it can't be thinned without opening holes: keep it whole
                break
            target = min(2 * target, MAX_FACES)   # it ate surface (thin plates, small holes): keep more triangles
        trimesh.Trimesh(v, fc, process=False).export(os.path.join(HERE, out))
        text = text.replace(f'"{f}"', f'"{out}"')
        print(f"{f}: {len(m.faces)} -> {len(fc)}")
    text = text.replace("<!-- written by make_urdf.py: edit that, not this file -->",
                        "<!-- converted by convert.py from Prometheus' make_urdf.py output: rerun convert.py, don't edit -->")
    text = re.sub(r'<robot name="[^"]*">', f'<robot name="{name}">', text, count=1)
    open(os.path.join(HERE, f"{name}.urdf"), "w").write(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
