"""Convert the Pioneer v2 URDF (Prometheus repo, humanoid_urdf/pioneer_v2) for MuJoCo here: every visual mesh
decimated to a binary STL (the CAD export is ~270 MB of OBJ; this is ~20 MB), the URDF rewritten to point at them.

    python convert.py <Prometheus>/Backend/cad_bench/projects/humanoid_urdf/pioneer_v2/pioneer_humanoid_v2_mujoco.urdf

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
MIN_FACES, MAX_FACES = 300, 30_000


def main(src):
    root = os.path.dirname(src)
    text = open(src).read()
    files = sorted(set(re.findall(r'filename="(meshes/[^"]+\.obj)"', text)))
    meshes = {f: trimesh.load(os.path.join(root, f), force="mesh", process=False) for f in files}
    total = sum(len(m.faces) for m in meshes.values())
    os.makedirs(os.path.join(HERE, "meshes"), exist_ok=True)
    for f, m in meshes.items():
        # budget by area share and face share: big smooth shells need fewer triangles than small detailed parts
        target = int(np.clip(FACE_BUDGET * len(m.faces) / total, MIN_FACES, MAX_FACES))
        v, fc = np.asarray(m.vertices, np.float32), np.asarray(m.faces)
        if len(fc) > target:
            v, fc = fast_simplification.simplify(v, fc, target_count=target)
        out = f.replace(".obj", ".stl")
        trimesh.Trimesh(v, fc, process=False).export(os.path.join(HERE, out))
        text = text.replace(f'"{f}"', f'"{out}"')
        print(f"{f}: {len(m.faces)} -> {len(fc)}")
    text = text.replace("<!-- written by make_urdf.py: edit that, not this file -->",
                        "<!-- converted by convert.py from Prometheus' make_urdf.py output: rerun convert.py, don't edit -->")
    open(os.path.join(HERE, "pioneer_humanoid_v2.urdf"), "w").write(text)


if __name__ == "__main__":
    main(sys.argv[1])
