"""Build the arm-swing whole-body URDF from the Oct 24 whole-body asset (git d104f229, extracted to assets/ here by make_robot.sh), with the _isaac_urdf / _bake_angle / FIST code of src/pioneer_humanoid/pioneer_humanoid/{leg_only,whole_body}.py."""
import ast, math, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..", "..", "pioneer_humanoid", "pioneer_humanoid")
ns = {"math": math, "os": os, "re": re, "HIP_HEIGHT": 0.847917,
      "_ASSET_DIR": os.path.join(HERE, "assets", "pioneer_leg_only_humanoid"), "_OUT_DIR": os.path.join(HERE, "pioneer_leg_only_humanoid")}
for f, names in (("leg_only.py", ("_isaac_urdf", "_bake_angle")), ("whole_body.py", ("FIST",))):
    tree = ast.parse(open(f"{PKG}/{f}").read())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names or (isinstance(n, ast.Assign) and n.targets[0].id in names if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else False)]
    exec(compile(ast.Module(nodes, []), f, "exec"), ns)
print(ns["_isaac_urdf"]("pioneer_whole_body_humanoid", "pioneer_whole_body_humanoid.urdf", legs_only=True,
                        keep=("l_shoulder_pitch", "r_shoulder_pitch", "l_elbow", "r_elbow"), name="pioneer_whole_body_humanoid_arm_swing", pose=ns["FIST"]))
