from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yourdfpy

# Find the repo's legs URDF relative to THIS file, so it works on any computer
# and keeps working if the folder moves (as long as it stays inside the repo).
REPO = next(p for p in Path(__file__).resolve().parents if (p / "assets").is_dir())
URDF_PATH = REPO / "assets" / "whole_body_humanoid" / "whole_body_humanoid.urdf"
MESH_DIR = URDF_PATH.parent / "meshes"


def find_mesh(fname):
    """The URDF says '../meshes/X.STL', which points to the wrong folder.
    Ignore that and look for X.STL in the meshes folder next to the URDF."""
    return str(MESH_DIR / Path(fname).name)


robot = yourdfpy.URDF.load(
    URDF_PATH,
    filename_handler=find_mesh,
    build_collision_scene_graph=True,
    load_collision_meshes=True,
)

# ---------- What's in the URDF ----------
print("links: ", list(robot.link_map))
print("joints:", robot.actuated_joint_names)

# ---------- Forward kinematics ----------
# Joint angles (radians) in the same order as robot.actuated_joint_names.
cfg = np.zeros(len(robot.actuated_joint_names))
cfg[robot.actuated_joint_names.index("Knee_L")] = np.radians(-50)   # bend the left knee
robot.update_cfg(cfg)

for link in ["base", "Thigh_L", "Calf_L", "Foot_L", "Foot_R"]:
    T_base_link = robot.get_transform(link, "base")      # pose of the link in the base frame
    print(f"{link:8s} position (m): {np.round(T_base_link[:3, 3], 3)}")

# ---------- Plot the robot's collision meshes as a point cloud ----------
# to_geometry() gives all collision meshes, already moved to their FK poses, in the base frame.
legs = robot.collision_scene.to_geometry()
pts = legs.sample(5000)   # random points on the mesh surfaces: like sensor hits on the robot

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_aspect("equal")
plt.show()
