from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yourdfpy

from self_filter.body_filter import body_mask

rng = np.random.default_rng(seed=0)

# ---------- Load the legs URDF (same as real_legs_fk.py) ----------
REPO = next(p for p in Path(__file__).resolve().parents if (p / "assets").is_dir())
URDF_PATH = REPO / "assets" / "whole_body_humanoid" / "whole_body_humanoid.urdf"
MESH_DIR = URDF_PATH.parent / "meshes"


def find_mesh(fname):
    return str(MESH_DIR / Path(fname).name)


robot = yourdfpy.URDF.load(URDF_PATH, filename_handler=find_mesh,
                           build_collision_scene_graph=True, load_collision_meshes=True)


# ---------- One box per link, measured from its collision mesh ----------
def fit_box(link_name):
    """Min and max corners of the link's collision mesh, in the link's own frame."""
    corners = []
    for collision in robot.link_map[link_name].collisions:
        mesh = trimesh.load(find_mesh(collision.geometry.mesh.filename))
        if collision.origin is not None:          # mesh may be offset inside the link
            mesh.apply_transform(collision.origin)
        corners.append(mesh.bounds)               # bounds = [[min x, y, z], [max x, y, z]]
    corners = np.vstack(corners)
    return corners.min(axis=0), corners.max(axis=0)


links = [name for name, link in robot.link_map.items() if link.collisions]
link_boxes = {name: fit_box(name) for name in links}
for name, (lo, hi) in link_boxes.items():
    print(f"{name:10s} box size (m): {np.round(hi - lo, 3)}")


# ---------- Pose the robot: bend the left knee ----------
cfg = np.zeros(len(robot.actuated_joint_names))
cfg[robot.actuated_joint_names.index("Knee_L")] = np.radians(-30)
robot.update_cfg(cfg)
link_poses = {name: robot.get_transform(name, "base") for name in links}


# ---------- Fake sensor data: robot surface + floor ----------
robot_pts = robot.collision_scene.to_geometry().sample(3000)   # hits on the robot itself
floor_z = robot_pts[:, 2].min()                                 # floor at the lowest foot point
n_floor = 3000
floor_pts = np.column_stack([
    rng.uniform(-1.0, 1.0, n_floor),
    rng.uniform(-1.0, 1.0, n_floor),
    floor_z + rng.normal(0.0, 0.005, n_floor),
])
points = np.vstack([robot_pts, floor_pts])
is_robot = np.concatenate([np.ones(len(robot_pts), bool), np.zeros(n_floor, bool)])


# ---------- Run the filter ----------
inside = body_mask(points, link_poses, link_boxes, padding=0.02)
print("robot points kept (fake obstacles):", np.sum(is_robot & ~inside))
print("floor points removed (holes):      ", np.sum(~is_robot & inside))


# ---------- Plot ----------
kept, removed = points[~inside], points[inside]
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(kept[:, 0], kept[:, 1], kept[:, 2], s=1, c="tab:blue", label="kept")
ax.scatter(removed[:, 0], removed[:, 1], removed[:, 2], s=1, c="tab:red", label="removed")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_aspect("equal")
ax.legend()
plt.show()
