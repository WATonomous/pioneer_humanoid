import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)


# ---------- Transform helpers ----------

def rot_y(angle):
    """Rotation matrix: turn by `angle` (radians) around the y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def make_T(R, t):
    """Pack a 3x3 rotation R and a position t into one 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    """Inverse transform: undoes T (rotate back, then move back)."""
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T, pts):
    """Apply transform T to every point (rows of pts): p_new = R @ p + t."""
    # einsum = "for each point n: R @ p_n". (Same as pts @ R.T, but avoids a
    # false-alarm warning from numpy's math library on Apple Silicon Macs.)
    return np.einsum("ij,nj->ni", T[:3, :3], pts) + T[:3, 3]


def inside_cylinder(pts, radius, height, padding):
    """Points given in the LEG's own frame: upright cylinder from z=0 to z=height."""
    dist_from_axis = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    return ((dist_from_axis < radius + padding)
            & (pts[:, 2] > 0.0 - padding)
            & (pts[:, 2] < height + padding))


# ---------- Fake world ----------

n_floor = 2000
floor = np.column_stack([
    rng.uniform(-1.0, 1.0, n_floor),
    rng.uniform(-1.0, 1.0, n_floor),
    rng.normal(0.0, 0.005, n_floor),
])

# Leg in its OWN frame: upright cylinder standing at the origin. Never changes.
n_leg = 500
leg_radius = 0.06
leg_height = 0.8
angle = rng.uniform(0, 2 * np.pi, n_leg)
leg_in_leg_frame = np.column_stack([
    leg_radius * np.cos(angle),
    leg_radius * np.sin(angle),
    rng.uniform(0.0, leg_height, n_leg),
])

# Where the leg REALLY is right now: foot at (0.3, 0, 0), tilted 30 degrees.
true_tilt = np.radians(30)
T_world_leg_true = make_T(rot_y(true_tilt), [0.3, 0.0, 0.0])
leg = transform_points(T_world_leg_true, leg_in_leg_frame)

points = np.vstack([floor, leg])
is_leg = np.concatenate([np.zeros(n_floor, dtype=bool), np.ones(n_leg, dtype=bool)])


# ---------- Self-filter ----------

# Where the robot BELIEVES the leg is (from joint angles). Try making this wrong!
believed_tilt = np.radians(20)
T_world_leg = make_T(rot_y(believed_tilt), [0.3, 0.0, 0.0])

# Move all points into the leg's frame, then use the simple upright test.
points_in_leg_frame = transform_points(inv_T(T_world_leg), points)
inside = inside_cylinder(points_in_leg_frame, leg_radius, leg_height, padding=0.03)

kept, removed = points[~inside], points[inside]
print("kept:", len(kept), " removed:", len(removed))
print("leg points kept (fake obstacles):", np.sum(is_leg & ~inside))
print("floor points removed (holes):   ", np.sum(~is_leg & inside))


# ---------- Plot ----------

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(kept[:, 0], kept[:, 1], kept[:, 2], s=1, c="tab:blue", label="kept")
ax.scatter(removed[:, 0], removed[:, 1], removed[:, 2], s=3, c="tab:red", label="removed")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_aspect("equal")
ax.legend()
plt.show()