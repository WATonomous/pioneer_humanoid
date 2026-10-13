import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)


# ---------- Transform helpers (same as before) ----------

def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T, pts):
    return np.einsum("ij,nj->ni", T[:3, :3], pts) + T[:3, 3]


# ---------- Leg model (what a URDF would tell us) ----------

hip_height = 0.75   # hip joint is 75 cm above the floor
thigh_len = 0.4
calf_len = 0.4
radius = 0.06


def forward_kinematics(hip_angle, knee_angle):
    """Joint angles in -> pose of each link out (in the world frame).

    Each link's own frame sits at the joint at its top, with the link
    hanging DOWN along its -z axis.
    """
    T_world_thigh = make_T(rot_y(hip_angle), [0.0, 0.0, hip_height])
    T_thigh_calf = make_T(rot_y(knee_angle), [0.0, 0.0, -thigh_len])   # knee = bottom of thigh
    T_world_calf = T_world_thigh @ T_thigh_calf                         # chain them!
    return T_world_thigh, T_world_calf


def link_points(n, length):
    """Fake sensor hits on a hanging cylinder, in the link's own frame."""
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([radius * np.cos(a),
                            radius * np.sin(a),
                            rng.uniform(-length, 0.0, n)])


def inside_hanging_cylinder(pts, length, padding):
    """Points in the link's own frame: cylinder from z=0 down to z=-length."""
    d = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    return ((d < radius + padding)
            & (pts[:, 2] < 0.0 + padding)
            & (pts[:, 2] > -length - padding))


# ---------- Fake world: floor + a bent leg ----------

n_floor = 2000
floor = np.column_stack([
    rng.uniform(-1.0, 1.0, n_floor),
    rng.uniform(-1.0, 1.0, n_floor),
    rng.normal(0.0, 0.005, n_floor),
])

# The leg's REAL joint angles right now (mid-step: thigh forward, knee bent).
true_hip = np.radians(-30)
true_knee = np.radians(60)
T_world_thigh_true, T_world_calf_true = forward_kinematics(true_hip, true_knee)

thigh = transform_points(T_world_thigh_true, link_points(300, thigh_len))
calf = transform_points(T_world_calf_true, link_points(300, calf_len))

points = np.vstack([floor, thigh, calf])
is_leg = np.concatenate([np.zeros(n_floor, dtype=bool), np.ones(600, dtype=bool)])

foot = transform_points(T_world_calf_true, np.array([[0.0, 0.0, -calf_len]]))[0]
print("foot position (m):", np.round(foot, 3))


# ---------- Self-filter: check every link ----------

# What the robot BELIEVES the joint angles are. Try making one of them wrong!
believed_hip = np.radians(-30)
believed_knee = np.radians(60)
T_world_thigh, T_world_calf = forward_kinematics(believed_hip, believed_knee)

padding = 0.03
in_thigh = inside_hanging_cylinder(transform_points(inv_T(T_world_thigh), points), thigh_len, padding)
in_calf = inside_hanging_cylinder(transform_points(inv_T(T_world_calf), points), calf_len, padding)
inside = in_thigh | in_calf      # | means "or": inside ANY link gets removed

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
