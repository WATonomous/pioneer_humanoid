import numpy as np
import matplotlib.pyplot as plt

# A random number generator with a fixed seed: same "random" points every run,
# so results are repeatable.
rng = np.random.default_rng(seed=0)

# ---- Floor: 2000 points scattered on a 2 m x 2 m square at height z = 0 ----
n_floor = 2000
floor = np.column_stack([
    rng.uniform(-1.0, 1.0, n_floor),   # x: anywhere from -1 to 1 m
    rng.uniform(-1.0, 1.0, n_floor),   # y: anywhere from -1 to 1 m
    rng.normal(0.0, 0.005, n_floor),   # z: about 0, with 5 mm of sensor noise
])

# ---- Leg: 500 points on the surface of an upright cylinder ----
n_leg = 500
leg_radius = 0.06      # 6 cm
leg_height = 0.8       # 80 cm tall
leg_x, leg_y = 0.3, 0.0  # where the leg stands

angle = rng.uniform(0, 2 * np.pi, n_leg)          # a random angle around the cylinder
leg = np.column_stack([
    leg_x + leg_radius * np.cos(angle),            # circle in x...
    leg_y + leg_radius * np.sin(angle),            # ...and y
    rng.uniform(0.0, leg_height, n_leg),           # any height up the leg
])

# ---- Combine into one cloud, like the sensor would give us ----
points = np.vstack([floor, leg])
# Ground-truth labels (only possible with fake data): True = this point is really leg.
is_leg = np.concatenate([np.zeros(n_floor, dtype=bool), np.ones(n_leg, dtype=bool)])

print("cloud shape:", points.shape)

# ---- Plot in 3D ----
# ---- Self-filter: remove points inside the leg (+ padding) ----
padding = 0.10   # 3 cm safety margin around the leg

# Horizontal distance of every point from the leg's centre line (Pythagoras).
dx = points[:, 0] - leg_x
dy = points[:, 1] - leg_y
dist_from_axis = np.sqrt(dx**2 + dy**2)

# A "mask": one True/False per point. True = this point is inside the padded leg.
inside = (
    (dist_from_axis < leg_radius + padding)
    & (points[:, 2] > 0.0 - padding)
    & (points[:, 2] < leg_height + padding)
)

kept = points[~inside]      # ~ means "not": keep points that are NOT inside
removed = points[inside]
print("kept:", len(kept), " removed:", len(removed))
leg_kept = np.sum(is_leg & ~inside)        # leg points we failed to remove -> fake obstacles
floor_removed = np.sum(~is_leg & inside)   # floor points we wrongly removed -> holes
print("leg points kept (fake obstacles):", leg_kept)
print("floor points removed (holes):   ", floor_removed)

# ---- Plot: kept points in blue, removed points in red ----
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

