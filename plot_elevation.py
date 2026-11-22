import json
import numpy as np
import matplotlib.pyplot as plt

map_path = "terrain mapping bags/elevation.npy"
metadata_path = "terrain mapping bags/metadata.json"

heights = np.load(map_path)

with open(metadata_path) as f:
    metadata = json.load(f)

resolution = metadata["resolution"]
rows, cols = heights.shape

# GridMap position is the map centre.
center_x = metadata["position"]["x"]
center_y = metadata["position"]["y"]

x_min = center_x - metadata["length_x"] / 2
x_max = center_x + metadata["length_x"] / 2
y_min = center_y - metadata["length_y"] / 2
y_max = center_y + metadata["length_y"] / 2

cmap = plt.cm.terrain.copy()
cmap.set_bad("black")

fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(
    np.ma.masked_invalid(heights),
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap=cmap,
    interpolation="nearest",
    aspect="equal"
)

fig.colorbar(im, ax=ax, label="Elevation (m)")
ax.set_title("Nov 6 Raw Elevation Map")
ax.set_xlabel("X position in odom (m)")
ax.set_ylabel("Y position in odom (m)")

ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("terrain mapping bags/elevation_map.png", dpi=200)
plt.show()

print(f"Shape: {rows} x {cols}")
print(f"Resolution: {resolution} m/cell")
print(f"Frame: {metadata['frame_id']}")
print(f"Map centre: ({center_x:.3f}, {center_y:.3f})")
print(f"Valid cells: {np.isfinite(heights).sum()} / {heights.size}")
