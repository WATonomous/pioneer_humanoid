import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add terrain_core package to Python's import path
sys.path.insert(0, str(Path("src/terrain_mapping/terrain_core").resolve()))

from terrain_core.traversability import compute_traversability

# Load real elevation map and metadata
data_dir = Path("terrain mapping bags")
elevation = np.load(data_dir / "elevation.npy")

with open(data_dir / "metadata.json") as f:
    metadata = json.load(f)

resolution = metadata["resolution"]

# Run existing traversability implementation with default parameters
results = compute_traversability(elevation, resolution)

traversable = results["traversable"]
score = results["traversability"]

# Cell classifications
valid_input = np.isfinite(elevation)
valid_estimate = np.isfinite(score)
unknown_input = ~valid_input
unestimated = valid_input & ~valid_estimate
accepted = valid_estimate & traversable
rejected = valid_estimate & ~traversable

total = elevation.size
valid_count = valid_input.sum()
estimated_count = valid_estimate.sum()

print("\n--- Traversability Results ---")
print(f"Total cells: {total}")
print(f"Unknown input cells: {unknown_input.sum()} ({unknown_input.mean()*100:.2f}%)")
print(f"Valid elevation cells: {valid_count}")
print(f"Valid cells without terrain estimate: {unestimated.sum()}")
print(f"Cells evaluated: {estimated_count}")
print(f"Traversable: {accepted.sum()}")
print(f"Non-traversable: {rejected.sum()}")

if estimated_count:
    print(f"Traversable among evaluated cells: {accepted.sum()/estimated_count*100:.2f}%")

if valid_count:
    print(f"Traversable among valid input cells: {accepted.sum()/valid_count*100:.2f}%")

# Plot elevation and traversability side by side
center_x = metadata["position"]["x"]
center_y = metadata["position"]["y"]
length_x = metadata["length_x"]
length_y = metadata["length_y"]

extent = [
    center_x - length_x / 2,
    center_x + length_x / 2,
    center_y - length_y / 2,
    center_y + length_y / 2,
]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

elevation_cmap = plt.cm.terrain.copy()
elevation_cmap.set_bad("black")

axes[0].imshow(
    np.ma.masked_invalid(elevation),
    origin="lower",
    extent=extent,
    cmap=elevation_cmap,
    interpolation="nearest",
    aspect="equal",
)
axes[0].set_title("Elevation Map")
axes[0].set_xlabel("X position in odom (m)")
axes[0].set_ylabel("Y position in odom (m)")

# 0 = non-traversable, 1 = traversable, NaN = unknown
display_mask = np.full(elevation.shape, np.nan)
display_mask[rejected] = 0
display_mask[accepted] = 1

mask_cmap = plt.cm.RdYlGn.copy()
mask_cmap.set_bad("black")

axes[1].imshow(
    display_mask,
    origin="lower",
    extent=extent,
    cmap=mask_cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="equal",
)
axes[1].set_title("Traversability Mask")
axes[1].set_xlabel("X position in odom (m)")
axes[1].set_ylabel("Y position in odom (m)")

plt.tight_layout()
plt.savefig(data_dir / "traversability_validation.png", dpi=200)
plt.show()
