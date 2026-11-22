import numpy as np
import matplotlib.pyplot as plt

heights = np.load("terrain mapping bags/elevation.npy")
valid = heights[np.isfinite(heights)]

plt.figure(figsize=(9, 5))
plt.hist(valid, bins=100)
plt.xlabel("Elevation (m)")
plt.ylabel("Number of cells")
plt.title("Nov 6 Elevation Height Distribution")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("terrain mapping bags/elevation_histogram.png", dpi=200)
plt.show()

print("Median:", np.median(valid))
print("Mean:", np.mean(valid))
print("Standard deviation:", np.std(valid))
