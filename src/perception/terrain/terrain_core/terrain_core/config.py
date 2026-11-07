"""Terrain map sizes.

Two grids:
- policy map: height map sent to locomotion
  (16 x 10 cells at 0.1 m, confirmed with locomotion)
- elevation map: the internal map built from sensor data. 
  (5 cm and bigger than the policy map)
"""

# Policy map
POLICY_LENGTH_M = 1.6
POLICY_WIDTH_M = 1.0
POLICY_RESOLUTION_M = 0.1

# Internal elevation map
ELEVATION_RESOLUTION_M = 0.05
# Size not set yet: it will be bigger than the policy map
# Add ELEVATION_LENGTH_M and ELEVATION_WIDTH_M


def cells(length_m, width_m, resolution_m):
    """Number of cells as (forward, sideways)."""
    return round(length_m / resolution_m), round(width_m / resolution_m)


def policy_shape():
    return cells(POLICY_LENGTH_M, POLICY_WIDTH_M, POLICY_RESOLUTION_M)
