"""Output map specification for the terrain height map."""

# Size of the robot-centric height map, in meters.
MAP_LENGTH_M = 1.6   
MAP_WIDTH_M = 1.0   

# Size of one grid cell (5 cm)
RESOLUTION_M = 0.05


def grid_shape():
    """Return the number of cells as (x, y)."""
    return (
        round(MAP_LENGTH_M / RESOLUTION_M),
        round(MAP_WIDTH_M / RESOLUTION_M),
    )