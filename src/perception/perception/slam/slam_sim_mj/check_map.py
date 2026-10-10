#!/usr/bin/env python3
"""Report what RTAB-Map actually built, by reading its database.

Use this instead of grepping the ROS log. RTAB-Map does not print "loop closure" at
default verbosity, so a log grep reports zero on a run that is closing loops
constantly. The database is ground truth.

    python3 check_map.py [path/to/rtabmap.db]

Two numbers matter:

  closures  loop closure and proximity links. Zero means the map is pure odometry
            with nothing correcting it -- the camera never recognised a place it had
            already been.
  z drift   spread of node heights. The trajectories here are flat by construction,
            so ANY spread is odometry error. Grid/MaxGroundHeight is 0.15 m, so once
            drift approaches that the 2D occupancy grid stops being able to tell
            floor from obstacle and degenerates into a blob. Measure this BEFORE
            touching any Grid/* parameter: those thresholds are usually
            correct, and what has broken is the datum they measure against.
"""

import sqlite3
import struct
import sys
from pathlib import Path

# rtabmap/core/Link.h. Only kNeighbor is odometry.
#
# NOTE 7/8/9 -- these values differ between rtabmap versions, and getting them wrong
# fails silently: counting Gravity as a closure type reports "OK, N closure links" on
# a run that closed nothing, because every node carries exactly one gravity
# constraint. That is the false positive this script exists to prevent, so keep these
# in sync with the header whenever the rtabmap version changes.
LINK_TYPES = {
    0: ("Neighbor", "sequential odometry links"),
    1: ("GlobalClosure", "appearance-based loop closure"),
    2: ("LocalSpaceClosure", "proximity detection"),
    3: ("LocalTimeClosure", "time-based proximity"),
    4: ("UserClosure", "manually added"),
    5: ("VirtualClosure", "virtual"),
    6: ("NeighborMerged", "merged neighbour"),
    7: ("PosePrior", "absolute pose prior (From == To)"),
    8: ("Landmark", "landmark observation"),
    9: ("Gravity", "IMU gravity constraint (From == To) -- NOT a closure"),
}
# From == To. These constrain a node against the world, not against a revisit.
SELF_LINKS = {7, 9}
CLOSURES = {1, 2, 3, 4}


def _xyz(blob: bytes) -> tuple[float, float, float]:
    """Node.pose is a 3x4 row-major float32 transform; translation is the last column."""
    f = struct.unpack("<12f", blob[:48])
    return f[3], f[7], f[11]


def main() -> int:
    db_path = Path(sys.argv[1] if len(sys.argv) > 1 else "~/.ros/rtabmap.db").expanduser()
    if not db_path.exists():
        print(f"no database at {db_path}")
        return 1

    con = sqlite3.connect(str(db_path))
    nodes = con.execute("SELECT COUNT(*) FROM Node").fetchone()[0]
    rows = con.execute("SELECT type, COUNT(*) FROM Link GROUP BY type ORDER BY type").fetchall()
    poses = [_xyz(b) for (b,) in con.execute("SELECT pose FROM Node WHERE pose IS NOT NULL")]
    con.close()

    print(f"database : {db_path}  ({db_path.stat().st_size / 1e6:.0f} MB)")
    print(f"nodes    : {nodes}")
    for t, count in rows:
        name, desc = LINK_TYPES.get(t, (f"type{t}", "UNKNOWN -- check Link.h"))
        print(f"  {name:18s} {count:5d}   {desc}")

    if poses:
        zs = [p[2] for p in poses]
        drift = max(zs) - min(zs)
        print(f"\nz drift  : {drift:.3f} m  (min {min(zs):+.3f}, max {max(zs):+.3f})")
        if drift > 0.15:
            print("  ^ EXCEEDS Grid/MaxGroundHeight (0.15 m). The 2D map will be a blob.")
            print("    This is an odometry problem, not a Grid/* tuning problem.")

    closures = sum(c for t, c in rows if t in CLOSURES)
    gravity = sum(c for t, c in rows if t == 9)
    print()
    if gravity:
        print(f"gravity  : {gravity} constraints — the IMU prior is reaching rtabmap.")
    elif poses:
        print("gravity  : NONE — no IMU prior. Expect z drift; check imu_filter_madgwick.")

    if closures:
        print(f"OK: {closures} loop closures — the map is corrected against revisits.")
        return 0
    print("NO LOOP CLOSURES: odometry only. Either the camera never revisited a place")
    print("(one lap gives only a brief revisit at the very end — try --laps 2), or the")
    print("scene is too texture-poor to recognise a view a second time.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
