"""Hypothetical thumb mounts for the design study (hand frame: fingers along +Y, index at -X, pinky at +X,
palm facing -Z). As built, the thumb base sits mid-palm between index and middle, hangs along -Z and its
MCP_A swing carries it toward the fingertips (+Y), so it meets the fingers head-on. The variants turn that
swing across the palm (toward the pinky) and/or move the base to the index side near the wrist, the way
human, Shadow and Allegro thumbs oppose.
"""
STOCK_POS = (0.0037, 0.0779, 0.0198)

# name: (base position in the palm frame, yaw of the thumb chain about the palm normal, deg)
THUMB_MOUNTS = {
    "stock": (STOCK_POS, 0.0),
    "yaw90": (STOCK_POS, -90.0),                  # same base, swings across the palm toward the pinky
    "radial45": ((-0.035, 0.045, 0.0198), -45.0),  # index side near the wrist, swings toward ring finger
    "radial90": ((-0.035, 0.06, 0.0198), -90.0),   # index side, swings straight across the palm
    # Best fingertip-opposition overlap from opposition.py's 3000-mount search (kinematics only):
    "fwd": ((0.003, 0.109, 0.0198), -2.0),        # stock, base 3 cm toward the knuckles (overlap .57 vs .41)
    "flip": ((0.052, 0.106, 0.0198), -175.0),     # pinky side near the knuckles, turned around (.59)
    "palm108": ((0.028, 0.087, 0.0198), 108.0),   # best with the base kept on the palm, y <= 0.09 (.53)
    # fwd, flip, palm108 (and yaw90) put the thumb base inside the finger roots -- not buildable.
    # Best mount whose base clears the fingers (opposition.py feasibility check):
    "near36": ((0.002, 0.090, 0.0198), 36.0),     # base 12 mm toward the knuckles, turned 36 deg (.50)
}


def thumb_kwargs(name: str) -> dict:
    if name == "stock":  # exactly as built (STOCK_POS is the URDF origin rounded to 0.1 mm)
        return {}
    pos, yaw = THUMB_MOUNTS[name]
    return dict(thumb_pos=pos, thumb_yaw_deg=yaw)
