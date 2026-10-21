"""CAD models (build123d) of the workshop props, exported as STL meshes (metres) for MuJoCo: a benchtop tool
chest (red powder-coated steel, a shallow and a deep drawer on slides, black bar pulls standing off the faces, a carry
handle on top), 3/4" drive sockets (hex mouth, square drive, groove) and the foam rail they stand in, a wooden tool
tray and the wall's hanging tools.

    python make_meshes.py        # writes meshes/*.stl

The chest's shape is set by the constants below; the scene (``tool_chest.py``) reads the same numbers from
``CHEST`` so the collision boxes and the joints line up with the meshes.
"""
import math
import os

from build123d import (add, RegularPolygon, Align, Axis, Box, Cylinder, Location, Locations, Plane, Pos, Rot, Sphere, Sketch, BuildSketch, BuildPart,
                       Circle, RectangleRounded, Rectangle, extrude, fillet, chamfer, revolve, Polyline, make_face,
                       BuildLine, Mode, export_stl, Compound)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "meshes")

# the chest (x: depth, the front at x = 0 facing -x; y: width; z: up from its base); metres
CHEST = dict(
    depth=0.28, width=0.52, wall=0.004,
    base_h=0.012,                    # the plinth
    drawers=(0.075, 0.165),          # drawer heights, bottom first (a shallow one a hand reaches into, a deep one)
    gap=0.004,                       # between drawers
    lid_h=0.022,                     # the top
    pull_len=0.22, pull_r=0.0085,    # the drawer pull: a bar along the drawer's face (y), its length and radius,
    pull_out=0.058, pull_down=0.022, # standing this far out of the face (bar axis: room for a gloved hand), below the top
    drawer_depth=0.26,
    box_h=(0.05, None),              # the box behind each face (None: full height): the bottom one a shallow tray,
                                     # a hand reaches in over its sides
)


def height():
    c = CHEST
    return c["base_h"] + sum(c["drawers"]) + c["gap"] * (len(c["drawers"]) + 1) + c["lid_h"]


def save(shape, name, tol=0.0004, ang=0.2):
    export_stl(shape, os.path.join(OUT, name + ".stl"), tolerance=tol, angular_tolerance=ang)
    print(name)


def chest_body():
    """The cabinet: an open-fronted steel box (top, sides, back, floor), rounded vertical edges, a plinth."""
    c = CHEST
    H = height()
    d, w, t = c["depth"], c["width"], c["wall"]
    with BuildPart() as p:
        with Locations(Location((0, 0, c["base_h"]))):
            Box(d, w, H - c["base_h"], align=(Align.MIN, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), 0.012)
        # hollow it from the front: one opening for the drawers
        oz = c["base_h"] + c["gap"]
        oh = sum(c["drawers"]) + c["gap"] * (len(c["drawers"]) - 1)
        with Locations(Location((-0.001, 0, oz))):
            Box(d - t, w - 2 * t, oh, align=(Align.MIN, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT )
        # the plinth, a little inset
        with Locations(Location((0.01, 0, 0))):
            Box(d - 0.02, w - 0.02, c["base_h"], align=(Align.MIN, Align.CENTER, Align.MIN))
        # a shallow lid recess on top (the rubber mat sits in it)
        with Locations(Location((d / 2, 0, H + 0.0005))):
            Box(d - 0.03, w - 0.03, 0.004, align=(Align.CENTER, Align.CENTER, Align.MAX), mode=Mode.SUBTRACT )
    return p.part


def chest_mat():
    c = CHEST
    return Box(c["depth"] - 0.032, c["width"] - 0.032, 0.003, align=(Align.CENTER, Align.CENTER, Align.MIN)
               ).locate(Location((c["depth"] / 2, 0, height() - 0.0035)))


def carry_handle():
    """A chrome bar handle on two posts across the top."""
    c = CHEST
    H = height()
    with BuildPart() as p:
        for s in (-1, 1):
            with Locations(Location((c["depth"] / 2, s * 0.09, H))):
                Cylinder(0.006, 0.03, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations(Location((c["depth"] / 2, 0, H + 0.035))):
            Cylinder(0.009, 0.20, rotation=(90, 0, 0))
        for s in (-1, 1):
            with Locations(Location((c["depth"] / 2, s * 0.10, H + 0.035))):
                Sphere(0.009)
    return p.part


def box_h(i):
    """The height of drawer ``i``'s box behind its face."""
    h = CHEST["drawers"][i]
    return min(h - 0.008, CHEST["box_h"][i] or h)


def drawer_front(i):
    """Drawer ``i``'s front (in the drawer's frame: its face at x = 0, the box behind it, z from the drawer's
    bottom): a pressed-steel face with a rolled lip, the box behind it (sides, back, floor)."""
    h = CHEST["drawers"][i]
    c = CHEST
    w = c["width"] - 2 * c["wall"] - 0.004
    dd = c["drawer_depth"]
    t = 0.0015
    with BuildPart() as p:
        Box(0.012, w, h, align=(Align.MIN, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Y).group_by(Axis.X)[0], 0.004)
        # the box behind
        bh = box_h(i)
        with Locations(Location((0.012, 0, 0.002))):
            Box(dd, w - 0.006, bh, align=(Align.MIN, Align.CENTER, Align.MIN))
        with Locations(Location((0.012 + t, 0, 0.002 + t))):
            Box(dd - 2 * t, w - 0.006 - 2 * t, bh, align=(Align.MIN, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT )
    return p.part


def drawer_pull():
    """A bar pull (in the drawer's frame: the face at x = 0, the bar's axis at z = 0, out along -x): a round bar along
    y on two posts, room behind it for the fingers."""
    c = CHEST
    L, r, out = c["pull_len"], c["pull_r"], c["pull_out"]
    with BuildPart() as p:
        with Locations(Location((-out, 0, 0))):
            Cylinder(r, L, rotation=(90, 0, 0))
        for y in (-L / 2 + 0.014, L / 2 - 0.014):
            with Locations(Location((-out / 2, y, 0))):
                Cylinder(0.0065, out, rotation=(0, 90, 0))
            with Locations(Location((-0.002, y, 0))):
                Cylinder(0.011, 0.004, rotation=(0, 90, 0))        # the post's foot on the face
        fillet(p.edges().group_by(Axis.Y)[0] + p.edges().group_by(Axis.Y)[-1], 0.004)
    return p.part


# the sockets in the bottom drawer: 3/4"-drive impact-style sockets standing in a foam rail, (size, outside diameter)
SOCKETS = (("22", 0.034), ("24", 0.037), ("27", 0.040), ("30", 0.043))
SOCKET_H = 0.045
SOCKET_Y = (0.12, 0.065, -0.06, -0.125)  # where each stands across the drawer (from its middle)
SOCKET_X = 0.148                 # ... and back from the foam's front edge
TAKE = 2                         # the one the robot takes (27 mm)
FOAM_L, FOAM_H, HOLE_Z = 0.245, 0.048, 0.024     # the foam (it fills the drawer), the holes' floor


def socket(af: str, od: float):
    """A socket: a chrome cylinder, the hex for the nut at the top (``af`` mm across flats), the 3/4" square drive in
    the bottom, a chamfered mouth, a groove round its drive end, the size stamped as a ring of flats (just a band)."""
    h, r = SOCKET_H, od / 2
    a = float(af) / 1000
    with BuildPart() as p:
        Cylinder(r, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        chamfer(p.edges().group_by(Axis.Z)[-1], 0.0012)
        chamfer(p.edges().group_by(Axis.Z)[0], 0.0008)
        with BuildSketch(Plane.XY.offset(h)) as hx:      # the hex, down from the mouth
            RegularPolygon(a / 2 / math.cos(math.pi / 6), 6)
        extrude(amount=-0.022, mode=Mode.SUBTRACT)
        with BuildSketch(Plane.XY) as sq:                # the square drive, up from the bottom
            Rectangle(0.0191, 0.0191)
        extrude(amount=0.014, mode=Mode.SUBTRACT)
        with Locations(Location((0, 0, 0.009))):         # the groove round the drive end
            Cylinder(r + 0.001, 0.002, mode=Mode.SUBTRACT)
            Cylinder(r - 0.0007, 0.002)
    return p.part


# the socket rail scene: loose sockets of four sizes (a red size band round each: what the head camera measures), a
# rail with a stud per socket
RAIL_SOCKETS = (("17", 0.030), ("22", 0.034), ("24", 0.038), ("30", 0.042))
BAND_Z = (0.029, 0.037)          # the band, up the socket's side
RAIL = dict(length=0.46, width=0.032, height=0.014, pitch=0.11, stud_r=0.0075, stud_h=0.013)


def socket_band(od: float):
    """The red size band round a socket (a separate mesh: its own material)."""
    r = od / 2
    with BuildPart() as p:
        with Locations(Location((0, 0, BAND_Z[0]))):
            Cylinder(r + 0.0004, BAND_Z[1] - BAND_Z[0], align=(Align.CENTER, Align.CENTER, Align.MIN))
            Cylinder(r - 0.001, BAND_Z[1] - BAND_Z[0], align=(Align.CENTER, Align.CENTER, Align.MIN),
                     mode=Mode.SUBTRACT)
    return p.part


def socket_rail():
    """A socket rail (in its frame: along y, centred, its base at z = 0): a black anodised aluminium channel with a
    rounded top, red end caps (a separate mesh), a chrome stud per socket with a spring ball in its side."""
    c = RAIL
    L, w, h = c["length"], c["width"], c["height"]
    with BuildPart() as p:
        Box(w, L, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Y).group_by(Axis.Z)[-1], 0.004)
        with Locations(Location((0, 0, 0))):           # the channel underneath
            Box(w - 0.008, L, 0.006, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
    return p.part


def rail_studs():
    c = RAIL
    n = len(RAIL_SOCKETS)
    with BuildPart() as p:
        for k in range(n):
            y = (k - (n - 1) / 2) * c["pitch"]
            with Locations(Location((0, y, c["height"]))):
                Cylinder(c["stud_r"], c["stud_h"], align=(Align.CENTER, Align.CENTER, Align.MIN))
                Cylinder(c["stud_r"] + 0.003, 0.002, align=(Align.CENTER, Align.CENTER, Align.MIN))   # its collar
            with Locations(Location((c["stud_r"], y, c["height"] + c["stud_h"] * 0.55))):
                Sphere(0.0022)
        chamfer(p.edges().group_by(Axis.Z)[-1], 0.002)
    return p.part


def rail_caps():
    c = RAIL
    L, w, h = c["length"], c["width"], c["height"]
    with BuildPart() as p:
        for s in (-1, 1):
            with Locations(Location((0, s * (L / 2 + 0.006), 0))):
                Box(w + 0.004, 0.012, h + 0.004, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Y), 0.003)
    return p.part


def foam_insert():
    """Grey foam on the bottom drawer's floor, a row of round holes the sockets stand in (sunk about half their
    height: they ride in and out with the drawer)."""
    c = CHEST
    w = c["width"] - 2 * c["wall"] - 0.012
    with BuildPart() as p:
        Box(FOAM_L, w, FOAM_H, align=(Align.MIN, Align.CENTER, Align.MIN))
        for (_, od), y in zip(SOCKETS, SOCKET_Y):
            with Locations(Location((SOCKET_X, y, HOLE_Z))):
                Cylinder(od / 2 + 0.0012, FOAM_H, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
        with Locations(Location((SOCKET_X, 0, FOAM_H - 0.002))):       # a printed strip of sizes along the row
            Box(0.012, w - 0.03, 0.002, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)
    return p.part


def tool_tray():
    """A small open wooden tray (the socket goes in it), 0.30 x 0.14, 4 cm deep, finger cut-outs at the ends."""
    with BuildPart() as p:
        Box(0.14, 0.30, 0.04, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), 0.008)
        with Locations(Location((0, 0, 0.008))):
            Box(0.124, 0.284, 0.04, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT )
        for s in (-1, 1):
            with Locations(Location((0, s * 0.15, 0.045))):
                Cylinder(0.02, 0.03, rotation=(90, 0, 0), mode=Mode.SUBTRACT)
    return p.part


# hanging tools for the pegboard (visual only)
def wrench(L=0.22):
    with BuildPart() as p:
        Box(L * 0.75, 0.016, 0.005, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        with Locations(Location((L * 0.4, 0, 0))):
            Cylinder(0.019, 0.006)
        with Locations(Location((L * 0.4, 0, 0))):
            Cylinder(0.009, 0.007, mode=Mode.SUBTRACT)
        with Locations(Location((-L * 0.42, 0, 0))):
            Cylinder(0.016, 0.006)
        with Locations(Location((-L * 0.45, 0, 0))):
            Box(0.02, 0.012, 0.007, mode=Mode.SUBTRACT)
    return p.part


def hammer():
    with BuildPart() as p:
        Box(0.03, 0.03, 0.26, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), 0.01)
        with Locations(Location((0.01, 0, 0.26))):
            Box(0.11, 0.028, 0.03, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def pliers():
    with BuildPart() as p:
        for s in (-1, 1):
            with Locations(Location((s * 0.012, 0, 0)) * Rot(0, s * -6, 0)):
                Box(0.016, 0.006, 0.13, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations(Location((0, 0, 0.13))):
            Cylinder(0.012, 0.008, rotation=(90, 0, 0))
        with Locations(Location((0, 0, 0.13))):
            Box(0.018, 0.006, 0.05, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def tape_measure():
    with BuildPart() as p:
        Box(0.07, 0.035, 0.07)
        fillet(p.edges(), 0.012)
    return p.part


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    save(chest_body(), "chest_body")
    save(chest_mat(), "chest_mat")
    save(carry_handle(), "chest_handle")
    for i, h in enumerate(CHEST["drawers"]):
        save(drawer_front(i), f"drawer{i}")
    save(drawer_pull(), "drawer_pull")
    for af, od in SOCKETS:
        save(socket(af, od), f"socket_{af}")
    for af, od in RAIL_SOCKETS:
        save(socket(af, od), f"rs_{af}")
        save(socket_band(od), f"rs_{af}_band")
    save(socket_rail(), "socket_rail")
    save(rail_studs(), "rail_studs")
    save(rail_caps(), "rail_caps")
    save(foam_insert(), "foam_insert")
    save(tool_tray(), "tool_tray")
    save(wrench(), "wrench")
    save(wrench(0.17), "wrench_small")
    save(hammer(), "hammer")
    save(pliers(), "pliers")
    save(tape_measure(), "tape_measure")
