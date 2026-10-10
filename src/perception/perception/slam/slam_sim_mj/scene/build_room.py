#!/usr/bin/env python3
"""Generate the corridor-loop room as MJCF: geometry, materials and textures.

An office corridor circuit -- painted walls, baseboards, doors with frames and
handles, notice boards, lockers, a tiled floor and ceiling light panels. It is built
rather than downloaded because MuJoCo's OBJ loader binds ONE texture per mesh, so a
multi-material room import arrives as untextured grey unless it is split per material
first. Generating it also means the loop is guaranteed to fit the walls.

    python3 build_room.py                  # writes room.xml + textures/
    python3 build_room.py --preview        # ...and POV renders to preview.png

Two things here are load-bearing for SLAM, not for looks:

  * NOTHING REPEATS ALONG A CORRIDOR. Every wall panel, door and notice board gets
    its own texture. MuJoCo's builtin `checker` is a repeating pattern, which is
    worse than no texture at all -- identical patches metres apart make visual
    odometry match the wrong one and loop closure fire on places the camera has
    never been.
  * GEOMETRY CARRIES THE FEATURES, not paint. Real corridor walls are close to
    featureless, which is the documented failure mode on the real rig ("Not enough
    inliers" against a blank wall). Door frames, baseboards, handles and locker
    edges give a corner detector actual corners, so the walls can be painted a
    realistic flat colour instead of noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# The walkable loop centreline is a rectangle of half-extents (HALF_X, HALF_Y); walls
# sit half a corridor width either side of it. These match corridor_pose()'s defaults
# in orbit_path.py -- keep them in sync or the camera walks through a wall.
HALF_X = 6.0
HALF_Y = 4.0
CORRIDOR_W = 2.2
WALL_H = 2.7

BASEBOARD_H = 0.12
DOOR_W, DOOR_H = 0.95, 2.05
PANEL_W = 2.4          # wall is divided into bays this wide, one texture each
WALL_T = 0.08


# --------------------------------------------------------------------------- textures

def _smooth(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Low-frequency noise in [0,1], by upsampling a coarse grid."""
    coarse = (rng.random((cells, cells)) * 255).astype(np.uint8)
    return np.array(Image.fromarray(coarse).resize((size, size), Image.BICUBIC)) / 255.0


def wall_tint(rng) -> np.ndarray:
    """One paint colour, near-neutral like real institutional paint.

    Chosen ONCE PER WALL RUN, not per bay. Randomising it per bay made every 2.4 m of
    corridor a different colour, which read as patchwork rather than a building. The
    per-bay textures still differ in grain and shading, which is all that non-repeat
    matching needs -- they just no longer differ in hue.
    """
    grey = rng.uniform(0.66, 0.80)
    hue = rng.normal(0.0, 0.035, 3)          # a slight, plausible cast
    return np.clip(grey + hue, 0.55, 0.88)


def tex_paint(rng, tint, size=256):
    """Painted plaster: a flat colour with faint grain and occasional scuffing.
    Deliberately low-contrast -- the geometry supplies the trackable features."""
    img = np.asarray(tint)[None, None, :] * (0.90 + 0.10 * _smooth(rng, size, 24)[..., None])
    img += 0.03 * (rng.random((size, size, 1)) - 0.5)          # fine grain
    img *= 0.94 + 0.06 * _smooth(rng, size, 5)[..., None]      # broad shading
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def tex_wood(rng, size=256):
    """Door slab: vertical grain, warm tone."""
    base = np.array([rng.uniform(0.36, 0.52), rng.uniform(0.22, 0.31), rng.uniform(0.13, 0.19)])
    grain = 0.5 + 0.5 * np.sin(np.linspace(0, rng.uniform(40, 70), size))[None, :]
    grain = grain * (0.75 + 0.25 * _smooth(rng, size, 30))
    img = base[None, None, :] * (0.72 + 0.38 * grain[..., None])
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def tex_notice(rng, size=256):
    """A notice board / poster: strong graphic shapes. These are the most
    distinctive things in the scene and do most of the work for loop closure."""
    img = np.ones((size, size, 3)) * rng.uniform(0.85, 0.97)
    for _ in range(rng.integers(4, 9)):
        h, w = rng.integers(14, 70), rng.integers(24, 110)
        y, x = rng.integers(0, size - h), rng.integers(0, size - w)
        img[y:y + h, x:x + w] = rng.random(3) * 0.8
    for _ in range(rng.integers(3, 7)):                        # text-like rules
        y = rng.integers(4, size - 6)
        x0 = rng.integers(0, size // 2)
        img[y:y + rng.integers(2, 5), x0:x0 + rng.integers(40, size - x0)] *= 0.35
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def tex_locker(rng, size=256):
    """Locker bank: vertical dividers and handle strips."""
    base = np.array([rng.uniform(0.30, 0.55), rng.uniform(0.38, 0.58), rng.uniform(0.42, 0.60)])
    img = np.tile(base, (size, size, 1)) * (0.92 + 0.08 * _smooth(rng, size, 20)[..., None])
    for x in range(0, size, size // 4):                        # dividers
        img[:, x:x + 3] *= 0.45
    img[size // 3: size // 3 + 6, :] *= 0.5                    # handle line
    img[2 * size // 3: 2 * size // 3 + 6, :] *= 0.5
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def tex_floor(rng, size=512):
    """Tiled floor. The grout grid repeats, which is fine underfoot: the camera looks
    along the corridor, and tiles are rarely what a frame is matched on. Broad
    blotching on top keeps large areas from being pixel-identical anyway."""
    img = np.ones((size, size, 3)) * np.array([0.52, 0.50, 0.47])
    img *= 0.88 + 0.12 * _smooth(rng, size, 16)[..., None]
    img += 0.04 * (rng.random((size, size, 1)) - 0.5)
    for g in range(0, size, size // 6):                        # grout
        img[g:g + 3, :] *= 0.62
        img[:, g:g + 3] *= 0.62
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def tex_ceiling(rng, size=512):
    """Acoustic tile: pale, lightly speckled, with a lay-in grid."""
    img = np.ones((size, size, 3)) * 0.87
    img *= 0.94 + 0.06 * _smooth(rng, size, 40)[..., None]
    img += 0.02 * (rng.random((size, size, 1)) - 0.5)
    for g in range(0, size, size // 4):
        img[g:g + 2, :] *= 0.80
        img[:, g:g + 2] *= 0.80
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- geometry

class Room:
    """Accumulates MJCF assets and geoms, with a wall-local coordinate helper."""

    def __init__(self, rng: np.random.Generator, tex_dir: Path):
        self.rng = rng
        self.tex_dir = tex_dir
        self.assets: list[str] = []
        self.geoms: list[str] = []
        self._n = 0

    def material(self, fn, *args, specular=0.15, shininess=0.2, repeat=(1, 1),
                 uniform=False) -> str:
        """Generate one unique texture and return the material name referencing it."""
        name = f"m{self._n}"
        self._n += 1
        Image.fromarray(fn(self.rng, *args)).save(self.tex_dir / f"{name}.png")
        self.assets.append(
            f'    <texture name="t{name}" type="2d" file="textures/{name}.png"/>\n'
            f'    <material name="{name}" texture="t{name}" texrepeat="{repeat[0]} {repeat[1]}" '
            f'texuniform="{str(uniform).lower()}" specular="{specular}" shininess="{shininess}"/>'
        )
        return name

    def flat(self, rgb: tuple[float, float, float], *, specular=0.1) -> str:
        name = f"m{self._n}"
        self._n += 1
        self.assets.append(
            f'    <material name="{name}" rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1" '
            f'specular="{specular}" shininess="0.3"/>'
        )
        return name

    def box(self, pos, size, material: str) -> None:
        self.geoms.append(
            f'    <geom type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" material="{material}"/>'
        )


class Wall:
    """A straight wall run. `u` runs along it, `n` is the outward offset from its
    face towards the corridor -- so anything mounted ON the wall has n > 0.

    Keeping this one class is what makes the four outer and four inner walls a loop
    rather than eight hand-written coordinate blocks, each with its own chance of a
    sign error that puts a door inside the concrete.
    """

    def __init__(self, axis: str, fixed: float, u0: float, u1: float, normal: int):
        self.axis, self.fixed, self.u0, self.u1, self.normal = axis, fixed, u0, u1, normal

    @property
    def length(self) -> float:
        return self.u1 - self.u0

    def place(self, room: Room, u: float, z: float, du: float, dz: float,
              depth: float, material: str, inset: float = 0.0) -> None:
        """Add a box centred at along-position `u`, height `z`, standing `inset` out
        from the wall face and `depth` thick (half-extents du/dz along/up)."""
        off = inset + depth
        if self.axis == "x":
            pos = (u, self.fixed + self.normal * off, z)
            size = (du, depth, dz)
        else:
            pos = (self.fixed + self.normal * off, u, z)
            size = (depth, du, dz)
        room.box(pos, size, material)


def build_wall(room: Room, wall: Wall, rng: np.random.Generator, *, furnish: bool) -> None:
    """Paint a wall in bays and hang things on it."""
    n_bays = max(1, int(round(wall.length / PANEL_W)))
    bay_w = wall.length / n_bays

    tint = wall_tint(rng)
    baseboard = room.flat((0.28, 0.27, 0.26))
    frame_mat = room.flat((0.88, 0.87, 0.84))
    handle_mat = room.flat((0.72, 0.68, 0.35), specular=0.9)

    for i in range(n_bays):
        u = wall.u0 + bay_w * (i + 0.5)
        wall.place(room, u, WALL_H / 2, bay_w / 2, WALL_H / 2, WALL_T,
                   room.material(tex_paint, tint))
        wall.place(room, u, BASEBOARD_H / 2, bay_w / 2, BASEBOARD_H / 2, 0.02, baseboard, inset=WALL_T * 2)

        if not furnish:
            continue

        roll = rng.random()
        if roll < 0.34 and bay_w > DOOR_W + 0.5:
            # Door: frame proud of the wall, slab inside it, handle on the slab.
            wall.place(room, u, DOOR_H / 2, DOOR_W / 2 + 0.09, DOOR_H / 2 + 0.09, 0.035,
                       frame_mat, inset=WALL_T * 2)
            wall.place(room, u, DOOR_H / 2, DOOR_W / 2, DOOR_H / 2, 0.025,
                       room.material(tex_wood), inset=WALL_T * 2 + 0.07)
            wall.place(room, u + DOOR_W / 2 - 0.13, 1.02, 0.055, 0.02, 0.035,
                       handle_mat, inset=WALL_T * 2 + 0.12)
        elif roll < 0.60:
            # Notice board, at eye level -- the strongest loop-closure landmark here.
            h = rng.uniform(0.45, 0.75)
            wall.place(room, u, rng.uniform(1.35, 1.65), rng.uniform(0.45, 0.8), h, 0.03,
                       room.material(tex_notice), inset=WALL_T * 2)
        elif roll < 0.74 and bay_w > 1.2:
            # Locker bank standing on the floor.
            wall.place(room, u, 0.9, bay_w / 2 - 0.15, 0.9, 0.16,
                       room.material(tex_locker), inset=WALL_T * 2)


def build(out_path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    tex_dir = out_path.parent / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    for old in tex_dir.glob("*.png"):
        old.unlink()

    room = Room(rng, tex_dir)

    ox, oy = HALF_X + CORRIDOR_W / 2, HALF_Y + CORRIDOR_W / 2   # outer shell
    ix, iy = HALF_X - CORRIDOR_W / 2, HALF_Y - CORRIDOR_W / 2   # inner block

    # Outer shell: normals point inward, towards the corridor.
    for wall in (Wall("x", oy, -ox, ox, -1), Wall("x", -oy, -ox, ox, +1),
                 Wall("y", ox, -oy, oy, -1), Wall("y", -ox, -oy, oy, +1)):
        build_wall(room, wall, rng, furnish=True)

    # Inner block: normals point outward, also towards the corridor.
    for wall in (Wall("x", iy, -ix, ix, +1), Wall("x", -iy, -ix, ix, -1),
                 Wall("y", ix, -iy, iy, +1), Wall("y", -ix, -iy, iy, -1)):
        build_wall(room, wall, rng, furnish=True)

    floor = room.material(tex_floor, repeat=(6, 6), uniform=True, specular=0.25, shininess=0.4)
    ceiling = room.material(tex_ceiling, repeat=(6, 6), uniform=True, specular=0.05)
    room.box((0, 0, -0.05), (ox + 0.5, oy + 0.5, 0.05), floor)
    room.box((0, 0, WALL_H + 0.05), (ox + 0.5, oy + 0.5, 0.05), ceiling)

    # Ceiling light panels down the middle of each corridor run. Bright flat material
    # rather than actual lights -- MuJoCo point lights fall off far too fast to carry
    # a 26 m corridor, so illumination comes from ambient (see <headlight> below) and
    # these are there to look right and to break up the ceiling.
    lamp = room.flat((0.97, 0.97, 0.92), specular=0.0)
    mid_x, mid_y = (ox + ix) / 2, (oy + iy) / 2
    for x in np.arange(-ox + 1.5, ox - 1.0, 3.0):
        room.box((x, mid_y, WALL_H - 0.03), (0.55, 0.14, 0.03), lamp)
        room.box((x, -mid_y, WALL_H - 0.03), (0.55, 0.14, 0.03), lamp)
    for y in np.arange(-oy + 1.5, oy - 1.0, 3.0):
        room.box((mid_x, y, WALL_H - 0.03), (0.14, 0.55, 0.03), lamp)
        room.box((-mid_x, y, WALL_H - 0.03), (0.14, 0.55, 0.03), lamp)

    xml = f"""<mujoco model="slam_corridor">
  <!-- GENERATED by scene/build_room.py -- edit that, not this. -->
  <compiler angle="radian" texturedir="."/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096" offsamples="4"/>
    <!-- Ambient-dominant, deliberately. MuJoCo's point lights attenuate, so a 26 m
         corridor lit only from the ceiling renders near-black a few metres out and
         ORB finds ~40 features per frame where it needs several hundred. Ambient
         lights every surface equally regardless of distance. A downloaded scene with
         baked lighting would not need this. -->
    <headlight ambient="0.48 0.48 0.50" diffuse="0.45 0.45 0.43" specular="0.1 0.1 0.1"/>
  </visual>

  <asset>
{chr(10).join(room.assets)}
  </asset>

  <worldbody>
    <light pos="0 0 {WALL_H - 0.15}" dir="0 0 -1" diffuse="0.25 0.25 0.24" castshadow="false"/>
{chr(10).join(room.geoms)}
  </worldbody>
</mujoco>
"""
    out_path.write_text(xml)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent / "room.xml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true",
                    help="Render POV frames from around the loop to preview.png.")
    args = ap.parse_args()
    p = build(Path(args.out), args.seed)
    print(f"wrote {p} (+ {p.parent / 'textures'})")

    if args.preview:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from mj_camera import MjCamera
        from orbit_path import corridor_pose

        cam = MjCamera(p, 640, 480)
        try:
            import cv2
            orb = cv2.ORB_create(1500)
        except ImportError:
            orb = cv2 = None

        tiles = []
        for t in (0, 25, 50, 75):
            pos, quat = corridor_pose(float(t), np.zeros(3), period=100.0, height=1.5,
                                      scan_yaw=0.35, scan_pitch=0.12)
            cam.set_pose(pos, quat)
            rgb, depth = cam.render()
            n = len(orb.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)) if orb else -1
            print(f"  t={t:3d}  orb features={n:5d}  depth {depth.min():.2f}-{depth.max():.2f} m")
            tiles.append(rgb)
        out = p.parent / "preview.png"
        Image.fromarray(np.concatenate(tiles, axis=1)).save(out)
        print(f"wrote {out}")
