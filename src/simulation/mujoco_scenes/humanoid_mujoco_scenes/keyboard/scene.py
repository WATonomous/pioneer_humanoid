"""Keyboard typing: pick the stylus out of its holder and type the word announced, one key per step.

Real-size keyboard (19.05 mm key pitch, 18 mm keycaps): number row, three letter rows and a space bar.
Each keycap is a 3 g body on a vertical slide joint with a preloaded spring, like a linear switch:
0.35 N to start moving, ~0.55 N at the 2 mm actuation point, ~0.75 N at the 4 mm bottom-out. A key
registers when it passes ACTUATE and must rise back past RELEASE before it can register again.
The keyboard is a free 0.6 kg body on rubber feet, so a hard sideways shove moves it.

The gripper's finger pads are 61 x 33 mm, wider than a key, so typing goes through a stylus: a 16 mm
square handle (pinch it with the gripper pointing down) and a rounded 8 mm tip. With the gripper
pointing down each arm reaches only its own side (right arm y <= 0, left arm y >= 0, x up to ~0.45),
so the keyboard sits centred at y = 0 and each arm has its own stylus, in a holder on its side:
the left arm types Q-T / A-G / Z-V, the right arm Y-P / H-L / B-M, like touch typing.

Steps (``progress``): "press <key>" for each character of the text: a word from WORDS picked at reset,
or any text (A-Z, 0-9, space) given with ``set_text``. Wrong keys are ignored; only the next one
advances. ``typer.py`` types any text with both arms: ``python -m humanoid_mujoco_scenes.keyboard.typer``.
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import add_floor, scene

TABLE_TOP_Z = 0.705
TABLE_X = (0.15, 0.85)
TABLE_HALF_Y = 0.6

KB_POS = (0.33, 0.0)            # keyboard centre on the table
KB_SIZE = (0.125, 0.235, 0.018) # case depth (x), width (y), height
KB_MASS = 0.6
KB_FRICTION = [1.2, 0.005, 0.0001]  # rubber feet
PITCH = 0.01905
CAP = 0.018                     # keycap footprint
CAP_H = 0.008
STEM = 0.012                    # keycap top above the case top, at rest
CAP_MASS = 0.003
TRAVEL = 0.004
SPRING_K = 100.0                # N/m
PRELOAD = 0.35                  # N at rest
SPRING_DAMPING = 0.5            # N s/m; critical is ~1.1 for a 3 g cap
ACTUATE = 0.002                 # m of travel that registers a press
RELEASE = 0.0015

# (keys, offset of the first key from the number row's first key, in pitches toward -y)
ROWS = [("1234567890", 0.0), ("QWERTYUIOP", 0.5), ("ASDFGHJKL", 0.75), ("ZXCVBNM", 1.25)]
SPACE_UNITS = 6.0
SPACE_COL = 3.75                # space bar centre, in pitches from the number row's first key

STYLUS_POS = {"right": (0.30, -0.24), "left": (0.30, 0.24)}  # one per arm, each in its own holder
STYLUS_YAW = 45.0               # deg; handle faces (and holders) turned so each arm's jaws reach them square
STYLUS_HANDLE = (0.016, 0.10)   # square side, length
STYLUS_TIP = (0.004, 0.03)      # radius, length (capsule, rounded end)
STYLUS_MASS = 0.015
HOLDER_INNER = 0.019             # 1.5 mm clearance round the handle: a stylus put back stands upright
HOLDER_WALL = 0.003
HOLDER_H = 0.05

WORDS = ["HI", "CAT", "WATO", "ROBOT", "HELLO", "TYPE"]   # reset picks one; set_text() takes any text
KEYS = [k for row, _ in ROWS for k in row] + ["SPACE"]
MAX_TEXT = 64

# data.userdata: [characters typed so far, text length, latched "down" flag per key..., text char codes...]
_U_DONE, _U_LEN, _U_DOWN = 0, 1, 2
_U_TEXT = _U_DOWN + len(KEYS)


def _keycap_body(key: str) -> str:
    return f"key_{key}"


def key_for(char: str) -> str:
    """Key that types `char` (case-insensitive; " " is SPACE). Raises ValueError if the keyboard has none."""
    key = "SPACE" if char == " " else char.upper()
    if key not in KEYS:
        raise ValueError(f"no key for {char!r}: the keyboard has A-Z, 0-9 and space")
    return key


def set_text(model: mujoco.MjModel, data: mujoco.MjData, text: str) -> None:
    """Make `text` the thing to type and clear progress. Letters, digits and spaces, up to MAX_TEXT."""
    if not 0 < len(text) <= MAX_TEXT:
        raise ValueError(f"text must be 1-{MAX_TEXT} characters, got {len(text)}")
    codes = [ord(c) for c in text.upper()]
    for c in text:
        key_for(c)
    data.userdata[_U_DONE] = 0
    data.userdata[_U_LEN] = len(codes)
    data.userdata[_U_TEXT:_U_TEXT + MAX_TEXT] = 0
    data.userdata[_U_TEXT:_U_TEXT + len(codes)] = codes


def get_text(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    n = int(data.userdata[_U_LEN])
    return "".join(chr(int(c)) for c in data.userdata[_U_TEXT:_U_TEXT + n])


# ----------------------------------------------------------------------------- hooks
def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Register key presses (with hysteresis) and advance through the text."""
    text = get_text(model, data)
    for i, key in enumerate(KEYS):
        depth = -data.qpos[model.joint(_keycap_body(key)).qposadr[0]]
        down = data.userdata[_U_DOWN + i]
        if not down and depth > ACTUATE:
            data.userdata[_U_DOWN + i] = 1
            k = int(data.userdata[_U_DONE])
            if k < len(text) and key == key_for(text[k]):
                data.userdata[_U_DONE] = k + 1
        elif down and depth < RELEASE:
            data.userdata[_U_DOWN + i] = 0


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    data.userdata[:] = 0
    set_text(model, data, WORDS[rng.integers(len(WORDS))])


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    text = get_text(model, data)
    k = int(data.userdata[_U_DONE])
    return k, len(text), f'type "{text}": press {key_for(text[min(k, len(text) - 1)])}'


def pressed_keys(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Keys currently held past the actuation point."""
    return [k for i, k in enumerate(KEYS) if data.userdata[_U_DOWN + i]]


def key_top(model: mujoco.MjModel, data: mujoco.MjData, key: str) -> np.ndarray:
    """World position of the centre of a keycap's top face."""
    b = model.body(_keycap_body(key)).id
    return data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.array([0, 0, CAP_H / 2])


# ----------------------------------------------------------------------------- scene
_CAP_RGB = [0.88, 0.88, 0.9]
_LABEL_PX = 64


def _add_label(spec: mujoco.MjSpec, key: str) -> str:
    """Cube texture with the letter on the keycap's top face (+Z, face 4), upright seen from the robot."""
    from PIL import Image, ImageDraw, ImageFont

    blank = Image.new("RGB", (_LABEL_PX, _LABEL_PX), tuple(int(255 * c) for c in _CAP_RGB))
    top = blank.copy()
    ImageDraw.Draw(top).text((_LABEL_PX / 2, _LABEL_PX / 2), key, fill=(25, 25, 30), anchor="mm",
                             font=ImageFont.load_default(size=int(_LABEL_PX * 0.6)))
    faces = [np.asarray(blank)] * 6
    faces[4] = np.rot90(np.asarray(top), k=1)
    tex = spec.add_texture(name=f"label_{key}", type=mujoco.mjtTexture.mjTEXTURE_CUBE,
                           width=_LABEL_PX, height=_LABEL_PX * 6, nchannel=3)
    tex.data = np.ascontiguousarray(np.concatenate(faces, axis=0)).tobytes()
    mat = spec.add_material(name=f"label_{key}")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    return mat.name


@scene("keyboard", camera=dict(lookat=[0.33, 0.0, 0.73], distance=0.9, azimuth=160, elevation=-45),
       step=step, reset=reset, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.nuserdata = _U_TEXT + MAX_TEXT
    world = spec.worldbody
    box, capsule = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE

    x0, x1 = TABLE_X
    world.add_geom(name="table", type=box, size=[(x1 - x0) / 2, TABLE_HALF_Y, TABLE_TOP_Z / 2],
                   pos=[(x0 + x1) / 2, 0, TABLE_TOP_Z / 2], rgba=[0.55, 0.42, 0.3, 1])

    # Keyboard case: frame at its top centre. Keycaps are its children, so MuJoCo already skips
    # case-keycap contacts (parent/child); the slide-joint limit is the bottom-out.
    dx, dy, h = KB_SIZE
    kb = world.add_body(name="keyboard", pos=[*KB_POS, TABLE_TOP_Z + h])
    kb.add_freejoint(name="keyboard")
    kb.add_geom(name="keyboard_case", type=box, size=[dx / 2, dy / 2, h / 2], pos=[0, 0, -h / 2],
                mass=KB_MASS, friction=KB_FRICTION, rgba=[0.12, 0.12, 0.14, 1])

    # Rows run along y; the number row is farthest from the robot (+x), Q is on the robot's left (+y).
    n_rows = len(ROWS) + 1
    width = (10 + 0.5) * PITCH
    y_first = width / 2 - PITCH / 2
    x_far = (n_rows - 1) * PITCH / 2
    cap_z = STEM - CAP_H / 2

    def add_key(name, x, y, size_y):
        cap = kb.add_body(name=_keycap_body(name), pos=[x, y, cap_z])
        j = cap.add_joint(name=_keycap_body(name), type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                          range=[-TRAVEL, 0], stiffness=SPRING_K, springref=PRELOAD / SPRING_K,
                          damping=SPRING_DAMPING)
        j.solref_limit = [0.004, 1.0]
        geom = cap.add_geom(name=_keycap_body(name), type=box, size=[CAP / 2, size_y / 2, CAP_H / 2],
                            mass=CAP_MASS, friction=[0.5, 0.005, 0.0001], rgba=_CAP_RGB + [1])
        if len(name) == 1:
            geom.material = _add_label(spec, name)

    for r, (keys, offset) in enumerate(ROWS):
        for c, key in enumerate(keys):
            add_key(key, x_far - r * PITCH, y_first - (offset + c) * PITCH, CAP)
    add_key("SPACE", x_far - len(ROWS) * PITCH, y_first - SPACE_COL * PITCH,
            SPACE_UNITS * PITCH - (PITCH - CAP))

    side, length = STYLUS_HANDLE
    tip_r, tip_len = STYLUS_TIP
    half = (HOLDER_INNER + HOLDER_WALL) / 2
    yaw = np.radians(STYLUS_YAW)
    quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    for arm, (hx, hy) in STYLUS_POS.items():
        holder = world.add_body(name=f"stylus_holder_{arm}", pos=[hx, hy, TABLE_TOP_Z + HOLDER_H / 2], quat=quat)
        for sx, sy, size in ((1, 0, [HOLDER_WALL / 2, half + HOLDER_WALL / 2, HOLDER_H / 2]),
                             (-1, 0, [HOLDER_WALL / 2, half + HOLDER_WALL / 2, HOLDER_H / 2]),
                             (0, 1, [half - HOLDER_WALL / 2, HOLDER_WALL / 2, HOLDER_H / 2]),
                             (0, -1, [half - HOLDER_WALL / 2, HOLDER_WALL / 2, HOLDER_H / 2])):
            holder.add_geom(type=box, size=size, pos=[sx * half, sy * half, 0], rgba=[0.3, 0.3, 0.32, 1])

        # Stylus frame: the tip end of the handle; handle up, tip down.
        st = world.add_body(name=f"stylus_{arm}", pos=[hx, hy, TABLE_TOP_Z + tip_len + 0.002], quat=quat)
        st.add_freejoint(name=f"stylus_{arm}")
        # solref 2 ms (the stiffest the 1 ms step allows): a pinched handle creeps ~25% less than at 4 ms
        st.add_geom(name=f"stylus_{arm}_handle", type=box, size=[side / 2, side / 2, length / 2],
                    pos=[0, 0, length / 2], mass=STYLUS_MASS * 0.85, friction=[1.0, 0.02, 0.001], condim=4,
                    solref=[0.002, 1.0], rgba=[0.15, 0.4, 0.8, 1])
        st.add_geom(name=f"stylus_{arm}_tip", type=capsule, size=[tip_r, (tip_len - 2 * tip_r) / 2, 0],
                    pos=[0, 0, -tip_len / 2], mass=STYLUS_MASS * 0.15,
                    friction=[1.0, 0.02, 0.001], condim=4, rgba=[0.1, 0.1, 0.1, 1])
