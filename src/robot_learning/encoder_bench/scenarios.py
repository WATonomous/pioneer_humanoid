"""Scenario shifts: "weather" and environment changes that hit BOTH the colour image and depth.

The first test shifts (light, color) change only the colour image, so an RGB-D encoder can fall back on untouched
depth. These scenarios model what a real D455 / wrist camera sees when the room changes, including what that does to
depth, so no input gets a free pass (except where physics says it should: visible shadows do not affect depth).

    name        colour image                                   depth
    fog         exponential haze from the true depth           holes growing with range + range noise
    glare       sun through a window: exposure, veiling glare,  IR pattern washed out: holes where the image
                bloom, a flare streak                           saturates
    lens        wet / dirty lens: blurred, brightened blobs     no reading under the blobs
    shadow      one low, hard side light, the rest dimmed      unchanged (shadows do not change geometry)
    clutter     5-9 distractor objects (4-9 cm) on the table            the objects are in depth too
    background  large boxes and a panel around the workspace   in depth too
    table       a new tablecloth: random colour + stripes        ~unchanged (1 mm thick, visual only)
                HELD OUT: added 2026-11-15 before any PioneerNet v6 result existed, because v6's training
                corruptions were designed after seeing v5 fail clutter/background; this one was not looked at.

Each scenario is drawn once per episode from rng [seed, 7] (as gen_data.py draws test shifts); per-frame noise uses
its own generator. Distractor geoms are added to the rendered scene only (mjvScene), never to physics, and are kept
away from every free body (the task objects) and from the path between them, so they never hide the target.

Usage (render loop):
    look = scenarios.make(name, model, data, rng)      # once per episode, after the task objects are placed
    scenarios.apply_model(model, look)                 # model-level changes (lights); caller restores fields
    r.update_scene(data, cam); scenarios.add_geoms(r.scene, look); rgb = r.render()   (same for depth)
    rgb, depth = scenarios.post(look, cam, rgb, depth, frame_rng)
"""
from __future__ import annotations

import numpy as np

SCENARIOS = ("fog", "glare", "lens", "shadow", "clutter", "background", "table")
DEPTH_HIT = {"fog", "glare", "lens", "clutter", "background"}          # also change the depth image
NEEDS_DEPTH = {"fog", "glare", "lens"}                                   # post-processing reads true depth
LIGHT_FIELDS = ("light_diffuse", "light_ambient", "light_specular", "light_pos", "light_dir", "light_castshadow")
TABLE_TOP = 0.78   # m, the lightbox table top (table geom centre 0.39 + half-height 0.39)


def make(name, model, data, rng: np.random.Generator) -> dict:
    """Draw one episode's scenario. data must already hold the episode's object layout."""
    import mujoco
    look = {"name": name, "geoms": [], "seed": int(rng.integers(2 ** 31))}
    if name == "fog":
        look["density"] = rng.uniform(0.7, 1.1)              # 1/m: at 1 m range 50-67% of the light is haze
        look["fog_rgb"] = np.array([0.78, 0.79, 0.80]) * rng.uniform(0.85, 1.05)
    elif name == "glare":
        look["exposure"] = rng.uniform(1.1, 1.3)
        look["sun_xy"] = rng.uniform([0.15, 0.1], [0.85, 0.45])   # glare source in normalised image coordinates
        look["veil"] = rng.uniform(0.15, 0.3)
    elif name == "lens":
        look["blobs"] = {}
    elif name == "shadow":
        look["side"] = rng.choice([-1.0, 1.0])
        look["elev"] = np.radians(rng.uniform(18, 30))
        look["key"] = rng.uniform(0.8, 1.1)
    elif name in ("clutter", "background"):
        free = []
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                free.append(data.xpos[model.jnt_bodyid[j]][:2].copy())
        for g in range(model.ngeom):     # fixed task geoms (e.g. the hole block's walls) count as task objects too
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if n.startswith("wall_"):
                free.append(data.geom_xpos[g][:2].copy())
        look["geoms"] = _clutter(rng, free) if name == "clutter" else _background(rng)
    elif name == "table":
        look["geoms"] = _tablecloth(rng)
    else:
        raise ValueError(name)
    return look


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def _clutter(rng, task_xy):
    """4-8 small objects on the table, >= 9 cm from every task object and 7 cm from the paths between them."""
    out, tries = [], 0
    n = rng.integers(5, 10)
    while len(out) < n and tries < 400:
        tries += 1
        p = rng.uniform([0.24, -0.35], [0.62, 0.55])
        if any(np.linalg.norm(p - q) < 0.09 for q in task_xy):
            continue
        if any(_seg_dist(p, a, b) < 0.07 for i, a in enumerate(task_xy) for b in task_xy[i + 1:]):
            continue
        if any(np.linalg.norm(p - o["pos"][:2]) < 0.06 for o in out):
            continue
        kind = rng.choice(["box", "cyl", "sphere", "capsule"])
        s = rng.uniform(0.02, 0.045)
        h = rng.uniform(0.02, 0.07)
        size, z = {"box": ([s, rng.uniform(0.02, 0.045), h], h), "cyl": ([s, h, 0], h),
                   "sphere": ([s, 0, 0], s), "capsule": ([s * 0.7, h, 0], h + s * 0.7)}[kind]
        yaw = rng.uniform(0, np.pi)
        out.append({"type": kind, "size": np.array(size, float), "pos": np.array([*p, TABLE_TOP + z]),
                    "yaw": yaw, "rgba": np.array([*rng.uniform(0.05, 0.95, 3), 1.0])})
    return out


def _background(rng):
    """A changed workspace: a tall panel behind the table and 2-4 large boxes at the table's far edge and sides."""
    out = [{"type": "box", "size": np.array([0.01, rng.uniform(0.5, 0.75), rng.uniform(0.25, 0.4)]),
            "pos": np.array([rng.uniform(0.76, 0.8), rng.uniform(-0.15, 0.25), TABLE_TOP + 0.3]), "yaw": 0.0,
            "rgba": np.array([*rng.uniform(0.15, 0.9, 3), 1.0])}]
    for _ in range(rng.integers(2, 5)):
        sx, sy, sz = rng.uniform(0.04, 0.09), rng.uniform(0.04, 0.12), rng.uniform(0.05, 0.14)
        side = rng.integers(3)
        p = ([rng.uniform(0.62, 0.72), rng.uniform(-0.4, 0.55)] if side == 0 else
             [rng.uniform(0.3, 0.75), rng.choice([-0.5, 0.65]) + rng.uniform(-0.04, 0.04)])
        out.append({"type": "box", "size": np.array([sx, sy, sz]), "pos": np.array([*p, TABLE_TOP + sz]),
                    "yaw": rng.uniform(0, np.pi), "rgba": np.array([*rng.uniform(0.1, 0.9, 3), 1.0])})
    return out


def _tablecloth(rng):
    """A visual-only cloth over the whole table top (x 0.24-0.84, y -0.76-0.76) in a random colour, with 2-6 stripes
    of a second colour, along or across the table. 1.2 mm above the table so it never hides the task objects' bases."""
    base = np.array([*rng.uniform(0.05, 0.95, 3), 1.0])
    stripe = np.array([*rng.uniform(0.05, 0.95, 3), 1.0])
    out = [{"type": "box", "size": np.array([0.302, 0.765, 0.0006]), "pos": np.array([0.54, 0.0, TABLE_TOP + 0.0006]),
            "yaw": 0.0, "rgba": base}]
    along_x = rng.random() < 0.5                      # stripes run along or across the table, clipped to it
    w = rng.uniform(0.01, 0.05)
    gap = rng.uniform(0.06, 0.16)
    half = 0.30 if not along_x else 0.76               # extent of the axis the stripes are spread over
    t = rng.uniform(-half, -half + gap)
    while t < half - w:
        if along_x:
            out.append({"type": "box", "size": np.array([0.302, w, 0.0003]), "pos": np.array([0.54, t, TABLE_TOP + 0.0015]),
                        "yaw": 0.0, "rgba": stripe})
        else:
            out.append({"type": "box", "size": np.array([w, 0.765, 0.0003]), "pos": np.array([0.54 + t, 0.0, TABLE_TOP + 0.0015]),
                        "yaw": 0.0, "rgba": stripe})
        t += gap
    return out


def apply_model(model, look):
    """Model-level part (lights). The caller snapshots LIGHT_FIELDS and the headlight and restores them."""
    if look["name"] != "shadow":
        return
    model.light_diffuse[:] *= 0.12
    model.light_ambient[:] *= 0.3
    model.light_specular[:] *= 0.3
    model.vis.headlight.ambient[:] = 0.08
    model.vis.headlight.diffuse[:] = 0.1
    i = 3                                          # a light that already casts shadows, re-aimed low from one side
    el, s = look["elev"], look["side"]
    d = np.array([0.35, -s * np.cos(el), -np.sin(el)])
    d /= np.linalg.norm(d)
    target = np.array([0.42, 0.1, TABLE_TOP])
    model.light_dir[i] = d
    model.light_pos[i] = target - d * 1.2
    model.light_diffuse[i] = np.array([1.0, 0.95, 0.85]) * look["key"]
    model.light_castshadow[i] = 1


def add_geoms(scene, look):
    """Append the scenario's distractors to an already-updated mjvScene (call after every update_scene)."""
    if not look["geoms"]:
        return
    import mujoco
    T = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cyl": mujoco.mjtGeom.mjGEOM_CYLINDER,
         "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}
    for o in look["geoms"]:
        if scene.ngeom >= scene.maxgeom:
            break
        c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
        mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float).flatten()
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, T[o["type"]], o["size"], o["pos"], mat, o["rgba"].astype(np.float32))
        g.category = mujoco.mjtCatBit.mjCAT_STATIC
        scene.ngeom += 1


def _blur(x, k):
    """Box blur of an (H, W[, C]) float image, k odd."""
    import cv2
    return x if k < 3 else cv2.blur(np.ascontiguousarray(x, np.float32), (k, k), borderType=cv2.BORDER_REPLICATE)


def _lens_blobs(look, cam, H, W):
    if cam not in look["blobs"]:
        rng = np.random.default_rng([look["seed"], {"ego": 0}.get(cam, 1)])
        yy, xx = np.mgrid[0:H, 0:W]
        a = np.zeros((H, W), np.float32)
        for _ in range(rng.integers(5, 10)):
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            r = rng.uniform(0.04, 0.11) * W
            e = rng.uniform(0.7, 1.3)
            d = ((xx - cx) / r) ** 2 + ((yy - cy) / (r * e)) ** 2
            a = np.maximum(a, np.clip(1.4 - d, 0, 1) ** 0.7)
        look["blobs"][cam] = a
    return look["blobs"][cam]


def post(look, cam, rgb, depth, rng: np.random.Generator):
    """Image-space part. rgb uint8 (H, W, 3) or None; depth float metres (H, W, 0 = no reading) or None.
    fog / glare / lens need the true depth or the image: pass both when the renderer can (NEEDS_DEPTH)."""
    name = look["name"]
    if name == "fog":
        z = depth if depth is not None else None
        if rgb is not None and z is not None:
            t = np.exp(-look["density"] * np.where(z > 0, z, 3.0))[..., None]
            x = rgb / 255.0 * t + look["fog_rgb"] * (1 - t)
            rgb = (np.clip(x + rng.normal(0, 0.01, x.shape), 0, 1) * 255).astype(np.uint8)
        if depth is not None:
            p_hole = np.clip((depth - 0.45) * 0.45 * look["density"], 0, 0.45)   # stereo matching fails in haze
            speck = _blur(rng.random(depth.shape).astype(np.float32), 7)            # holes come in patches
            hole = (speck - 0.5) * 3.5 + 0.5 < p_hole
            depth = np.where(hole, 0.0, depth * (1 + rng.normal(0, 0.01 * look["density"], depth.shape)))
    elif name == "glare":
        img = rgb if rgb is not None else depth
        H, W = img.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]
        sx, sy = look["sun_xy"]
        r2 = ((xx - sx) * W / H) ** 2 + (yy - sy) ** 2
        veil = np.clip(look["veil"] * 2.2 * np.exp(-r2 / 0.15) + np.exp(-r2 / 0.004)
                       + 0.4 * np.exp(-((yy - sy) ** 2) / 0.0008) * np.exp(-((xx - sx) ** 2) / 0.3), 0, 1)
        if rgb is not None:   # veiling glare: light scattered in the lens lifts and washes out the image
            x = np.clip(rgb / 255.0 * look["exposure"], 0, 1)
            x = x * (1 - veil[..., None]) + veil[..., None] * np.array([1.0, 0.96, 0.85])
            rgb = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        if depth is not None:   # sunlight swamps the IR projector pattern: no stereo match where the glare is strong
            hole = _blur(rng.random(depth.shape).astype(np.float32), 5) < np.clip((veil - 0.15) * 1.6, 0, 0.9) * 0.6 + \
                np.clip((veil - 0.15) * 1.6, 0, 0.9) * 0.4
            depth = np.where(hole, 0.0, depth)
    elif name == "lens":
        img = rgb if rgb is not None else depth
        a = _lens_blobs(look, cam, *img.shape[:2])
        if rgb is not None:
            x = rgb / 255.0
            b = _blur(x, 21) * 1.08 + 0.04
            rgb = (np.clip(x * (1 - a[..., None]) + b * a[..., None], 0, 1) * 255).astype(np.uint8)
        if depth is not None:
            depth = np.where(a > 0.35, 0.0, depth)
    return rgb, depth
