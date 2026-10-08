"""More hand actions with the arm-like wrist of pick.py: place, stack, push, press a button, gestures.

Scripted, no learning: the arm stand-in moves the wrist in straight lines; the hand switches between a few
targets (the grasp pick.py found, open, a gesture). Each action has a success test and runs over many
randomized trials (object / button placement).

  place   pick (pitch -30 deg), carry PLACE_OFFSET across the table, lower until the object's bottom is
          just above the table, open, back off. Success: it rests within PLACE_TOL of the target.
  stack   the same onto a box (the "pedestal"). Success: it rests on top of the box.
  push    fingers pointing down (pitch -90), back of the fingers behind the object, sweep PUSH_M along +Y.
          Success: it moved >= PUSH_MIN_M along +Y and drifted < PUSH_DRIFT_M sideways.
  press   point gesture (index straight, the rest curled), index pointing down, press a spring-loaded
          button placed with +-PRESS_JITTER_M error. Success: the button went down >= PRESS_MIN_M.
  gestures  a still grid of hand poses (open, fist, point, thumbs-up, peace, OK).

    python -m hand_synergies.actions place --trials 300
    MUJOCO_GL=osmesa python -m hand_synergies.actions stack --gif out/stack.gif
    MUJOCO_GL=osmesa python -m hand_synergies.actions gestures --png out/gestures.png
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import mujoco
import numpy as np
from pioneer_humanoid.mujoco_hand import JOINT_NAMES

from .pick import BUTTON_TRAVEL, PEDESTAL_HALF, Picker

PLACE_OFFSET = np.array([0.15, 0.0])
PLACE_TOL = 0.025
RELEASE_GAP = 0.004        # m between the object's bottom and the surface when the hand opens
RELEASE_OPEN_S = 0.4
PUSH_M = 0.15
PUSH_MIN_M = 0.08
PUSH_DRIFT_M = 0.03
PUSH_CLEARANCE = 0.002     # m, the hand's lowest point above the table while pushing
PRESS_JITTER_M = 0.008
PRESS_MIN_M = 0.006
PICK_PITCH = -30.0
DOWN_PITCH = -90.0


def _curl(frac: dict) -> np.ndarray:
    """Hand targets from {joint or joint prefix: fraction of the way from straight to closed}."""
    from pioneer_humanoid.mujoco_hand import FLEX_JOINTS

    from .pick import make_model

    ranges = _RANGES.get("r")
    if ranges is None:
        from pioneer_humanoid.mujoco_hand import joint_ranges

        ranges = _RANGES.setdefault("r", joint_ranges(make_model()))
    q = np.zeros(len(JOINT_NAMES))
    for i, name in enumerate(JOINT_NAMES):
        f = next((v for k, v in frac.items() if name == k or name.startswith(k)), 0.0)
        lo, hi = ranges[i]
        if name in FLEX_JOINTS or name == "MCP_A_thumb":
            closed = hi if abs(hi) > abs(lo) else lo
            q[i] = f * closed
        else:  # spread / circumduction: f is the angle itself (rad)
            q[i] = np.clip(f, lo, hi)
    return q


_RANGES: dict = {}


def gestures() -> dict[str, np.ndarray]:
    fingers = ("MCP_1", "PIP_1", "DIP_1", "MCP_2", "PIP_2", "DIP_2", "MCP_3", "PIP_3", "DIP_3",
               "MCP_4", "PIP_4", "DIP_4")
    fist_fingers = {j: 1.0 for j in fingers}
    tucked_thumb = {"MCP_A_thumb": 0.6, "PIP_thumb": 0.8, "DIP_thumb": 0.8}
    return {
        "open": _curl({}),
        "fist": _curl({**fist_fingers, **tucked_thumb}),
        "point": _curl({**{j: 1.0 for j in fingers if not j.endswith("_1")}, **tucked_thumb}),
        "thumbs-up": _curl(fist_fingers),
        "peace": _curl({**{j: 1.0 for j in fingers if j[-1] in "34"}, **tucked_thumb,
                        "MCP_A_1": -0.35, "MCP_A_2": 0.15}),
        # thumb tip meets the index tip (4.6 mm apart; searched over a small grid)
        "ok": _curl({"MCP_1": 0.45, "PIP_1": 0.5, "DIP_1": 0.4, "MCP_A_thumb": 0.7, "circumduction": -0.3}),
    }


def _lowest_vertex(m: mujoco.MjModel, d: mujoco.MjData, body: str) -> np.ndarray:
    """World position of the lowest collision-mesh vertex of ``body`` (current kinematics)."""
    bid = m.body(body).id
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == bid and m.geom_contype[i])
    mesh = m.geom_dataid[g]
    verts = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
    world = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return world[np.argmin(world[:, 2])]


# --- actions ----------------------------------------------------------------------------------------
def _settled(p: Picker, seconds=0.5) -> bool:
    p.step(seconds, hand=p.data.ctrl[p.gen.idx.act].copy())
    v = p.data.qvel[p.gen.idx.obj_dof:p.gen.idx.obj_dof + 3]
    return float(np.linalg.norm(v)) < 0.02


def _lower_onto(p: Picker, surface_z: float, half_h: float, hand) -> None:
    """Lower the wrist until the held object's bottom is RELEASE_GAP above ``surface_z``."""
    w = p.wrist_pos()
    dz = (p.obj_pos()[2] - half_h) - (surface_z + RELEASE_GAP)
    p.move(w, w - [0, 0, dz], 0.8, hand=hand)


def _release(p: Picker, grip: np.ndarray) -> None:
    """Open gradually (snapping a squeezing hand open can blow the simulation up), then back off."""
    w, open_q = p.wrist_pos(), gestures()["open"]
    n = int(RELEASE_OPEN_S / 0.001)
    for i in range(n):
        p.step(0.001, hand=grip + (open_q - grip) * (i + 1) / n)
    p.step(0.2, hand=open_q)
    p.move(w, w + [0, 0, 0.08], 0.6, hand=open_q)


def place(p: Picker, seed: int, pca, frames=None, stack: bool = False) -> tuple[bool, bool]:
    """(picked up, placed)."""
    def put_pedestal(obj_pos):  # the box to stack on is there from the start
        ped = p.model.body("pedestal").mocapid[0]
        p.data.mocap_pos[ped] = [obj_pos[0] + PLACE_OFFSET[0], obj_pos[1] + PLACE_OFFSET[1], PEDESTAL_HALF[2]]

    g = p.grasp_and_lift(seed, "synergy-3" if pca is not None else "prior", pca, frames,
                         setup=put_pedestal if stack else None)
    if not g["ok"]:
        return False, False
    target = g["start_pos"][:2] + PLACE_OFFSET
    surface = 2 * PEDESTAL_HALF[2] if stack else 0.0
    w = p.wrist_pos()
    above = w + [PLACE_OFFSET[0], PLACE_OFFSET[1], 0.0]
    p.move(w, above, 1.5, hand=g["ctrl"])
    _lower_onto(p, surface, g["half_h"], g["ctrl"])
    _release(p, g["ctrl"])
    if p.gen.unstable() or not _settled(p):
        return True, False
    obj = p.obj_pos()
    if np.linalg.norm(obj[:2] - target) > (PEDESTAL_HALF[0] + 0.01 if stack else PLACE_TOL):
        return True, False
    if stack:
        return True, bool(abs(obj[2] - surface - g["half_h"]) < 0.015)
    return True, bool(obj[2] < g["half_h"] + 0.015)


def push(p: Picker, seed: int, frames=None) -> bool:
    """Back of the straight fingers (pointing down) sweeps the object along +Y."""
    rng = np.random.default_rng(seed)
    m, d, gen = p.model, p.data, p.gen
    p.frames = frames
    kind, size, pos, quat, half_h = p._object(rng)
    hand = gestures()["open"]
    reach = max(size[:2]) if kind != "cylinder" else size[1]  # object's horizontal extent (generous)
    mujoco.mj_resetData(m, d)
    d.qpos[p.pitch_q] = p.pitch
    d.ctrl[p.pitch_act] = p.pitch
    from .scene import set_object

    set_object(m, gen.idx, kind, size)
    d.qpos[gen.idx.qpos] = hand
    mujoco.mj_kinematics(m, d)  # wrist at the origin: where are the fingertips?
    tips = np.array([d.site_xpos[m.site(f"tip_distal_{f}").id] for f in "1234"])
    low = min(p._geom_bottom(gg) for gg in p.hand_geoms)
    start = np.array([pos[0] - tips[:, 0].mean(), pos[1] - reach - 0.025 - tips[:, 1].max(), PUSH_CLEARANCE - low])
    d.qpos[p.wrist_q] = start
    p._set_wrist(start)
    d.ctrl[gen.idx.act] = hand
    d.qpos[gen.idx.obj_qpos:gen.idx.obj_qpos + 3] = pos
    d.qpos[gen.idx.obj_qpos + 3:gen.idx.obj_qpos + 7] = quat
    mujoco.mj_forward(m, d)
    p.step(0.2, hand=hand)
    p.move(start, start + [0, PUSH_M, 0], 2.0, hand=hand)
    p.step(0.3, hand=hand)
    moved = p.obj_pos() - pos
    return bool(not gen.unstable() and moved[1] >= PUSH_MIN_M and abs(moved[0]) < PUSH_DRIFT_M)


def press(p: Picker, seed: int, frames=None) -> bool:
    """Point gesture, index straight down onto a button placed with some error."""
    rng = np.random.default_rng(seed)
    m, d, gen = p.model, p.data, p.gen
    p.frames = frames
    hand = gestures()["point"]
    mujoco.mj_resetData(m, d)
    d.qpos[p.pitch_q] = p.pitch
    d.ctrl[p.pitch_act] = p.pitch
    d.qpos[gen.idx.qpos] = hand
    mujoco.mj_kinematics(m, d)
    tip = _lowest_vertex(m, d, "distal_1")  # what touches first, wrist at the origin (not the tip site)
    base = m.body("button_base").mocapid[0]
    bxy = rng.uniform(-PRESS_JITTER_M, PRESS_JITTER_M, size=2)  # where it really is; the hand aims at 0
    d.mocap_pos[base] = [bxy[0], bxy[1], 0.0]
    top = 0.02 + 0.012
    start = np.array([-tip[0], -tip[1], top + 0.03 - tip[2]])
    d.qpos[p.wrist_q] = start
    p._set_wrist(start)
    d.ctrl[gen.idx.act] = hand
    mujoco.mj_forward(m, d)
    bq = m.joint("button").qposadr[0]
    lowest = [0.0]

    def watch():
        lowest[0] = min(lowest[0], d.qpos[bq])
        p.record()

    p.step(0.2, hand=hand)
    n = int(1.0 / 0.001)
    for i in range(n):  # down 4.5 cm: the tip ends 1.5 cm below the button's resting top
        p._set_wrist(start - [0, 0, 0.045 * (i + 1) / n])
        d.ctrl[gen.idx.act] = hand
        mujoco.mj_step(m, d)
        watch()
    p.step(0.3, hand=hand)
    p.move(p.wrist_pos(), start, 0.6, hand=hand)
    return bool(not gen.unstable() and -lowest[0] >= PRESS_MIN_M)


# --- runner -----------------------------------------------------------------------------------------
_P = {}


def _picker(action: str, fc: bool) -> Picker:
    key = (action, fc)
    if key not in _P:
        pitch = DOWN_PITCH if action in ("push", "press") else PICK_PITCH
        _P[key] = Picker("stock", fc, pitch_deg=pitch, extras=True)
    return _P[key]


def run(action: str, p: Picker, seed: int, pca, frames=None) -> tuple[bool, bool]:
    """(precondition met -- for place/stack the pick worked, success)."""
    if action in ("place", "stack"):
        return place(p, seed, pca, frames, stack=action == "stack")
    return True, {"push": push, "press": press}[action](p, seed, frames)


def _work(args):
    action, seed, fc, pca_path = args
    pca = dict(np.load(pca_path)) if pca_path else None
    return run(action, _picker(action, fc), seed, pca)


def render_gestures(path: str) -> None:
    from PIL import Image, ImageDraw

    p = Picker("stock", False)
    m, d = p.model, p.data
    renderer = mujoco.Renderer(m, 300, 300)
    tiles = []
    for name, q in gestures().items():
        mujoco.mj_resetData(m, d)
        d.qpos[p.wrist_q] = [0, 0, 0.3]
        d.qpos[p.gen.idx.qpos] = q
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.02, 0.1, 0.27]
        cam.distance, cam.azimuth, cam.elevation = 0.36, 200, 25  # from below the palm, thumb side
        renderer.update_scene(d, cam)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((8, 8), name, fill=(255, 255, 255))
        tiles.append(np.asarray(img))
    Image.fromarray(np.hstack(tiles)).save(path)
    print(f"wrote {path}")


def render_gestures_gif(path: str, hold_s: float = 0.8, move_s: float = 0.6) -> None:
    """The hand (servos, gravity on, finger collisions on) moving through every gesture and back to open,
    with a slowly orbiting camera."""
    from PIL import Image, ImageDraw

    p = Picker("stock", True)
    m, d = p.model, p.data
    m.vis.headlight.ambient[:] = 0.35  # the scene light is above; this view looks up at the palm
    m.vis.headlight.diffuse[:] = 0.6
    renderer = mujoco.Renderer(m, 360, 480)
    poses = gestures()
    order = [*poses, "open"]
    mujoco.mj_resetData(m, d)
    d.qpos[p.wrist_q] = [0, 0, 0.3]
    p._set_wrist([0, 0, 0.3])
    q = poses["open"]
    d.qpos[p.gen.idx.qpos] = q
    imgs, frame_dt = [], 0.04

    def shoot(label):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.02, 0.1, 0.27]
        cam.distance, cam.elevation = 0.38, 25
        cam.azimuth = 200 + 40 * np.sin(d.time * 0.6)
        renderer.update_scene(d, cam)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((10, 10), label, fill=(255, 255, 255))
        imgs.append(img)

    for name in order:
        target = poses[name]
        n = int(move_s / frame_dt)
        for i in range(n):  # smooth (cosine) blend to the next pose
            a = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / n)
            p.step(frame_dt, hand=q + (target - q) * a)
            shoot(name)
        q = target
        for _ in range(int(hold_s / frame_dt)):
            p.step(frame_dt, hand=q)
            shoot(name)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(frame_dt * 1000), loop=0)
    renderer.close()
    print(f"wrote {path} ({len(imgs)} frames)")


def render_gif(action: str, path: str, fc: bool, pca, seeds) -> None:
    from PIL import Image

    p = _picker(action, fc)
    for seed in seeds:
        if run(action, p, seed, pca)[1]:
            break
    else:
        raise SystemExit(f"no successful {action} to render")
    renderer = mujoco.Renderer(p.model, 360, 480)
    imgs = []
    lookat = [0.07, 0.0, 0.06] if action in ("place", "stack") else [0.0, 0.05, 0.04]

    class Recorder(list):
        def append(self, t):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = lookat
            cam.distance, cam.azimuth, cam.elevation = 0.5, 135, -18
            renderer.update_scene(p.data, cam)
            imgs.append(Image.fromarray(renderer.render()))
            super().append(t)

    run(action, p, seed, pca, frames=Recorder())
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
    renderer.close()
    print(f"wrote {path} (seed {seed}, {len(imgs)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=("place", "stack", "push", "press", "gestures"))
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--finger-collisions", action="store_true")
    p.add_argument("--synergies", default="out/synergies.npz", help="pre-shapes for place/stack ('' = prior)")
    p.add_argument("--gif", default=None)
    p.add_argument("--png", default="out/gestures.png")
    args = p.parse_args()
    if args.action == "gestures":
        if args.gif:
            render_gestures_gif(args.gif)
        else:
            render_gestures(args.png)
        return
    pca_path = args.synergies if args.action in ("place", "stack") else ""
    if args.gif:
        pca = dict(np.load(pca_path)) if pca_path else None
        render_gif(args.action, args.gif, args.finger_collisions, pca, range(args.trials))
        return
    jobs = [(args.action, s, args.finger_collisions, pca_path) for s in range(args.trials)]
    with mp.Pool() as pool:
        res = np.array(pool.map(_work, jobs, chunksize=8))
    pre, ok = res[:, 0], res[:, 1]
    msg = f"{args.action}: {ok.mean():.1%} of {len(ok)} trials"
    if args.action in ("place", "stack"):
        done = {"place": "placed", "stack": "stacked"}[args.action]
        msg += f" (picked up {pre.mean():.1%}; {ok.sum() / max(pre.sum(), 1):.0%} of those {done})"
    print(msg)


if __name__ == "__main__":
    main()
