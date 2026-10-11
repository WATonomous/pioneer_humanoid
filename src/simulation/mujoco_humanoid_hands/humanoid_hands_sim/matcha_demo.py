"""Scripted matcha by the Pioneer humanoid, both hands: scoop the matcha in, pour the water, whisk (the right
hand holds the cup), serve the cup on the tray. Open loop on a script, closed loop on where each tool actually
sits in the hand. Records qpos at 30 Hz for render.py.

    python -m humanoid_hands_sim.matcha_demo --out matcha.npy      # ~75 s simulated, ~10 min on 4 CPU cores
    python -m humanoid_hands_sim.render matcha.npy matcha.mp4
"""
import argparse
import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from humanoid_mujoco_scenes.matcha import scene as M

from . import matcha_hands as S
from .control import READY, Sim
from .robot import HANDSHAKE

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--out", default="matcha.npy", help="qpos record (30 Hz) for render.py")
ap.add_argument("--phases", default="ladle,pitcher,whisk,serve", help="comma list: ladle, pitcher, whisk, serve")
ap.add_argument("--verbose", action="store_true", help="print the targets and where things really went")
args = ap.parse_args()
only = args.phases.split(",")
sim = Sim()
m, d = sim.m, sim.d
def log(msg):
    if args.verbose:
        print(f"    {d.time:5.1f}s {msg}", flush=True)
def Rx(deg): return Rot.from_euler("x", deg, degrees=True).as_matrix()
def report(tag):
    k, n, txt = S.progress(m, d)
    print(f"[{d.time:5.1f}s] {tag}: step {k}/{n}  matcha in cup {M.balls_in_cup(m, d, 'powder')}/8  water {M.balls_in_cup(m, d, 'water')}/24  "
          f"whisk path {M.whisk_path(d):.2f} m", flush=True)

def post_grasp_point(geom, below_top=S.GRASP_BELOW_TOP):
    """Point on the handle geom's axis `below_top` under its top, from the geom's actual (possibly leaning) pose."""
    gid = m.geom(geom).id
    return d.geom_xpos[gid] + d.geom_xmat[gid].reshape(3, 3) @ [0, 0, m.geom_size[gid][2] - below_top]

def grab(side, geom, sign=1, g=None, c=0.9, thumb=0.8, tool=None, short=0.006, until=None, follow_yaw=True):
    """Handshake-grasp a square post handle (or point g), coming in from the hand's outside (+Y for the left
    hand, sign=-1: -Y for the right). follow_yaw: turn the hand to the handle's actual heading (a jug's handle
    points along its body +X; it turns a little when things bump it)."""
    g = post_grasp_point(geom) if g is None else g
    out = np.array([0.0, sign, 0.0])
    if follow_yaw and geom:
        x = d.xmat[m.geom(geom).bodyid[0]].reshape(3, 3)[:, 0]
        x = x[:2] / np.linalg.norm(x[:2]); yaw = math.atan2(x[1], x[0]) - math.atan2(sign, 0.0)
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw) > math.radians(35): raise RuntimeError(f"{geom} turned {math.degrees(yaw):.0f} deg")
        out = np.array([x[0], x[1], 0.0])
    else:
        yaw = 0.0
    Rg = Rot.from_euler("z", yaw).as_matrix() @ HANDSHAKE
    sim.set_hand(side, c=0.0, thumb=0.0)
    # The little finger hangs ~60 mm below the grasp point: come over high, then down beside the post.
    sim.move(side, g + 0.08 * out + [0, 0, 0.17], Rg, 1.2)
    sim.move(side, g + 0.08 * out, t=0.8)
    g = g + short * out                            # stop short of the palm: the closing fingers pull the post in
    sim.move(side, g, t=0.8); sim.wait(0.1)
    log(f"{side} at {geom or tool}: site error {np.linalg.norm(sim.site(side)[0] - g) * 1000:.0f} mm, post at "
        f"{(post_grasp_point(geom) if geom else g).round(3)}, hand at {sim.site(side)[0].round(3)}")
    if until:                                      # close fingers and thumb together until they touch
        sim.ramp_hand(side, c, thumb, t=1.0, until=tool or geom.split("_")[0])
    else:
        sim.ramp_hand(side, c, 0.0, t=0.6); sim.wait(0.1)      # fingers wrap the post, then the thumb closes over it
        sim.ramp_hand(side, c, thumb, t=0.4); sim.wait(0.1)
    sim.attach(side, tool or geom.split("_")[0])
    return g

POUR_REST = np.array([0.24, -0.84, 0.36, -1.49, 2.43, -0.3, -0.02])   # elbow out, forearm rolled (IK search)
def pour(side, body, lip_off, up=0.12, angle=110, hold=1.5, h=0.15, shakes=3, dy=0.0):
    """Lift the held tool, bring it over the cup and roll it about X so its contents fall over the -Y lip.

    Closed loop on where the tool actually sits in the hand (it twists a little in the grip when carried)."""
    def container():
        p, R = sim.body(body); return p + R @ np.array(lip_off)
    def v_hand():                                  # site -> container centre, in the site's frame
        s, Rs = sim.site(side); return Rs.T @ (container() - s)
    s, _ = sim.site(side)
    sim.move(side, s + [0, 0, up], t=0.8)
    cup, _ = sim.body("cup")
    over = cup + [0, dy, h]                        # container centre goes here (dy: its mouth over the cup)
    for t in (1.2, 0.5, 0.3):                      # re-aim twice on the measured offset
        sim.move(side, over - HANDSHAKE @ v_hand() + [0, 0.02, 0], HANDSHAKE, t=t)
    vh = v_hand()
    log(f"over the cup: container {container().round(3)} cup {cup.round(3)}")
    rest = sim.arms[side].rest; sim.arms[side].rest = POUR_REST
    for a in (30, 60, angle):
        R = Rx(a) @ HANDSHAKE
        sim.move(side, over + [0, 0, -0.01] - R @ vh, R, t=0.7)
        log(f"pour {a}: tool tilt {np.degrees(np.arccos(min(1, sim.body(body)[1][2, 2]))):.0f} deg, container "
            f"{container().round(3)}, site error {np.linalg.norm(sim.site(side)[0] - sim.tgt[side][0]) * 1000:.0f} mm")
    for k in range(shakes):                        # tap it out: rock the roll a little
        for da in (-15, 12):
            R = Rx(angle + da) @ HANDSHAKE; sim.move(side, over + [0, 0, -0.01] - R @ vh, R, t=0.18)
    sim.wait(hold)
    sim.move(side, over - HANDSHAKE @ vh + [0, 0.02, 0], HANDSHAKE, t=1.0)
    sim.arms[side].rest = rest

def upright_pose(side, tool, place):
    """Site target and rotation that put `tool` upright (same heading) with its origin at `place`: the grasp
    tips a tool 10-25 deg in the hand, and set down tipped it falls over."""
    s_, Rs = sim.site(side); p, Rt = sim.body(tool)
    yaw = math.atan2(Rt[1, 0], Rt[0, 0]); Rt_des = Rot.from_euler("z", yaw).as_matrix()
    rel_R, rel_p = Rt.T @ Rs, Rt.T @ (s_ - p)                  # site in the tool's frame
    return place + Rt_des @ rel_p, Rt_des @ rel_R

BALLS = [f"powder{i}" for i in range(M.POWDER["count"])] + [f"water{i}" for i in range(M.WATER["count"])]
def free_spot(tool, home_p):
    """Nearest spot to home_p (2 cm grid, up to 6 cm off) with no spilled ball under the tool's footprint:
    a tool set down on a 10 mm ball tips and falls over when let go."""
    _, Rt = sim.body(tool)
    on_table = [d.xpos[m.body(b).id] for b in BALLS if d.xpos[m.body(b).id][2] < S.T + 0.015]
    reach = 0.065 if tool != "whisk" else 0.0       # the whisk goes into its holder, nothing to tip it there
    pts = [np.array([0, 0, 0]), Rt @ [0.07, 0, 0]] if tool != "whisk" else []   # container centre, post
    best = None
    for dx in (0, 0.02, -0.02, 0.04, -0.04, 0.06):
        for dy in (0, 0.02, -0.02, 0.04, -0.04):
            c = home_p + [dx, dy, 0]
            clear = all(np.hypot(*(b - (c + q))[:2]) > reach for b in on_table for q in pts) if reach else True
            if clear and (best is None or np.hypot(dx, dy) < np.hypot(*(best - home_p)[:2])): best = c
    if best is None: best = home_p
    if np.linalg.norm(best - home_p) > 0: print(f"  {tool}: spilled balls on its spot, setting it down {np.round((best - home_p) * 1000)} mm off")
    return best

def put_back(side, tool, up=1.05):
    """Carry the held tool back over a high waypoint (it and the open fingers hang well below the palm) and set it
    down upright where it started, aiming the tool itself (closed loop on where it sits in the hand), then let go."""
    home_p = free_spot(tool, HOME[tool])
    s_, _ = sim.site(side)
    sim.move(side, [s_[0], s_[1], max(s_[2], up)], HANDSHAKE, t=0.6)
    p, R = upright_pose(side, tool, home_p); sim.move(side, [p[0], p[1], up], R, t=1.2)
    for dz, t in ((0.06, 0.8), (0.04, 0.4), (0.003, 0.7)):
        p, R = upright_pose(side, tool, home_p + [0, 0, dz]); sim.move(side, p, R, t=t)
    sim.wait(0.2)
    sim.detach(side); sim.ramp_hand(side, 0.0, 0.0, t=0.5); sim.wait(0.2)
    # Up first: the opened thumb still reaches past the post, and backing straight out would hook it.
    s_, _ = sim.site(side); out = np.array([0, 0.08 if side == "l" else -0.08, 0])
    sim.move(side, s_ + [0, 0, 0.07], t=0.7); sim.move(side, s_ + out + [0, 0, 0.17], HANDSHAKE, t=0.8)

HOME = {n: sim.body(n)[0] for n in ("ladle", "pitcher", "whisk", "cup")}   # where each tool starts
if "ladle" in only:
    g = grab("l", "ladle_handle")
    pour("l", "ladle", [0, 0, 0.015], angle=95, h=0.11, shakes=0, dy=0.015)
    put_back("l", "ladle"); report("ladle")
if "pitcher" in only:
    g = grab("l", "pitcher_handle")
    pour("l", "pitcher", [0, 0, 0.03], angle=105, h=0.135, shakes=0, dy=0.035, hold=2.0)
    put_back("l", "pitcher"); report("pitcher")
def home(side):
    """Back to the ready pose over a high waypoint (the little finger hangs ~60 mm under the grasp point)."""
    s_, _ = sim.site(side); ready = READY[side]
    sim.move(side, [s_[0], s_[1], max(s_[2], 1.03)], HANDSHAKE, 0.5)
    sim.move(side, [ready[0], ready[1], 1.03], HANDSHAKE, 1.0)
    sim.move(side, ready, HANDSHAKE, 0.5)
if "whisk" in only or "serve" in only:
    # Right hand takes the cup by its handle, from the outside (-Y), and holds it while the left hand whisks.
    gc = grab("r", "cup_handle", sign=-1, until=True)          # the full cup is heavy: stop closing once gripped
if "whisk" in only:
    g = grab("l", "whisk_handle", g=post_grasp_point("whisk_handle", 0.03), follow_yaw=False)
    def tip():
        return d.site_xpos[m.site("whisk_tip").id].copy()
    def tip_v():
        s_, Rs = sim.site("l"); return Rs.T @ (tip() - s_)
    w0, _ = sim.body("whisk")                                                  # straight up out of the holder,
    for dz, t in ((0.0, 0.4), (0.06, 0.8), (0.12, 0.6)):                       # held upright (the grasp tips it)
        p_, R_ = upright_pose("l", "whisk", w0 + [0, 0, dz]); sim.move("l", p_, R_, t=t)
    cup, _ = sim.body("cup")
    s_, _ = sim.site("l"); sim.move("l", [s_[0], s_[1], 1.06], t=0.6)          # high: the tines hang ~10 cm under the palm
    sim.move("l", [cup[0], cup[1], 1.06] - HANDSHAKE @ tip_v() * [1, 1, 0], HANDSHAKE, t=1.4)
    for z, t in ((0.15, 0.8), (0.15, 0.4), (0.035, 1.0)):                   # over the cup, then tines in
        sim.move("l", cup + [0, 0, z] - HANDSHAKE @ tip_v(), HANDSHAKE, t=t)
    vh = tip_v(); c0 = cup + [0, 0, 0.055]
    rad, revs, per = 0.018, 12, 0.9
    n = int(revs * per / 0.01)
    for i in range(n + 1):                                                  # circles, as with a chasen
        a = 2 * math.pi * i / (per / 0.01)
        r_ = rad * min(1.0, i / 30)
        sim.tgt["l"] = (c0 + [r_ * math.cos(a), r_ * math.sin(a), 0] - HANDSHAKE @ vh, HANDSHAKE); sim.tick()
    report("whisked")
    sim.move("l", c0 + [0, 0, 0.12] - HANDSHAKE @ vh, t=0.8)
    put_back("l", "whisk")
    home("l")
if "serve" in only:
    tray = np.array([np.mean(S.TRAY[0]), np.mean(S.TRAY[1]), S.T + 0.006])
    s_, _ = sim.site("r"); cup, _ = sim.body("cup"); off = s_ - cup              # site relative to the cup
    sim.move("r", s_ + [0, 0, 0.03], t=0.6)
    sim.move("r", tray + off + [0, 0, 0.03], t=1.6)
    sim.move("r", tray + off + [0, 0, 0.002], t=0.6); sim.wait(0.3)
    sim.detach("r"); sim.ramp_hand("r", 0.0, 0.0, t=0.6); sim.wait(0.4)
    s_, _ = sim.site("r"); sim.move("r", s_ + [0, 0, 0.07], t=0.7); sim.move("r", s_ + [0, -0.08, 0.12], t=0.6); home("r")
    sim.wait(1.0); report("served")
np.save(args.out, np.array(sim.traj))
report("end")
for n in ("ladle", "pitcher", "whisk", "cup"):
    p, R = sim.body(n); print(f"  {n:8s} at {p.round(3)} tilt {np.degrees(np.arccos(min(1, R[2, 2]))):.0f} deg")
