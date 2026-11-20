"""Does every joint move cleanly? Each whole-body model on a fixed base, under gravity, every joint's motor a position
servo capped at the URDF's effort limit. Each joint in turn is swept back and forth over most of its range (a 2 s sine, at
most half its velocity limit, the others holding), then all of them at once: how closely it follows, how fast it goes against its velocity limit,
and whether anything blows up (a joint spinning off, a NaN, a solver warning) - the "wonky at certain torques" sort of
trouble a near-massless link or an impossible inertia brings.

    python check_motion.py [model.urdf ...]        # default: both whole-body URDFs here

A joint fails if it lags its target by more than 10 degrees (10 mm for a jaw) at any point, or the model blows up.
MuJoCo is more forgiving than PhysX (Isaac Sim): passing here is necessary, not sufficient.
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERIOD, CYCLES = 2.0, 2
LAG_RAD, LAG_M = np.radians(10.0), 0.010


def load(urdf):
    text = open(urdf).read()
    root = ET.fromstring(text)
    limits = {j.get("name"): (float(j.find("limit").get("effort")), float(j.find("limit").get("velocity")))
              for j in root.iter("joint") if j.find("limit") is not None}
    mimics = {j.get("name") for j in root.iter("joint") if j.find("mimic") is not None}
    text = re.sub(r"<visual>.*?</visual>", "", text, flags=re.S).replace('discardvisual="false"', "")
    spec = mujoco.MjSpec.from_string(text)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for g in spec.geoms:
        g.contype = g.conaffinity = 0
    for j in spec.joints:
        if j.name in limits and j.name not in mimics:
            e, _ = limits[j.name]
            slide = j.type == mujoco.mjtJoint.mjJNT_SLIDE
            kp = e / (0.002 if slide else 0.02)            # full effort at ~1.1 deg (2 mm) of lag
            a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=kp, kv=0.01 * kp)              # (light damping: its own lag ~0.6 deg at 1 rad/s)
            a.forcelimited, a.forcerange = True, [-e, e]
            j.armature = 0.01
    return spec.compile(), limits


def run(m, sweep, vmax):
    """sweep: actuator ids to move (the rest hold at rest). Returns per-actuator max lag, peak speed; blew_up"""
    d = mujoco.MjData(m)
    lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
    jid = m.actuator_trnid[:, 0]
    for a in range(m.nu):
        lo[a], hi[a] = m.jnt_range[jid[a]]
    # over 80 % of its range, but never asked past half its velocity limit (a sine's peak speed: amp 2 pi / period)
    mid, amp = (lo + hi) / 2, np.minimum(0.4 * (hi - lo), 0.5 * vmax * PERIOD / (2 * np.pi))
    rest = np.clip(0.0, lo, hi)
    d.qpos[m.jnt_qposadr[jid]] = rest
    d.ctrl[:] = rest
    mujoco.mj_forward(m, d)
    lag, speed = np.zeros(m.nu), np.zeros(m.nu)
    t_end = PERIOD * CYCLES + 0.5
    while d.time < t_end:
        ramp = min(1.0, d.time / 0.5)                  # eased in from rest
        phase = np.sin(2 * np.pi * d.time / PERIOD)
        target = rest.copy()
        target[sweep] = rest[sweep] + ramp * ((mid[sweep] - rest[sweep]) + amp[sweep] * phase)
        d.ctrl[:] = np.clip(target, lo, hi)
        mujoco.mj_step(m, d)
        q = d.qpos[m.jnt_qposadr[jid]]
        v = d.qvel[m.jnt_dofadr[jid]]
        if not np.all(np.isfinite(d.qpos)) or np.abs(d.qvel).max() > 100.0:
            return lag, speed, True
        if d.time > 0.6:
            lag = np.maximum(lag, np.abs(q - d.ctrl))
            speed = np.maximum(speed, np.abs(v))
    warn = any(d.warning[i].number for i in range(len(d.warning)))
    return lag, speed, warn


def main(paths):
    bad = 0
    for p in paths:
        m, limits = load(p)
        names = [m.joint(m.actuator_trnid[a, 0]).name for a in range(m.nu)]
        slide = [m.jnt_type[m.actuator_trnid[a, 0]] == mujoco.mjtJoint.mjJNT_SLIDE for a in range(m.nu)]
        fails, worst = [], []
        vmaxes = np.array([limits[n][1] for n in names])
        for a in range(m.nu):                           # one joint at a time
            lag, speed, blew = run(m, np.array([a]), vmaxes)
            tol = LAG_M if slide[a] else LAG_RAD
            vmax = limits[names[a]][1]
            worst.append((lag[a] / tol, names[a], lag[a], speed[a], vmax, slide[a]))
            if blew or lag[a] > tol:
                fails.append(f"{names[a]}: {'blew up' if blew else 'lags'}")
        lag, speed, blew = run(m, np.arange(m.nu), vmaxes)     # all together
        if blew:
            fails.append("all joints together: blew up")
        print(f"{os.path.relpath(p)}: {m.nu} joints swept one at a time, then all at once")
        for r, n, lg, sp, vm, sl in sorted(worst, reverse=True)[:6]:
            u, k = ("mm", 1000.0) if sl else ("deg", np.degrees(1.0))
            print(f"  {n:18s} max lag {lg * k:6.2f} {u:3s}  peak speed {sp:6.2f} (limit {vm:g})")
        print("  all together:", "blew up" if blew else "stable")
        for f in fails:
            print("  FAIL", f)
        bad += len(fails)
    print("every joint moves cleanly" if not bad else f"{bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    default = [os.path.join(HERE, f) for f in ("pioneer_whole_body_humanoid.urdf", "pioneer_whole_body_humanoid_gripper.urdf")]
    sys.exit(main(sys.argv[1:] or default))
