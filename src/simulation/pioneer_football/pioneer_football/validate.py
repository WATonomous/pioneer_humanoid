"""Run CPU model/physics checks and write machine-readable results."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .model import JOINTS, PARAMS, URDF, audit_source, ground_spec, rest_data


def run_case(dt, seed, duration=10.0, push=False, friction=0.8):
    m = ground_spec(timestep=dt).compile()
    for side in ("l", "r"):
        m.geom_friction[m.geom(f"{side}_foot_collision_0").id, 0] = friction
    d = rest_data(m)
    assert m.nu == 12 and m.nv == 18 and m.neq == 0
    assert np.count_nonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
    assert not np.any(m.body_gravcomp)
    assert abs(m.body_mass.sum() - audit_source()) < 1e-6
    assert not any(c.dist < -1e-4 for c in d.contact), "Initial self penetration"
    assert d.xpos[m.body("l_foot").id, 1] > d.xpos[m.body("r_foot").id, 1]
    # Prove gravity acts on a free body, independently of its initial support.
    drop = mujoco.MjData(m)
    drop.qpos[:] = d.qpos
    drop.qpos[2] += 3.0
    mujoco.mj_forward(m, drop)
    z0 = drop.subtree_com[1, 2]
    mujoco.mj_step(m, drop, nstep=round(0.1 / dt))
    mujoco.mj_forward(m, drop)
    fall_distance = float(z0 - drop.subtree_com[1, 2])
    assert abs(fall_distance - 0.04905) < 0.003
    if seed:
        rng = np.random.default_rng(seed)
        for name in JOINTS:
            d.qpos[m.joint(name).qposadr[0]] += rng.uniform(-0.005, 0.005)
        d.qpos[2] += 0.015
        mujoco.mj_forward(m, d)
    start = d.qpos[:3].copy()
    feet = {m.body("l_foot").id, m.body("r_foot").id}
    peak_effort, peak_speed, min_height = 0.0, 0.0, float(d.qpos[2])
    floor_contacts, fallen = 0, False
    for i in range(round(duration / dt)):
        if push:
            d.xfrc_applied[m.body("base").id, 0] = 30.0 if 1.0 <= d.time < 1.1 else 0.0
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError("Non-finite physics state")
        min_height = min(min_height, float(d.qpos[2]))
        peak_effort = max(
            peak_effort,
            float(np.max(np.abs(d.actuator_force) / m.actuator_forcerange[:, 1])),
        )
        for name in JOINTS:
            speed = (
                abs(d.qvel[m.joint(name).dofadr[0]]) / PARAMS[name.rsplit("_", 1)[0]][3]
            )
            peak_speed = max(peak_speed, float(speed))
        for c in d.contact:
            bodies = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
            if 0 in bodies:
                floor_contacts += 1
                if not bodies.intersection(feet):
                    fallen = True
        fallen |= d.qpos[2] < 0.55
    assert floor_contacts > 0
    assert not np.any(d.warning.number), str(d.warning.number)
    assert peak_effort <= 1.0001
    return {
        "timestep": dt,
        "seed": seed,
        "duration_s": duration,
        "push_30N_100ms": push,
        "sole_friction": friction,
        "mass_kg": float(m.body_mass.sum()),
        "freefall_drop_m": fall_distance,
        "initial_height_m": float(start[2]),
        "minimum_height_m": min_height,
        "horizontal_drift_m": float(np.linalg.norm(d.qpos[:2] - start[:2])),
        "peak_effort_fraction": peak_effort,
        "peak_velocity_fraction": peak_speed,
        "fixed_command_standing": not bool(fallen),
        "numerical_checks_pass": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/validation.json"))
    p.add_argument(
        "--render", type=Path, help="Optional CAD-mesh render of the initial stance"
    )
    args = p.parse_args()
    rows = [run_case(dt, seed) for dt in (0.001, 0.002) for seed in (0, 1, 2)]
    rows.append(run_case(0.002, 0, push=True))
    rows.extend(run_case(0.002, 1, friction=mu) for mu in (0.6, 1.0))
    result = {
        "source_urdf": str(URDF),
        "mujoco_version": mujoco.__version__,
        "cases": rows,
        "limitations": [
            "Upper body rigid at defined pose; no arm balance control.",
            "Position servos: capped effort, assumed gains/armature, no identified torque-speed curve or latency.",
            "Perturbed standing is a diagnostic, not a walking or push-recovery result.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.render:
        from PIL import Image

        m = ground_spec(visuals=True).compile()
        d = rest_data(m)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.8]
        cam.distance = 2.8
        cam.azimuth = 135
        cam.elevation = -15
        with mujoco.Renderer(m, height=480, width=640) as r:
            r.update_scene(d, camera=cam)
            args.render.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(r.render()).save(args.render)
    # Require the unperturbed support checks to hold. Disturbances are diagnostics.
    if not all(
        r["fixed_command_standing"]
        for r in rows
        if r["seed"] == 0 and not r["push_30N_100ms"]
    ):
        raise SystemExit("Nominal fixed-command support failed; inspect the report.")


if __name__ == "__main__":
    main()
