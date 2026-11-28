"""Current demo regressions: physics, navigation, policy history, handoffs and match rules."""

import itertools
import math
import unittest
from dataclasses import asdict, replace
from itertools import pairwise
from types import SimpleNamespace as N

import mujoco
import numpy as np
import torch
from mjlab.utils.buffers import CircularBuffer
from mjviser.conversions import get_geom_texture_id
from tensordict import TensorDict

from pioneer_football.approach import (
    ApproachCfg,
    ApproachController,
    plan_route,
    positioning_target,
    segment_clearance,
)
from pioneer_football.ball import (
    DEFAULT_BALL_CFG,
    DEFAULT_PITCH_CFG,
    BallCfg,
    add_ball,
    add_contact_pairs,
    add_pitch,
    physics_spec,
)
from pioneer_football.demo import (
    PolicyHistory,
    anticipatory_strike_window,
    ball_path_command,
    blend_fraction,
    canonical_history,
    field_failure,
    goal_target,
    immediate_strike_command,
    policy_role,
    preparation_speed,
    ready_for_strike,
    recovery_command,
    retain_dribbling,
    retry_unreleased_strike,
    shadow_target,
    skill_changes,
    stable_shot_motion,
    stationary_restart,
    striker_contact,
    travel_command,
    turning_command,
    turning_navigation,
)
from pioneer_football.match_audit import MatchAudit
from pioneer_football.match_rules import MatchRules
from pioneer_football.model import JOINTS, REST, audit_source, ground_spec, rest_data
from pioneer_football.navigation import StagedApproachController, StagedNavigationCfg
from pioneer_football.navigation_frame import in_navigation_frame
from pioneer_football.runtime import actuated_robot_spec as original
from pioneer_football.runtime import directed_release, make_env_cfg, robot_spec
from pioneer_football.strike_history import restore_buffer, rotate_history
from pioneer_football.twist_control import heading_twist, waypoint_twist


def ball_energy(model, data, cfg):
    j = model.joint("ball_root")
    q, v = j.qposadr[0], j.dofadr[0]
    vel = data.qvel[v : v + 6]
    return float(
        0.5 * cfg.mass * np.dot(vel[:3], vel[:3])
        + 0.5 * cfg.inertia * np.dot(vel[3:], vel[3:])
        - cfg.mass * model.opt.gravity[2] * data.qpos[q + 2]
    )


def passive_case(kind, cfg=DEFAULT_BALL_CFG, timestep=0.0005):
    # Measure free rolling on the original unbounded floor, without goal-net impacts.
    model = physics_spec(cfg, timestep, stadium=False).compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = [0, 0, cfg.radius]
    duration = {"rest": 5, "freefall": 0.1, "rolling": 30, "sliding": 0.3, "bounce": 3}[
        kind
    ]
    if kind in ("freefall", "bounce"):
        data.qpos[2] += 1.0
    if kind in ("rolling", "sliding"):
        data.qvel[0] = 2.0
    if kind == "rolling":
        data.qvel[4] = 2.0 / cfg.radius
    mujoco.mj_forward(model, data)
    initial_energy = ball_energy(model, data, cfg)
    trace = []
    max_energy = initial_energy
    penetration = 0.0
    for _ in range(round(duration / timestep)):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"Nonfinite {kind} state")
        max_energy = max(max_energy, ball_energy(model, data, cfg))
        penetration = max(penetration, cfg.radius - data.qpos[2])
        trace.append([data.time, *data.qpos[:3], *data.qvel])
    a = np.array(trace)
    checks = {
        "finite_no_warnings": not bool(np.any(data.warning.number)),
        "penetration_below_5mm": penetration < 0.005,
        # Discrete stepping can have small energy error; forbid material gain.
        "passive_energy_bound": max_energy <= initial_energy * 1.02 + 1e-4,
    }
    row = {
        "kind": kind,
        "timestep": timestep,
        "parameters": asdict(cfg),
        "max_penetration_m": penetration,
        "max_energy_over_initial": max_energy / initial_energy,
    }
    if kind == "rest":
        row["horizontal_drift_m"] = float(np.linalg.norm(data.qpos[:2]))
        checks["rest_drift_below_1cm"] = row["horizontal_drift_m"] < 0.01
    elif kind == "freefall":
        row["drop_m"] = float(cfg.radius + 1 - data.qpos[2])
        checks["gravity_error_below_2mm"] = (
            abs(row["drop_m"] - 0.5 * 9.81 * duration**2) < 0.002
        )
    elif kind == "rolling":
        speed = a[:, 4]
        moving = (a[:, 0] >= 1) & (a[:, 0] <= 5)
        row["deceleration_mps2"] = float(-np.polyfit(a[moving, 0], speed[moving], 1)[0])
        stop = np.where(np.abs(speed) < 0.02)[0]
        row["stop_time_s"] = float(a[stop[0], 0]) if len(stop) else None
        row["distance_m"] = float(data.qpos[0])
        checks["deceleration_in_range"] = 0.05 < row["deceleration_mps2"] < 0.6
        checks["stops_without_reversal"] = (
            row["stop_time_s"] is not None and float(speed.min()) > -0.02
        )
    elif kind == "sliding":
        early = a[np.argmin(abs(a[:, 0] - 0.1))]
        row["slip_at_100ms_mps"] = float(abs(early[4] - cfg.radius * early[8]))
        row["final_slip_mps"] = float(abs(data.qvel[0] - cfg.radius * data.qvel[4]))
        row["final_spin_rps"] = float(data.qvel[4])
        checks["friction_reduces_slip_and_generates_spin"] = (
            row["final_slip_mps"] < 0.3 and row["final_spin_rps"] > 1
        )
    else:
        h = a[:, 3] - cfg.radius
        peaks = (
            np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]) & (h[1:-1] > 0.002))[0] + 1
        )
        row["rebound_heights_m"] = h[peaks].tolist()
        first = row["rebound_heights_m"][0] if len(peaks) else 0
        row["effective_restitution"] = float(np.sqrt(first))
        checks["first_rebound_in_range"] = 0.2 < first < 0.9
        checks["successive_bounces_dissipate"] = len(peaks) >= 2 and bool(
            np.all(np.diff(h[peaks]) < 0)
        )
    checks = {key: bool(value) for key, value in checks.items()}
    row["checks"] = checks
    row["passed"] = all(checks.values())
    return row


def impact_case(speed=1.0, offset=0.0, cfg=DEFAULT_BALL_CFG, timestep=0.0005):
    """Passive finite-mass impactor with the real left-foot geom and inertia.

    Gravity is zero in this isolated momentum/energy check; no prescribed motion,
    PD control, or injected ball velocity supplies an unbounded energy source.
    """
    source = ground_spec().compile()
    stance = rest_data(source)
    gid = source.geom("l_foot_collision_0").id
    bid = source.geom_bodyid[gid]
    rotation = stance.xmat[bid].reshape(3, 3)
    geom_rot = stance.geom_xmat[gid].reshape(3, 3)
    extent = float(np.abs(geom_rot[0]) @ source.geom_size[gid])
    spec = mujoco.MjSpec()
    spec.option.timestep = timestep
    spec.option.gravity = [0, 0, 0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    add_pitch(spec)
    add_ball(spec, cfg, position=[extent + cfg.radius + 0.06, offset, 1])
    foot = spec.worldbody.add_body(
        name="foot",
        pos=np.array([0, 0, 1]) - rotation @ source.geom_pos[gid],
        quat=stance.xquat[bid],
        mass=float(source.body_mass[bid]),
        inertia=source.body_inertia[bid],
        ipos=source.body_ipos[bid],
        iquat=source.body_iquat[bid],
        explicitinertial=True,
    )
    foot.add_freejoint(name="foot_root")
    foot.add_geom(
        name="foot_geom",
        type=source.geom_type[gid],
        size=source.geom_size[gid],
        pos=source.geom_pos[gid],
        quat=source.geom_quat[gid],
        rgba=[0.2, 0.2, 0.25, 1],
    )
    add_contact_pairs(spec, cfg, feet=("foot_geom",))
    model = spec.compile()
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    data = mujoco.MjData(model)
    fv = model.joint("foot_root").dofadr[0]
    bv = model.joint("ball_root").dofadr[0]
    data.qvel[fv] = speed
    mujoco.mj_forward(model, data)
    initial_ke = float(data.energy[1])
    momentum0 = source.body_mass[bid] * speed
    contacts, max_energy, max_penetration = 0, initial_ke, 0.0
    for _ in range(round(0.6 / timestep)):
        mujoco.mj_step(model, data)
        mujoco.mj_energyVel(model, data)
        max_energy = max(max_energy, float(data.energy[1]))
        for c in data.contact:
            if {int(c.geom1), int(c.geom2)} == {
                model.geom("ball_geom").id,
                model.geom("foot_geom").id,
            }:
                contacts += 1
                max_penetration = max(max_penetration, -c.dist)
    mujoco.mj_forward(model, data)
    # cvel is COM-centered, so angular momentum around an offset inertial frame
    # does not contaminate this linear momentum check.
    momentum = sum(
        model.body_mass[b] * data.cvel[b, 3]
        for b in (model.body("ball").id, model.body("foot").id)
    )
    vx, vy = float(data.qvel[bv]), float(data.qvel[bv + 1])
    checks = {
        "finite_no_warnings": bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and not np.any(data.warning.number)
        ),
        "actual_contact": contacts > 0 if speed else contacts == 0,
        "forward_transfer": vx > 0.05 if speed else abs(vx) < 1e-8,
        "passive_energy_bound": max_energy <= initial_ke * 1.02 + 1e-6,
        "momentum_error_below_1percent": abs(momentum - momentum0)
        <= abs(momentum0) * 0.01 + 1e-6,
        "penetration_below_5mm": max_penetration < 0.005,
    }
    if offset:
        checks["lateral_deflection"] = vy * offset > 0
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "kind": "foot_impact",
        "timestep": timestep,
        "foot_speed_mps": speed,
        "offset_m": offset,
        "foot_mass_kg": float(source.body_mass[bid]),
        "ball_vx_mps": vx,
        "ball_vy_mps": vy,
        "ball_spin_rps": data.qvel[bv + 3 : bv + 6].tolist(),
        "contact_steps": contacts,
        "max_penetration_m": max_penetration,
        "max_energy_over_initial": max_energy / initial_ke if initial_ke else 0,
        "momentum_error": float(momentum - momentum0),
        "checks": checks,
        "passed": all(checks.values()),
    }


def strike_gate(report):
    rows = report["episodes"]
    passed = [
        r
        for r in rows
        if r["functional_pass"]
        and r.get("settled_recovery_s", 0) >= 2.0
        and r.get("post_release_recontacts", 0) == 0
    ]
    coverage = all(
        any(r["ball_lateral_start_m"] == side for r in passed)
        for side in {r["ball_lateral_start_m"] for r in rows}
    )
    falls = sum(r["fall"] for r in rows)
    goals = sum(r["scored"] for r in passed)
    speed_coverage = all(
        any(row["requested_mps"] == speed for row in passed)
        for speed in {row["requested_mps"] for row in rows}
    )
    return {
        "passed": report.get("qualification_eligible", True)
        and len(rows) >= 18
        and len(passed) >= (2 * len(rows) + 2) // 3
        and falls == 0
        and goals >= (len(rows) + 2) // 3
        and coverage
        and speed_coverage,
        "functional_trials": len(passed),
        "trials": len(rows),
        "physical_falls": falls,
        "goals": goals,
        "all_goal_crossings": sum(r["scored"] for r in rows),
        "lateral_coverage": coverage,
        "speed_coverage": speed_coverage,
    }


def match_gate(report):
    match = report.get("match") or {}
    touches = match.get("touch_bouts", [])
    return (
        match.get("phase") == "FULL_TIME"
        and match.get("elapsed_s", 0) >= 179.9
        and match.get("unexpected_live_resets", 1) == 0
        and len(touches) == 2
        and min(touches) >= 1
        and sum(match.get("score", [])) >= 1
        and match.get("physical_falls") == [0, 0]
        and len(match.get("in_pitch_fraction", [])) == 2
        and min(match["in_pitch_fraction"]) >= 0.95
        and max(match.get("maximum_outside_pitch_m", [float("inf")])) <= 0.75
        and not any(event.get("event") == "fall" for event in match.get("events", []))
    )


def corner_gate(report):
    rows = report.get("episodes", [])
    if {row.get("side") for row in rows} != {-1, 1}:
        return False
    for row in rows:
        events = row.get("events", [])
        goals = [e for e in events if e.get("event") == "goal" and e.get("team") == 0]
        restarts = [e for e in events if e.get("event") == "kickoff"]
        if not (
            goals
            and any(
                restart["time_s"] - goal["time_s"] >= 2.0 - 1e-6
                for goal in goals
                for restart in restarts
            )
            and row.get("touches", 0) > 0
            and row.get("physical_falls") == [0, 0]
            and row.get("unexpected_live_resets", 1) == 0
            and len(row.get("in_pitch_fraction", [])) == 2
            and min(row["in_pitch_fraction"]) >= 0.95
            and len(row.get("maximum_outside_pitch_m", [])) == 2
            and max(row["maximum_outside_pitch_m"]) <= 0.75
            and any(t.get("to") == "DRIBBLE" for t in row.get("transitions", []))
        ):
            return False
    return True


def sequence_gate(report):
    rows = report.get("episodes", [])
    return (
        len(rows) >= 6
        and sum(
            r["success"]
            and r.get("settled_recovery_s", 0) >= 2.0
            and r.get("scored_after_release", True)
            and r.get("post_release_recontacts", 0) == 0
            for r in rows
        )
        >= (2 * len(rows) + 2) // 3
        and not any(
            r["reason"] == "fall" or r.get("unexpected_live_resets", 1) for r in rows
        )
    )


def navigation_dependencies_match(report, expected, opponent):
    if opponent:
        return (
            report.get("match") is not None
            and report.get("controller_dependencies") == expected
        )
    return (
        "match" in report
        and report["match"] is None
        and report.get("opponent_tactics") is None
        and report.get("controller_dependencies") == {}
    )


class ApproachTests(unittest.TestCase):
    def test_alignment_corrects_drift_outside_success_tolerance(self):
        c = ApproachController(1, "cpu")
        b, g = torch.tensor([[0.0, 0]]), torch.tensor([[6.0, 0]])
        p = torch.tensor([[-0.95, 0.0]])
        c.reset(torch.tensor([0]), p, b, g)
        c.update(p, torch.tensor([0.0]), b, g)
        self.assertTrue(bool(c.aligning[0]))
        # Reproduce the reported 20–23 cm dead zone after entering alignment.
        drifted = torch.tensor([[-1.06, 0.0]])
        for _ in range(20):
            command, metrics = c.update(drifted, torch.tensor([0.0]), b, g)
        self.assertGreater(float(metrics["position_error"][0]), 0.20)
        self.assertGreater(float(command[0, 0]), 0.05)
        self.assertFalse(bool(metrics["success"][0]))

    def test_alignment_repositions_when_reverse_would_be_required(self):
        c = ApproachController(1, "cpu")
        b, g = torch.tensor([[0.0, 0]]), torch.tensor([[6.0, 0]])
        p = torch.tensor([[-0.85, 0.0]])
        c.reset(torch.tensor([0]), p, b, g)
        c.update(p, torch.tensor([0.0]), b, g)
        command, _ = c.update(torch.tensor([[-0.635, 0.0]]), torch.tensor([0.0]), b, g)
        self.assertFalse(bool(c.aligning[0]))
        self.assertGreater(abs(float(command[0, 2])), 0)

    def test_target_behind_ball_for_either_goal(self):
        np.testing.assert_allclose(positioning_target([1, 0], [6, 0]), [0.15, 0])
        np.testing.assert_allclose(positioning_target([1, 0], [-6, 0]), [1.85, 0])

    def test_goal_side_routes_avoid_ball(self):
        for angle in np.linspace(-math.pi, math.pi, 17):
            start = 2 * np.array([math.cos(angle), math.sin(angle)])
            route = plan_route(start, [0, 0], [6, 0])
            path = np.vstack((start, route))
            for a, b in itertools.pairwise(path):
                self.assertGreaterEqual(segment_clearance(a, b, [0, 0]), 0.75 - 1e-8)

    def test_turns_in_place_instead_of_walking_backward(self):
        c = ApproachController(1, "cpu")
        position = torch.tensor([[-2.0, 0]])
        ball, goal = torch.tensor([[0.0, 0]]), torch.tensor([[6.0, 0]])
        c.reset(torch.tensor([0]), position, ball, goal)
        cmd, _ = c.update(position, torch.tensor([math.pi]), ball, goal)
        self.assertEqual(float(cmd[0, 0]), 0)
        self.assertGreater(abs(float(cmd[0, 2])), 0)

    def test_success_requires_continuous_position_and_heading_hold(self):
        c = ApproachController(1, "cpu")
        p, b, g = (
            torch.tensor([[-0.85, 0]]),
            torch.tensor([[0.0, 0]]),
            torch.tensor([[6.0, 0]]),
        )
        c.reset(torch.tensor([0]), p, b, g)
        for _ in range(50):
            _, m = c.update(p, torch.tensor([0.0]), b, g)
        self.assertFalse(bool(m["success"][0]))
        c.update(p, torch.tensor([1.0]), b, g)
        self.assertEqual(float(c.hold[0]), 0)
        for _ in range(102):
            _, m = c.update(p, torch.tensor([0.0]), b, g)
        self.assertTrue(bool(m["success"][0]))

    def test_rejects_unsafe_configuration_and_start(self):
        with self.assertRaises(ValueError):
            ApproachCfg(max_forward=1)
        with self.assertRaises(ValueError):
            plan_route([0.2, 0], [0, 0], [6, 0])


class ArmPoseTests(unittest.TestCase):
    def test_configs_do_not_change_shared_action_scales(self):
        first, second = make_env_cfg(), make_env_cfg()
        for cfg in (first, second):
            for joint in ("l_elbow", "r_elbow"):
                scale = cfg.actions["joint_pos"].scale[joint]
                rest = cfg.scene.entities["robot"].init_state.joint_pos[joint + "$"]
                self.assertEqual(scale, 0.1)
                self.assertLess(rest + 3 * scale, 0)
                self.assertGreater(rest - 3 * scale, -2.1817)
        first.actions["joint_pos"].scale["r_elbow"] = 99
        self.assertEqual(second.actions["joint_pos"].scale["r_elbow"], 0.1)


class BallTests(unittest.TestCase):
    def test_shell_mass_inertia_and_center(self):
        cfg = BallCfg()
        m = physics_spec(cfg).compile()
        bid = m.body("ball").id
        self.assertEqual(m.nv, 6)
        self.assertEqual(m.nu, 0)
        self.assertAlmostEqual(m.body_mass[bid], cfg.mass)
        np.testing.assert_allclose(m.body_inertia[bid], cfg.inertia)
        np.testing.assert_allclose(m.body_ipos[bid], 0)
        self.assertFalse(np.any(m.body_gravcomp))
        self.assertGreaterEqual(2 * np.pi * cfg.radius, 0.68)
        self.assertLessEqual(2 * np.pi * cfg.radius, 0.70)

    def test_robot_interface_and_contact_parameters(self):
        cfg = BallCfg()
        m = physics_spec(robot=True).compile()
        self.assertEqual(m.nv, 24)
        self.assertEqual(m.nu, 12)
        self.assertEqual([m.actuator(i).name for i in range(m.nu)], list(JOINTS))
        self.assertAlmostEqual(m.body_mass.sum(), audit_source() + cfg.mass)
        d = rest_data(m)
        j = m.joint("ball_root").qposadr[0]
        d.qpos[j : j + 3] = [1, 0, cfg.radius - 0.001]
        mujoco.mj_forward(m, d)
        ball, floor = m.geom("ball_geom").id, m.geom("floor").id
        contacts = [
            c for c in d.contact if {int(c.geom1), int(c.geom2)} == {ball, floor}
        ]
        self.assertTrue(contacts)
        for c in contacts:
            self.assertEqual(c.dim, 6)
            np.testing.assert_allclose(
                c.friction,
                [cfg.sliding_friction] * 2
                + [cfg.torsional_friction]
                + [cfg.rolling_friction] * 2,
            )
        for side in ("l", "r"):
            foot = m.geom(f"{side}_foot_collision_0").id
            self.assertEqual(m.geom_priority[foot], 1)
        for friction in (0.6, 1.0):
            feet = [m.geom(f"{side}_foot_collision_0").id for side in ("l", "r")]
            m.geom_friction[feet, 0] = friction
            d = rest_data(m, ground_clearance=-0.001)
            support = [
                c
                for c in d.contact
                if floor in (c.geom1, c.geom2)
                and any(g in (c.geom1, c.geom2) for g in feet)
            ]
            self.assertTrue(support)
            for c in support:
                self.assertAlmostEqual(c.friction[0], friction)
        lines = [
            g
            for g in range(m.ngeom)
            if m.geom(g).name.startswith(("pitch_line_", "goal_marker_"))
        ]
        self.assertTrue(lines)
        self.assertFalse(np.any(m.geom_contype[lines]))
        self.assertFalse(np.any(m.geom_conaffinity[lines]))
        env = make_env_cfg()
        self.assertEqual(env.decimation * env.sim.mujoco.timestep, 0.02)
        self.assertIn("joga_layout", env.events)

    def test_roll_slip_and_bounce_are_passive(self):
        for kind in ("rolling", "sliding", "bounce"):
            with self.subTest(kind=kind):
                result = passive_case(kind)
                self.assertTrue(result["passed"], result)

    def test_foot_transfers_momentum_without_injected_energy(self):
        stationary = impact_case(0)
        moving = impact_case(1)
        self.assertTrue(stationary["passed"], stationary)
        self.assertTrue(moving["passed"], moving)
        self.assertGreater(moving["ball_vx_mps"], 0.1)

    def test_invalid_parameters_rejected(self):
        for name, value in (
            ("mass", 0),
            ("radius", -1),
            ("rolling_friction", float("nan")),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(BallCfg(), **{name: value})


class DeliveryGateTests(unittest.TestCase):
    def test_helper_provenance_follows_the_roles_actually_loaded(self):
        helpers = {"turn.py": "exact tested source"}
        solo = {"match": None, "opponent_tactics": None, "controller_dependencies": {}}
        match = {"match": {"phase": "FULL_TIME"}, "controller_dependencies": helpers}
        self.assertTrue(navigation_dependencies_match(solo, helpers, False))
        self.assertTrue(navigation_dependencies_match(match, helpers, True))
        self.assertFalse(navigation_dependencies_match(match, helpers, False))
        self.assertFalse(navigation_dependencies_match(solo, helpers, True))
        self.assertFalse(
            navigation_dependencies_match(
                {**match, "controller_dependencies": {}}, helpers, True
            )
        )
        self.assertFalse(
            navigation_dependencies_match(
                {**match, "controller_dependencies": {"turn.py": "changed source"}},
                helpers,
                True,
            )
        )
        self.assertFalse(navigation_dependencies_match({}, helpers, False))

    def test_corner_recovery_requires_both_sides_and_full_pause_without_falls(self):
        row = {
            "touches": 1,
            "physical_falls": [0, 0],
            "unexpected_live_resets": 0,
            "in_pitch_fraction": [1.0, 1.0],
            "maximum_outside_pitch_m": [0.0, 0.0],
            "transitions": [{"to": "DRIBBLE"}],
            "events": [
                {"event": "goal", "team": 0, "time_s": 18.0},
                {"event": "kickoff", "time_s": 20.0},
            ],
        }
        rows = [{**row, "side": side} for side in (-1, 1)]
        self.assertTrue(corner_gate({"episodes": rows}))
        self.assertFalse(corner_gate({"episodes": rows[:1]}))
        for field, value in (
            ("physical_falls", [1, 0]),
            ("unexpected_live_resets", 1),
            ("touches", 0),
            ("transitions", []),
            ("events", row["events"][:1]),
            ("events", [row["events"][0], {"event": "kickoff", "time_s": 19.0}]),
        ):
            self.assertFalse(
                corner_gate({"episodes": [{**rows[0], field: value}, rows[1]]})
            )

    def rows(self):
        return [
            {
                "functional_pass": True,
                "settled_recovery_s": 2.5,
                "requested_mps": (1.2, 1.8, 2.4)[i % 3],
                "fall": False,
                "strike": True,
                "scored": i % 2 == 0,
                "ball_lateral_start_m": (0.0, -0.6, 0.6)[i % 3],
            }
            for i in range(18)
        ]

    def test_shooting_requires_broad_balance_exit_and_real_goal_evidence(self):
        rows = self.rows()
        self.assertTrue(strike_gate({"episodes": rows})["passed"])
        self.assertFalse(
            strike_gate({"episodes": rows, "qualification_eligible": False})["passed"]
        )
        self.assertFalse(strike_gate({"episodes": rows[:9]})["passed"])
        for field, value in (
            ("fall", True),
            ("functional_pass", False),
            ("scored", False),
            ("settled_recovery_s", 0.0),
            ("post_release_recontacts", 1),
        ):
            changed = [{**row, field: value} for row in rows]
            self.assertFalse(strike_gate({"episodes": changed})["passed"])
        rows[0] = {**rows[0], "fall": True}
        self.assertFalse(strike_gate({"episodes": rows})["passed"])

    def test_one_missing_approach_side_cannot_be_hidden_by_total_pass_count(self):
        rows = [
            {**row, "functional_pass": row["ball_lateral_start_m"] != 0.6}
            for row in self.rows()
        ]
        result = strike_gate({"episodes": rows})
        self.assertEqual(result["functional_trials"], 12)
        self.assertFalse(result["lateral_coverage"])
        self.assertFalse(result["passed"])

    def test_dribbling_goals_cannot_qualify_other_shots_that_never_score(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            row.update(
                scored=i >= 12,
                functional_pass=i < 12,
                post_release_recontacts=1 if i >= 12 else 0,
            )
        result = strike_gate({"episodes": rows})
        self.assertEqual(result["functional_trials"], 12)
        self.assertEqual(result["all_goal_crossings"], 6)
        self.assertEqual(result["goals"], 0)
        self.assertFalse(result["passed"])

    def test_match_requires_both_players_and_a_goal_without_hidden_resets(self):
        match = {
            "phase": "FULL_TIME",
            "elapsed_s": 180.0,
            "touch_bouts": [4, 7],
            "score": [1, 0],
            "unexpected_live_resets": 0,
            "physical_falls": [0, 0],
            "in_pitch_fraction": [1.0, 1.0],
            "maximum_outside_pitch_m": [0.0, 0.0],
        }
        self.assertTrue(match_gate({"match": match}))
        for key, value in (
            ("phase", "PLAY"),
            ("elapsed_s", 30),
            ("touch_bouts", [0, 8]),
            ("score", [0, 0]),
            ("unexpected_live_resets", 1),
            ("physical_falls", [0, 1]),
            ("in_pitch_fraction", [1.0, 0.5]),
            ("maximum_outside_pitch_m", [0.0, 1.5]),
        ):
            self.assertFalse(match_gate({"match": {**match, key: value}}))
        self.assertFalse(match_gate({}))
        self.assertFalse(
            match_gate({"match": {**match, "events": [{"event": "fall"}]}})
        )

    def test_sequence_cannot_hide_falls_or_resets_behind_goal_counts(self):
        rows = [
            {
                "success": i < 4,
                "reason": "goal_and_balance" if i < 4 else "timeout",
                "unexpected_live_resets": 0,
                "settled_recovery_s": 2.5,
            }
            for i in range(6)
        ]
        self.assertTrue(sequence_gate({"episodes": rows}))
        for field, value in (
            ("scored_after_release", False),
            ("post_release_recontacts", 1),
        ):
            self.assertFalse(
                sequence_gate({"episodes": [{**row, field: value} for row in rows]})
            )
        self.assertFalse(
            sequence_gate(
                {"episodes": [{**row, "settled_recovery_s": 0.0} for row in rows]}
            )
        )
        self.assertFalse(sequence_gate({"episodes": rows[:3]}))
        self.assertFalse(
            sequence_gate({"episodes": [{**rows[0], "reason": "fall"}] + rows[1:]})
        )
        self.assertFalse(
            sequence_gate(
                {"episodes": [{**rows[0], "unexpected_live_resets": 1}] + rows[1:]}
            )
        )


class LiveHandoffTests(unittest.TestCase):
    def test_dribbled_goal_uses_standing_recovery_but_existing_strike_keeps_its_actor(
        self,
    ):
        phase = torch.tensor([0, 1, 2, 3, 4, 1])
        defending = torch.tensor([False, False, False, False, False, True])
        self.assertEqual(
            stationary_restart(phase, defending, True, True).tolist(),
            [True, True, False, False, False, False],
        )
        self.assertFalse(bool(stationary_restart(phase, defending, False, True).any()))
        self.assertFalse(bool(stationary_restart(phase, defending, True, False).any()))

    def test_turning_commands_keep_bearing_and_qualified_speed_without_changing_other_skills(
        self,
    ):
        command = torch.tensor([[0.18, -0.24, 0.0], [0.0, 0.5, 0.0], [2.4, 0.0, 0.0]])
        result = turning_command(command, torch.tensor([True, True, False]))
        torch.testing.assert_close(result[0], torch.tensor([0.24, -0.32, 0.0]))
        torch.testing.assert_close(result[1:], command[1:])
        torch.testing.assert_close(command[0], torch.tensor([0.18, -0.24, 0.0]))

    def test_goal_mouth_aim_retains_forward_intent_at_both_goal_lines(self):
        goals = torch.tensor([[6.0, 0.0], [-6.0, 0.0], [6.0, 0.0], [6.0, 0.0]])
        ball = torch.tensor([[5.95, 0.6], [-6.06, -0.67], [4.0, 0.4], [5.95, 1.2]])
        target = goal_target(goals, ball)
        torch.testing.assert_close(target[0], torch.tensor([6.75, 0.0]))
        torch.testing.assert_close(target[1], torch.tensor([-6.86, 0.0]))
        torch.testing.assert_close(target[2:], goals[2:])

    def test_stoppage_keeps_a_shooters_target_despite_body_yaw(self):
        command = torch.tensor([[-0.6, -2.3, 0.0], [0.5, 0.1, 0.0], [0.2, 0.3, 0.0]])
        result = recovery_command(
            command,
            torch.zeros(3),
            torch.tensor([True, False, False]),
            torch.tensor([False, False, True]),
            2.4,
        )
        torch.testing.assert_close(result[0], command[0])
        torch.testing.assert_close(result[1], torch.tensor([2.4, 0.0, 0.0]))
        torch.testing.assert_close(result[2], command[2])
        torch.testing.assert_close(command[1], torch.tensor([0.5, 0.1, 0.0]))

    def test_shot_readiness_waits_for_a_quieter_stride(self):
        linear = torch.tensor([[0.9, 0.0, 0.0], [0.5, 0.1, 0.0], [0.5, 0.0, 0.0]])
        angular = torch.tensor([[0.1, 0.1, 0.0], [1.1, 0.4, 0.0], [0.2, -0.2, 0.6]])
        self.assertEqual(
            stable_shot_motion(linear, angular).tolist(), [False, False, True]
        )

    def test_turning_expert_does_not_replace_initial_approach_or_patrol(self):
        phase = torch.tensor([0, 0, 0, 1, 2])
        defending = torch.tensor([False, False, True, False, False])
        had_dribbled = torch.tensor([False, True, True, True, True])
        self.assertEqual(
            turning_navigation(phase, defending, had_dribbled, True).tolist(),
            [False, True, False, False, False],
        )
        self.assertFalse(
            bool(turning_navigation(phase, defending, had_dribbled, False).any())
        )

    def test_referee_out_of_play_does_not_fail_the_off_ball_controller(self):
        ball = torch.tensor([[6.2, 0.0], [0.0, 4.2], [6.2, 0.0]])
        scored = torch.tensor([False, False, True])
        self.assertEqual(
            field_failure(ball, scored, False).tolist(), [True, True, False]
        )
        self.assertEqual(
            field_failure(ball, scored, True).tolist(), [False, False, False]
        )

    def test_patrol_is_a_separate_expert_and_recovery_keeps_the_striker(self):
        phase = torch.tensor([0, 0, 1, 2, 3, 4])
        defending = torch.tensor([False, True, False, False, False, False])
        self.assertEqual(policy_role(phase, defending).tolist(), [0, 3, 1, 2, 2, 2])

    def test_opponent_keeps_possession_beyond_entry_threshold_but_releases_a_lost_ball(
        self,
    ):
        delta = torch.tensor([[1.2, 0.0], [1.7, 0.0], [-0.3, 0.0], [0.4, 0.0]])
        self.assertEqual(
            retain_dribbling(
                torch.ones(4, dtype=torch.bool),
                torch.tensor([False, False, False, True]),
                delta,
                torch.tensor([[1.0, 0.0]] * 4),
            ).tolist(),
            [True, False, False, False],
        )

    def test_defender_orbit_remains_inside_field_even_when_ball_is_at_goal(self):
        positions = torch.tensor([[4.8, 0.0], [-4.8, 0.0], [0.0, 2.8], [0.0, -2.8]])
        balls = torch.tensor([[6.0, 0.0], [-6.0, 0.0], [0.0, 4.0], [0.0, -4.0]])
        target = shadow_target(positions, balls)
        self.assertTrue(bool((target.abs() <= torch.tensor([5.0, 3.0])).all()))

    def test_retry_never_erases_a_released_shot_or_goal_and_is_bounded(self):
        phase = torch.tensor([2, 2, 2, 2, 2, 2, 3])
        age = torch.tensor([3.0, 3.0, 3.0, 3.0, 1.0, 3.0, 3.0])
        released = torch.tensor([False, True, False, False, False, False, True])
        scored = torch.tensor([False, False, True, False, False, False, False])
        speed = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0])
        retries = torch.tensor([0, 0, 0, 2, 0, 0, 0])
        self.assertEqual(
            retry_unreleased_strike(
                phase, age, released, scored, speed, retries, 2
            ).tolist(),
            [True, False, False, False, False, False, False],
        )

    def test_direct_goal_command_aims_at_goal_without_reacting_to_lateral_foot_impacts(
        self,
    ):
        ball = torch.tensor([[2.0, 1.0]])
        goal = torch.tensor([[6.0, 0.0]])
        expected = 0.7 * torch.tensor([[4.0, -1.0]]) / (17.0**0.5)
        for velocity in (torch.tensor([[0.0, 2.0]]), torch.tensor([[0.0, -2.0]])):
            torch.testing.assert_close(
                ball_path_command(ball, velocity, goal, 0.7, direct_goal=True), expected
            )

    def test_anticipation_requires_closing_on_a_separated_reachable_ball(self):
        delta = torch.tensor([[0.7, 0.0], [0.7, 0.0], [0.4, 0.0], [0.7, 0.0]])
        robot = torch.tensor([[0.7, 0.0]] * 4)
        ball = torch.tensor([[0.3, 0.0], [0.9, 0.0], [0.3, 0.0], [0.3, 0.0]])
        quiet = torch.tensor([0.2, 0.2, 0.2, 0.0])
        self.assertEqual(
            anticipatory_strike_window(delta, robot, ball, quiet).tolist(),
            [True, False, False, False],
        )

    def test_predecessor_touch_is_not_credited_before_the_striker_has_acted(self):
        active = torch.tensor([True, True, False])
        newly_started = torch.tensor([True, False, False])
        strong = torch.ones(3, dtype=torch.bool)
        self.assertEqual(
            striker_contact(active, newly_started, strong).tolist(),
            [False, True, False],
        )

    def test_shot_preparation_slows_only_near_goal_and_preserves_default_speed(self):
        distance = torch.tensor([4.0, 3.4, 2.8, 2.2, 1.0])
        self.assertTrue(
            torch.allclose(
                preparation_speed(distance, 0.7, 2.2, 3.4, 0.35).flatten(),
                torch.tensor([0.7, 0.7, 0.525, 0.35, 0.35]),
            )
        )
        self.assertTrue(
            torch.allclose(
                preparation_speed(distance, 0.7, 2.2, None, 0.35),
                torch.full((5, 1), 0.7),
            )
        )

    def test_immediate_intent_changes_only_command_features_and_keeps_source_history(
        self,
    ):
        from tensordict import TensorDict

        original = torch.arange(207, dtype=torch.float32).reshape(1, -1)
        obs = TensorDict({"actor": original.clone()}, batch_size=[1])
        current = torch.tensor([[3.0, 0.0, 0.0]])
        result = immediate_strike_command(obs, current)["actor"]
        self.assertTrue(torch.equal(obs["actor"], original))
        self.assertTrue(torch.equal(result[:, :162], original[:, :162]))
        self.assertTrue(torch.equal(result[:, 171:], original[:, 171:]))
        self.assertEqual(result[:, 162:171].tolist(), [[3.0, 0.0, 0.0] * 3])

    def test_direct_switch_preserves_the_first_action_instead_of_diluting_it(self):
        age = torch.tensor([0.02, 0.04, 0.08])
        self.assertEqual(blend_fraction(age, 0).tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(blend_fraction(age, 0.08).tolist(), [0.25, 0.5, 1.0])
        with self.assertRaises(ValueError):
            blend_fraction(age, -0.1)

    def test_contact_timing_experiment_waits_for_the_next_controlled_touch(self):
        ones = torch.ones(4)
        ready = ready_for_strike(
            ones,
            ones,
            2 * ones,
            2 * ones,
            0.5 * ones,
            0.1 * ones,
            torch.tensor([0.0, 0.1, 0.24, 0.44]),
            2.2,
            quiet_limit=0.12,
        )
        self.assertEqual(ready.tolist(), [True, True, False, False])

    def test_recovery_does_not_delay_the_same_striker_with_another_blend(self):
        phase = torch.tensor([0, 1, 2, 3, 4])
        previous = torch.tensor([-1, 0, 1, 2, 3])
        self.assertEqual(
            skill_changes(phase, previous).tolist(), [True, True, True, False, False]
        )

    def test_reachable_sideways_stance_does_not_wait_until_after_the_goal(self):
        ones = torch.ones(4)
        ready = ready_for_strike(
            ones,
            ones,
            2 * ones,
            2 * ones,
            0.5 * ones,
            torch.tensor([0.1, 0.6, 0.79, 0.81]),
            torch.zeros(4),
            2.2,
        )
        self.assertEqual(ready.tolist(), [True, True, True, False])

    def test_current_touch_is_ready_but_stale_possession_is_not(self):
        quiet = torch.tensor([0.0, 0.04, 0.08, 0.5, 1.05])
        ones = torch.ones(5)
        ready = ready_for_strike(
            ones, ones, 2 * ones, 2 * ones, 0.5 * ones, 0.1 * ones, quiet, 2.2
        )
        self.assertEqual(ready.tolist(), [True, True, True, True, False])

    def test_genuine_dribbling_and_reachable_geometry_remain_required(self):
        ones = torch.ones(4)
        ready = ready_for_strike(
            torch.tensor([1, 1, 1, 2]),
            torch.tensor([0.2, 1.0, 1.0, 1.0]),
            torch.tensor([2, 1, 2, 2]),
            torch.tensor([2.0, 2.0, 3.0, 2.0]),
            0.5 * ones,
            0.1 * ones,
            torch.zeros(4),
            2.2,
        )
        self.assertFalse(ready.any())


class MatchAuditTests(unittest.TestCase):
    def test_physical_falls_are_independent_of_score_and_survive_kickoffs(self):
        audit = MatchAudit()
        up, down = [0.0, 0.0, 0.8, -1.0], [0.0, 0.0, 0.3, -0.2]
        self.assertEqual(audit.observe([up, down]), [False, True])
        self.assertEqual(audit.observe([up, down]), [False, False])
        audit.kickoff()
        audit.observe([up, up])
        audit.observe([down, up])
        self.assertEqual(audit.report()["physical_falls"], [1, 1])

    def test_both_robots_are_checked_against_real_pitch_bounds(self):
        audit = MatchAudit()
        audit.observe([[0.0, 0.0, 0.8, -1.0], [0.0, 4.8, 0.8, -1.0]])
        report = audit.report()
        self.assertEqual(report["in_pitch_fraction"], [1.0, 0.0])
        self.assertAlmostEqual(report["maximum_outside_pitch_m"][1], 0.8)
        self.assertEqual(report["physical_falls"], [0, 0])

    def test_nonfinite_physical_states_cannot_be_reported_as_safe(self):
        with self.assertRaises(ValueError):
            MatchAudit().observe(
                [[float("nan"), 0.0, 0.8, -1.0], [0.0, 0.0, 0.8, -1.0]]
            )


class MatchRulesTests(unittest.TestCase):
    def test_whole_ball_and_interpolated_goal_mouth(self):
        game = MatchRules()
        self.assertIsNone(game.observe([5.9, 0.0, 0.11], [6.05, 0.0, 0.11], 0.02))
        event = game.observe([6.05, 0.0, 0.11], [6.3, 0.0, 0.11], 0.02)
        self.assertEqual(event["event"], "goal")
        self.assertEqual(game.score, [1, 0])
        game.observe([6.3, 0.0, 0.11], [6.4, 0.0, 0.11], 0.02)
        self.assertEqual(game.score, [1, 0])
        self.assertEqual(game.kickoff, 1)

    def test_left_goal_counts_for_opponent(self):
        game = MatchRules()
        game.observe([-6.0, 0.0, 0.11], [-6.3, 0.0, 0.11], 0.02)
        self.assertEqual(game.score, [0, 1])

    def test_wide_and_high_crossings_never_score(self):
        for y, z in ((0.95, 0.11), (0.0, 1.1)):
            game = MatchRules()
            self.assertEqual(
                game.observe([6.0, y, z], [6.3, y, z], 0.02)["event"], "out_of_play"
            )
            self.assertEqual(game.score, [0, 0])

    def test_restart_wait_and_match_clock(self):
        game = MatchRules(duration=10, restart_delay=0.1)
        game.observe([0.0, 0.0, 0.11], [0.0, 0.0, 0.11], 0.02, player_fall=True)
        self.assertEqual(game.phase, "RESTART")
        self.assertIsNone(game.observe([0.0, 0.0, 0.11], [0.0, 0.0, 0.11], 0.04))
        self.assertEqual(
            game.observe([0.0, 0.0, 0.11], [0.0, 0.0, 0.11], 0.07)["event"], "kickoff"
        )
        self.assertEqual(game.phase, "PLAY")
        game.elapsed = 9.99
        self.assertEqual(
            game.observe([6.0, 0.0, 0.11], [6.3, 0.0, 0.11], 0.02)["event"], "full_time"
        )
        self.assertEqual(game.score, [0, 0])

    def test_live_play_does_not_request_resets(self):
        game = MatchRules()
        for _ in range(100):
            self.assertIsNone(game.observe([0.0, 0.0, 0.11], [0.01, 0.0, 0.11], 0.02))
        self.assertEqual(game.events, [])

    def test_touchline_uses_actual_eight_metre_pitch_width(self):
        game = MatchRules()
        self.assertIsNone(game.observe([0.0, 4.0, 0.11], [0.0, 4.05, 0.11], 0.02))
        self.assertEqual(
            game.observe([0.0, 4.05, 0.11], [0.0, 4.2, 0.11], 0.02)["event"],
            "out_of_play",
        )


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = ground_spec().compile()

    def test_free_root_and_mass(self):
        m = self.m
        self.assertEqual(m.nv, 18)
        self.assertEqual(m.nu, 12)
        self.assertEqual(m.neq, 0)
        self.assertEqual(m.body(1).name, "base")
        self.assertFalse(np.any(m.body_gravcomp))
        self.assertAlmostEqual(m.body_mass.sum(), audit_source(), places=6)
        self.assertEqual([m.actuator(i).name for i in range(m.nu)], list(JOINTS))

    def test_stance_axes_and_contacts(self):
        m = self.m
        d = rest_data(m)
        self.assertGreater(d.xpos[m.body("l_foot").id, 1], 0.1)
        self.assertLess(d.xpos[m.body("r_foot").id, 1], -0.1)
        self.assertGreater(d.qpos[2], 0.8)
        self.assertFalse(any(c.dist < -1e-4 for c in d.contact))
        for j in JOINTS:
            lo, hi = m.joint(j).range
            self.assertLessEqual(lo, REST[j])
            self.assertGreaterEqual(hi, REST[j])

    def test_hip_and_knee_directions(self):
        m = self.m
        for side in ("L", "R"):
            d = rest_data(m)
            knee = m.joint("Knee_" + side).id
            ankle = m.joint("Ankle_P_" + side).id
            before = d.xanchor[knee].copy()
            d.qpos[m.joint("Hip_F_" + side).qposadr[0]] += 0.2
            mujoco.mj_forward(m, d)
            self.assertGreater(d.xanchor[knee, 0] - before[0], 0.03)
            d = rest_data(m)
            before = d.xanchor[ankle].copy()
            d.qpos[m.joint("Knee_" + side).qposadr[0]] -= 0.2
            mujoco.mj_forward(m, d)
            self.assertLess(d.xanchor[ankle, 0] - before[0], -0.03)

    def test_nonadjacent_leg_contacts_are_enabled(self):
        m = self.m
        left = m.geom("l_foot_collision_0").id
        right = m.geom("r_foot_collision_0").id
        self.assertTrue(m.geom_contype[left] & m.geom_conaffinity[right])
        self.assertEqual(m.nexclude, 2)  # only same-side ankle assembly exclusions

    def test_nominal_support(self):
        m = self.m
        d = rest_data(m)
        mujoco.mj_step(m, d, nstep=1500)
        self.assertGreater(d.qpos[2], 0.8)
        self.assertLess(np.linalg.norm(d.qpos[:2]), 0.03)
        self.assertFalse(np.any(d.warning.number))

    def test_sole_friction_controls_actual_ground_contacts(self):
        m = ground_spec().compile()
        floor = m.geom("floor").id
        m.geom_friction[floor, 0] = 1.0
        feet = [m.geom(f"{side}_foot_collision_0").id for side in ("l", "r")]
        for friction in (0.6, 1.0):
            m.geom_friction[feet, 0] = friction
            d = rest_data(m, ground_clearance=-0.001)
            contacts = [
                c
                for c in d.contact
                if floor in (c.geom1, c.geom2)
                and any(foot in (c.geom1, c.geom2) for foot in feet)
            ]
            self.assertGreater(len(contacts), 0)
            for contact in contacts:
                self.assertAlmostEqual(contact.friction[0], friction)


class NavigationTests(unittest.TestCase):
    # Ending a goal-facing turn near the final target must correct position rather than spin again.
    def test_stage_handoff_corrects_lateral_drift_while_facing_goal(self):
        cfg = StagedNavigationCfg(
            align_exit=0.35,
            waypoint_tolerance=0.06,
            staging_tolerance=0.06,
            staging_distance=1.05,
            staging_clearance=0.90,
            staging_orbit=1.08,
        )
        c = StagedApproachController(1, "cpu", cfg)
        ball, goal = torch.zeros(1, 2), torch.tensor([[6.0, 0.0]])
        c.reset(torch.tensor([0]), torch.tensor([[-1.20, 0.0]]), ball, goal)
        self.assertGreater(len(c.routes[0]), 1)
        c.staged_hold[:] = True
        commands, _ = c.update(
            torch.tensor([[-1.05, 0.18]]), torch.zeros(1), ball, goal
        )
        self.assertTrue(bool(c.aligning[0]))
        self.assertEqual(c.cursor[0], len(c.routes[0]) - 1)
        self.assertGreater(float(commands[0, 0]), 0)
        self.assertLess(float(commands[0, 1]), 0)
        self.assertEqual(float(commands[0, 2]), 0)

    # Compact recovery must reach its staging tolerance before advancing into the final leg.
    def test_compact_stage_cannot_skip_goal_facing_turn(self):
        cfg = StagedNavigationCfg(
            waypoint_tolerance=0.06,
            staging_tolerance=0.06,
            staging_distance=1.05,
            staging_clearance=0.90,
            staging_orbit=1.08,
        )
        c = StagedApproachController(1, "cpu", cfg)
        ball, goal = torch.zeros(1, 2), torch.tensor([[6.0, 0.0]])
        c.reset(torch.tensor([0]), torch.tensor([[-1.13, 0.0]]), ball, goal)
        c.update(torch.tensor([[-1.13, 0.0]]), torch.tensor([math.pi / 2]), ball, goal)
        self.assertEqual(c.cursor[0], 0)
        self.assertFalse(bool(c.staged_hold[0]))
        commands, _ = c.update(
            torch.tensor([[-1.09, 0.0]]), torch.tensor([math.pi / 2]), ball, goal
        )
        self.assertTrue(bool(c.staged_hold[0]))
        self.assertEqual(float(commands[0, :2].norm()), 0.0)
        self.assertLess(float(commands[0, 2]), 0.0)

    # The orbit stays wide and only the final straight segment enters the stopping zone.
    def test_staged_route_clearance(self):
        c = StagedApproachController(1, "cpu")
        start = torch.tensor([[2.0, 0.0]])
        ball, goal = torch.tensor([[0.0, 0.0]]), torch.tensor([[6.0, 0.0]])
        c.reset(torch.tensor([0]), start, ball, goal)
        route = c.routes[0].numpy()
        np.testing.assert_allclose(route[-2], [-c.cfg.staging_distance, 0.0])
        np.testing.assert_allclose(route[-1], [-c.cfg.standoff, 0.0])
        path = np.vstack((start.numpy(), route))
        for first, second in pairwise(path[:-1]):
            self.assertGreaterEqual(
                segment_clearance(first, second, [0.0, 0.0]),
                c.cfg.staging_clearance - 1e-6,
            )
        self.assertGreaterEqual(
            segment_clearance(path[-2], path[-1], [0.0, 0.0]), c.cfg.standoff - 1e-6
        )

    # Turning is done far from the ball, then the last leg resumes without a reverse command.
    def test_staging_turn_preserves_real_route(self):
        c = StagedApproachController(1, "cpu")
        position = torch.tensor([[-c.cfg.staging_distance, 0.0]])
        ball, goal = torch.tensor([[0.0, 0.0]]), torch.tensor([[6.0, 0.0]])
        c.reset(torch.tensor([0]), position, ball, goal)
        before = c.routes[0].clone()
        commands, _ = c.update(position, torch.tensor([math.pi / 2]), ball, goal)
        self.assertTrue(bool(c.staged_hold[0]))
        self.assertEqual(float(commands[0, :2].norm()), 0.0)
        self.assertLess(float(commands[0, 2]), 0.0)
        torch.testing.assert_close(c.routes[0], before)
        commands, _ = c.update(position, torch.tensor([0.0]), ball, goal)
        self.assertFalse(bool(c.staged_hold[0]))
        self.assertGreater(float(commands[0, 0]), 0.0)

    # Already-positioned starts should not be sent back around the field.
    def test_positioned_reset_and_safety_rejection(self):
        c = StagedApproachController(1, "cpu")
        ball, goal = torch.tensor([[0.0, 0.0]]), torch.tensor([[6.0, 0.0]])
        c.reset(torch.tensor([0]), torch.tensor([[-0.85, 0.0]]), ball, goal)
        self.assertEqual(len(c.routes[0]), 1)
        with self.assertRaises(ValueError):
            c.reset(torch.tensor([0]), torch.tensor([[0.2, 0.0]]), ball, goal)


class NavigationFrameTests(unittest.TestCase):
    def test_same_physical_history_has_same_inputs_at_arbitrary_global_yaws(self):
        generator = torch.Generator().manual_seed(7601)
        source = torch.randn(16, 207, generator=generator)
        yaw = torch.randn(16, 3, generator=generator)
        source[:, 171:177] = torch.stack((yaw.sin(), yaw.cos()), dim=-1).flatten(1)
        angle = torch.linspace(-math.pi, math.pi, 16)
        rotated = rotate_history(source, angle)
        original = TensorDict({"actor": source}, batch_size=[16])
        changed = TensorDict({"actor": rotated}, batch_size=[16])
        a = in_navigation_frame(original, math.pi)["actor"]
        b = in_navigation_frame(changed, math.pi)["actor"]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a[:, :162], source[:, :162])
        torch.testing.assert_close(a[:, 177:189], source[:, 177:189])
        torch.testing.assert_close(
            a[:, 175:177],
            source.new_tensor([0.0, -1.0]).expand(16, -1),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(original["actor"], source)

    def test_default_keeps_original_interface_and_rejects_nonfinite_reference(self):
        obs = TensorDict({"actor": torch.zeros(1, 207)}, batch_size=[1])
        self.assertIs(in_navigation_frame(obs, None), obs)
        with self.assertRaises(ValueError):
            in_navigation_frame(obs, float("nan"))


class PitchSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = physics_spec().compile()

    # Exercise both ends of the pitch with actual integration and physical collision contacts.
    def shoot(self, sign, y):
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:3] = [sign * 5.3, y, 0.11]
        d.qvel[0] = sign * 3
        d.qvel[4] = sign * 3 / 0.11
        contacts, farthest = set(), 0
        for _ in range(2000):
            mujoco.mj_step(m, d)
            farthest = max(farthest, sign * d.qpos[0])
            for c in d.contact:
                contacts.update((m.geom(c.geom1).name, m.geom(c.geom2).name))
        self.assertTrue(np.isfinite(d.qpos).all())
        self.assertFalse(np.any(d.warning.number))
        return farthest, contacts

    # A central shot must cross the open mouth and then hit the modeled net.
    def test_center_shot_enters_and_is_caught_at_both_ends(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                farthest, contacts = self.shoot(sign, 0)
                self.assertGreater(farthest, 6.11)
                self.assertLess(farthest, 6.86)
                self.assertIn(f"goal_{sign}_net_collision_back", contacts)
                self.assertNotIn(f"goal_{sign}_crossbar", contacts)

    # A shot aimed into a post must make a real frame contact rather than passing through it.
    def test_posts_block_ball_at_both_ends(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                farthest, contacts = self.shoot(sign, 1.04)
                self.assertLess(farthest, 6.02)
                self.assertIn(f"goal_{sign}_post_1", contacts)

    # Browser conversion must retain the grass texture, which primitive planes do not support.
    def test_turf_has_browser_compatible_uv_texture(self):
        m = self.model
        geom = m.geom("stadium_turf").id
        self.assertGreaterEqual(get_geom_texture_id(m, geom), 0)
        self.assertEqual(m.geom_contype[geom], 0)
        self.assertEqual(m.geom_conaffinity[geom], 0)
        self.assertEqual(m.nv, 6)
        self.assertEqual(m.nu, 0)

    # Scenery, paint, and individual net strands must never become extra ball/foot obstacles.
    def test_only_goal_frame_and_panels_add_contacts(self):
        m = self.model
        for i in range(m.ngeom):
            name = m.geom(i).name
            if name.startswith(("stadium_", "pitch_line_")) or (
                name.startswith("goal_")
                and not any(
                    t in name for t in ("_post_", "_crossbar", "_net_collision_")
                )
            ):
                self.assertEqual(m.geom_contype[i], 0, name)
                self.assertEqual(m.geom_conaffinity[i], 0, name)
        self.assertEqual(DEFAULT_PITCH_CFG.goal_height, 1)


class SharedSkillTests(unittest.TestCase):
    def test_ball_path_corrects_drift_and_continues_through_both_goal_planes(self):
        ball = torch.tensor([[0.0, 0.2], [6.05, 0.0], [-6.05, 0.0]])
        velocity = torch.tensor([[0.5, 0.2], [0.5, 0.0], [-0.5, 0.0]])
        goals = torch.tensor([[6.0, 0.0], [6.0, 0.0], [-6.0, 0.0]])
        command = ball_path_command(ball, velocity, goals, 0.7)
        self.assertLess(float(command[0, 1]), -0.1)
        self.assertGreater(float(command[1, 0]), 0)
        self.assertLess(float(command[2, 0]), 0)
        torch.testing.assert_close(command.norm(dim=1), torch.full((3,), 0.7))

    def test_canonical_policy_history_is_invariant_to_world_yaw(self):
        torch.manual_seed(7)
        values = torch.randn(4, 207)
        headings = torch.tensor([[-0.02, -0.01, 0.0]]) + torch.arange(4)[:, None] * 0.2
        values[:, 171:177] = torch.stack(
            (headings.sin(), headings.cos()), dim=-1
        ).flatten(1)
        original = values.clone()
        rotated = values.clone()
        angle = 0.73
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        for start, width, vector_offsets in ((162, 3, (0,)), (189, 6, (0, 3))):
            history = rotated[:, start : start + width * 3].reshape(4, 3, width)
            for offset in vector_offsets:
                x, y = history[..., offset].clone(), history[..., offset + 1].clone()
                history[..., offset], history[..., offset + 1] = (
                    c * x - s * y,
                    s * x + c * y,
                )
        headings += angle
        rotated[:, 171:177] = torch.stack(
            (headings.sin(), headings.cos()), dim=-1
        ).flatten(1)
        torch.testing.assert_close(
            canonical_history(rotated), canonical_history(values), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(values, original, atol=0, rtol=0)
        result = canonical_history(values)
        torch.testing.assert_close(result[:, :162], values[:, :162], atol=0, rtol=0)
        torch.testing.assert_close(
            result[:, 177:189], values[:, 177:189], atol=0, rtol=0
        )

    def test_shadow_route_keeps_clearance_and_stays_inside_pitch(self):
        position = torch.tensor([[1.8, -0.5], [-1.8, -0.15]])
        ball = torch.zeros_like(position)
        targets = shadow_target(position, ball)
        torch.testing.assert_close((targets - ball).norm(dim=1), torch.full((2,), 2.0))
        edge = shadow_target(torch.tensor([[5.5, 3.5]]), torch.tensor([[5.0, 3.0]]))
        self.assertTrue((edge.abs() <= torch.tensor([[5.0, 3.0]])).all())

    def test_history_retains_applied_actions_and_commands_when_skill_changes(self):
        data = N(
            root_link_ang_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            joint_pos=torch.zeros(1, 16),
            default_joint_pos=torch.zeros(1, 16),
            joint_vel=torch.zeros(1, 16),
            default_joint_vel=torch.zeros(1, 16),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_pos_w=torch.zeros(1, 3),
            root_link_lin_vel_w=torch.zeros(1, 3),
        )
        robot = N(data=data, find_joints=lambda *a, **k: (list(range(16)), None))
        ball = N(
            data=N(
                root_link_pos_w=torch.tensor([[0.7, 0.0, 0.11]]),
                root_link_lin_vel_w=torch.zeros(1, 3),
            )
        )
        history = PolicyHistory(robot, ball, 1, "cpu")
        first = torch.tensor([[0.4, 0.0, 0.0]])
        history.sample(first)
        history.last.fill_(1.0)
        second = history.sample(torch.tensor([[0.7, 0.0, 0.0]]))["actor"]
        torch.testing.assert_close(
            second[:, 114:162].reshape(1, 3, 16),
            torch.stack((torch.zeros(16), torch.zeros(16), torch.ones(16)))[None],
        )
        first.fill_(99.0)
        third = history.sample(torch.tensor([[1.8, 0.0, 0.0]]))["actor"]
        torch.testing.assert_close(
            third[:, 162:171].reshape(1, 3, 3),
            torch.tensor([[[0.4, 0.0, 0.0], [0.7, 0.0, 0.0], [1.8, 0.0, 0.0]]]),
        )

    def test_world_rotation_rotates_travel_without_changing_speed(self):
        position = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[2.0, 1.0]])
        command = travel_command(position, torch.tensor([0.0]), target)
        rotated = travel_command(
            position, torch.tensor([torch.pi / 2]), torch.tensor([[-1.0, 2.0]])
        )
        torch.testing.assert_close(
            rotated[:, :2],
            torch.stack((-command[:, 1], command[:, 0]), dim=1),
            atol=1e-6,
            rtol=1e-5,
        )


class CadCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = original().compile()
        cls.new = robot_spec().compile()

    def pairs(self, model, pitch, roll):
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        for side in ("L", "R"):
            for prefix, value in [
                ("Hip_F", pitch),
                ("Hip_A", roll),
                ("Knee", -0.5),
                ("Ankle_P", -0.25),
            ]:
                data.qpos[model.joint(prefix + "_" + side).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        base = int(model.geom("base_link_collision_0").bodyid[0])
        hips = {model.body(n).id for n in ("l_hip_roll_link", "r_hip_roll_link")}
        return [
            c
            for c in data.contact
            if base in [model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]]
            and hips.intersection(
                [int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])]
            )
        ]

    def test_source_inertia_limits_and_exclusions_are_preserved(self):
        for name in ("body_mass", "body_inertia", "jnt_range", "qpos0"):
            np.testing.assert_allclose(
                getattr(self.old, name), getattr(self.new, name), atol=1e-12, rtol=0
            )
        self.assertEqual(self.old.nexclude, self.new.nexclude)

    def test_false_box_overlap_removed_and_real_mesh_contact_retained(self):
        self.assertGreater(len(self.pairs(self.old, 0.25, -0.1)), 0)
        self.assertEqual(len(self.pairs(self.new, 0.25, -0.1)), 0)
        self.assertGreater(len(self.pairs(self.new, -0.25, -0.17)), 0)

    def test_headless_model_retains_physical_cad_hulls(self):
        model = robot_spec(visuals=False).compile()
        physical = [
            i for i in range(model.ngeom) if "_collision_cad_" in model.geom(i).name
        ]
        self.assertEqual(len(physical), 5)
        self.assertTrue(
            all(model.geom_contype[i] and model.geom_conaffinity[i] for i in physical)
        )


class ShotReleaseTests(unittest.TestCase):
    def test_weak_grazes_pushing_and_wrong_direction_do_not_unlock_post_shot_rewards(
        self,
    ):
        velocity = torch.tensor(
            [
                [0.3, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        )
        command = torch.tensor([[2.0, 0.0, 0.0]] * 5)
        seen = torch.tensor([True, True, True, True, False])
        quiet = torch.tensor([0.2, 0.0, 0.1, 0.2, 0.2])
        release = directed_release(velocity, command, seen, quiet)
        torch.testing.assert_close(
            release, torch.tensor([False, False, True, False, False])
        )


class StrikeHistoryTests(unittest.TestCase):
    def test_restoration_keeps_pointer_other_worlds_and_next_append_order(self):
        buffer = CircularBuffer(3, 3, "cpu")
        for step in range(5):
            buffer.append(
                torch.tensor([[step], [10 + step], [20 + step]], dtype=torch.float32)
            )
        before, pointer = buffer.buffer.clone(), buffer._pointer
        history = torch.tensor([[[100.0], [101.0], [102.0]]])
        restore_buffer(buffer, torch.tensor([1]), history)
        self.assertEqual(buffer._pointer, pointer)
        torch.testing.assert_close(
            buffer.buffer[[0, 2]], before[[0, 2]], atol=0, rtol=0
        )
        torch.testing.assert_close(buffer.buffer[1:2], history, atol=0, rtol=0)
        buffer.append(torch.tensor([[5.0], [103.0], [25.0]]))
        torch.testing.assert_close(
            buffer.buffer[1:2], torch.tensor([[[101.0], [102.0], [103.0]]])
        )

    def test_world_frame_reconstruction_preserves_all_local_features_and_is_invertible(
        self,
    ):
        history = torch.randn(4, 207, dtype=torch.float64)
        heading = torch.randn(4, 3, dtype=torch.float64)
        history[:, 171:177] = torch.stack(
            (heading.sin(), heading.cos()), dim=-1
        ).flatten(1)
        angle = torch.tensor([0.3, -1.4, 2.1, -0.8], dtype=torch.float64)
        rotated = rotate_history(history, angle)
        torch.testing.assert_close(rotated[:, :162], history[:, :162], atol=0, rtol=0)
        torch.testing.assert_close(
            rotated[:, 177:189], history[:, 177:189], atol=0, rtol=0
        )
        torch.testing.assert_close(
            rotate_history(rotated, -angle), history, atol=1e-12, rtol=1e-12
        )


class TwistControlTests(unittest.TestCase):
    def test_opposite_bearing_requests_rotation_without_forward_walking(self):
        command = heading_twist(
            torch.tensor([0.0]), torch.tensor([math.pi]), torch.tensor([0.7])
        )
        torch.testing.assert_close(command[:, :2], torch.zeros(1, 2))
        self.assertAlmostEqual(float(command[0, 2].abs()), 0.8, places=6)
        command = waypoint_twist(
            torch.zeros(1, 2), torch.tensor([1.5]), torch.zeros(1, 2)
        )
        torch.testing.assert_close(command, torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
