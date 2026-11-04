"""Tests for DelayedDCMotor (#335 Task 3).

Run inside the simulation_isaac container:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $ISAACLAB/isaaclab.sh -p -m pytest \
        src/pioneer_humanoid/tests/test_actuators.py -v -p no:cacheprovider
"""

# Start Kit first: isaaclab imports pxr (USD), which only exists once Kit runs.
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app  # headless = no window

import re  # noqa: E402
from dataclasses import MISSING  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.types import ArticulationActions  # noqa: E402  (targets: positions / velocities / efforts)

from pioneer_humanoid.delayed_dc_motor_cfg import DelayedDCMotorCfg  # noqa: E402
from pioneer_humanoid.leg_params import JOINT_MOTOR  # noqa: E402
from pioneer_humanoid.whole_body import WHOLE_BODY_HUMANOID_CFG  # noqa: E402

# Test-only sizes. The class uses whatever num_envs/device it is given, so training
# (cuda:0, 4096 envs) runs the same code.
NUM_ENVS = 4
DEVICE = "cpu"
JOINT_NAMES = ["j0", "j1"]

# Round test numbers, not motor values (real ones come from leg_params in Task 4).
# Chosen so a small PD torque stays far below every limit -> clipping can't hide a math error.
KP = 100.0  # Nm/rad, inside MIT kp range 0-500
KD = 2.0  # Nm·s/rad, inside MIT kd range 0-5 (D4)
EFFORT_LIMIT = 50.0  # Nm, flat cap (like peak torque, D13)
SATURATION_EFFORT = 100.0  # Nm, stall torque at zero speed (DCMotor's name for it)
VELOCITY_LIMIT = 10.0  # rad/s, no-load speed: torque-speed line hits 0 here


def make_actuator(num_envs=NUM_ENVS, **overrides):
    """Build an actuator the same way the robot does: cfg.class_type(...).

    `overrides` replaces any default below, e.g. make_actuator(max_delay=2).
    `num_envs` defaults to 4; the delay-range test needs many more (see there).
    """
    params = dict(
        joint_names_expr=JOINT_NAMES,
        stiffness=KP,
        damping=KD,
        effort_limit=EFFORT_LIMIT,
        saturation_effort=SATURATION_EFFORT,
        velocity_limit=VELOCITY_LIMIT,
        min_delay=0,
        max_delay=0,
    )
    params.update(overrides)
    cfg = DelayedDCMotorCfg(**params)
    return cfg.class_type(
        cfg,
        joint_names=JOINT_NAMES,
        joint_ids=list(range(len(JOINT_NAMES))),
        num_envs=num_envs,
        device=DEVICE,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
    )


def full(value):
    """A (num_envs, num_joints) tensor filled with `value`."""
    return torch.full((NUM_ENVS, len(JOINT_NAMES)), value, device=DEVICE)


def test_pd_math():
    act = make_actuator()
    act.reset(None)  # reset picks the delay per env; None = all envs

    # Targets (what the policy asks for)
    q_des = 0.3  # rad (~17 deg)
    qd_des = 0.0  # rad/s: "hold still at q_des"
    tau_ff = 2.0  # Nm, extra torque added on top of PD

    # Current joint state (what the sim measures)
    q = 0.2  # rad (~11 deg): 0.1 rad short of the target
    qd = 0.5  # rad/s (~29 deg/s): slow, far below the 10 rad/s limit

    # By hand: kp term 100*(0.3-0.2) = 10, kd term 2*(0-0.5) = -1, ff = 2 -> 11 Nm.
    # Each term is a different size, so dropping one or flipping a sign gives a
    # different total (1, 12, 9, 13 or -9) and the test catches it.
    expected = KP * (q_des - q) + KD * (qd_des - qd) + tau_ff
    assert abs(expected - 11.0) < 1e-9  # guards against a typo in the numbers above

    action = ArticulationActions(
        joint_positions=full(q_des),
        joint_velocities=full(qd_des),
        joint_efforts=full(tau_ff),
    )
    act.compute(action, joint_pos=full(q), joint_vel=full(qd))

    # PD output before clipping
    torch.testing.assert_close(act.computed_effort, full(expected))
    # 11 Nm at 0.5 rad/s is inside the limits (+50 / -50 Nm here), so clipping must leave it alone
    torch.testing.assert_close(act.applied_effort, full(expected))


@pytest.mark.parametrize("unset", [None, MISSING])  # MISSING = field left out; None = Isaac's "use USD value"
@pytest.mark.parametrize("name", ["saturation_effort", "velocity_limit", "effort_limit"])
def test_missing_limit_raises(name, unset):
    with pytest.raises(ValueError, match=name):  # message must name the missing field
        make_actuator(**{name: unset})


@pytest.mark.parametrize("n", [0, 1, 2])  # runs the test 3 times: delay 0, 1, 2 steps
def test_delay_n_steps(n):
    act = make_actuator(min_delay=n, max_delay=n)  # min = max -> delay is exactly n
    act.reset(None)
    step_at = 3  # call where the target jumps from 0 to 0.1 rad
    torques = []
    for k in range(step_at + 4):  # 7 calls: enough to see the jump even at n = 2
        q_des = 0.1 if k >= step_at else 0.0
        # new ArticulationActions every call: compute() changes the one you pass in
        action = ArticulationActions(
            joint_positions=full(q_des), joint_velocities=full(0.0), joint_efforts=full(0.0)
        )
        act.compute(action, joint_pos=full(0.0), joint_vel=full(0.0))
        torques.append(act.computed_effort.clone())  # clone = keep a copy of this call's torque
    for k, tau in enumerate(torques):
        expected = KP * 0.1 if k >= step_at + n else 0.0  # 10 Nm once the delayed target arrives
        torch.testing.assert_close(tau, full(expected), msg=f"call {k}, delay {n}")


# 1000 envs, not 4: each env draws 0, 1 or 2, so with 4 envs some value is missing ~56% of runs.
# With 1000 the chance is <= 3*(2/3)^1000 (~1e-176), i.e. never. Still cheap: 1000 ints on CPU.
MANY_ENVS = 1000


def test_delay_range_at_reset():
    act = make_actuator(num_envs=MANY_ENVS, min_delay=0, max_delay=2)
    act.reset(None)  # all envs draw a delay
    lags = act.positions_delay_buffer.time_lags  # one whole number per env, shape (num_envs,)

    # Every delay inside [min_delay, max_delay] ...
    assert lags.min().item() >= 0 and lags.max().item() <= 2
    # ... and every value in the range actually used (not stuck on one value).
    assert set(lags.tolist()) == {0, 1, 2}
    # All three buffers (position / velocity / effort targets) must share the same delay per env.
    torch.testing.assert_close(act.velocities_delay_buffer.time_lags, lags)
    torch.testing.assert_close(act.efforts_delay_buffer.time_lags, lags)


def test_partial_reset_redraws_only_those_envs():
    # In training, envs reset one by one (when a robot falls), so this is the common path.
    act = make_actuator(num_envs=MANY_ENVS, min_delay=0, max_delay=2)
    act.reset(None)
    before = act.positions_delay_buffer.time_lags.clone()  # clone: keep a copy, the buffer is updated in place

    # Tensor of env ids, like the env manager passes in training.
    reset_ids = torch.arange(MANY_ENVS // 2, device=DEVICE)  # first half
    act.reset(reset_ids)
    after = act.positions_delay_buffer.time_lags
    half = MANY_ENVS // 2

    # Untouched half: exactly the same delays as before.
    torch.testing.assert_close(after[half:], before[half:])
    # Reset half: still in range, all values used, and actually redrawn
    # (each env keeps its old value with chance 1/3, so all 500 unchanged is (1/3)^500, i.e. never).
    assert set(after[:half].tolist()) == {0, 1, 2}
    assert not torch.equal(after[:half], before[:half])


# Torque-speed clip (inherited from DCMotor). With the test numbers above
# (stall 100 Nm, no-load 10 rad/s, flat cap 50 Nm) the allowed torque at speed v is:
#   max = min(100 * (1 - v/10), 50)     min = max(100 * (-1 - v/10), -50)
# kp = kd = 0 so the commanded torque is exactly tau_ff (no PD term to work out).
@pytest.mark.parametrize(
    "vel, tau_cmd, expected",
    [
        (0.0, 80.0, 50.0),  # standing still: only the flat cap applies
        (7.5, 40.0, 25.0),  # on the slope: 100 * (1 - 0.75) = 25
        (10.0, 30.0, 0.0),  # at no-load speed: driving torque -> 0
        (10.0, -30.0, -30.0),  # ... but braking (against the motion) is still allowed
        (10.0, -80.0, -50.0),  # ... up to the flat cap
        (-10.0, -30.0, 0.0),  # same, moving the other way
        (-10.0, 30.0, 30.0),
        (12.0, 0.0, -20.0),  # above no-load: even a 0 command becomes braking, 100 * (1 - 1.2) = -20
    ],
)
def test_torque_speed_clip(vel, tau_cmd, expected):
    act = make_actuator(stiffness=0.0, damping=0.0)
    act.reset(None)
    action = ArticulationActions(
        joint_positions=full(0.0), joint_velocities=full(0.0), joint_efforts=full(tau_cmd)
    )
    act.compute(action, joint_pos=full(0.0), joint_vel=full(vel))
    torch.testing.assert_close(act.computed_effort, full(tau_cmd))  # before the clip: untouched
    torch.testing.assert_close(act.applied_effort, full(expected))  # after the clip


# --- Task 4: leg actuator config in whole_body.py (no robot spawn) ----------------------------
LEG_ACTUATORS = WHOLE_BODY_HUMANOID_CFG.actuators


def _groups_matching(joint):
    """Names of actuator groups whose joint_names_expr matches `joint` (Isaac uses full regex matches)."""
    return [name for name, cfg in LEG_ACTUATORS.items() if any(re.fullmatch(expr, joint) for expr in cfg.joint_names_expr)]


@pytest.mark.parametrize("joint", sorted(JOINT_MOTOR))
def test_every_leg_joint_in_exactly_one_group(joint):
    # A pattern typo only gives a spawn-time warning in Isaac ("Not all actuators are configured!").
    assert len(_groups_matching(joint)) == 1, _groups_matching(joint)


@pytest.mark.parametrize("group", sorted(LEG_ACTUATORS))
def test_group_values_come_from_its_motor(group):
    cfg = LEG_ACTUATORS[group]
    joints = [j for j in JOINT_MOTOR if group in _groups_matching(j)]
    motor = JOINT_MOTOR[joints[0]]
    assert isinstance(cfg, DelayedDCMotorCfg)
    assert cfg.effort_limit == pytest.approx(motor.peak_torque)  # D13
    assert cfg.saturation_effort == pytest.approx(motor.saturation_effort)  # D14
    assert cfg.velocity_limit == pytest.approx(motor.torque_speed_limit)  # D14
    assert cfg.velocity_limit_sim == pytest.approx(1.5 * motor.no_load_speed)  # D15
    assert cfg.armature == pytest.approx(motor.armature)
    assert cfg.friction == pytest.approx(motor.backdrive_torque)
    assert cfg.dynamic_friction == pytest.approx(motor.backdrive_torque)  # moving joints feel friction too
    assert cfg.damping <= motor.kd_max  # D4
    assert (cfg.min_delay, cfg.max_delay) == (0, 2)
