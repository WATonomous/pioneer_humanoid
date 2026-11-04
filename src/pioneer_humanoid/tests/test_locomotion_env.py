"""Env-level tests for #335 Task 5: actuator randomization + rated-torque metric/penalty.

Builds the Flat Pioneer locomotion env once (spawns the robot, so the USD is regenerated if the URDF changed).
Run inside the simulation_isaac container, as its own pytest call (one Kit app per process):
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $ISAACLAB/isaaclab.sh -p -m pytest \
        src/pioneer_humanoid/tests/test_locomotion_env.py -v -p no:cacheprovider
"""

# Start Kit first: isaaclab imports pxr (USD), which only exists once Kit runs.
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app  # headless = no window

import pytest  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from humanoid_rl_tasks.locomotion.config.pioneer_humanoid_v1.flat_env_cfg import PioneerHumanoidFlatEnvCfg  # noqa: E402
from humanoid_rl_tasks.locomotion.config.pioneer_humanoid_v1.rough_env_cfg import (  # noqa: E402
    ARMATURE_SCALE,
    FRICTION_SCALE,
    GAIN_SCALE,
    PioneerHumanoidRoughEnvCfg,
)
from pioneer_humanoid.leg_params import JOINT_MOTOR  # noqa: E402

# 64 envs x 2 joints per group: enough draws that "values vary" and "range is used" can't fail by chance.
NUM_ENVS = 64
TOL = 1e-5


@pytest.fixture(scope="module")
def env():
    cfg = PioneerHumanoidFlatEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRLEnv(cfg=cfg)
    yield env
    env.close()


def _within(x, lo, hi):
    return bool(torch.all(x >= lo - TOL) and torch.all(x <= hi + TOL))


def test_rough_cfg_kd_upper_bound_respects_mit_max():
    """kd 5 groups may only scale down (0.8-1.0); Hip_R (kd 4) keeps 0.8-1.2 -> max 4.8 (D4)."""
    events = PioneerHumanoidRoughEnvCfg().events
    expected_hi = {"hip_flexion": 1.0, "hip_abduction": 1.0, "hip_rotation": 1.2, "knee": 1.0, "ankle": 1.0}
    for group, hi in expected_hi.items():
        term = getattr(events, f"randomize_gains_{group}")
        assert term.mode == "reset"
        assert term.params["stiffness_distribution_params"] == GAIN_SCALE
        assert term.params["damping_distribution_params"] == pytest.approx((0.8, hi))
    assert events.randomize_joint_parameters.mode == "startup"


def test_gains_randomized_per_env_at_reset(env):
    env.reset()
    robot = env.scene["robot"]
    before = {g: a.stiffness.clone() for g, a in robot.actuators.items()}
    for group, act in robot.actuators.items():
        kp0, kd0 = act.cfg.stiffness, act.cfg.damping
        kd_max = min(JOINT_MOTOR[n].kd_max for n in act.joint_names)
        assert _within(act.stiffness / kp0, *GAIN_SCALE), group
        assert _within(act.damping / kd0, GAIN_SCALE[0], min(GAIN_SCALE[1], kd_max / kd0)), group
        assert float(act.damping.max()) <= kd_max + TOL, group
        assert float(act.stiffness.std()) > 0 and float(act.damping.std()) > 0, group
    # Hip_R's range goes above nominal (4 -> up to 4.8); with 128 draws some must be above 4.
    assert float(robot.actuators["hip_rotation"].damping.max()) > 4.0
    env.reset()
    for group, act in robot.actuators.items():
        assert not torch.equal(act.stiffness, before[group]), f"{group}: reset did not redraw"


def test_armature_and_friction_randomized_in_physx(env):
    robot = env.scene["robot"]
    armature = robot.root_physx_view.get_dof_armatures().cpu()  # what PhysX actually uses
    friction = robot.root_physx_view.get_dof_friction_properties().cpu()  # [..., 0/1/2] = static/dynamic/viscous
    for j, name in enumerate(robot.joint_names):
        motor = JOINT_MOTOR[name]
        static, dynamic, viscous = friction[:, j, 0], friction[:, j, 1], friction[:, j, 2]
        assert _within(armature[:, j] / motor.armature, *ARMATURE_SCALE), name
        assert _within(static / motor.backdrive_torque, *FRICTION_SCALE), name
        assert _within(dynamic / motor.backdrive_torque, *FRICTION_SCALE), name
        assert torch.all(dynamic <= static + TOL), name  # PhysX requires dynamic <= static
        assert torch.all(viscous == 0.0), name
        assert float(armature[:, j].std()) > 0 and float(static.std()) > 0 and float(dynamic.std()) > 0, name


def test_metric_counts_fraction_above_rated(env):
    """Hand-made torques: joint 0 above rated in 3 of 4 steps (sign ignored) -> 0.75, others 0."""
    env.reset()
    robot = env.scene["robot"]
    term = env.command_manager.get_term("torque_rating")
    j, name, rated = term.joint_ids[0], term.joint_names[0], float(term.rated_torque[0])
    key = f"frac_above_rated_{name}"
    env.episode_length_buf[:] = 1  # mid-episode, so every step counts
    for scale in (2.0, -1.5, 3.0, 0.5):
        robot.data.applied_torque[:] = 0.0
        robot.data.applied_torque[:, j] = scale * rated
        env.command_manager.compute(dt=env.step_dt)
    assert torch.allclose(term.metrics[key], torch.full((NUM_ENVS,), 0.75, device=env.device))
    for other in term.joint_names[1:]:
        assert torch.all(term.metrics[f"frac_above_rated_{other}"] == 0.0)
    # An env that just reset (episode_length_buf 0) must not count the old episode's last torque.
    env.episode_length_buf[0] = 0
    robot.data.applied_torque[:, j] = 2.0 * rated
    env.command_manager.compute(dt=env.step_dt)
    assert float(term.metrics[key][0]) == pytest.approx(0.75)  # unchanged
    assert float(term.metrics[key][1]) == pytest.approx(0.8)  # 4 of 5
    # Logged at reset as the mean over the reset envs: (0.75 + 63 * 0.8) / 64.
    _, extras = env.reset()
    logged = extras["log"][f"Metrics/torque_rating/{key}"]
    assert logged == pytest.approx((0.75 + (NUM_ENVS - 1) * 0.8) / NUM_ENVS)
    assert set(f"Metrics/torque_rating/frac_above_rated_{n}" for n in JOINT_MOTOR) <= set(extras["log"])


def test_penalty_is_linear_excess(env):
    """2x rated -> 1, -1.5x rated -> 0.5, 0.5x rated -> 0: total 1.5 per env. Weight stays 0."""
    env.reset()
    robot = env.scene["robot"]
    term_cfg = env.reward_manager.get_term_cfg("torque_above_rated")
    assert term_cfg.weight == 0.0
    penalty = term_cfg.func  # class terms are instantiated by the manager
    ids, rated = penalty.joint_ids, penalty.rated_torque
    robot.data.applied_torque[:] = 0.0
    robot.data.applied_torque[:, ids[0]] = 2.0 * rated[0]
    robot.data.applied_torque[:, ids[1]] = -1.5 * rated[1]
    robot.data.applied_torque[:, ids[2]] = 0.5 * rated[2]
    value = penalty(env, **term_cfg.params)
    assert value.shape == (NUM_ENVS,)
    assert torch.allclose(value, torch.full((NUM_ENVS,), 1.5, device=env.device))


def test_rollout_10_steps(env):
    """Zero actions (hold default pose): finite obs/reward/torque, metric matches a manual count."""
    env.reset()
    robot = env.scene["robot"]
    term = env.command_manager.get_term("torque_rating")
    actions = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim, device=env.device)
    above = torch.zeros(NUM_ENVS, len(term.joint_ids), device=env.device)
    for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert not torch.any(terminated | truncated), "an env ended within 10 steps; manual count would be off"
        tau = robot.data.applied_torque
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all() and torch.isfinite(tau).all()
        above += (tau[:, term.joint_ids].abs() > term.rated_torque).float()
    for i, name in enumerate(term.joint_names):
        assert torch.allclose(term.metrics[f"frac_above_rated_{name}"], above[:, i] / 10), name
