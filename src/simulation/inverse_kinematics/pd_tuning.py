"""PD and action-scale tuning (task A3): how fast and how cleanly do the arm's joints follow their reference?

Running it prints three tables:
1. Step responses: each joint's 90% rise time, overshoot and final error (gravity sag, no compensation).
2. Ramp tests: lag and leftover wobble when the reference moves by eta rad per control step (gravity comp on).
3. Tool step sizes: how far one smallest step (eta * beta_min) moves and turns the tool point.

Findings (2026-11-24):
- Joints respond first-order (time constant = damping / stiffness), no overshoot, 90% of a step in 0.13-0.2 s.
- eta <= 0.05 is clean. At 0.0625-0.08 the forearm roll keeps swinging ~3 deg after an all-joint ramp stops;
  it needs the forearm to move together with other joints (the forearm alone, or any group without it, is clean).
- At eta = 0.05, beta_min = 0.1, one full step near the goal moves the tool 1-3 mm and turns it 0.29 deg.
Recommended: eta = 0.05 rad/step, beta_min = 0.1, the paper's beta thresholds, the robot's real PD gains.
"""
import numpy as np

from arm_sim import Sim
from arm_model import LeftArm
from metrics import CONTROL_DT, pose_errors
from sampling import sample_goal

JOINT_NAMES = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
               "forearm_roll", "wrist_pitch", "wrist_yaw"]


def step_response(sim, q0, joint, size, steps=30):
    """Hold the arm at q0, then move ONE joint's reference by `size` rad and hold it there.
    Returns the joint's movement (rad, relative to q0) after each control step: an array (steps,)."""
    sim.reset(q0)
    target = q0.copy()
    target[joint] += size
    moved = []
    for _ in range(steps):
        sim.step(target)
        moved.append(sim.data.qpos[sim.arm.qadr[joint]] - q0[joint])
    return np.array(moved)


def describe(moved, size):
    """From a step response: time to reach 90% of the step (s), overshoot (% of the step), final error (rad)."""
    t90 = np.flatnonzero(moved >= 0.9 * size)
    t90 = (t90[0] + 1) * CONTROL_DT if t90.size else np.nan
    overshoot = max(0.0, (moved.max() - size) / size * 100)
    final_err = size - moved[-1]
    return t90, overshoot, final_err


def ramp_test(sim, q0, eta, distance=0.5, joints=range(7), settle_steps=40):
    """Move the reference of `joints` from q0 toward q0 + distance by eta rad per control step (like the policy
    at full action), then hold it there.
    Returns (during, after): reference minus joint angle at each step, during the ramp (n_ramp, 7)
    and after it stops (settle_steps, 7)."""

    sim.reset(q0)
    ref = q0.copy()
    mask = np.zeros(7)
    mask[joints] = 1
    n_ramp = int(np.ceil(distance / eta))
    during = []
    after = []
    for k in range(n_ramp + settle_steps):
        if k < n_ramp:
            ref = np.minimum(ref + eta * mask, q0 + distance * mask)
        sim.step(ref)
        if k < n_ramp:
            during.append(ref - sim.data.qpos[sim.arm.qadr])
        else:
            after.append(ref - sim.data.qpos[sim.arm.qadr])
    return np.array(during), np.array(after)


def tool_step_sizes(arm, rng, step, n=200):
    """For n random goals: how far the tool point moves (m) and turns (rad) when ONE joint moves by `step` rad.
    Returns (moves, turns), each (n, 7): one row per goal, one column per joint."""
    moves, turns = np.zeros((n, 7)), np.zeros((n, 7))
    for i in range(n):
        q, p, R = sample_goal(arm, rng)
        for j in range(7):
            q2 = q.copy(); q2[j] += step
            p2, R2 = arm.tcp_pose(q2)
            moves[i, j], turns[i, j] = pose_errors(p2, R2, p, R)
    return moves, turns


if __name__ == "__main__":
    # --- 1. Step responses (no gravity compensation, so the final error shows the sag) ---
    sim = Sim(gravity_comp=False)
    kp = sim.model.actuator_gainprm[sim.act_ids, 0]          # each motor's stiffness
    limit = sim.model.actuator_forcerange[sim.act_ids, 1]    # each motor's torque limit
    # A motor maxes out when (angle error x stiffness) is bigger than its torque limit:
    for name, k, lim in zip(JOINT_NAMES, kp, limit):
        print(f"{name:15s} stiffness {k:6.0f}  limit {lim:5.1f} Nm  -> maxes out above {np.degrees(lim / k):.2f} deg of error")

    q0 = sim.arm.rest.copy()
    q0[0], q0[1] = 0.3, 0.3                  # a pose away from the joint limits
    print()
    for j in range(7):
        for s in (0.01, 0.05, 0.2):
            moved = step_response(sim, q0, j, s)
            t90, overshoot, final_err = describe(moved, s)
            print(f"{JOINT_NAMES[j]:15s} step {s:4.2f} rad   t90 {t90:.3f} s   "
                  f"overshoot {overshoot:4.1f}%   final error {np.degrees(final_err):+.5f} deg")

    # --- 2. Ramp tests: gravity compensation ON, so sag doesn't get mixed in with lag ---
    sim = Sim(gravity_comp=True)
    maxout = limit / kp                      # rad: the error at which each motor hits its torque limit
    np.set_printoptions(precision=2, suppress=True)

    print("\nRamp tests: all joints, 0.5 rad")
    for eta in (0.01, 0.02, 0.03, 0.05, 0.0625, 0.08, 0.1):
        during, after = ramp_test(sim, q0, eta)
        ramp_time = len(during) * CONTROL_DT
        lag = np.max(np.abs(during), axis=0)
        wobble = np.max(np.abs(after[4:]), axis=0)
        print(f"eta {eta:.4f}  ramp {ramp_time:.2f} s  max lag {np.degrees(lag)} deg  "
              f"lag/maxout {np.max(lag / maxout):.1f}  wobble {np.degrees(wobble)} deg")

    print("\nWhich joints shake the forearm roll? eta = 0.0625")
    for label, joints in [("all", range(7)), ("forearm only", [4]),
                          ("shoulder+elbow", [0, 1, 2, 3]), ("wrist only", [5, 6])]:
        during, after = ramp_test(sim, q0, 0.0625, joints=joints)
        wobble = np.max(np.abs(after[4:, 4]))
        print(f"{label:15s} forearm roll wobble {np.degrees(wobble):.2f} deg")

    # --- 3. Tool step sizes: geometry only, no physics ---
    arm = LeftArm()
    rng = np.random.default_rng(1)       # NOT EVAL_SEED: tuning must never touch the evaluation set
    print("\nTool movement from one step of eta * beta_min (beta_min = 0.1)")
    for eta in (0.03, 0.05, 0.0625):
        step = eta * 0.1
        moves, turns = tool_step_sizes(arm, rng, step)
        print(f"eta {eta:.4f}  step {np.degrees(step):.2f} deg")
        print(f"   median moves (mm) {np.median(moves, axis=0) * 1000}")
        print(f"   max moves (mm)    {np.max(moves, axis=0) * 1000}")
        print(f"   median turns (deg) {np.degrees(np.median(turns, axis=0))}")
