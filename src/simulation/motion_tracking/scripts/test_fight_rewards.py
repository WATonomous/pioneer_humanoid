"""Force each fight situation in the arena and check the rewards / round
endings that fire (red's side unless noted).

  uv run scripts/test_fight_rewards.py

Prints one line per scenario: PASS/FAIL, what was expected, and the
non-zero reward terms of the deciding step.
"""

import os
import re
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from wato_boxing import mdp as fight  # noqa: E402
from wato_boxing.arena_env_cfg import boxing_arena_env_cfg  # noqa: E402
from wato_tracking.robot import WATO_ACTION_SCALE  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TOUCH_DISTANCE = 0.69  # red's guard glove just touches blue's head


def make(**kw) -> ManagerBasedRlEnv:
  env = ManagerBasedRlEnv(cfg=boxing_arena_env_cfg(**kw), device=DEVICE)
  env.reset()
  return env


def root_state(ent) -> torch.Tensor:
  d = ent.data
  return torch.cat([d.root_link_pos_w, d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w], -1)


def push(env, team: str, speed: float, spin: float) -> None:
  """Shove a fighter backwards (away from the centre) and tip it over."""
  ent = env.scene[team]
  root = root_state(ent)
  sign = -1.0 if team == "red" else 1.0
  root[:, 7] = sign * speed
  root[:, 11] = sign * spin
  ent.write_root_state_to_sim(root)


def actions_holding(env, team: str, offsets: dict[str, float]) -> torch.Tensor:
  """Action that holds `team`'s joints at stance + offsets (others zero)."""
  a = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  start = 0 if team == "red" else len(env.scene["red"].actuator_names)
  for i, n in enumerate(env.scene[team].actuator_names):
    if n in offsets:
      scale = next(v for k, v in WATO_ACTION_SCALE.items() if re.fullmatch(k, n))
      a[:, start + i] = offsets[n] / scale
  return a


def run(env, steps: int, action=None, every=None):
  """Step until the round ends (or `steps`); return per-step rewards of env 0."""
  names = env.reward_manager.active_terms
  log = []
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  for k in range(steps):
    if every is not None:
      every(env, k)
    env.step(action(env, k) if callable(action) else (action if action is not None else zero))
    r = env.reward_manager._step_reward[0].cpu().numpy()
    st = env._fight_state
    log.append(
      dict(
        rewards={n: float(v) for n, v in zip(names, r, strict=True)},
        terminated=bool(env.reset_terminated[0]),
        time_out=bool(env.reset_time_outs[0]),
        red_down=bool(st.teams["red"].down[0]),
        blue_down=bool(st.teams["blue"].down[0]),
        self_col=bool(st.teams["red"].self_collision[0]),
      )
    )
    if log[-1]["terminated"] or log[-1]["time_out"]:
      break
  return log


def nonzero(rewards: dict, tol=1e-6) -> str:
  return ", ".join(f"{k} {v:+.3g}" for k, v in rewards.items() if abs(v) > tol)


RESULTS = []


def check(name: str, ok: bool, expected: str, detail: str) -> None:
  RESULTS.append((name, ok))
  print(f"[{'PASS' if ok else 'FAIL'}] {name}: expected {expected}\n        {detail}")


def total(log, term):
  return sum(s["rewards"][term] for s in log)


def main():
  # ---------------------------------------------------------------- normal distance
  env = make()
  log = run(env, 100)
  last = log[-1]["rewards"]
  check(
    "standing still, 2 s",
    not any(s["terminated"] for s in log) and last["stability"] > 0.019 and total(log, "rope_contact") == 0
    and total(log, "joint_limits") == 0
    and total(log, "airborne") == 0 and total(log, "self_collision") == 0,
    "no round end, stability ~+0.02/step (full), no joint-limit/rope/airborne/self-collision penalty",
    f"steps {len(log)}, last step: {nonzero(last)}",
  )

  env.reset()
  log = run(env, 150, every=lambda e, k: push(e, "blue", 1.5, 4.0) if k == 0 else None)
  end = log[-1]
  check(
    "blue falls without being hit",
    end["terminated"] and end["blue_down"] and end["rewards"]["opponent_slip"] == 10 and end["rewards"]["knockdown"] == 0,
    "round ends, opponent_slip +10, knockdown 0",
    f"ended after {len(log)} steps: {nonzero(end['rewards'])}",
  )

  env.reset()
  log = run(env, 150, every=lambda e, k: push(e, "red", 1.5, 4.0) if k == 0 else None)
  end = log[-1]
  check(
    "red falls",
    end["terminated"] and end["red_down"] and end["rewards"]["knocked_down"] == -100,
    "round ends, knocked_down -100",
    f"ended after {len(log)} steps: {nonzero(end['rewards'])}",
  )

  env.reset()
  log = run(env, 150, every=lambda e, k: (push(e, "red", 1.5, 4.0), push(e, "blue", 1.5, 4.0)) if k == 0 else None)
  end = log[-1]
  check(
    "both fall together",
    end["terminated"] and end["rewards"]["double_down"] == -50,
    "round ends, double_down -50",
    f"ended after {len(log)} steps: {nonzero(end['rewards'])}",
  )

  # feet sliding: drag red sideways along the floor for a few steps
  env.reset()

  def drag(e, k):
    if k < 5:
      ent = e.scene["red"]
      root = root_state(ent)
      root[:, 8] = 0.8
      ent.write_root_state_to_sim(root)

  log = run(env, 6, every=drag)
  check(
    "feet dragged along the floor",
    any(s["rewards"]["foot_slip"] < -0.05 for s in log),
    "foot_slip penalty",
    f"foot_slip per step: {[round(s['rewards']['foot_slip'], 3) for s in log]}",
  )

  env.reset()

  def lift(e, k):
    if k == 0:
      ent = e.scene["red"]
      root = root_state(ent)
      root[:, 2] += 0.12
      ent.write_root_state_to_sim(root)

  log = run(env, 15, every=lift)
  check(
    "dropped from 12 cm (both feet off the floor)",
    any(s["rewards"]["airborne"] == -2 for s in log),
    "airborne -2 while in the air",
    f"airborne per step: {[s['rewards']['airborne'] for s in log]}",
  )

  env.reset()
  rng = np.random.default_rng(0)
  n_red = len(env.scene["red"].actuator_names)

  def thrash(e, k):
    a = torch.zeros(e.num_envs, e.action_manager.total_action_dim, device=e.device)
    a[:, :n_red] = torch.as_tensor(rng.uniform(-6, 6, n_red), dtype=torch.float32)
    return a

  log = run(env, 15, action=thrash)
  check(
    "thrashing (random large actions)",
    total(log, "overspeed") < 0 and total(log, "torque_saturation") < 0 and total(log, "jerkiness") < 0,
    "overspeed, torque_saturation and jerkiness penalties",
    f"sums over {len(log)} steps: overspeed {total(log, 'overspeed'):.2f}, torque_saturation "
    f"{total(log, 'torque_saturation'):.2f}, jerkiness {total(log, 'jerkiness'):.3f}, joint_limits {total(log, 'joint_limits'):.2f}",
  )

  # self-collision: put the left leg through the right one (pose found by search)
  env.reset()
  red = env.scene["red"]
  crossed = None
  for hip_r in np.linspace(-1.2, 1.2, 13):
    for thigh in np.linspace(-1.0, 1.0, 9):
      q = red.data.default_joint_pos.clone()
      q[:, red.joint_names.index("left_hip_r_rs04")] += float(hip_r)
      q[:, red.joint_names.index("left_thigh_rs03")] += float(thigh)
      red.write_joint_state_to_sim(q, torch.zeros_like(q))
      env.sim.forward()
      if bool(fight._self_collision(env, "red")[0]):
        crossed = (float(hip_r), float(thigh), q)
        break
    if crossed:
      break
  env.reset()
  hold = actions_holding(env, "red", {"left_hip_r_rs04": crossed[0], "left_thigh_rs03": crossed[1]})

  def cross(e, k):
    if k == 0:
      e.scene["red"].write_joint_state_to_sim(crossed[2], torch.zeros_like(crossed[2]))

  log = run(env, 5, action=hold, every=cross)
  end = log[-1]
  check(
    "left leg put through the right leg",
    end["terminated"] and end["self_col"] and end["rewards"]["self_collision"] == -100,
    "round ends, self_collision -100",
    f"pose: left hip roll {crossed[0]:+.1f}, hip yaw {crossed[1]:+.2f} rad; ended after {len(log)} steps: {nonzero(end['rewards'])}",
  )
  env.close()

  # ---------------------------------------------------------------- self-collision geometry vs MuJoCo
  env = make()
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  red = env.scene["red"]
  arm_joints = [j for j in red.joint_names if "shoulder" in j or "elbow" in j]
  leg_joints = [j for j in red.joint_names if "hip" in j or "thigh" in j or "knee" in j]
  pairs_leg = [(f"red/leg_left_{a}", f"red/leg_right_{b}") for a in ("thigh_1", "calf_1") for b in ("thigh_1", "calf_1")]
  pairs_arm = [("red/torso", f"red/arm_{b}") for b in ("Mirrorlink3__1__1", "Mirrorlink5__1__1", "link3__1__1", "link5__1__1")]
  agree, clear_fp, deep_fn, n = 0, 0, 0, 0
  for _ in range(150):
    q = red.data.default_joint_pos.clone()
    for j in arm_joints + leg_joints:
      i = red.joint_names.index(j)
      q[:, i] += float(rng.uniform(-1.2, 1.2))
    red.write_joint_state_to_sim(q, torch.zeros_like(q))
    env.sim.forward()
    flag = bool(fight._self_collision(env, "red")[0])
    d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    depth = min(
      mujoco.mj_geomDistance(m, d, m.geom(a).id, m.geom(b).id, 0.0, None) for a, b in pairs_leg + pairs_arm
    )
    n += 1
    deep, clear = depth < -0.08, depth > 0.0
    deep_fn += deep and not flag
    clear_fp += clear and flag
    agree += (deep and flag) or (clear and not flag) or (not deep and not clear)
  check(
    "self-collision test vs MuJoCo distances (150 random leg/arm poses)",
    clear_fp == 0 and deep_fn == 0,
    "flags every pose MuJoCo shows >8 cm deep in, never a pose with clearance",
    f"missed deep overlaps {deep_fn}, false alarms with clearance {clear_fp}",
  )
  env.close()

  # ---------------------------------------------------------------- touching distance (glove on head)
  env = make(distance=TOUCH_DISTANCE)
  log0 = run(env, 3)
  hit_seen = any(s["rewards"]["clean_hits"] > 0 for s in log0)
  env.reset()
  log = run(env, 150, every=lambda e, k: push(e, "blue", 1.5, 4.0) if k == 1 else None)
  end = log[-1]
  check(
    "glove contact, then blue falls (knockdown)",
    hit_seen and end["terminated"] and end["rewards"]["knockdown"] == 100,
    "clean_hits > 0 while touching; round ends, knockdown +100",
    f"clean_hits first steps {[round(s['rewards']['clean_hits'], 3) for s in log0]}; ended after {len(log)} steps: {nonzero(end['rewards'])}",
  )
  env.close()

  env = make(distance=TOUCH_DISTANCE, learner="blue")
  log = run(env, 150, every=lambda e, k: push(e, "blue", 1.5, 4.0) if k == 1 else None)
  end = log[-1]
  check(
    "same knockdown from blue's side",
    end["terminated"] and end["rewards"]["knocked_down"] == -100 and end["rewards"]["knockdown"] == 0,
    "knocked_down -100",
    f"ended after {len(log)} steps: {nonzero(end['rewards'])}",
  )
  env.close()

  # ---------------------------------------------------------------- ropes and the bell
  env = make(distance=3.75)
  log = run(env, 5)
  check(
    "backed onto the ropes",
    any(s["rewards"]["rope_contact"] == -2 for s in log),
    "rope_contact -2",
    f"rope_contact per step: {[s['rewards']['rope_contact'] for s in log]}",
  )
  env.close()

  env = make(round_s=1.0)
  log = run(env, 60)
  end = log[-1]
  check(
    "bell after a 1 s round, nobody down",
    end["time_out"] and not end["terminated"] and "time_up_points" in end["rewards"],
    "time out (not a knockdown), time_up_points = 10 x tanh(landed difference / 3) = 0 with no hits",
    f"ended after {len(log)} steps, time_up_points {end['rewards']['time_up_points']:+.2f}",
  )
  env.close()

  print(f"\n{sum(ok for _, ok in RESULTS)}/{len(RESULTS)} scenarios passed")


if __name__ == "__main__":
  main()
