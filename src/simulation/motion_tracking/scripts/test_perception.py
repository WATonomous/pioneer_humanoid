"""Check the opponent perception (wato_boxing/perception.py) in the arena.

  uv run scripts/test_perception.py

1. Camera occlusion vs MuJoCo ray casting, over random arm poses of both
   fighters: is a target hidden exactly when MuJoCo's ray hits a glove or arm
   first?
2. Ultrasonic range vs MuJoCo's ray straight down the sensor axis (the cone
   reading can only be shorter or equal), and how often it sees its own guard.
3. EKF accuracy, standing and with the opponent moving in and out, for
   RGB only, RGB + ultrasonic and RGB + D455 depth: position and velocity
   error per target, uncertainty, and how often each target is in view.
"""

import os
import sys
from dataclasses import replace

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from wato_boxing import perception as vision  # noqa: E402
from wato_boxing.arena_env_cfg import boxing_arena_env_cfg  # noqa: E402
from wato_boxing.perception import TARGETS, CameraCfg, PerceptionCfg, UltrasonicCfg  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS = []


def check(name, ok, detail):
  RESULTS.append(ok)
  print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")


def root_state(ent):
  d = ent.data
  return torch.cat([d.root_link_pos_w, d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w], -1)


def random_arms(env, rng, spread=1.0):
  for team in ("red", "blue"):
    ent = env.scene[team]
    q = ent.data.default_joint_pos.clone()
    for i, n in enumerate(ent.joint_names):
      if "shoulder" in n or "elbow_ak80_1" in n or "elbow_ak80_2" in n:
        q[:, i] += float(rng.uniform(-spread, spread))
    ent.write_joint_state_to_sim(q, torch.zeros_like(q))
  env.sim.forward()


def main():
  rng = np.random.default_rng(0)
  env = ManagerBasedRlEnv(cfg=boxing_arena_env_cfg(distance=1.1), device=DEVICE)
  env.reset()
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  tr = vision.tracker(env, "red")
  sc = tr.scene
  groups = np.array([0, 1, 0, 1, 0, 0], dtype=np.uint8)  # gloves (1) + hitboxes (3)
  occluder_geoms = set(sum(sc.glove_geoms.values(), []))
  for team in ("red", "blue"):
    for a, _ in vision.ARM_SEGMENTS:
      occluder_geoms.add(m.geom(f"{team}/arm_{a}").id)
  geomid = np.zeros(1, dtype=np.int32)

  # 1. camera occlusion vs MuJoCo rays
  agree = total = skipped = hidden_n = 0
  for _ in range(200):
    random_arms(env, rng)
    _, _, _, info = vision.simulate_camera(sc, replace(CameraCfg(), dropout=0.0), None)
    d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    cam = d.site_xpos[m.site("red/d455").id]
    for ti, g in enumerate(sc.target_geoms):
      if not bool(info["in_fov"][0, ti]):
        continue
      vec = d.geom_xpos[g] - cam
      dist = np.linalg.norm(vec)
      # the camera sits inside red's own head hitbox (the head target is the camera bar)
      hit_d = mujoco.mj_ray(m, d, cam, vec / dist, groups, 0, m.body("red/baseLink__1__1").id, geomid)
      first = int(geomid[0])
      if first == g or first == -1:
        truth = False
      elif first in occluder_geoms and (g not in occluder_geoms or first != g):
        # a glove's own forearm is excluded by design
        if (TARGETS[ti] == "glove_l" and m.geom(first).name == "blue/arm_Mirrorlink5__1__1") or (
          TARGETS[ti] == "glove_r" and m.geom(first).name == "blue/arm_link5__1__1"
        ):
          skipped += 1
          continue
        truth = True
      else:
        skipped += 1  # blocked by a body part the camera model does not use as an occluder
        continue
      ours = bool(info["hidden"][0, ti])
      total += 1
      agree += ours == truth
      hidden_n += truth
  check(
    "camera occlusion matches MuJoCo ray casting (200 random arm poses, both fighters)",
    total > 100 and agree / total > 0.95,
    f"agree {agree}/{total} ({agree / max(total, 1):.0%}), {hidden_n} hidden by a glove/arm; "
    f"{skipped} rays skipped (first hit a head/torso/leg, not modelled as an occluder)",
  )

  # 2. ultrasonic vs the axis ray
  env.reset()
  own_hits, short, n_valid = 0, 0, 0
  diffs = []
  for _ in range(100):
    random_arms(env, rng, spread=0.6)
    _, valid, true_rng, hit_own = vision.simulate_ultrasonic(sc, replace(UltrasonicCfg(), dropout=0.0), None)
    d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    sid = m.site("red/ultrasonic").id
    axis = -d.site_xmat[sid].reshape(3, 3)[:, 2]
    ray = mujoco.mj_ray(m, d, d.site_xpos[sid], axis, groups, 0, m.body("red/Torso_1").id, geomid)
    if bool(valid[0]):
      n_valid += 1
      own_hits += bool(hit_own[0])
      if ray >= 0:
        diffs.append(float(true_rng[0]) - ray)
        short += float(true_rng[0]) <= ray + 0.02
  check(
    "ultrasonic cone reading is never longer than the axis ray",
    len(diffs) >= 10 and short == len(diffs),
    f"{short}/{len(diffs)} readings <= axis ray (+2 cm); cone minus axis: median {np.median(diffs) * 100:+.1f} cm; "
    f"nearest thing was its own guard in {own_hits}/{n_valid} readings",
  )
  env.close()

  # 3. EKF accuracy
  configs = {
    "RGB only": PerceptionCfg(ultrasonic=UltrasonicCfg(enabled=False)),
    "RGB + ultrasonic": PerceptionCfg(),
    "RGB + D455 depth": PerceptionCfg(camera=CameraCfg(use_depth=True), ultrasonic=UltrasonicCfg(enabled=False)),
  }
  print("\nEKF error after 1 s settle (cm); moving = blue slides upright 0.6 m in at 0.4 m/s, then back")
  print(f"{'sensors':18s} {'target':8s} {'in view':>7s} {'still pos':>9s} {'still std':>9s} {'moving pos':>10s} {'moving vel':>10s} {'tracked':>8s}")
  summary = {}
  for label, pcfg in configs.items():
    env = ManagerBasedRlEnv(cfg=boxing_arena_env_cfg(distance=1.3, perception=pcfg), device=DEVICE)
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    still, still_std, moving, moving_v, seen = [], [], [], [], []
    blue = env.scene["blue"]
    start = root_state(blue).clone()
    resets = 0
    for k in range(250):
      if k >= 100:  # blue slides (upright, kinematically) towards red for 1 s, then back
        t = (k - 100) * env.step_dt
        offset = -0.4 * t if t < 1.5 else -0.4 * 1.5 + 0.4 * (t - 1.5)
        root = start.clone()
        root[:, 0] += offset
        root[:, 7] = -0.4 if t < 1.5 else 0.4
        blue.write_root_state_to_sim(root)
      env.step(zero)
      resets += int(env.reset_buf[0])
      tr = env._opponent_trackers["red"]
      true_p, true_v = tr.scene.targets()
      err = (tr.x[..., :3] - true_p).norm(dim=-1)[0].cpu().numpy()
      verr = (tr.x[..., 3:] - true_v).norm(dim=-1)[0].cpu().numpy()
      std = torch.diagonal(tr.P[..., :3, :3], dim1=-2, dim2=-1).sum(-1).sqrt()[0].cpu().numpy()
      info = tr.last_cam_info
      seen.append((info["in_fov"] & ~info["hidden"])[0].cpu().numpy())
      if not bool(tr.init[0].all()):
        err = np.where(tr.init[0].cpu().numpy(), err, np.nan)
        verr = np.where(tr.init[0].cpu().numpy(), verr, np.nan)
      if 50 <= k < 100:
        still.append(err)
        still_std.append(std)
      elif k >= 110:
        moving.append(err)
        moving_v.append(verr)
    still, still_std, moving, moving_v, seen = map(np.array, (still, still_std, moving, moving_v, seen))
    tracked = np.isfinite(moving).mean(0)
    for ti, t in enumerate(TARGETS):
      print(
        f"{label:18s} {t:8s} {seen[:, ti].mean():6.0%} {still[:, ti].mean() * 100:9.1f} {still_std[:, ti].mean() * 100:9.1f}"
        f" {np.sqrt(np.nanmean(moving[:, ti] ** 2)) * 100:10.1f} {np.sqrt(np.nanmean(moving_v[:, ti] ** 2)) * 100:10.1f}"
        f" {tracked[ti]:8.0%}"
      )
    summary[label] = (np.nanmean(still), np.sqrt(np.nanmean(moving**2)))
    if resets:
      print(f"  ({label}: round restarted {resets}x during the run)")
    env.close()

  rgb, us, depth = summary["RGB only"], summary["RGB + ultrasonic"], summary["RGB + D455 depth"]
  check(
    "EKF tracks the opponent (mean error, standing still)",
    max(rgb[0], us[0], depth[0]) < 0.15,
    f"RGB only {rgb[0] * 100:.1f} cm, + ultrasonic {us[0] * 100:.1f} cm, + depth {depth[0] * 100:.1f} cm",
  )
  check(
    "adding a range sensor helps while the opponent moves",
    us[1] < rgb[1] and depth[1] < rgb[1],
    f"still: RGB {rgb[0] * 100:.1f} -> +US {us[0] * 100:.1f} / +depth {depth[0] * 100:.1f} cm; "
    f"moving RMS: RGB {rgb[1] * 100:.1f} -> +US {us[1] * 100:.1f} / +depth {depth[1] * 100:.1f} cm",
  )
  print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")


if __name__ == "__main__":
  main()
