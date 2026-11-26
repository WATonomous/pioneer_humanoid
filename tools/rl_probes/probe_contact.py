"""Is the terrain-contact sensor alive on this course?

The swing-clearance / knee-tuck / air-time rewards are gated on the two
``*_foot_terrain_contact`` sensors, which filter ``{terrain.prim_path}/terrain/mesh``.
On the spiral those terms log exactly 0.0000 while the contact_forces-based
terms fire, so the suspicion is that the filter matches nothing and the feet
never register as touching terrain. Print the prim tree under the terrain root,
the resolved filter expressions, and the force the sensors actually see.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src/simulation/humanoid_rl/humanoid_rl/scripts"))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_rl_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Usd  # noqa: E402

TASK = os.environ.get("TASK_OVERRIDE",
                      "Isaac-Locomotion-StairsSpiralFoothold-PioneerHumanoid-Play-v0")

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.scene.env_spacing = 0.0
cfg.observations.policy.enable_corruption = False
env = gym.make(TASK, cfg=cfg)
u = env.unwrapped

root = getattr(u.scene.terrain, "prim_path", None) or u.scene.terrain.cfg.prim_path
print("[C] terrain prim_path", root)
stage = u.sim.stage
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith("/World/ground"):
        print("[C] prim", path, prim.GetTypeName())

for name in ("left_foot_terrain_contact", "right_foot_terrain_contact"):
    sensor = u.scene.sensors[name]
    cfg_ = sensor.cfg
    print(f"[C] {name} prim {getattr(sensor, 'prim_path', None)} bodies {sensor.body_names}")
    print(f"[C]   filter_attr {getattr(cfg_, 'filter_prim_paths_expr', None)}")
    print(f"[C]   filter {getattr(cfg_, 'filter_prim_paths_expr', None)}")

print("[C] height scanner prims", getattr(u.scene.sensors.get("height_scanner"), "cfg", None) and u.scene.sensors["height_scanner"].cfg.mesh_prim_paths)

obs, _ = env.reset()
peak = {"left_foot_terrain_contact": 0.0, "right_foot_terrain_contact": 0.0}
for step in range(400):
    with torch.inference_mode():
        act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)
        obs, rew, term, trunc, info = env.step(act)
    for name in peak:
        sensor = u.scene.sensors[name]
        f = sensor.data.net_forces_w
        if f is not None and f.numel():
            peak[name] = max(peak[name], float(f.abs().max()))

print("[C] terrain-contact peak |F| over 400 idle steps", peak)
print("[C] contact_forces peak |F|",
      float(u.scene.sensors["contact_forces"].data.net_forces_w.abs().max()))
print("[C] reward term values (raw, last step):")
for name in ("foot_swing_clearance", "knee_swing_flexion", "feet_air_time",
             "foot_backswing_cost", "feet_impact"):
    term_ = u.reward_manager.get_term_cfg(name)
    try:
        val = term_.func(u, **term_.params)
        print(f"[C]   {name:24s} {float(val.mean()):+.5f}  weight {term_.weight}")
    except Exception as exc:  # noqa: BLE001
        print(f"[C]   {name:24s} FAILED {type(exc).__name__}: {exc}")
print("[C] OK")
sys.stdout.flush()
env.close()
simulation_app.close()
