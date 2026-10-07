"""Build the two-fighter arena, check its starting geometry and render it.

Prints the distances that matter (heads, guards, jab range, ropes) and any
contact between the fighters at the start, then saves stills from a few
camera angles. --seconds > 0 also steps the physics with zero actions (the
motors hold the stance, no balance policy) and saves a video.

  MUJOCO_GL=osmesa uv run scripts/view_arena.py --out-dir videos/arena
  # interactive: uv run scripts/play.py Mjlab-Boxing-Arena-Wato --agent zero --viewer viser
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("MUJOCO_GL") == "osmesa":
  # OSMesa and triton each bundle LLVM; importing triton after OSMesa is
  # loaded segfaults, so load it (via torch._dynamo) first.
  import torch._dynamo  # noqa: F401

import mujoco  # noqa: E402

import mjlab  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.viewer.offscreen_renderer import OffscreenRenderer  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402

from wato_boxing.arena_env_cfg import boxing_arena_env_cfg  # noqa: E402

JAB_REACH = 0.79  # lead glove, fully extended at head height, ahead of the stance centre [m]

VIEWS = {
  "side": dict(azimuth=90.0, elevation=-10.0, distance=3.2, lookat=(0.0, 0.0, 0.8)),
  "behind_red": dict(azimuth=180.0, elevation=-15.0, distance=3.0, lookat=(0.0, 0.0, 0.9)),
  "high": dict(azimuth=135.0, elevation=-40.0, distance=6.5, lookat=(0.0, 0.0, 0.3)),
}


def report(env: ManagerBasedRlEnv) -> None:
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  mujoco.mj_forward(m, d)

  def gpos(name):
    return d.geom_xpos[m.geom(name).id]

  def gsize(name):
    return m.geom_size[m.geom(name).id][0]

  head_gap = np.linalg.norm(gpos("red/head") - gpos("blue/head"))
  print(f"head centre to head centre:           {head_gap:.2f} m")
  for side in "lr":
    for other in "lr":
      gap = np.linalg.norm(gpos(f"red/glove_{side}") - gpos(f"blue/glove_{other}")) - 2 * gsize("red/glove_l")
      if side == other == "l" or gap < 0.15:
        print(f"red glove_{side} to blue glove_{other} (surface gap): {gap:.2f} m")
  # how far the red jab falls short of the blue head
  red_feet = [d.xpos[m.body(f"red/{f}").id][:2] for f in ("left_foot_1", "right_foot_1")]
  centre = np.mean(red_feet, 0)
  head_front = gpos("blue/head")[0] - gsize("blue/head")
  short = head_front - (centre[0] + JAB_REACH)
  print(f"red jab falls short of blue's head by: {short:.2f} m  (each step is 0.10 m)")

  fighter_contacts = []
  for c in d.contact[: d.ncon]:
    n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
    if n1.split("/")[0] != n2.split("/")[0] and "terrain" not in (n1 + n2) and "floor" not in (n1 + n2):
      fighter_contacts.append((n1, n2, c.dist))
  print(f"contacts between fighters/ropes at start: {fighter_contacts or 'none'}")
  floor = sorted(
    {m.geom(c.geom1 if "terrain" in m.geom(c.geom2).name else c.geom2).name for c in d.contact[: d.ncon]}
    - {""}
  )
  print(f"geoms touching the floor at start: {floor}")
  names = [m.geom(i).name for i in range(m.ngeom)]
  print(f"ring: {sum(n.startswith('ring/rope') for n in names)} ropes, {sum(n.startswith('ring/post') for n in names)} posts")


def main(out_dir: str = "videos/arena", distance: float = 1.3, seconds: float = 0.0, width: int = 960, height: int = 540):
  """Args:
    out_dir: where the PNGs / MP4 go.
    distance: fighter separation (foot-centre to foot-centre) [m].
    seconds: physics to simulate with zero actions for the video (0 = stills only).
    width / height: image size.
  """
  import mediapy as media

  cfg = boxing_arena_env_cfg(distance=distance)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
  env.reset()
  report(env)

  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  for name, v in VIEWS.items():
    r = OffscreenRenderer(
      model=env.sim.mj_model,
      cfg=ViewerConfig(origin_type=ViewerConfig.OriginType.WORLD, width=width, height=height, **v),
      scene=env.scene,
    )
    r.initialize()
    r.update(env.sim.data)
    media.write_image(str(out / f"{name}.png"), r.render())
    r.close()
  print(f"saved stills to {out}/")

  if seconds > 0:
    r = OffscreenRenderer(
      model=env.sim.mj_model,
      cfg=ViewerConfig(origin_type=ViewerConfig.OriginType.WORLD, width=width, height=height, **VIEWS["side"]),
      scene=env.scene,
    )
    r.initialize()
    frames = []
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(int(seconds / env.step_dt)):
      env.step(zero)
      r.update(env.sim.data)
      frames.append(r.render())
    media.write_video(str(out / "zero_action.mp4"), frames, fps=round(1 / env.step_dt))
    print(f"saved {out}/zero_action.mp4")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
