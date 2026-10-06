"""Retargeted CSV -> BeyondMimic NPZ for the Wato robot (mjlab port of
whole_body_tracking/scripts/csv_to_npz_wato.py).

Plays the CSV kinematically on the Wato model, resamples it to 50 fps (the
rate the policy runs at: sim dt 0.005 x decimation 4) and saves joint and body
states to a local NPZ. --video also renders the reference motion to MP4.

CSV row: root pos (3), root quat xyzw (4), 28 joint angles in
CSV_JOINT_NAMES order.

  uv run scripts/csv_to_npz.py --input-file data/motions/boxing.csv \
      --input-fps 120 --output-file data/motions/boxing.npz --video True

Passing a larger --input-fps than the recording's plays the motion faster
(e.g. 240 for a 120 fps recording = 2x speed).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("MUJOCO_GL") == "osmesa":
  # OSMesa and triton each bundle LLVM; importing triton after OSMesa is
  # loaded segfaults, so load it (via torch._dynamo) first.
  import torch._dynamo  # noqa: F401

import mjlab  # noqa: E402
from mjlab.scene import Scene  # noqa: E402
from mjlab.scripts.csv_to_npz import MotionLoader  # noqa: E402
from mjlab.sim.sim import Simulation, SimulationCfg  # noqa: E402
from mjlab.viewer.offscreen_renderer import OffscreenRenderer  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402

from wato_tracking.env_cfg import wato_flat_tracking_env_cfg  # noqa: E402
from wato_tracking.robot import CSV_JOINT_NAMES  # noqa: E402


def main(
  input_file: str,
  output_file: str,
  input_fps: float = 120.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  video: bool = False,
  video_file: str | None = None,
  line_range: tuple[int, int] | None = None,
):
  """Args:
    input_file: retargeted CSV.
    output_file: NPZ to write.
    input_fps: frame rate of the CSV (Xsens BVH: 120).
    output_fps: rate of the NPZ; keep 50 (the policy rate).
    device: torch / MuJoCo Warp device; falls back to CPU without CUDA.
    video: also render the reference motion to MP4.
    video_file: MP4 path (default: output_file with .mp4).
    line_range: only use these CSV lines (1-based, inclusive).
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available, using CPU.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(wato_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if video:
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=ViewerConfig(
        height=480,
        width=640,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=2.5,
        elevation=-10.0,
        azimuth=160.0,
      ),
      scene=scene,
    )
    renderer.initialize()

  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot = scene["robot"]
  joint_ids = robot.find_joints(CSV_JOINT_NAMES, preserve_order=True)[0]
  if motion.motion_dof_poss.shape[1] != len(joint_ids):
    raise ValueError(
      f"CSV has {motion.motion_dof_poss.shape[1]} joint columns, expected {len(joint_ids)}"
    )

  log: dict[str, list] = {
    k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
  }
  frames = []
  scene.reset()
  for _ in tqdm(range(motion.output_frames), desc="frames", ncols=80):
    (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), _ = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = dof_pos
    joint_vel[:, joint_ids] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

  out = Path(output_file)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, fps=[output_fps], **{k: np.stack(v) for k, v in log.items()})
  print(f"Saved {motion.output_frames} frames ({motion.output_frames / output_fps:.1f} s) to {out}")

  if renderer is not None:
    import mediapy as media

    mp4 = Path(video_file) if video_file else out.with_suffix(".mp4")
    media.write_video(str(mp4), frames, fps=output_fps)
    print(f"Saved reference video to {mp4}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
