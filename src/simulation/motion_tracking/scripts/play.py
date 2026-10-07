"""mjlab play entry point: runs a trained policy (or replays the reference
with --agent zero) in the viewer, or records a video headless.

  uv run scripts/play.py Mjlab-Tracking-Flat-Wato \
      --motion-file data/motions/boxing.npz \
      --checkpoint-file logs/rsl_rl/wato_tracking/<run>/model_<n>.pt \
      --viewer viser                       # http://localhost:8080

  # headless: --video True --video-length 1770 (frames at 50 fps)

  # the two-fighter arena (no policy yet; the motors hold the stance)
  uv run scripts/play.py Mjlab-Boxing-Arena-Wato --agent zero --viewer viser
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("MUJOCO_GL") == "osmesa":
  # OSMesa and triton each bundle LLVM; importing triton after OSMesa is
  # loaded segfaults, so load it (via torch._dynamo) first.
  import torch._dynamo  # noqa: F401

import wato_boxing  # noqa: E402,F401  (registers the arena)
import wato_tracking  # noqa: E402,F401  (registers the tasks)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  main()
