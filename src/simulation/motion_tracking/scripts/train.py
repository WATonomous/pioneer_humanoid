"""mjlab training entry point: registers the Wato tracking tasks, then hands
off to mjlab's train CLI (tyro; task id is the first positional argument).

  uv run scripts/train.py Mjlab-Tracking-Flat-Wato \
      --env.commands.motion.motion-file data/motions/boxing.npz \
      --env.scene.num-envs 4096
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wato_tracking  # noqa: E402,F401  (registers the tasks)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
