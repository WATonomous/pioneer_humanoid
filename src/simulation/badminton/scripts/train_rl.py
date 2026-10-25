"""mjlab training entry point: registers the badminton tasks, then hands
off to mjlab's train CLI (tyro; task id is the first positional argument).

  uv run scripts/train_rl.py Mjlab-Badminton-Shared-Teacher \
      --env.scene.num-envs 4096
"""

import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import humanoid_badminton  # noqa: F401  (registers the tasks)
from mjlab.scripts.train import main  # noqa: E402


def _sigterm(signum, frame):
    # scancel delivers SIGTERM to every process in the job; turn it into
    # KeyboardInterrupt so wandb syncs and closes the run instead of
    # showing it crashed
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    try:
        main()
    except KeyboardInterrupt:
        # mark the run finished (not killed) and sync before exiting
        import wandb
        if wandb.run is not None:
            wandb.finish()
