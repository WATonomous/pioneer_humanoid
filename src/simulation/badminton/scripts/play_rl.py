"""mjlab play/replay entry point (viser web viewer on headless machines).

  uv run scripts/play_rl.py Mjlab-Badminton-Shared-Teacher --viewer viser
  # then open http://localhost:8080 (or SSH-tunnel the port)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import humanoid_badminton  # noqa: F401  (registers the tasks)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
    main()
