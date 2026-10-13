"""lerobot-train with the DINOv2 patch-token backbone registered (see patch_backbone.py). Same arguments."""
import patch_backbone  # noqa: F401
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
