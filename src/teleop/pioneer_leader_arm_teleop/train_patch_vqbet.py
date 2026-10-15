"""lerobot-train with Patch Policy on VQ-BeT (see patch_vqbet.py). Same arguments, with --policy.type=vqbet."""
import os

os.environ["PATCH_VQBET"] = "1"
import patch_vqbet  # noqa: E402,F401
from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
