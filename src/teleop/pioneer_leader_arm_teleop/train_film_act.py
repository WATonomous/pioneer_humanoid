"""lerobot-train with FiLM language conditioning in ACT (see film_act.py). Same arguments."""
import os

os.environ["ACT_FILM"] = "1"
import film_act  # noqa: E402,F401
from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
