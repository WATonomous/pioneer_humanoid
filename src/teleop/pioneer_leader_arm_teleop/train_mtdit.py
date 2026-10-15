"""lerobot-train with the Multi-Task DiT policy (mtdit/, ported from LeRobot 0.6). Same arguments; --policy.type=multi_task_dit."""
import mtdit

mtdit.install()
from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
