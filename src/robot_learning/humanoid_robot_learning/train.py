"""``lerobot-train`` with this package's policies registered. Same arguments, e.g. ``--policy.type=multi_task_dit``."""
from humanoid_robot_learning.policies import register

register()

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
