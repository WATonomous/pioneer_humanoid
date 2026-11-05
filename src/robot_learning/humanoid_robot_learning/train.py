"""``lerobot-train`` with our policies registered. Same arguments, e.g. ``--policy.type=humanoid_dit``."""
from humanoid_robot_learning.policies import register


def main():
    register()
    from lerobot.scripts.lerobot_train import main as train
    train()


if __name__ == "__main__":
    main()
