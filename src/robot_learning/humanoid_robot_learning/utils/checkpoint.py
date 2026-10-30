"""Load a LeRobot checkpoint for a rollout loop (sim evaluate, real-arm deploy). No sim imports."""

from __future__ import annotations

import json
from pathlib import Path


def load_policy(path: str, device: str):
    """-> (policy, preprocessor, postprocessor) of a pretrained_model directory, in eval mode."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from humanoid_robot_learning.policies import register

    register()
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def training_task(path: str) -> str | None:
    """The task text of the policy's training dataset, if it has exactly one."""
    import pandas as pd

    config = Path(path) / "train_config.json"
    root = json.loads(config.read_text())["dataset"].get("root") if config.exists() else None
    tasks = Path(root or "") / "meta" / "tasks.parquet"
    if not root or not tasks.exists():
        return None
    names = list(pd.read_parquet(tasks).index)
    return names[0] if len(names) == 1 else None
