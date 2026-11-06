from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML


def _pkg_root() -> Path:
    # .../elevation_mapping_cupy/elevation_mapping_cupy/elevation_mapping_cupy/tests/test_repo_config_sanity.py
    # parents[2] = ROS package root (contains config/).
    return Path(__file__).resolve().parents[2]


def _supported_config_files() -> list[Path]:
    cfg_root = _pkg_root() / "config"
    assert cfg_root.is_dir(), f"Missing config dir: {cfg_root}"
    yamls = sorted(cfg_root.rglob("*.yaml"))
    supported = []
    for p in yamls:
        # Keep experimental configs in the tree, but don't treat them as supported.
        if "experimental" in p.parts:
            continue
        supported.append(p)
    return supported


@pytest.mark.parametrize("path", _supported_config_files())
def test_supported_configs_are_ros2_clean(path: Path):
    text = path.read_text(encoding="utf-8")

    # No ROS1-style or non-standard substitutions in supported configs.
    banned = [
        "$(rospack find",
        "$(find_package_share",
        "$(find-pkg-share",
    ]
    for token in banned:
        assert token not in text, f"{path} contains banned substitution token: {token}"

    # Kill the old typo forever.
    assert "drift_compensation_variance_inler" not in text, (
        f"{path} contains deprecated typo key drift_compensation_variance_inler"
    )

    # Parseable YAML.
    yaml = YAML(typ="safe")
    data = yaml.load(text) or {}
    assert isinstance(data, dict), f"{path} must parse to a dict, got {type(data)}"

    # Supported configs must keep a valid ROS2 subscriber/publisher schema.
    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k, v
                yield from _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _walk(v)

    for k, v in _walk(data):
        if k == "data_type":
            assert v in {"pointcloud", "image"}, f"{path} sets unsupported data_type={v!r}"

    # Validate subscriber schema for any file that defines subscribers.
    for node_name, node_cfg in data.items():
        if not isinstance(node_cfg, dict):
            continue
        params = node_cfg.get("ros__parameters", {})
        if not isinstance(params, dict):
            continue
        subs = params.get("subscribers")
        if subs is None:
            continue
        assert isinstance(subs, dict), f"{path} subscribers must be a dict"
        for sub_name, sub_cfg in subs.items():
            assert isinstance(sub_cfg, dict), f"{path} subscriber '{sub_name}' must be a dict"
            data_type = sub_cfg.get("data_type")
            assert data_type in {"pointcloud", "image"}, (
                f"{path} subscriber '{sub_name}' has unsupported data_type={data_type!r}"
            )
            assert sub_cfg.get("topic_name"), f"{path} subscriber '{sub_name}' missing topic_name"
            if data_type == "image":
                assert sub_cfg.get("camera_info_topic_name") or sub_cfg.get("topic_name_camera_info"), (
                    f"{path} image subscriber '{sub_name}' missing camera_info_topic_name"
                )
                assert sub_cfg.get("channels") or sub_cfg.get("channel_info_topic_name"), (
                    f"{path} image subscriber '{sub_name}' needs channels or channel_info_topic_name"
                )
