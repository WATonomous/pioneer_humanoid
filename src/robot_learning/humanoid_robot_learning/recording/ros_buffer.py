"""ROS 2 subscribers that hold the latest ArmPose / images for the record loop."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from humanoid_robot_learning.recording.arm_pose_io import arm_pose_to_vector
from humanoid_robot_learning.recording.observation import fit_image, image_msg_to_hwc
from humanoid_robot_learning.recording.schema import enabled_images
from humanoid_robot_learning.recording.snapshot import ObservationSnapshot

logger = logging.getLogger(__name__)


class RosRecordBuffer:
    def __init__(self, node: Any, cfg: dict[str, Any]) -> None:
        self._lock = threading.Lock()
        self._cfg = cfg
        self._ros_unit = (cfg.get("ros") or {}).get("arm_pose_unit", "deg")
        self._dataset_unit = cfg.get("state_unit", "rad")

        self._latest_action: np.ndarray | None = None
        self._latest_state: np.ndarray | None = None
        self._latest_images: dict[str, np.ndarray] = {}

        self._import_ros_types()
        self._create_subscriptions(node, cfg)

    def _import_ros_types(self) -> None:
        try:
            from common_msgs.msg import ArmPose
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise ImportError(
                "ROS messages not found. Source the workspace, e.g.\n"
                "  source install/setup.bash\n"
                "after building common_msgs."
            ) from exc
        self._ArmPose = ArmPose
        self._Image = Image

    def _create_subscriptions(self, node: Any, cfg: dict[str, Any]) -> None:
        ros_cfg = cfg.get("ros") or {}
        action_topic = ros_cfg.get("action_topic", "/arm/left/joint_targets")
        state_topic = ros_cfg.get("state_topic")

        node.create_subscription(
            self._ArmPose, action_topic, self._on_action, 10
        )
        logger.info("Subscribed action: %s", action_topic)

        if state_topic:
            node.create_subscription(
                self._ArmPose, state_topic, self._on_state, 10
            )
            logger.info("Subscribed state: %s", state_topic)

        for key, spec in enabled_images(cfg).items():
            topic = spec["topic"]
            node.create_subscription(
                self._Image,
                topic,
                lambda msg, k=key: self._on_image(k, msg),
                10,
            )
            logger.info("Subscribed image %s: %s", key, topic)

    def _vector_from_pose(self, msg) -> np.ndarray:
        return arm_pose_to_vector(
            msg,
            input_unit=self._ros_unit,
            output_unit=self._dataset_unit,
        )

    def _on_action(self, msg) -> None:
        vec = self._vector_from_pose(msg)
        with self._lock:
            self._latest_action = vec
            if (self._cfg.get("ros") or {}).get("state_topic") is None:
                self._latest_state = vec.copy()

    def _on_state(self, msg) -> None:
        with self._lock:
            self._latest_state = self._vector_from_pose(msg)

    def _on_image(self, key: str, msg) -> None:
        try:
            spec = self._cfg["images"][key]
            image = fit_image(image_msg_to_hwc(msg), int(spec["height"]), int(spec["width"]))
        except Exception:
            logger.exception("Failed to decode image for %s", key)
            return
        with self._lock:
            self._latest_images[key] = image

    def snapshot(self) -> ObservationSnapshot:
        with self._lock:
            return ObservationSnapshot(
                state=None
                if self._latest_state is None
                else self._latest_state.copy(),
                action=None
                if self._latest_action is None
                else self._latest_action.copy(),
                images={k: v.copy() for k, v in self._latest_images.items()},
            )
