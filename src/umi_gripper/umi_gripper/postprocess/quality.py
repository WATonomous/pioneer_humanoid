"""Which synced episodes enter the dataset.

Invalid steps at the very start and end (pressing the button before the controller was in view, or turning away
to press it again) are trimmed. Anything left that is wrong rejects the whole take: a policy would learn from an
interior tracking gap or jump as if the hand had really done it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from umi_gripper import geometry
from umi_gripper.postprocess.sync import Episode


@dataclass
class Verdict:
    ok: bool
    episode: Episode
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def trim_invalid_ends(episode: Episode) -> Episode:
    good = np.flatnonzero(episode.valid)
    if len(good) == 0:
        return episode.trimmed(0, 0)
    return episode.trimmed(int(good[0]), int(good[-1]) + 1)


def speeds(episode: Episode) -> tuple[np.ndarray, np.ndarray]:
    """TCP linear (m/s) and angular (rad/s) speed between consecutive steps."""
    if len(episode) < 2:
        return np.zeros(0), np.zeros(0)
    dt = np.diff(episode.times)
    linear = np.linalg.norm(np.diff(episode.tcp[:, :3, 3], axis=0), axis=1) / dt
    angular = np.array([np.linalg.norm(geometry.rotation_vector(a[:3, :3].T @ b[:3, :3]))
                        for a, b in zip(episode.tcp[:-1], episode.tcp[1:])]) / dt
    return linear, angular


def check_episode(episode: Episode, post: dict) -> Verdict:
    reasons = list(episode.issues)
    trimmed = trim_invalid_ends(episode)
    stats: dict = {"steps_raw": len(episode), "steps": len(trimmed)}
    if len(trimmed) == 0:
        reasons.append("no valid step (controller never tracked, or no camera frames)")
        return Verdict(False, trimmed, reasons, stats)
    fps = float(post["fps"])
    duration = (len(trimmed) - 1) / fps
    stats["duration_s"] = round(duration, 3)
    invalid = int((~trimmed.valid).sum())
    stats["interior_invalid_steps"] = invalid
    if invalid:
        reasons.append(f"{invalid} steps inside the take lost tracking or camera (> gap limits)")
    if duration < float(post["min_duration_s"]):
        reasons.append(f"too short: {duration:.2f} s < {post['min_duration_s']} s")
    linear, angular = speeds(trimmed)
    if len(linear):
        stats["max_speed_m_s"] = round(float(linear.max()), 3)
        stats["max_angular_speed_rad_s"] = round(float(angular.max()), 3)
        if linear.max() > float(post["max_speed_m_s"]):
            reasons.append(f"TCP jump: {linear.max():.2f} m/s > {post['max_speed_m_s']} m/s (tracking glitch?)")
        if angular.max() > float(post["max_angular_speed_rad_s"]):
            reasons.append(f"rotation jump: {angular.max():.1f} rad/s > {post['max_angular_speed_rad_s']} rad/s")
    return Verdict(not reasons, trimmed, reasons, stats)
