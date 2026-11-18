"""Restore the real predecessor observation history when a moving striking episode begins."""

import torch
from mjlab.managers.observation_manager import ObservationManager

TERM_SLICES = {
    "angular_velocity": (0, 9, 3),
    "gravity": (9, 18, 3),
    "joint_positions": (18, 66, 16),
    "joint_velocities": (66, 114, 16),
    "actions": (114, 162, 16),
    "ball_command": (162, 171, 3),
    "heading": (171, 177, 2),
    "ball_observation": (177, 189, 4),
    "simulation_velocities": (189, 207, 6),
}


# Use one rigid yaw transform for all historical world-frame vectors, preserving local joints and actions.
def rotate_history(history, angle):
    history = history.clone()
    c, s = angle.cos()[:, None], angle.sin()[:, None]

    def rotate(vectors):
        x, y = vectors[..., 0].clone(), vectors[..., 1].clone()
        vectors[..., 0], vectors[..., 1] = c * x - s * y, s * x + c * y

    rotate(history[:, 162:171].reshape(-1, 3, 3))
    headings = history[:, 171:177].reshape(-1, 3, 2)
    old_s, old_c = headings[..., 0].clone(), headings[..., 1].clone()
    headings[..., 0], headings[..., 1] = c * old_s + s * old_c, c * old_c - s * old_s
    velocities = history[:, 189:207].reshape(-1, 3, 6)
    rotate(velocities[..., :3])
    rotate(velocities[..., 3:])
    return history


# mjlab 1.6 has no public history setter; preserve its global pointer and all untouched worlds.
def restore_buffer(buffer, ids, chronological):
    if not buffer.is_initialized or buffer.max_length != chronological.shape[1]:
        raise ValueError(
            "Predecessor history must match an initialized circular buffer"
        )
    slots = (
        torch.arange(buffer.max_length, device=chronological.device)
        + buffer._pointer
        + 1
    ) % buffer.max_length
    buffer._buffer[slots[:, None], ids[None, :]] = chronological.transpose(0, 1)
    buffer._num_pushes[ids] = buffer.max_length


class ChainedObservationManager(ObservationManager):
    # Restore historical frames only for newly reset worlds; the final frame uses their new strike command.
    def compute(self, update_history=False, env_ids=None):
        result = super().compute(update_history=update_history, env_ids=env_ids)
        pending = getattr(self._env, "_strike_pending_history", None)
        if pending is None:
            return result
        ids, history = pending
        for group in ("actor", "critic"):
            buffers = self._group_obs_term_history_buffer[group]
            for name, (start, stop, width) in TERM_SLICES.items():
                if name not in buffers:
                    continue
                buffer = buffers[name]
                values = history[:, start:stop].reshape(-1, 3, width).clone()
                values[:, -1] = buffer.buffer[ids, -1]
                restore_buffer(buffer, ids, values)
            result[group] = self.compute_group(group, update_history=False)
        self._env._strike_pending_history = None
        self._obs_buffer = result
        return result
