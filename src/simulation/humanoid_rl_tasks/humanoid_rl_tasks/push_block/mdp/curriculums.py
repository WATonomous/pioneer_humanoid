"""Curriculum terms for the push-block task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase

from .rewards import block_on_floor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg

__all__ = ["spawn_offset_curriculum", "dr_ramp", "spawn_ramp", "success_rate_logger"]


class spawn_offset_curriculum(ManagerTermBase):
    """Performance-gated spawn-offset curriculum for the block reset.

    Widens the block's spawn *offset* range one stage at a time by mutating, in
    place, the ``pose_range`` of the ``reset_object_position`` event term (the
    :class:`EventManager` reads ``term_cfg.params`` fresh on every reset, so the
    new range takes effect on the next spawn). Only the offset *magnitude* grows
    -- the spawn stays a fixed default anchor + random xy offset + full yaw, the
    same mechanism as the moderate setup.

    Advancement is gated on task performance AND a minimum dwell: a stage
    advances only once (a) the policy has trained at that stage for at least
    ``min_stage_steps`` environment steps, and (b) the rolling success rate over
    the last ``window`` completed episodes then reaches ``threshold``. The dwell
    is essential -- at thousands of envs the ``window`` refills many times within
    a single PPO iteration, so without it a competent resumed policy would
    cascade through several stages in one or two iterations (skipping the actual
    graduated widening) before ever training on the wider spawns. After each
    advance the window is cleared so the next gate measures competence *at the
    new, wider spawn range*.

    Per-episode success reuses the existing success reward term
    (``success_reward_term``, e.g. ``block_on_floor``) with zero duplicated
    geometry: this term runs inside ``_reset_idx`` *before* the reward manager
    zeroes its episode sums, so a positive accumulated sum means the block
    reached the interior floor at some point during the episode.

    It also toggles an optional repositioning reward term
    (``reposition_reward_term``) on -- from weight 0 to ``reposition_weight`` --
    once the curriculum reaches ``reposition_start_stage``, tying that shaping
    term's activation directly to the curriculum stage.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._stages: list[dict] = list(p["stages"])
        self._event_term: str = p["event_term_name"]
        self._success_term: str = p["success_reward_term"]
        self._window: int = int(p["window"])
        self._threshold: float = float(p["threshold"])
        self._min_stage_steps: int = int(p.get("min_stage_steps", 0))
        self._min_ep_len: int = int(p.get("min_episode_len", 2))
        self._reposition_term: str | None = p.get("reposition_reward_term")
        self._reposition_weight: float = float(p.get("reposition_weight", 0.0))
        self._reposition_start_stage: int = int(p.get("reposition_start_stage", len(self._stages)))

        self._stage = 0
        self._stage_start_step = int(env.common_step_counter)  # for the dwell gate
        # ring buffer of recent completed-episode outcomes (1.0 success / 0.0 fail)
        self._buf = torch.zeros(self._window, device=env.device)
        self._count = 0  # valid entries written so far (saturates at window)
        self._ptr = 0

        # apply stage 0 range and set the repositioning term to its stage-0 state
        self._apply_stage()

    def _apply_stage(self):
        env = self._env
        # widen the spawn offset range in place
        ev_cfg = env.event_manager.get_term_cfg(self._event_term)
        ev_cfg.params["pose_range"] = dict(self._stages[self._stage])
        env.event_manager.set_term_cfg(self._event_term, ev_cfg)
        # gate the repositioning reward on the curriculum stage
        if self._reposition_term is not None:
            rw_cfg = env.reward_manager.get_term_cfg(self._reposition_term)
            active = self._stage >= self._reposition_start_stage
            rw_cfg.weight = self._reposition_weight if active else 0.0
            env.reward_manager.set_term_cfg(self._reposition_term, rw_cfg)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: list[dict],
        event_term_name: str,
        success_reward_term: str,
        window: int,
        threshold: float,
        min_stage_steps: int = 0,
        min_episode_len: int = 2,
        reposition_reward_term: str | None = None,
        reposition_weight: float = 0.0,
        reposition_start_stage: int | None = None,
    ) -> dict[str, float]:
        # record outcomes only for genuinely completed episodes (skip freshly
        # initialized envs whose episode has not run yet)
        ep_len = env.episode_length_buf[env_ids]
        valid = ep_len >= self._min_ep_len
        if bool(valid.any()):
            ep_sum = env.reward_manager._episode_sums[self._success_term][env_ids]
            outcomes = (ep_sum > 0.0).float()[valid]
            n = int(outcomes.numel())
            idx = (self._ptr + torch.arange(n, device=env.device)) % self._window
            self._buf[idx] = outcomes
            self._ptr = int((self._ptr + n) % self._window)
            self._count = min(self._count + n, self._window)

        if self._count >= self._window:
            rate = float(self._buf.mean())
        elif self._count > 0:
            rate = float(self._buf[: self._count].mean())
        else:
            rate = 0.0

        # advance only after a minimum dwell at this stage AND once the window is
        # full and performance clears the threshold (dwell prevents cascading
        # through stages within a single iteration; see class docstring).
        steps_in_stage = int(env.common_step_counter) - self._stage_start_step
        if (
            self._stage < len(self._stages) - 1
            and steps_in_stage >= self._min_stage_steps
            and self._count >= self._window
            and rate >= self._threshold
        ):
            self._stage += 1
            self._stage_start_step = int(env.common_step_counter)
            self._apply_stage()
            self._buf.zero_()
            self._count = 0
            self._ptr = 0

        return {
            "stage": float(self._stage),
            "success_rate": rate,
            "x_off_hi": float(self._stages[self._stage]["x"][1]),
            "reposition_active": float(self._stage >= self._reposition_start_stage),
        }


class dr_ramp(ManagerTermBase):
    """Linearly ramps domain randomization from none to full strength between two PPO iterations.

    Each ``event_ranges`` entry maps an event term to {param name: (full range, center)}; the param is set to
    ``center + alpha * (full range - center)``, with alpha going 0 -> 1 from ``start_iter`` to ``end_iter``.
    ``noise_terms`` maps policy-group observation terms to a full noise half-width, ramped the same way.
    Friction is not ramped: its materials are sampled once at startup and cannot be changed afterwards.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._start = p["start_iter"] * p["steps_per_iter"]
        self._end = p["end_iter"] * p["steps_per_iter"]
        self._offset = p["iter_offset"] * p["steps_per_iter"]
        self._event_ranges: dict = p["event_ranges"]
        self._noise_terms: dict = p["noise_terms"]
        self._alpha = None
        self._apply(self._alpha_at(0))

    def _alpha_at(self, step: int) -> float:
        return min(max((step + self._offset - self._start) / (self._end - self._start), 0.0), 1.0)

    def _apply(self, alpha: float):
        env = self._env
        for event_name, params in self._event_ranges.items():
            event_cfg = env.event_manager.get_term_cfg(event_name)
            for key, (full, center) in params.items():
                event_cfg.params[key] = tuple(center + alpha * (v - center) for v in full)
        # the observation manager works on its own copies of the term configs, so edit those (private API, Isaac Lab 2.3.2)
        manager = env.observation_manager
        for name, term_cfg in zip(manager._group_obs_term_names["policy"], manager._group_obs_term_cfgs["policy"]):
            if name in self._noise_terms and term_cfg.noise is not None:
                half_width = self._noise_terms[name]
                term_cfg.noise.n_min, term_cfg.noise.n_max = -alpha * half_width, alpha * half_width
        self._alpha = alpha

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        start_iter: int,
        end_iter: int,
        steps_per_iter: int,
        iter_offset: int,
        event_ranges: dict,
        noise_terms: dict,
    ):
        alpha = self._alpha_at(env.common_step_counter)
        if alpha != self._alpha:
            self._apply(alpha)
        return alpha


class spawn_ramp(ManagerTermBase):
    """Linearly widens the block spawn range from ``start_range`` to ``end_range`` between two PPO iterations."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._start = p["start_iter"] * p["steps_per_iter"]
        self._end = p["end_iter"] * p["steps_per_iter"]
        self._offset = p["iter_offset"] * p["steps_per_iter"]
        self._from: dict = p["start_range"]
        self._to: dict = p["end_range"]
        self._event_term: str = p["event_term_name"]
        self._alpha = None
        self._apply(self._alpha_at(0))

    def _alpha_at(self, step: int) -> float:
        return min(max((step + self._offset - self._start) / (self._end - self._start), 0.0), 1.0)

    def _apply(self, alpha: float):
        pose_range = {k: tuple(a + alpha * (b - a) for a, b in zip(self._from[k], self._to[k])) for k in self._from}
        self._env.event_manager.get_term_cfg(self._event_term).params["pose_range"] = pose_range
        self._alpha = alpha

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        start_iter: int,
        end_iter: int,
        steps_per_iter: int,
        iter_offset: int,
        start_range: dict,
        end_range: dict,
        event_term_name: str,
    ):
        alpha = self._alpha_at(env.common_step_counter)
        if alpha != self._alpha:
            self._apply(alpha)
        return alpha


class success_rate_logger(ManagerTermBase):
    """Logs the fraction of recent episodes that ended with the block on the floor at rest (``block_on_floor``).

    Evaluated on the final state of each episode, at reset time. Changes nothing in training.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._window: int = int(cfg.params["window"])
        self._buf = torch.zeros(self._window, device=env.device)
        self._count = 0
        self._ptr = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        window: int,
        x_min: float,
        x_max: float,
        y_half: float,
        floor_z: float,
        z_tol: float,
        rest_speed: float,
        min_episode_len: int = 2,
    ) -> dict[str, float]:
        valid = env.episode_length_buf[env_ids] >= min_episode_len
        if bool(valid.any()):
            outcomes = block_on_floor(
                env, x_min=x_min, x_max=x_max, y_half=y_half, floor_z=floor_z, z_tol=z_tol, rest_speed=rest_speed
            )[env_ids][valid]
            n = int(outcomes.numel())
            idx = (self._ptr + torch.arange(n, device=env.device)) % self._window
            self._buf[idx] = outcomes
            self._ptr = int((self._ptr + n) % self._window)
            self._count = min(self._count + n, self._window)
        rate = float(self._buf[: self._count].mean()) if self._count > 0 else 0.0
        return {"success_rate": rate}
