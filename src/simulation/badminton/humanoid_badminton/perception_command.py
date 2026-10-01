"""Simulated perception as a mjlab command term.

Owns the batched shuttle EKF (perception_torch) and publishes two feature
blocks each control tick, both laid out as [p(3), v(3), traj(n_traj*3)]:

  student_features  from the EKF fed one noisy position measurement per tick
                    — twitchy early, converging as measurements accumulate,
                    matching the real perception model refining its fit
  teacher_features  the same layout computed from the true state (the
                    privileged trajectory prior)

It also owns the bimanual arm assignment (0 right, 1 left), written to
env._badminton["assign"] each tick: the arm on the side (x relative to the
stand centre) where the predicted flight crosses the strike plane y =
control.assign_strike_y. The prediction comes from the true state
(assign_source "true", the teacher task) or the EKF state ("ekf", the
student tasks, deployable as is). The EKF assignment follows the estimate
until control.assign_latch_s into the episode and is fixed after that.
BadmintonAction holds the other arm at its ready pose.

The command lifecycle fits exactly: _resample_command fires on episode reset
(after the shuttle reset event wrote the new true state) and re-initializes
the filter rows; _update_command runs once per control tick after
sim.forward(), so measurements always come from the current true state and
observations read fresh features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

import aero
import perception_torch as pt

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class PerceptionCommand(CommandTerm):
    cfg: "PerceptionCommandCfg"

    def __init__(self, cfg: "PerceptionCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        params = aero.load_params()
        pp = params["perception"]
        self._k = params["shuttle"]["k"]
        self._g = params["gravity"]
        self._sigma_meas = pp["sigma_meas"]
        self._sigma_p0 = pp["sigma_p0"]
        self._sigma_v0 = pp["sigma_v0"]
        self._sigma_acc = pp["sigma_acc"]
        self._n_traj = pp["n_traj"]
        self._traj_dt = pp["traj_dt"]
        self._sub_dt = pp["traj_sub_dt"]

        self._shuttle = env.scene[cfg.entity_name]
        idx = self._shuttle.data.indexing
        self._q_adr = idx.free_joint_q_adr
        self._v_adr = idx.free_joint_v_adr

        n, dev = self.num_envs, self.device
        self._x = torch.zeros(n, 6, device=dev)
        self._P = torch.zeros(n, 6, 6, device=dev)
        self._just_reset = torch.zeros(n, dtype=torch.bool, device=dev)
        dim = 6 + 3 * self._n_traj
        self.student_features = torch.zeros(n, dim, device=dev)
        self.teacher_features = torch.zeros(n, dim, device=dev)
        # student-only: how much to trust the estimate. [EKF position std
        # (3), velocity std (3), tick-to-tick jump of the prior's last
        # point (1)]. A real EKF exposes its covariance the same way; the
        # jump is what a human watching the prior would call "still
        # settling". Without it the memoryless student cannot tell an early
        # (unreliable) estimate from a converged one and commits equally.
        self.student_uncertainty = torch.zeros(n, 7, device=dev)
        # ground-truth task geometry (logged via metrics): per-episode min
        # face->p* distance and the distance at the tick nearest t*
        self._face_cfg = None
        self._min_dist = torch.full((n,), float("inf"), device=dev)
        self._dist_at_tstar = torch.zeros(n, device=dev)
        self._tstar_seen = torch.zeros(n, dtype=torch.bool, device=dev)

        c = params["control"]
        self._strike_y = c["assign_strike_y"]
        self._latch_s = c["assign_latch_s"]
        self._center_x = params["arm"].get("base_x", 0.0)
        # per-episode agreement of the EKF assignment with the true one at
        # the latch (logged; 1 for the true-source teacher by construction)
        self._assign_agree = torch.ones(n, device=dev)

    def _strike_x(self, p: torch.Tensor, v: torch.Tensor,
                  sub_dt: float = 0.05, max_steps: int = 40) -> torch.Tensor:
        """x where the drag-model flight from (p, v) crosses the strike
        plane; the current x if it does not cross within the horizon (the
        fresh EKF has v = 0)."""
        x = p[:, 0].clone()
        done = p[:, 1] <= self._strike_y
        for _ in range(max_steps):
            if bool(done.all()):
                break
            p_next, v_next = pt.rk4_step(p, v, self._k, sub_dt, self._g)
            cross = (~done) & (p_next[:, 1] <= self._strike_y)
            if bool(cross.any()):
                t = ((p[cross, 1] - self._strike_y)
                     / (p[cross, 1] - p_next[cross, 1]).clamp_min(1e-9))
                x[cross] = p[cross, 0] + t * (p_next[cross, 0] - p[cross, 0])
            done |= cross
            p, v = p_next, v_next
        return x

    def _update_assign(self, p_true, v_true, env_ids=None) -> None:
        from humanoid_badminton import mdp  # lazy: mdp imports this module
        store = mdp._state(self._env)
        true_side = (self._strike_x(p_true, v_true) <= self._center_x).long()
        if self.cfg.assign_source == "true":
            store["assign"][:] = true_side
            self._assign_agree = torch.ones_like(self._assign_agree)
            return
        ekf_side = (self._strike_x(self._x[:, :3], self._x[:, 3:])
                    <= self._center_x).long()
        t_now = self._env.episode_length_buf.float() * self._env.step_dt
        live = t_now <= self._latch_s
        if env_ids is not None:
            live[env_ids] = True
        store["assign"][:] = torch.where(live, ekf_side, store["assign"])
        self._assign_agree = torch.where(
            live, (ekf_side == true_side).float(), self._assign_agree)

    @property
    def command(self) -> torch.Tensor:
        return self.student_features

    def _true_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._shuttle.data.data
        return (data.qpos[:, self._q_adr[:3]].clone(),
                data.qvel[:, self._v_adr[:3]].clone())

    def _measure(self, p_true: torch.Tensor) -> torch.Tensor:
        return p_true + self._sigma_meas * torch.randn_like(p_true)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        p_true, _ = self._true_state()
        z0 = self._measure(p_true[env_ids])
        x, P = pt.ekf_init(z0, self._sigma_p0, self._sigma_v0)
        self._x[env_ids] = x
        self._P[env_ids] = P
        self._just_reset[env_ids] = True
        self._min_dist[env_ids] = float("inf")
        self._dist_at_tstar[env_ids] = 0.0
        self._tstar_seen[env_ids] = False

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        p_true, v_true = self._true_state()
        if env_ids is None:
            # per-tick path: EKF predict+update for rows that were not reset
            # this very tick (reset rows already hold their init measurement)
            run = ~self._just_reset
            if bool(run.any()):
                ids = run.nonzero(as_tuple=False).squeeze(-1)
                x, P = pt.ekf_predict(self._x[ids], self._P[ids],
                                      self._env.step_dt, self._k, self._g,
                                      self._sigma_acc)
                x, P = pt.ekf_update(x, P, self._measure(p_true[ids]),
                                     self._sigma_meas)
                self._x[ids] = x
                self._P[ids] = P
            self._just_reset[:] = False
        prev_last = self.student_features[:, -3:].clone()
        self.student_features[:] = pt.feature_layout(
            self._x[:, :3], self._x[:, 3:], self._k, self._n_traj,
            self._traj_dt, self._sub_dt, self._g)
        jump = (self.student_features[:, -3:] - prev_last).norm(dim=-1)
        if env_ids is not None:
            jump[env_ids] = 0.0     # fresh episode: no previous prior
        std = torch.diagonal(self._P, dim1=-2, dim2=-1).clamp_min(0.0).sqrt()
        self.student_uncertainty[:] = torch.cat([std, jump.unsqueeze(-1)],
                                                dim=-1)
        self.teacher_features[:] = pt.feature_layout(
            p_true, v_true, self._k, self._n_traj,
            self._traj_dt, self._sub_dt, self._g)
        self._update_assign(p_true, v_true, env_ids)

    def _update_metrics(self) -> None:
        p_true, v_true = self._true_state()
        self.metrics["ekf_pos_err"] = (self._x[:, :3] - p_true).norm(dim=-1)
        self.metrics["ekf_vel_err"] = (self._x[:, 3:] - v_true).norm(dim=-1)
        self.metrics["assign_agree"] = self._assign_agree

        # lazy: mdp imports this module, so import it only at call time
        from humanoid_badminton import mdp
        store = getattr(self._env, "_badminton", None)
        if store is None:
            return
        if self._face_cfg is None:
            from mjlab.managers.scene_entity_config import SceneEntityCfg
            from humanoid_badminton.assets import FACE_SITES
            cfg = SceneEntityCfg("robot", site_names=FACE_SITES,
                                 preserve_order=True)
            cfg.resolve(self._env.scene)
            self._face_cfg = cfg
        dist = mdp.nearest_face_dist(self._env, self._face_cfg,
                                     store["p_star"])
        self._min_dist = torch.minimum(self._min_dist, dist)
        t_now = self._env.episode_length_buf.float() * self._env.step_dt
        at_tstar = (t_now >= store["t_star"]) & ~self._tstar_seen
        self._dist_at_tstar[at_tstar] = dist[at_tstar]
        self._tstar_seen |= at_tstar
        # logged at episode reset (last assigned value), so these read as
        # per-episode min / at-t* distances
        self.metrics["face_pstar_min_dist"] = torch.where(
            torch.isinf(self._min_dist), dist, self._min_dist)
        self.metrics["face_pstar_dist_at_tstar"] = self._dist_at_tstar


@dataclass(kw_only=True)
class PerceptionCommandCfg(CommandTermCfg):
    entity_name: str = "shuttle"
    assign_source: str = "true"     # "true" (teacher) or "ekf" (student)
    resampling_time_range: tuple[float, float] = field(
        default=(1.0e9, 1.0e9))  # never; reset() re-inits per episode

    def build(self, env: "ManagerBasedRlEnv") -> PerceptionCommand:
        return PerceptionCommand(self, env)
