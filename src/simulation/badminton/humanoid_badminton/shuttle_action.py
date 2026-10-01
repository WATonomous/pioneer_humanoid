"""Arm position action + shuttle custom physics, applied every substep.

mjlab's sanctioned per-substep hook is ActionTerm.apply_actions, so the term
that writes the arm targets also owns the two shuttle physics pieces the CPU
sim applies around mj_step (mjsim.Sim.step):

  drag     F = -m k |v_est| v_est with the half-step velocity estimate
           v_est = v + dt/2 (g - k|v|v), written to xfrc_applied
  orient   kinematic cork-first alignment: angular qvel zeroed, body +z
           rotated toward the velocity with first-order lag orient_tau

The orientation moves the cork collision sphere, so it is part of the
dynamics, not a visual. Ordering note: apply_actions runs before mj_step,
so the orientation update acts on the *previous* substep's post-step state —
the same phase the CPU sim's post-step hook sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import quat_mul

import aero

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def quat_z_axis(q: torch.Tensor) -> torch.Tensor:
    """World z-axis of the body frame for quats (N, 4), wxyz."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        dim=-1)


def quat_z2vec(v_hat: torch.Tensor) -> torch.Tensor:
    """Quat rotating +z onto unit vectors v_hat (N, 3). Torch port of
    mju_quatZ2Vec, antiparallel case handled with the x-axis."""
    z = torch.tensor([0.0, 0.0, 1.0], dtype=v_hat.dtype, device=v_hat.device)
    z = z.expand_as(v_hat)
    cross = torch.cross(z, v_hat, dim=-1)
    s = cross.norm(dim=-1)
    c = v_hat[..., 2].clamp(-1.0, 1.0)
    angle = torch.atan2(s, c)
    axis = cross / s.clamp_min(1e-12).unsqueeze(-1)
    anti = s < 1e-8
    x_axis = torch.zeros_like(v_hat)
    x_axis[..., 0] = 1.0
    axis = torch.where(anti.unsqueeze(-1), x_axis, axis)
    half = 0.5 * angle
    return torch.cat([torch.cos(half).unsqueeze(-1),
                      axis * torch.sin(half).unsqueeze(-1)], dim=-1)


def axis_angle_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    return torch.cat([torch.cos(half).unsqueeze(-1),
                      axis * torch.sin(half).unsqueeze(-1)], dim=-1)


class BadmintonAction(JointPositionAction):
    cfg: "BadmintonActionCfg"

    def __init__(self, cfg: "BadmintonActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)
        p = aero.load_params()
        self._k = p["shuttle"]["k"]
        self._mass = p["shuttle"]["mass"]
        self._g = p["gravity"]
        dt = env.physics_dt
        self._dt = dt
        self._orient_frac = 1.0 - float(
            torch.exp(torch.tensor(-dt / p["shuttle"]["orient_tau"])))
        self._shuttle = env.scene[cfg.shuttle_entity_name]
        idx = self._shuttle.data.indexing
        self._q_adr = idx.free_joint_q_adr
        self._v_adr = idx.free_joint_v_adr
        self._g_vec = torch.tensor([0.0, 0.0, -self._g], device=self.device)

        # Command moderation, mirroring autonomy/behaviour/joint_command
        # (armPoseToMotorCmds): per-tick cap on target motion, then low-pass
        # q <- alpha*q_prev + (1-alpha)*q, then the position clamp. The real
        # arm cannot receive bang-bang targets; without this the policy
        # exploits a command channel the hardware does not have (runs 8/10:
        # every joint pinned at its torque clamp most of the episode).
        c = p["control"]
        # per-joint caps, same motors on both arms (right first)
        self._step_max = torch.tensor(
            c["target_velocity_max"] * 2, device=self.device) * env.step_dt
        self._alpha = float(c["low_pass_alpha"])
        self._prev_target = self._offset.clone() if torch.is_tensor(
            self._offset) else torch.full_like(self._raw_actions, self._offset)
        # after the hit the target is no longer the policy's: the arm returns
        # to the rest pose through the same moderation (a deployment wrapper
        # does the same, so post-hit behaviour is deterministic and slow)
        # (both arms; the left rest pose is the mirrored right one)
        arm = p["arm"]
        rest = dict(zip(arm["joints"], arm["rest_joint_pos"]))
        rest.update({j: m * q for j, m, q in zip(
            arm["joints_left"], arm["left_mirror"], arm["rest_joint_pos"])})
        # target names carry the model prefix ("arm_joint4")
        self._arm_side = torch.tensor(
            [0 if n.removeprefix("arm_") in arm["joints"] else 1
             for n in self._target_names], device=self.device)
        self._rest = torch.tensor(
            [rest[n.removeprefix("arm_")] for n in self._target_names],
            device=self.device).unsqueeze(0)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        from humanoid_badminton import mdp  # lazy: mdp imports this module
        tgt = self._processed_actions
        store = mdp._state(self._env)
        # bimanual: the arm not assigned to the shot holds the ready pose
        # (the action offset), so only one racket goes for the shuttle
        ready = self._offset if torch.is_tensor(self._offset) else \
            torch.full_like(tgt, self._offset)
        idle = self._arm_side.unsqueeze(0) != store["assign"].unsqueeze(-1)
        tgt = torch.where(idle, ready, tgt)
        hit = store["hit"]
        tgt = torch.where(hit.unsqueeze(-1), self._rest.expand_as(tgt), tgt)
        prev = self._prev_target
        q = prev + (tgt - prev).clamp(-self._step_max, self._step_max)
        q = self._alpha * prev + (1.0 - self._alpha) * q
        if self.cfg.clip is not None:
            q = torch.clamp(q, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self._prev_target = q
        self._processed_actions = q

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        # the arm restarts at the default (ready) pose: moderate from there
        # (_offset is the per-env default joint pos when use_default_offset)
        if env_ids is None:
            env_ids = slice(None)
        self._prev_target[env_ids] = (self._offset[env_ids] if torch.is_tensor(
            self._offset) else self._offset)

    def apply_actions(self) -> None:
        super().apply_actions()
        data = self._shuttle.data.data
        qpos = data.qpos
        qvel = data.qvel

        v = qvel[:, self._v_adr[:3]]
        # drag at the half-step velocity estimate (see mjsim.Sim.apply_drag)
        a = self._g_vec - self._k * v.norm(dim=-1, keepdim=True) * v
        v_est = v + 0.5 * self._dt * a
        force = -self._mass * self._k * v_est.norm(dim=-1, keepdim=True) * v_est
        self._shuttle.write_external_wrench_to_sim(
            forces=force.unsqueeze(1), torques=torch.zeros_like(force).unsqueeze(1),
            body_ids=[0])

        # kinematic cork-first orientation lag
        qvel[:, self._v_adr[3:6]] = 0.0
        speed = v.norm(dim=-1)
        active = speed > 0.5
        if not bool(active.any()):
            return
        quat = qpos[:, self._q_adr[3:7]]
        z_world = quat_z_axis(quat)
        v_hat = v / speed.clamp_min(1e-9).unsqueeze(-1)
        cross = torch.cross(z_world, v_hat, dim=-1)
        s = cross.norm(dim=-1)
        c = (z_world * v_hat).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.atan2(s, c)
        x_axis = torch.zeros_like(v)
        x_axis[..., 0] = 1.0
        y_axis = torch.zeros_like(v)
        y_axis[..., 1] = 1.0
        anti_x = torch.cross(z_world, x_axis, dim=-1)
        anti_y = torch.cross(z_world, y_axis, dim=-1)
        anti = torch.where((anti_x.norm(dim=-1, keepdim=True) < 1e-8),
                           anti_y, anti_x)
        anti = anti / anti.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        axis = torch.where((s < 1e-8).unsqueeze(-1),
                           anti, cross / s.clamp_min(1e-12).unsqueeze(-1))
        dq = axis_angle_quat(axis, self._orient_frac * angle)
        new_q = quat_mul(dq, quat)
        new_q = new_q / new_q.norm(dim=-1, keepdim=True)
        apply = active & (angle > 1e-6)
        qpos[:, self._q_adr[3:7]] = torch.where(apply.unsqueeze(-1), new_q, quat)


@dataclass(kw_only=True)
class BadmintonActionCfg(JointPositionActionCfg):
    shuttle_entity_name: str = "shuttle"

    def build(self, env: "ManagerBasedRlEnv") -> BadmintonAction:
        return BadmintonAction(self, env)
