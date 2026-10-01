"""MDP terms for the badminton receive task.

Episode initial conditions come from launcher.sample_specs — the same
rejection-sampled bank the CPU gates use — built once on the host at the
first reset and indexed on the device afterwards. The reset event stores the
per-env episode targets (p*, t*) plus a has-hit flag in env._badminton;
reward/observation terms read that store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

import aero
import launcher
from humanoid_badminton.perception_command import PerceptionCommand
from humanoid_badminton.shuttle_action import quat_z2vec

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

BANK_SIZE = 4096
BANK_SEED = 0


def _state(env: "ManagerBasedRlEnv") -> dict:
    store = getattr(env, "_badminton", None)
    if store is None:
        if not launcher.workspace_exists():
            launcher.build_workspace(verbose=False)
        specs = launcher.sample_specs(BANK_SIZE, seed=BANK_SEED)
        dev = env.device

        import numpy as np

        def t(rows):
            return torch.tensor(np.array(rows), dtype=torch.float32,
                                device=dev)

        n = env.num_envs
        store = {
            "bank_p0": t([s.p0 for s in specs]),
            "bank_v0": t([s.v0 for s in specs]),
            "bank_p_star": t([s.p_star for s in specs]),
            "bank_t_star": t([s.t_star for s in specs]),
            "p_star": torch.zeros(n, 3, device=dev),
            "t_star": torch.zeros(n, device=dev),
            "hit": torch.zeros(n, dtype=torch.bool, device=dev),
            "first": torch.zeros(n, dtype=torch.bool, device=dev),
            # arm assigned to the shot (0 right, 1 left), written each tick
            # by PerceptionCommand
            "assign": torch.zeros(n, dtype=torch.long, device=dev),
            "p0_xy": torch.zeros(n, 2, device=dev),
            # per actuated joint, right arm then left (same motors)
            "tau_rated": t(aero.load_params()["arm"]["torque_rated"] * 2),
        }
        env._badminton = store
    return store


# -- events ---------------------------------------------------------------

def reset_badminton_episode(env: "ManagerBasedRlEnv",
                            env_ids: torch.Tensor) -> None:
    """Launch a fresh episode: place the shuttle on a bank row, cork-first."""
    store = _state(env)
    shuttle = env.scene["shuttle"]
    idx = torch.randint(0, store["bank_p0"].shape[0], (len(env_ids),),
                        device=env.device)
    p0 = store["bank_p0"][idx]
    v0 = store["bank_v0"][idx]
    quat = quat_z2vec(v0 / v0.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    state = torch.cat(
        [p0, quat, v0, torch.zeros_like(v0)], dim=-1)
    shuttle.write_root_state_to_sim(state, env_ids=env_ids)
    store["p_star"][env_ids] = store["bank_p_star"][idx]
    store["t_star"][env_ids] = store["bank_t_star"][idx]
    store["hit"][env_ids] = False
    store["p0_xy"][env_ids] = p0[:, :2]


# -- helpers --------------------------------------------------------------

def _sensor_hit(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """(num_envs,) bool: any contact on the sensor within the last tick.

    Contact history has no found_history field, so per-substep detection
    uses force_history (nonzero force implies contact); the found field
    alone would only see the final substep of the tick."""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if data.force_history is not None:
        # [B, N, H, 3] -> any substep with measurable force
        hit |= (data.force_history.norm(dim=-1) > 1e-6).any(dim=-1).any(dim=-1)
    if data.found is not None:
        hit |= (data.found > 0).any(dim=-1)
    return hit


def _shuttle_state(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
    shuttle = env.scene["shuttle"]
    d = shuttle.data
    idx = d.indexing
    return (d.data.qpos[:, idx.free_joint_q_adr[:3]],
            d.data.qvel[:, idx.free_joint_v_adr[:3]])


def _face_pose(env: "ManagerBasedRlEnv",
               asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Face center positions and outward normals (site z-axes), world
    frame, shape (num_envs, S, 3) for the S sites in asset_cfg (right
    face first, then left)."""
    robot = env.scene[asset_cfg.name]
    pos = robot.data.site_pos_w[:, asset_cfg.site_ids]
    mat = robot.data.data.site_xmat[:, robot.data.indexing.site_ids]
    mat = mat[:, asset_cfg.site_ids].reshape(env.num_envs, -1, 3, 3)
    return pos, mat[..., 2]


def nearest_face_dist(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg,
                      target: torch.Tensor) -> torch.Tensor:
    """(num_envs,) distance from target (num_envs, 3) to the closer face."""
    pos, _ = _face_pose(env, asset_cfg)
    return (pos - target.unsqueeze(1)).norm(dim=-1).min(dim=-1).values


# -- observations ---------------------------------------------------------

def student_perception(env: "ManagerBasedRlEnv",
                       command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, PerceptionCommand)
    return term.student_features


def student_uncertainty(env: "ManagerBasedRlEnv",
                        command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, PerceptionCommand)
    return term.student_uncertainty


def teacher_perception(env: "ManagerBasedRlEnv",
                       command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, PerceptionCommand)
    return term.teacher_features


def intercept_target(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Privileged: p* and remaining time to t*. Shape (num_envs, 4)."""
    store = _state(env)
    t_now = env.episode_length_buf.float() * env.step_dt
    return torch.cat(
        [store["p_star"], (store["t_star"] - t_now).unsqueeze(-1)], dim=-1)


def arm_assign(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """One-hot of the arm assigned to this shot, [right, left]."""
    return torch.nn.functional.one_hot(_state(env)["assign"], 2).float()


def face_state(env: "ManagerBasedRlEnv",
               asset_cfg: SceneEntityCfg) -> torch.Tensor:
    pos, normal = _face_pose(env, asset_cfg)
    # per face [pos, normal], right face first
    return torch.cat([pos, normal], dim=-1).flatten(1)


# -- rewards --------------------------------------------------------------

def face_contact(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Sparse: 1 on the tick the racket face first meets the cork. Also
    latches the has-hit flag, so list this term before return_flight."""
    store = _state(env)
    hit_now = _sensor_hit(env, sensor_name)
    first = hit_now & ~store["hit"]
    store["hit"] |= hit_now
    store["first"] = first          # this tick only; return_landing reads it
    return first.float()


def approach_intercept(env: "ManagerBasedRlEnv", std: float,
                       asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Dense shaping before the intercept: pull the assigned arm's face
    toward p* (asset_cfg lists the faces right first, so the assignment
    index selects the face)."""
    store = _state(env)
    pos, _ = _face_pose(env, asset_cfg)
    pos = pos[torch.arange(env.num_envs, device=env.device), store["assign"]]
    d2 = ((pos - store["p_star"]) ** 2).sum(dim=-1)
    t_now = env.episode_length_buf.float() * env.step_dt
    pending = (~store["hit"]) & (t_now < store["t_star"] + 0.1)
    return torch.exp(-d2 / std**2) * pending.float()


def return_flight(env: "ManagerBasedRlEnv", v_scale: float = 6.0) -> torch.Tensor:
    """After the hit: reward outgoing velocity toward the far court with an
    upward component (clears the net; landing shaping comes later)."""
    store = _state(env)
    _, v = _shuttle_state(env)
    quality = (v[:, 1] + 0.5 * v[:, 2]).clamp(0.0) / v_scale
    return quality.clamp(max=1.0) * store["hit"].float()


def predict_landing(p: torch.Tensor, v: torch.Tensor,
                    sub_dt: float = 0.05, max_steps: int = 80
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll (B,3) shuttle states through the drag model to the floor.

    Returns (landing_xy, ok): ok = the flight crosses the net line going
    forward with clearance above the net top, and lands in the far court.
    Positions where the flight never lands within the horizon get ok False.
    """
    import perception_torch as pt
    prm = aero.load_params()
    g = prm["gravity"]
    k = g / prm["shuttle"]["v_t"] ** 2
    net_top = prm["net"]["height_top"]
    land_xy = p[:, :2].clone()
    landed = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
    cleared = torch.zeros_like(landed)
    for _ in range(max_steps):
        p_next, v_next = pt.rk4_step(p, v, k, sub_dt, g)
        # net-line crossing inside this step: interpolate z at y = 0
        crossing = (~landed) & (p[:, 1] < 0.0) & (p_next[:, 1] >= 0.0)
        if bool(crossing.any()):
            t = (-p[crossing, 1] / (p_next[crossing, 1] - p[crossing, 1]
                                    ).clamp_min(1e-9)).clamp(0.0, 1.0)
            z_at_net = p[crossing, 2] + t * (p_next[crossing, 2]
                                             - p[crossing, 2])
            cleared[crossing] = z_at_net > net_top
        # floor contact inside this step: interpolate xy at z = 0
        touch = (~landed) & (p_next[:, 2] <= 0.0)
        if bool(touch.any()):
            t = (p[touch, 2] / (p[touch, 2] - p_next[touch, 2]
                                ).clamp_min(1e-9)).clamp(0.0, 1.0)
            land_xy[touch] = (p[touch, :2]
                              + t.unsqueeze(-1) * (p_next[touch, :2]
                                                   - p[touch, :2]))
            landed |= touch
        p, v = p_next, v_next
        if bool(landed.all()):
            break
    ok = landed & cleared & (land_xy[:, 1] > 0.0)
    return land_xy, ok


def return_landing(env: "ManagerBasedRlEnv", sigma: float = 1.5
                   ) -> torch.Tensor:
    """Sparse, paid once at the hit tick: gaussian in the predicted landing
    point's distance to the episode's launch origin (the target zone), zero
    if the predicted flight does not clear the net into the far court. The
    prediction rolls the post-contact shuttle state through the same drag
    model the launcher bank was built with, so the reward needs no extra
    sim time beyond the hit tick."""
    store = _state(env)
    out = torch.zeros(env.num_envs, device=env.device)
    first = store["first"]
    if bool(first.any()):
        pos, vel = _shuttle_state(env)
        land_xy, ok = predict_landing(pos[first], vel[first])
        d2 = ((land_xy - store["p0_xy"][first]) ** 2).sum(dim=-1)
        out[first] = torch.exp(-d2 / (2.0 * sigma**2)) * ok.float()
    return out


def torque_over_rated(env: "ManagerBasedRlEnv",
                      asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Thermal proxy (penalty): sum over joints of max(|tau|/tau_rated - 1, 0).

    The sim caps torque at datasheet peaks but has no heat model, so nothing
    stops a policy from living above the rated (continuous) torque. Run-8
    eval: the GL40 wrist sat above rated on 95% of ticks. Normalized per
    joint so a 0.5 Nm wrist overload weighs like a 35 Nm shoulder overload.
    Torque is qfrc_actuator (post jnt_actfrcrange clamp)."""
    store = _state(env)
    robot = env.scene[asset_cfg.name]
    idx = robot.data.indexing.joint_v_adr
    tau = robot.data.data.qfrc_actuator[:, idx].abs()
    return (tau / store["tau_rated"] - 1.0).clamp_min(0.0).sum(dim=-1)


def arm_clash(env: "ManagerBasedRlEnv",
              sensor_names: tuple[str, ...]) -> torch.Tensor:
    """(num_envs,) bool: the two arms (rackets, forearms) touched this tick.
    Used as a penalty and as a termination."""
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in sensor_names:
        hit |= _sensor_hit(env, name)
    return hit


# -- terminations ---------------------------------------------------------

def shuttle_grounded(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    return _sensor_hit(env, sensor_name)


def shuttle_net(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    return _sensor_hit(env, sensor_name)
