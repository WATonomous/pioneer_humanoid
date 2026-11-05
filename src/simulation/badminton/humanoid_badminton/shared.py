"""Shared-arm architecture: one single-arm policy drives either arm, and the
critic picks the arm.

The policy sees and moves only the active arm, in that arm's own frame:

  right arm  world x shifted by -ARM_PLANE_X, so its swing plane is x = 0
  left arm   the world mirrored about x = 0 (x -> -x, joint angles negated)
             and then shifted the same way, so the left arm sees what the
             right arm would see in the mirrored shot

Both arms run the same weights, so the pair is left/right symmetric by
construction.

Who takes the shot:

  training ("random")     the active arm is drawn 50/50 at reset and holds
                          the start pose for a random go delay in
                          [0, control.shared_go_max_s]; the critic learns
                          how well that arm does from that state, including
                          shots it cannot reach
  deployment ("external") both arms hold the start pose while ChooserPolicy
                          compares the critic on the right-arm frame with
                          the critic on the left-arm frame, and locks the
                          higher-value arm once the gap exceeds the margin
                          (or at the go delay cap)

After the lock the other arm goes to rest, and after the hit both do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

import aero
from humanoid_badminton import mdp
from humanoid_badminton.perception_command import PerceptionCommand
from humanoid_badminton.shuttle_action import BadmintonAction, BadmintonActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

N_ARM = 6
# x of the right arm's swing plane with the stand centred (arm.base_x 0);
# x_frame = x - ARM_PLANE_X
ARM_PLANE_X = 0.245
FRAMES = ("active", "right", "left")


def _arm(env: "ManagerBasedRlEnv", frame: str) -> torch.Tensor:
    """(n,) arm index of the frame: 0 right, 1 left."""
    store = mdp._state(env)
    if frame == "active":
        return store["assign"]
    return torch.full_like(store["assign"], 0 if frame == "right" else 1)


def _sign(arm: torch.Tensor) -> torch.Tensor:
    """(n, 1) +1 for the right arm, -1 for the left (the mirror)."""
    return (1.0 - 2.0 * arm.float()).unsqueeze(-1)


def _pick_arm(x12: torch.Tensor, arm: torch.Tensor) -> torch.Tensor:
    """(n, 6) of the chosen arm from (n, 12) laid out [right 6, left 6]."""
    return torch.where(arm.unsqueeze(-1) == 0, x12[:, :N_ARM], x12[:, N_ARM:])


def _frame_x(x: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """World x coordinate -> arm frame."""
    return sign * x - ARM_PLANE_X


# -- observations ---------------------------------------------------------

def joint_pos_frame(env: "ManagerBasedRlEnv", frame: str) -> torch.Tensor:
    robot = env.scene["robot"]
    rel = robot.data.joint_pos - robot.data.default_joint_pos
    arm = _arm(env, frame)
    return _sign(arm) * _pick_arm(rel, arm)


def joint_vel_frame(env: "ManagerBasedRlEnv", frame: str) -> torch.Tensor:
    robot = env.scene["robot"]
    rel = robot.data.joint_vel - robot.data.default_joint_vel
    arm = _arm(env, frame)
    return _sign(arm) * _pick_arm(rel, arm)


def face_state_frame(env: "ManagerBasedRlEnv", frame: str,
                     asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """[face position, outward normal] of the frame's racket, arm frame."""
    pos, normal = mdp._face_pose(env, asset_cfg)
    arm = _arm(env, frame)
    idx = torch.arange(env.num_envs, device=env.device)
    p, nrm = pos[idx, arm].clone(), normal[idx, arm].clone()
    s = _sign(arm)
    p[:, :1] = _frame_x(p[:, :1], s)
    nrm[:, :1] = s * nrm[:, :1]
    return torch.cat([p, nrm], dim=-1)


def last_action_frame(env: "ManagerBasedRlEnv", frame: str) -> torch.Tensor:
    """The policy's last action if it drives this frame's arm, else zeros.
    Before the go tick nobody drives an arm, so every frame reads zeros
    (the policy's output is ignored while the arms hold ready)."""
    store = mdp._state(env)
    act = env.action_manager.action
    drives = store["committed"] & (_arm(env, frame) == store["assign"])
    return act * drives.unsqueeze(-1).float()


def _shuttle_frame(feat: torch.Tensor, arm: torch.Tensor) -> torch.Tensor:
    """[p, v, traj points] (all 3-vectors) -> arm frame: x of positions
    mirrored and shifted, x of the velocity mirrored."""
    s = _sign(arm)
    out = feat.clone()
    xs = out[:, 0::3]                       # x of p, v, traj points
    xs = s * xs
    xs[:, 0] -= ARM_PLANE_X                 # p
    xs[:, 2:] -= ARM_PLANE_X                # traj points (column 1 is v)
    out[:, 0::3] = xs
    return out


def teacher_perception_frame(env: "ManagerBasedRlEnv", command_name: str,
                             frame: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, PerceptionCommand)
    return _shuttle_frame(term.teacher_features, _arm(env, frame))


def student_perception_frame(env: "ManagerBasedRlEnv", command_name: str,
                             frame: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, PerceptionCommand)
    return _shuttle_frame(term.student_features, _arm(env, frame))


def intercept_frame(env: "ManagerBasedRlEnv", frame: str) -> torch.Tensor:
    out = mdp.intercept_target(env).clone()
    s = _sign(_arm(env, frame))
    out[:, :1] = _frame_x(out[:, :1], s)
    return out


# -- rewards --------------------------------------------------------------

def approach_assigned(env: "ManagerBasedRlEnv", std: float,
                      asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Dense shaping before the intercept: pull the active arm's face
    toward p*."""
    store = mdp._state(env)
    pos, _ = mdp._face_pose(env, asset_cfg)
    idx = torch.arange(env.num_envs, device=env.device)
    d2 = ((pos[idx, store["assign"]] - store["p_star"]) ** 2).sum(dim=-1)
    t_now = env.episode_length_buf.float() * env.step_dt
    pending = (~store["hit"]) & (t_now < store["t_star"] + 0.1)
    return torch.exp(-d2 / std**2) * pending.float()


# -- action ---------------------------------------------------------------

class SharedArmAction(BadmintonAction):
    """6 joint targets for the active arm, in its frame (offsets from its
    ready pose; the left arm gets the mirrored targets). Until the go tick
    (store "committed") both arms hold the start pose
    (control.shared_start_pose); after it the other arm goes to rest; after
    the hit both go to rest."""

    cfg: "SharedArmActionCfg"

    def __init__(self, cfg: "SharedArmActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)
        assert self._num_targets == 2 * N_ARM, self._num_targets
        # the 12-target buffers (offset, clip, moderation) stay; only the
        # policy-facing dimension is one arm
        self._action_dim = N_ARM
        c = aero.load_params()["control"]
        self._go_max = float(c["shared_go_max_s"])
        assert c["shared_start_pose"] in ("rest", "ready")
        self._start_rest = c["shared_start_pose"] == "rest"
        n = self.num_envs
        self._go_t = torch.zeros(n, device=self.device)
        self._raw6 = torch.zeros(n, N_ARM, device=self.device)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw6

    def process_actions(self, actions: torch.Tensor) -> None:
        store = mdp._state(self._env)
        self._raw6[:] = actions
        if self.cfg.assign_mode == "random":
            t_now = self._env.episode_length_buf.float() * self._env.step_dt
            go = (~store["committed"]) & (t_now >= self._go_t)
            store["committed"] |= go
            store["commit_t"] = torch.where(go, t_now, store["commit_t"])
            store["commit_rule"] = torch.where(go, store["rule_side"],
                                               store["commit_rule"])
        arm = store["assign"].unsqueeze(-1)
        zero = torch.zeros_like(actions)
        # [right 6, left 6]: the active arm's targets, mirrored for the left
        raw12 = torch.cat([torch.where(arm == 0, actions, zero),
                           torch.where(arm == 1, -actions, zero)], dim=-1)
        self._raw_actions[:] = raw12
        tgt = raw12 * self._scale + self._offset
        rest = self._rest.expand_as(tgt)
        start = rest if self._start_rest else (
            self._offset if torch.is_tensor(self._offset)
            else torch.full_like(tgt, self._offset))
        go = store["committed"].unsqueeze(-1)
        other = self._arm_side.unsqueeze(0) != arm
        tgt = torch.where(go, tgt, start)
        tgt = torch.where(go & other, rest, tgt)
        tgt = torch.where(store["hit"].unsqueeze(-1), rest, tgt)
        self._moderate(tgt)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        store = mdp._state(self._env)
        if self._start_rest:
            # the arms start at rest (reset_arms_to_rest): moderate from there
            self._prev_target[env_ids] = self._rest
        self._raw6[env_ids] = 0.0
        store["committed"][env_ids] = False
        store["commit_t"][env_ids] = 0.0
        if self.cfg.assign_mode == "random":
            n = self._go_t[env_ids].shape[0]
            store["assign"][env_ids] = torch.randint(
                0, 2, (n,), device=self.device)
            self._go_t[env_ids] = self._go_max * torch.rand(
                n, device=self.device)
        else:
            store["assign"][env_ids] = 0


def reset_arms_to_rest(env: "ManagerBasedRlEnv",
                       env_ids: torch.Tensor | None) -> None:
    """Reset event: both arms at arm.rest_joint_pos (left mirrored), at
    rest. The default joint pos (the action offset and the joint_pos obs
    origin) stays the ready pose."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    rest = env.action_manager.get_term("joint_pos")._rest   # (1, 12)
    pos = robot.data.default_joint_pos[env_ids].clone()
    assert pos.shape[-1] == rest.shape[-1], (pos.shape, rest.shape)
    pos[:] = rest
    vel = torch.zeros_like(pos)
    robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


@dataclass(kw_only=True)
class SharedArmActionCfg(BadmintonActionCfg):
    # "random": training (arm drawn at reset, random go delay);
    # "external": ChooserPolicy sets assign/committed (eval, demo, robot)
    assign_mode: str = "random"

    def build(self, env: "ManagerBasedRlEnv") -> SharedArmAction:
        return SharedArmAction(self, env)


# -- deployment: the critic picks the arm ---------------------------------

class ChooserPolicy:
    """Wraps the inference policy for an env built with assign_mode
    "external" and the per-arm observation groups (<group>_right,
    <group>_left). Each tick before the lock it evaluates the critic on
    both arms' frames and locks the higher-value arm once the gap exceeds
    `margin`, or at `t_max` regardless. The actor then always runs on the
    locked arm's frame."""

    def __init__(self, env, policy, critic, group: str, margin: float,
                 t_max: float):
        self.uenv = env.unwrapped
        self.policy, self.critic = policy, critic
        self.group, self.margin, self.t_max = group, margin, t_max
        self.v = None       # last (n, 2) critic values, for logging

    @torch.no_grad()
    def __call__(self, obs):
        store = mdp._state(self.uenv)
        g = self.group
        o_r, o_l = obs[f"{g}_right"], obs[f"{g}_left"]
        from tensordict import TensorDict
        n = o_r.shape[0]
        both = TensorDict({g: torch.cat([o_r, o_l])}, batch_size=[2 * n],
                          device=o_r.device)
        v = self.critic(both).reshape(2, n).T          # (n, 2)
        self.v = v
        t_now = self.uenv.episode_length_buf.float() * self.uenv.step_dt
        gap = v[:, 0] - v[:, 1]
        lock = (~store["committed"]) & ((gap.abs() > self.margin)
                                        | (t_now >= self.t_max))
        store["assign"] = torch.where(lock, (gap < 0).long(), store["assign"])
        store["committed"] |= lock
        store["commit_t"] = torch.where(lock, t_now, store["commit_t"])
        store["commit_rule"] = torch.where(lock, store["rule_side"],
                                           store["commit_rule"])
        # the actor's group follows the (possibly just locked) arm; before
        # the lock its output is ignored (both arms hold ready)
        left = (store["assign"] == 1).unsqueeze(-1)
        obs = obs.clone()
        obs[g] = torch.where(left, o_l, o_r)
        return self.policy(obs)


def make_chooser(env, policy, critic_task: str, critic_checkpoint: str,
                 device: str, margin: float, t_max: float | None = None
                 ) -> ChooserPolicy:
    """ChooserPolicy around `policy`, with the critic of `critic_task`
    loaded from `critic_checkpoint` (a PPO checkpoint; a distilled student
    has no critic, so pass its teacher's). `env` must be built from the
    task's play cfg (chooser=True)."""
    from dataclasses import asdict

    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg

    cfg = load_rl_cfg(critic_task)
    runner = MjlabOnPolicyRunner(env, asdict(cfg), device=device)
    runner.load(critic_checkpoint, load_cfg={"critic": True}, strict=True,
                map_location=device)
    critic = runner.alg.critic
    critic.eval()
    if t_max is None:
        t_max = float(aero.load_params()["control"]["shared_go_max_s"])
    return ChooserPolicy(env, policy, critic, cfg.obs_groups["critic"][0],
                         margin, t_max)
