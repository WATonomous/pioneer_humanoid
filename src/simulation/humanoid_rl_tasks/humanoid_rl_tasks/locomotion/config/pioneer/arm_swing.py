"""Scripted arm swing for the whole-body humanoid: each shoulder swings opposite its leg on the gait clock (as people
walk), more at higher commanded speed and not at all standing still; the elbows stay a little bent. It takes no
actions from the policy, so the legs' policy keeps the leg-only robot's observations and actions.

Both shoulder-pitch and elbow axes are the robot's lateral axis with the same sign on each side: negative swings the arm
(bends the forearm) forward.
"""
from __future__ import annotations

import math

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import gait


class ArmSwingAction(ActionTerm):
    cfg: ArmSwingActionCfg

    def __init__(self, cfg: ArmSwingActionCfg, env):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.asset_name]
        self._shoulders, _ = self._robot.find_joints(["l_shoulder_pitch", "r_shoulder_pitch"], preserve_order=True)
        self._elbows, _ = self._robot.find_joints(["l_elbow", "r_elbow"], preserve_order=True)
        self._empty = torch.zeros(env.num_envs, 0, device=env.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        speed = self._env.command_manager.get_command("base_velocity")[:, 0].abs()
        amplitude = self.cfg.amplitude * torch.clamp(speed / self.cfg.full_speed, 0.0, 1.0)
        # left arm furthest forward while the right leg is mid-swing: at phase stance / 2 of the left leg's clock
        phase = gait.leg_phase(self._env)[:, 0]
        swing = amplitude * torch.sin(2 * math.pi * (phase - gait.STANCE / 2 + 0.25))
        self._robot.set_joint_position_target(torch.stack([-swing, swing], dim=1), joint_ids=self._shoulders)
        self._robot.set_joint_position_target(torch.full_like(torch.stack([swing, swing], dim=1), -self.cfg.elbow_bend),
                                              joint_ids=self._elbows)


@configclass
class ArmSwingActionCfg(ActionTermCfg):
    class_type: type = ArmSwingAction
    asset_name: str = "robot"
    amplitude: float = 0.35  # rad of shoulder pitch at full speed
    full_speed: float = 0.8  # m/s
    elbow_bend: float = 0.3  # rad
