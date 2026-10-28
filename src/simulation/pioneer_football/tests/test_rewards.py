import unittest
from types import SimpleNamespace as N

import torch

from pioneer_football import mdp


class RewardTests(unittest.TestCase):
    def env(self, commands, velocity):
        cmd = torch.tensor(commands, dtype=torch.float32)
        d = N(
            root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]),
            root_link_lin_vel_w=torch.tensor(velocity, dtype=torch.float32),
            root_link_ang_vel_w=torch.zeros(1, 3),
        )
        return N(
            scene={"robot": N(data=d)}, command_manager=N(get_command=lambda _: cmd)
        )

    def test_tracking_rewards_motion_over_standing(self):
        e = self.env([[0.6, 0, 0]], [[0.6, 0, 0]])
        self.assertAlmostEqual(float(mdp.track_linear(e)), 1.0)
        e.scene["robot"].data.root_link_lin_vel_w.zero_()
        self.assertLess(float(mdp.track_linear(e)), 0.25)

    def test_yaw_frame_tracking(self):
        e = self.env([[0.6, 0, 0]], [[0, 0.6, 0]])
        e.scene["robot"].data.root_link_quat_w[:] = torch.tensor(
            [[2**-0.5, 0, 0, 2**-0.5]]
        )
        self.assertAlmostEqual(float(mdp.track_linear(e)), 1.0, places=5)

    def test_single_stance_reward_and_standing_gate(self):
        e = self.env([[0.3, 0, 0]], [[0, 0, 0]])
        e.scene["feet"] = N(
            data=N(
                current_contact_time=torch.tensor([[0.3, 0]]),
                current_air_time=torch.tensor([[0, 0.15]]),
            )
        )
        self.assertAlmostEqual(float(mdp.air_time(e)), 0.15, places=6)
        e.command_manager = N(get_command=lambda _: torch.zeros(1, 3))
        self.assertEqual(float(mdp.air_time(e)), 0.0)
        e.command_manager = N(get_command=lambda _: torch.tensor([[0.3, 0, 0]]))
        e.scene["feet"].data.current_contact_time[:] = 0.3
        self.assertEqual(float(mdp.air_time(e)), 0.0)


if __name__ == "__main__":
    unittest.main()
