"""The guards that keep the emulator apart from the real arm, and its config from drifting off the real one."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from arm_emulator.config import REPO_ACTUATORS
from arm_emulator.safety_guard import environment_failures, graph_failures
from arm_emulator.sim_config import MARKER, SIM_OVERRIDES, allowed_changes, differences, overlay, write


class TestEnvironment(unittest.TestCase):
    GOOD = {"ROS_DOMAIN_ID": "42", "ROS_LOCALHOST_ONLY": "1"}

    def test_good(self) -> None:
        self.assertEqual(environment_failures(self.GOOD, Path("/nonexistent")), [])

    def test_real_arm_domain_is_refused(self) -> None:
        for env in ({"ROS_LOCALHOST_ONLY": "1"}, {"ROS_DOMAIN_ID": "0", "ROS_LOCALHOST_ONLY": "1"}):
            self.assertTrue(environment_failures(env, Path("/nonexistent")))

    def test_network_visible_is_refused(self) -> None:
        self.assertTrue(environment_failures({"ROS_DOMAIN_ID": "42"}, Path("/nonexistent")))

    def test_can_interface_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as net:
            (Path(net) / "can0").mkdir()
            (Path(net) / "can0" / "type").write_text("280\n")
            (Path(net) / "lo").mkdir()
            (Path(net) / "lo" / "type").write_text("772\n")
            failures = environment_failures(self.GOOD, Path(net))
        self.assertEqual(len(failures), 1)
        self.assertIn("can0", failures[0])


class TestGraph(unittest.TestCase):
    def test_can_node_is_refused(self) -> None:
        self.assertTrue(graph_failures(["arm_emulator", "can_node"], ["arm_emulator"], "arm_emulator"))

    def test_second_feedback_publisher_is_refused(self) -> None:
        self.assertTrue(graph_failures(["arm_emulator"], ["arm_emulator", "other"], "arm_emulator"))

    def test_clean_graph(self) -> None:
        self.assertEqual(graph_failures(["arm_emulator", "joint_command_node"], ["arm_emulator"], "arm_emulator"), [])


class TestSimConfig(unittest.TestCase):
    def test_only_declared_values_differ_from_the_real_config(self) -> None:
        real = yaml.safe_load(REPO_ACTUATORS.read_text(encoding="utf-8"))
        self.assertLessEqual(set(differences(real, overlay(real))), allowed_changes())

    def test_overrides_stay_few_and_named(self) -> None:
        self.assertEqual(list(SIM_OVERRIDES), [("wrist", "pitch", "gravity_ff_scale")])

    def test_written_files_carry_the_marker_and_the_real_gains(self) -> None:
        real = yaml.safe_load(REPO_ACTUATORS.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as out:
            write(Path(out))
            sim = yaml.safe_load((Path(out) / "arm_actuators.yaml").read_text(encoding="utf-8"))
            self.assertTrue((Path(out) / "arm_calibration.yaml").exists())
        self.assertEqual(sim["emulator_overlay"], MARKER)
        self.assertIs(sim["safety"]["left"]["gripper"]["open_close"]["active"], True)
        for key, value in real["safety"]["left"]["gripper"]["open_close"].items():
            if key != "active":
                self.assertEqual(sim["safety"]["left"]["gripper"]["open_close"][key], value)


if __name__ == "__main__":
    unittest.main()
