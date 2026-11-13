"""Empty local sessions must resume without losing recordings or contacting the Hub."""
import json
from pathlib import Path
import tempfile
import unittest

from humanoid_robot_learning.sim.cargo_data import _archive_empty_session


class EmptySessionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name) / "corrections"
        (self.root / "meta").mkdir(parents=True)
        self.info = self.root / "meta/info.json"
        self.info.write_text(json.dumps({"total_episodes": 0, "total_frames": 0}))
        (self.root / "cargo_contract.json").write_text('{"version": 1}')

    def test_empty_shell_is_preserved_as_a_backup(self):
        original = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        backup = _archive_empty_session(self.root)
        self.assertFalse(self.root.exists())
        self.assertTrue(backup.is_dir())
        self.assertEqual({p.relative_to(backup): p.read_bytes() for p in backup.rglob("*") if p.is_file()}, original)

    def test_recorded_episode_or_frame_counter_prevents_archiving(self):
        for episodes, frames in ((1, 10), (0, 10), (1, 0)):
            self.info.write_text(json.dumps({"total_episodes": episodes, "total_frames": frames}))
            self.assertIsNone(_archive_empty_session(self.root))
            self.assertTrue(self.info.is_file())

    def test_zero_counters_cannot_hide_recording_artifacts(self):
        for relative in ("data/chunk-000/file-000.parquet", "videos/ego/file-000.mp4",
                         "images/ego/frame-000000.png", "collection_manifest.jsonl"):
            with self.subTest(relative=relative):
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"preserve this recording")
                with self.assertRaisesRegex(ValueError, "preserving it"):
                    _archive_empty_session(self.root)
                self.assertEqual(path.read_bytes(), b"preserve this recording")
                path.unlink()

    def test_missing_or_invalid_counters_are_not_assumed_empty(self):
        for info in ({}, {"total_episodes": -1, "total_frames": 0},
                     {"total_episodes": 0, "total_frames": "0"}):
            self.info.write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                _archive_empty_session(self.root)
            self.assertTrue(self.root.exists())


if __name__ == "__main__":
    unittest.main()
