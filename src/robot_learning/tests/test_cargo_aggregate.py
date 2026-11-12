"""Regression for repeated LeRobot aggregation after parquet compaction."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Requires the cargo container's pyarrow")
class AggregateIndexTests(unittest.TestCase):
    def test_compacted_files_remain_usable_in_next_round(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from humanoid_robot_learning.sim.cargo_data import _repair_aggregate_indices

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data/chunk-000"
            meta = root / "meta/episodes/chunk-000"
            data.mkdir(parents=True)
            meta.mkdir(parents=True)
            pq.write_table(pa.table({"episode_index": [0, 0, 1, 1]}), data / "file-000.parquet")
            pq.write_table(pa.table({"episode_index": [2, 2]}), data / "file-001.parquet")
            path = meta / "file-000.parquet"
            pq.write_table(pa.table({
                "episode_index": [0, 1, 2],
                "data/chunk_index": [0, 0, 0],
                "data/file_index": [0, 1, 2],
                "meta/episodes/chunk_index": [0, 0, 0],
                "meta/episodes/file_index": [0, 1, 2],
                "videos/ego/from_timestamp": [0., 1.2, 2.4],
            }), path)
            _repair_aggregate_indices(root)
            result = pq.read_table(path).to_pydict()
            self.assertEqual(result["data/file_index"], [0, 0, 1])
            self.assertEqual(result["meta/episodes/file_index"], [0, 0, 0])
            self.assertEqual(result["videos/ego/from_timestamp"], [0., 1.2, 2.4])


if __name__ == "__main__":
    unittest.main()
