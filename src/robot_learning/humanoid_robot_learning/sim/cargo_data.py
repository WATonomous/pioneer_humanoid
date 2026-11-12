"""CPU-buffered LeRobot episode writing and dataset validation/aggregation tools."""
import argparse
import json
from pathlib import Path

import numpy as np

REPO_ID = "humanoid/nasa_cargo_left_arm"


class EpisodeWriter:
    def __init__(self, root, contract, resume=False):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from humanoid_robot_learning.sim.sim_recorder import _install_subprocess_video_encoder, _open_for_append

        _install_subprocess_video_encoder()
        self.root = Path(root)
        self.contract = contract
        self._open_for_append = _open_for_append
        if self.root.exists():
            if not resume:
                raise FileExistsError(f"Dataset exists: {root}; use --resume to append")
            if json.loads((self.root / "cargo_contract.json").read_text()) != contract:
                raise ValueError("Existing dataset has a different cargo observation/action contract")
            self.dataset = _open_for_append(REPO_ID, self.root)
        else:
            features = {
                "observation.state": {"dtype": "float32", "shape": (7,), "names": contract["action_names"]},
                "action": {"dtype": "float32", "shape": (7,), "names": contract["action_names"]},
            }
            for name in contract["cameras"]:
                features[f"observation.images.{name}"] = {
                    "dtype": "video", "shape": (contract["height"], contract["width"], 3),
                    "names": ["height", "width", "channels"],
                }
            self.dataset = LeRobotDataset.create(
                REPO_ID, fps=contract["fps"], features=features, root=self.root,
                robot_type="pioneer_v1_left_arm", use_videos=True, image_writer_threads=4,
            )
            (self.root / "cargo_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    @property
    def episodes(self):
        return self.dataset.meta.total_episodes

    def save(self, frames, provenance):
        if not frames:
            return
        episode_id = self.episodes
        for frame in frames:
            self.dataset.add_frame(dict(frame))
        self.dataset.save_episode(parallel_encoding=False)
        self.dataset.finalize()
        with (self.root / "collection_manifest.jsonl").open("a") as stream:
            stream.write(json.dumps({"episode": episode_id, "frames": len(frames), **provenance}) + "\n")
        self.dataset = self._open_for_append(REPO_ID, self.root)
        print(f"[Dataset] Saved episode {episode_id}: {len(frames)} frames -> {self.root}", flush=True)

    def close(self):
        self.dataset.finalize()


def validate(root, expected=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    contract = json.loads((root / "cargo_contract.json").read_text())
    dataset = LeRobotDataset(REPO_ID, root=root)
    assert dataset.fps == contract["fps"] == 25
    assert tuple(dataset.features["action"]["shape"]) == (7,)
    if expected is not None:
        assert dataset.num_episodes == expected, (dataset.num_episodes, expected)
    manifest = [json.loads(line) for line in (root / "collection_manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == dataset.num_episodes
    assert sum(row["frames"] for row in manifest) == dataset.num_frames
    sample_indices = set()
    offset = 0
    for episode, row in enumerate(manifest):
        assert row["episode"] == episode
        assert row["frames"] > 0
        assert row["source"] in {"synthetic_expert", "human_demo", "human_intervention"}
        sample_indices.update((offset, offset + row["frames"] - 1))
        offset += row["frames"]
        metadata = dataset.meta.episodes[episode]
        assert (root / dataset.meta.info["data_path"].format(
            chunk_index=metadata["data/chunk_index"], file_index=metadata["data/file_index"])).is_file()
        if row["source"] == "synthetic_expert":
            assert row["success"] and row["released"] and row["lifted"]
            assert row.get("failure") is None
            assert row["cargo_motion"] == "robot_contact_only"
    # Decode the first and last frame of every episode, including video boundaries.
    for i in sorted(sample_indices):
        sample = dataset[i]
        assert sample["action"].shape == sample["observation.state"].shape == (7,)
        assert np.isfinite(sample["action"].numpy()).all()
        assert np.isfinite(sample["observation.state"].numpy()).all()
        for name in contract["cameras"]:
            img = sample[f"observation.images.{name}"]
            assert tuple(img.shape) == (3, contract["height"], contract["width"])
            assert float(img.max() - img.min()) > .02, "Blank observation image"
    print(f"CARGO_DATASET_VALID episodes={dataset.num_episodes} frames={dataset.num_frames} fps={dataset.fps}")


def merge(inputs, output):
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    roots = [Path(path).resolve() for path in inputs]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate input dataset would double-count demonstrations")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    contracts = [json.loads((root / "cargo_contract.json").read_text()) for root in roots]
    if any(c != contracts[0] for c in contracts):
        raise ValueError("Cannot aggregate different cargo layouts, cameras, FPS or action contracts")
    datasets = [LeRobotDataset(REPO_ID, root=root) for root in roots]
    result = merge_datasets(datasets, output_repo_id=REPO_ID, output_dir=output)
    result.finalize()
    _repair_aggregate_indices(output)
    (output / "cargo_contract.json").write_text(json.dumps(contracts[0], indent=2) + "\n")
    with (output / "collection_manifest.jsonl").open("w") as stream:
        index = 0
        for root in roots:
            for line in (root / "collection_manifest.jsonl").read_text().splitlines():
                row = json.loads(line)
                row.setdefault("source_dataset", str(root))
                row["episode"] = index
                stream.write(json.dumps(row) + "\n")
                index += 1
    print(f"Aggregated {index} episodes into {output}")


def _repair_aggregate_indices(root):
    """Remap parquet references after LeRobot 0.4 compacts source files.

    Its aggregator adds source file numbers to destination numbers even when
    many source files become one. Reading frames still works, but a subsequent
    aggregation then references nonexistent files. Derive references from the
    actual output, without changing source datasets or video timestamps.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    locations = {}
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        chunk, file = int(path.parent.name.split("-")[1]), int(path.stem.split("-")[1])
        episodes = set(pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist())
        for episode in episodes:
            if episode in locations:
                raise ValueError(f"Episode {episode} unexpectedly spans multiple parquet files")
            locations[episode] = (chunk, file)
    for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        episodes = table["episode_index"].to_pylist()
        chunk, file = int(path.parent.name.split("-")[1]), int(path.stem.split("-")[1])
        values = {
            "data/chunk_index": [locations[e][0] for e in episodes],
            "data/file_index": [locations[e][1] for e in episodes],
            "meta/episodes/chunk_index": [chunk] * len(episodes),
            "meta/episodes/file_index": [file] * len(episodes),
        }
        for name, column in values.items():
            index = table.schema.get_field_index(name)
            table = table.set_column(index, name, pa.array(column, type=table.schema.field(name).type))
        pq.write_table(table, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("root", type=Path)
    check.add_argument("--episodes", type=int)
    aggregate = sub.add_parser("merge")
    aggregate.add_argument("inputs", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        validate(args.root, args.episodes)
    else:
        merge(args.inputs, args.output)


if __name__ == "__main__":
    main()
