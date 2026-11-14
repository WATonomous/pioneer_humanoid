# Save accepted corrections locally without depending on a simulator.
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from .core import Contract


# Store accepted human segments with their metadata and data contract.
class CorrectionStore:
    # Open a correction folder and reject incompatible existing data.
    def __init__(self, root, contract):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        path = self.root / "contract.json"
        if path.exists():
            if Contract.from_json(path.read_text()) != contract:
                raise ValueError("Correction dataset contract mismatch; use a new directory")
        else:
            if any(self.root.iterdir()):
                raise ValueError("Nonempty correction directory is missing contract.json")
            # Create the contract only if another collector has not already written it.
            with path.open("x") as f:
                f.write(contract.json() + "\n")

    # Check and save all accepted segments in one atomic transaction.
    def save(self, segments, metadata):
        if not segments or any(not s for s in segments):
            raise ValueError("Cannot save empty interventions")
        frames = [f for segment in segments for f in segment]
        for f in frames:
            self.contract.check_observation({k: v for k, v in f.items() if k != "action"})
            self.contract.action(f["action"])
        arrays = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
        arrays["segment_lengths"] = np.array(list(map(len, segments)), dtype=np.int64)
        arrays["metadata"] = np.array(json.dumps(metadata))
        arrays["contract"] = np.array(self.contract.json())
        fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".npz", dir=self.root)
        try:
            # Finish writing the temporary file before publishing the recording.
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            target = self.root / f"accepted-{uuid.uuid4().hex}.npz"
            os.replace(temporary, target)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

    # Read and validate each uninterrupted human intervention separately.
    def episodes(self):
        for path in sorted(self.root.glob("accepted-*.npz")):
            with np.load(path, allow_pickle=False) as data:
                if Contract.from_json(str(data["contract"])) != self.contract:
                    raise ValueError(f"Contract mismatch: {path}")
                lengths = data["segment_lengths"]
                if lengths.ndim != 1 or not len(lengths) or (lengths <= 0).any():
                    raise ValueError(f"Invalid intervention boundaries: {path}")
                keys = {"observation.state", "action", *self.contract.images}
                # Decompress each array once instead of reopening it for every frame.
                arrays = {k: data[k] for k in keys}
                total = int(lengths.sum())
                if any(len(arrays[k]) != total for k in keys):
                    raise ValueError(f"Inconsistent frame counts: {path}")
                metadata = json.loads(str(data["metadata"]))
                start = 0
                for length in lengths:
                    frames = []
                    for i in range(start, start + int(length)):
                        frame = {k: arrays[k][i].copy() for k in keys}
                        self.contract.check_observation({k: v for k, v in frame.items() if k != "action"})
                        self.contract.action(frame["action"])
                        frames.append(frame)
                    yield frames, metadata
                    start += int(length)


# Reopen a correction folder using its saved contract.
def open_store(root):
    root = Path(root)
    return CorrectionStore(root, Contract.from_json((root / "contract.json").read_text()))


# Validate every saved frame and count accepted segments across collection sessions.
def validate(root):
    segments = frames = 0
    for episode, _ in open_store(root).episodes():
        segments += 1
        frames += len(episode)
    return {"segments": segments, "frames": frames}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate and count saved HG-DAgger corrections")
    parser.add_argument("root", help="Correction directory")
    print(json.dumps(validate(parser.parse_args().root)))
