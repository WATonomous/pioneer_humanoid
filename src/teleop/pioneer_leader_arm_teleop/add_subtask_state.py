"""Tell a language-blind policy (ACT) which step of a multi-step scene it is on: append an encoding of the frame's
instruction to ``observation.state``, in place.

    python add_subtask_state.py <dataset_root> [onehot|words]

``onehot``: one slot per distinct instruction. ``words`` (default): the instruction is split at its preposition
("put the red brick | on the yellow brick") and each side becomes a bag of words, so "red brick" is the same signal
whatever it goes onto: nine duplo instructions share 3 object + 4 target words instead of being nine unrelated classes.

Multi-step scenes (drawer_stow, duplo) announce a new order every episode; the recorder stores it as each frame's
task. ACT does not read the task, so without this it cannot know which block or brick comes next.
The instruction list is written to ``meta/subtask_state.json`` (index = one-hot position); rollouts read it
(qplan.Proposer / sim_policy_eval) and append the same one-hot from the scene's current instruction.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STOP = {"the", "a", "put", "brick", "block"}
PREPOSITIONS = (" on ", " in ", " into ", " onto ")


def make_spec(names, mode):
    if mode == "onehot":
        return {"mode": "onehot", "names": names}
    sides = [split(n) for n in names]
    return {"mode": "words", "object": sorted({w for a, _ in sides for w in a}), "target": sorted({w for _, b in sides for w in b})}


def split(instruction):
    cut = max((instruction.rfind(p), p) for p in PREPOSITIONS)
    a, b = (instruction[:cut[0]], instruction[cut[0] + len(cut[1]):]) if cut[0] >= 0 else (instruction, "")
    words = lambda t: [w for w in t.lower().split() if w not in STOP]  # noqa: E731
    return words(a), words(b)


def encode(instruction, spec):
    """The instruction as the vector appended to the state (zeros for an unknown instruction, e.g. after the last step)."""
    if isinstance(spec, list):                      # first version of the file: a plain list of instructions
        spec = {"mode": "onehot", "names": spec}
    if spec["mode"] == "onehot":
        v = np.zeros(len(spec["names"]), dtype=np.float32)
        if instruction in spec["names"]:
            v[spec["names"].index(instruction)] = 1.0
        return v
    a, b = split(instruction)
    return np.array([float(w in a) for w in spec["object"]] + [float(w in b) for w in spec["target"]], dtype=np.float32)


def main(root: str, mode: str = "words") -> None:
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    feat = info["features"]["observation.state"]
    if (root / "meta/subtask_state.json").exists():
        raise SystemExit("already done")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas().reset_index()
    name_col = [c for c in tasks.columns if c != "task_index"][0]
    names = [str(t) for t in tasks.sort_values("task_index")[name_col]]
    spec = make_spec(names, mode)
    table_enc = np.stack([encode(n, spec) for n in names])
    k, dim = table_enc.shape[1], feat["shape"][0]
    states = []
    for path in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        table = pq.read_table(path)
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
        onehot = table_enc[table["task_index"].to_numpy()]
        new = np.concatenate([state, onehot], 1)
        states.append(new)
        col = pa.FixedSizeListArray.from_arrays(pa.array(new.reshape(-1), pa.float32()), dim + k) \
            if pa.types.is_fixed_size_list(table.schema.field("observation.state").type) else pa.array(list(new), pa.list_(pa.float32()))
        meta = dict(table.schema.metadata or {})
        if b"huggingface" in meta:
            hf = json.loads(meta[b"huggingface"])
            hf["info"]["features"]["observation.state"]["length"] = dim + k
            meta[b"huggingface"] = json.dumps(hf).encode()
        i = table.schema.get_field_index("observation.state")
        pq.write_table(table.set_column(i, "observation.state", col).replace_schema_metadata(meta), path)
    feat["shape"] = [dim + k]
    feat["names"] = list(feat["names"]) + [f"subtask_{i}" for i in range(k)]
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    allv = np.concatenate(states)
    stats = json.loads((root / "meta/stats.json").read_text())
    old = stats["observation.state"]
    new = {"mean": allv.mean(0), "std": allv.std(0), "min": allv.min(0), "max": allv.max(0)}
    for q in [key for key in old if key.startswith("q")]:
        new[q] = np.quantile(allv, float(q[1:]) / 100, axis=0)
    for key, v in new.items():
        v = np.asarray(v, dtype=np.float64)
        if key == "std":
            v[dim:] = 1.0                     # leave the one-hot unscaled (and un-shifted, below)
        if key == "mean":
            v[dim:] = 0.0
        old[key] = v.tolist()
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    (root / "meta/subtask_state.json").write_text(json.dumps(spec, indent=1))
    print(f"[SUBTASK] {root}: state {dim} -> {dim + k}; {spec}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
