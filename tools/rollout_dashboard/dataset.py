"""Read-only view of one LeRobot v3 dataset, plus its optional RECAP sidecar, for the dashboard.

Only metadata is loaded up front: the episode table (meta/episodes + recap/episodes.parquet) and the
per-frame value arrays (a few float32s per frame). Per-frame state/action, recap step counters and video
are read per episode on request, so memory stays flat with dataset size.

An episode subset (--episodes) is a parquet/CSV with an `episode_index` column (or index): only those episodes are
listed in the file's row order, and its other numeric/bool columns become sortable list columns (e.g. a score).

Value sources (one checkbox each in the RL tab):
  * recap/labels_r*.parquet — auto-discovered; `value` (+ `positive`) keyed by dataset `index`.
  * --values NAME=PATH — a .npy with one float per dataset frame (dataset-index order), or a labels-style parquet.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RECAP_DIR = "recap"
DEMO_SOURCE = "demo"  # dataset episodes with no recap/episodes.parquet row are demos (successful by construction)
FLOAT_DIGITS = 4  # rounding of per-frame floats sent to the browser; keeps episode JSON small
RECAP_COLUMNS = ["source", "round", "policy", "seed", "success", "advantage", "parent_episode"]


@dataclass
class ValueSource:
    name: str
    values: np.ndarray  # float per dataset frame (global `index`), NaN where the source has no value
    positive: np.ndarray | None  # int8 per frame: 1 positive, 0 negative, -1 unknown; None if the source has no labels


def load_value_source(name: str, path: Path, total_frames: int) -> ValueSource:
    """Load a per-frame value function from a values.npy or a labels-style parquet (index, value[, positive])."""
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r")
        if values.shape not in ((total_frames,), (total_frames, 1)):
            raise ValueError(f"{path}: shape {values.shape}, expected one value per dataset frame ({total_frames},)")
        return ValueSource(name, np.asarray(values, dtype=np.float32).reshape(-1), None)
    columns = [c for c in ("index", "value", "positive") if c in pq.read_schema(path).names]
    table = pq.read_table(path, columns=columns)
    index = table["index"].to_numpy()
    values = np.full(total_frames, np.nan, dtype=np.float32)
    values[index] = table["value"].to_numpy()
    positive = None
    if "positive" in columns:
        positive = np.full(total_frames, -1, dtype=np.int8)
        positive[index] = table["positive"].to_numpy().astype(np.int8)
    return ValueSource(name, values, positive)


def load_subset(path: Path) -> pd.DataFrame:
    """Episode subset file -> DataFrame with an `episode_index` column (accepts it as the pandas index too)."""
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "episode_index" not in table.columns:
        table = table.reset_index()
    if "episode_index" not in table.columns:
        raise ValueError(f"{path}: needs an episode_index column or index")
    return table


def short_policy(path: str) -> str:
    """`.../runs/<run>/train/checkpoints/<step>/pretrained_model` -> `<run>@<step>`; other strings unchanged."""
    if not isinstance(path, str) or not path:
        return ""
    parts = Path(path).parts
    if "checkpoints" not in parts:
        return Path(path).name
    i = parts.index("checkpoints")
    run = parts[i - 2] if i >= 2 and parts[i - 1] == "train" else parts[i - 1]
    step = parts[i + 1] if i + 1 < len(parts) else ""
    return f"{run}@{step}"


def _rounded(array: np.ndarray) -> list:
    """JSON-ready list with NaN -> None."""
    out = np.round(np.asarray(array, dtype=np.float64), FLOAT_DIGITS)
    return [None if np.isnan(x) else x for x in out.tolist()] if np.isnan(out).any() else out.tolist()


class Dataset:
    def __init__(self, root: Path, extra_values: dict[str, Path] | None = None, subset: Path | None = None):
        self.root = Path(root).resolve()
        self.name = self.root.name
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(self.info["fps"])
        self.total_frames = int(self.info["total_frames"])
        features = self.info["features"]
        self.cameras = [k for k, f in features.items() if f["dtype"] == "video"]
        self.joint_names = {k: features[k].get("names") for k in ("observation.state", "action") if k in features}
        self.has_subtask = "subtask_index" in features
        self.tasks = self._load_tasks()
        self.recap_dir = self.root / RECAP_DIR
        self.has_recap = (self.recap_dir / "episodes.parquet").exists()
        self.values = self._load_values(extra_values or {})
        self.episodes = self._load_episode_table()
        self.subset_name = Path(subset).name if subset else None
        self.extra_columns: list[str] = []
        if subset:
            self._apply_subset(load_subset(Path(subset)))

    # ---- loading (once, at startup) ----
    def _load_tasks(self) -> list[str]:
        tasks = pd.read_parquet(self.root / "meta" / "tasks.parquet")
        by_index = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
        return [by_index.get(i, "") for i in range(max(by_index, default=-1) + 1)]

    def _load_values(self, extra: dict[str, Path]) -> dict[str, ValueSource]:
        paths = {}
        if self.recap_dir.is_dir():
            labels = sorted(self.recap_dir.glob("labels_r*.parquet"), key=lambda p: int(re.search(r"r(\d+)", p.stem).group(1)))
            paths.update({p.stem: p for p in labels})
        paths.update(extra)
        return {name: load_value_source(name, Path(p), self.total_frames) for name, p in paths.items()}

    def _load_episode_table(self) -> pd.DataFrame:
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        columns = [c for c in pq.read_schema(files[0]).names if not c.startswith("stats/")]
        eps = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
        eps["tasks"] = eps["tasks"].map(lambda t: " | ".join(t) if t is not None and not isinstance(t, str) else (t or ""))
        eps["seconds"] = eps["length"] / self.fps
        eps = self._merge_recap(eps)
        for name, source in self.values.items():
            end = np.asarray(source.values)[eps["dataset_to_index"].to_numpy() - 1]
            eps[f"value_end:{name}"] = end  # value on the last frame: separates "almost made it" from stalled failures
        return eps.sort_values("episode_index").reset_index(drop=True)

    def _merge_recap(self, eps: pd.DataFrame) -> pd.DataFrame:
        if not self.has_recap:
            eps["source"] = DEMO_SOURCE
            eps["success"] = True
            return eps
        recap = pd.read_parquet(self.recap_dir / "episodes.parquet")
        recap = recap[["episode_index"] + [c for c in RECAP_COLUMNS if c in recap.columns]]
        eps = eps.merge(recap, on="episode_index", how="left")
        eps["source"] = eps["source"].fillna(DEMO_SOURCE)
        eps["success"] = eps["success"].astype("boolean").fillna(True).astype(bool)
        for col in ("round", "seed", "parent_episode"):
            if col in eps:
                eps[col] = eps[col].astype("Int64")  # NaN for demos would otherwise turn them into floats
        if "policy" in eps:
            eps["policy"] = eps["policy"].map(short_policy)
        return eps

    def _apply_subset(self, subset: pd.DataFrame):
        """Keep only the subset's episodes; its numeric/bool columns not already in the table become list columns."""
        self.extra_columns = [c for c in subset.columns if c != "episode_index" and c not in self.episodes
                              and (pd.api.types.is_numeric_dtype(subset[c]) or pd.api.types.is_bool_dtype(subset[c]))]
        missing = len(set(subset["episode_index"]) - set(self.episodes["episode_index"]))
        if missing:
            print(f"warning: {missing} subset episodes are not in {self.name}")
        subset = subset.assign(subset_rank=np.arange(1, len(subset) + 1))  # the file's own order is the default sort
        self.extra_columns = ["subset_rank"] + self.extra_columns
        self.episodes = self.episodes.merge(subset[["episode_index"] + self.extra_columns], on="episode_index", how="inner")

    def playable(self) -> np.ndarray:
        """True where every camera's mp4 exists. Re-checked per request, so videos copied in while serving show up."""
        ok = np.ones(len(self.episodes), dtype=bool)
        for cam in self.cameras:
            keys = list(zip(self.episodes[f"videos/{cam}/chunk_index"], self.episodes[f"videos/{cam}/file_index"]))
            exists = {k: self.video_path(cam, int(k[0]), int(k[1])).exists() for k in set(keys)}
            ok &= np.array([exists[k] for k in keys], dtype=bool)
        return ok

    # ---- summaries for the browser ----
    def describe(self) -> dict:
        return {
            "name": self.name,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "num_episodes": len(self.episodes),
            "cameras": self.cameras,
            "codecs": sorted({self.info["features"][c].get("info", {}).get("video.codec", "") for c in self.cameras}),
            "joint_names": self.joint_names,
            "tasks": self.tasks,
            "has_recap": self.has_recap,
            "values": [{"name": n, "has_positive": s.positive is not None} for n, s in self.values.items()],
            "subset": self.subset_name,
            "extra_columns": self.extra_columns,
        }

    def episode_table(self) -> dict:
        """Column-oriented episode table (no per-frame data): ~100 bytes per episode."""
        keep = ["episode_index", "length", "seconds", "tasks", "source", "success"]
        keep += [c for c in RECAP_COLUMNS if c in self.episodes and c not in keep]
        keep += [c for c in self.episodes if c.startswith("value_end:")] + self.extra_columns
        table = self.episodes[keep].astype(object).where(self.episodes[keep].notna(), None)
        out = {c: [x.item() if isinstance(x, np.generic) else x for x in table[c]] for c in keep}  # float32 etc. -> JSON
        out["playable"] = self.playable().tolist()
        return out

    # ---- per-episode data (on request) ----
    def row(self, episode: int) -> pd.Series:
        rows = self.episodes.index[self.episodes["episode_index"] == episode]
        if len(rows) == 0:
            raise KeyError(f"episode {episode} not in {self.name}")
        return self.episodes.loc[rows[0]]

    def video_path(self, camera: str, chunk: int, file: int) -> Path:
        return self.root / self.info["video_path"].format(video_key=camera, chunk_index=chunk, file_index=file)

    def video_segments(self, row: pd.Series) -> dict:
        segments = {}
        for cam in self.cameras:
            path = self.video_path(cam, int(row[f"videos/{cam}/chunk_index"]), int(row[f"videos/{cam}/file_index"]))
            segments[cam] = {
                "path": path.relative_to(self.root).as_posix(),
                "available": path.exists(),
                "from": float(row[f"videos/{cam}/from_timestamp"]),
                "to": float(row[f"videos/{cam}/to_timestamp"]),
            }
        return segments

    def _frames(self, row: pd.Series) -> pd.DataFrame:
        path = self.root / self.info["data_path"].format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        columns = [c for c in ("frame_index", "observation.state", "action", "task_index", "subtask_index") if c in pq.read_schema(path).names]
        table = pq.read_table(path, columns=columns, filters=[("episode_index", "==", int(row["episode_index"]))])
        return table.to_pandas().sort_values("frame_index")

    def _recap_progress(self, episode: int) -> list | None:
        """The scene's step counter per frame from recap/frames (only that column: the state vectors are large)."""
        path = self.recap_dir / "frames" / f"episode_{episode:06d}.parquet"
        if not path.exists():
            return None
        return pq.read_table(path, columns=["frame_index", "progress"]).to_pandas().sort_values("frame_index")["progress"].tolist()

    def episode_payload(self, episode: int) -> dict:
        row = self.row(episode)
        frames = self._frames(row)
        start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        payload = {
            "episode_index": episode,
            "length": int(row["length"]),
            "videos": self.video_segments(row),
            "task_index": frames["task_index"].astype(int).tolist() if "task_index" in frames else None,
            "subtask_index": frames["subtask_index"].fillna(-1).astype(int).tolist() if "subtask_index" in frames else None,
            "values": {n: _rounded(s.values[start:stop]) for n, s in self.values.items()},
            "positive": {n: s.positive[start:stop].tolist() for n, s in self.values.items() if s.positive is not None},
            "progress": self._recap_progress(episode),
        }
        for key in ("observation.state", "action"):
            if key in frames:
                payload[key] = [_rounded(col) for col in np.stack(frames[key].to_numpy()).T]
        return payload
