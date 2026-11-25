# rollout_dashboard

A local web page for browsing large sets of robot episodes: SFT demo datasets, ACT/policy training
data and RL (RECAP) rollouts. It reads LeRobot v3 datasets directly (pandas/pyarrow + the dataset's
mp4s) and needs no LeRobot or repo imports.

- **SFT tab**: episode list with search (episode / task / policy) and a length filter. Shows both
  cameras side by side, the current step instruction, a coloured strip of the episode's instructions
  (click one to jump to it) and per-joint state (solid) vs action (dashed).
- **RL tab**: adds filters for outcome (success / failure), source (rollout / correction / demo), round,
  policy and seed range. Every value function in the dataset gets a checkbox, and the checked ones are
  overlaid on one chart with a time bar that follows the video. Options: the scene's step changes
  (`recap/frames` `progress`) as dashed marks, and a labels file's `positive` frames as a shaded band.
  Table columns show each value source's value at the last frame, so sorting by it finds stalled failures.

Click or drag on either chart to seek. Keys: `space` play/pause, `←/→` one frame, `shift+←/→` one second,
`↑/↓` previous/next episode in the filtered list. The URL keeps the dataset, tab and episode, so a link
reopens the same view.

## Run

Needs `numpy`, `pandas` and `pyarrow` (any LeRobot env has them), plus `ffmpeg` only for browsers
without AV1 support.

```bash
python tools/rollout_dashboard/dashboard.py datasets/recap/two_r4_all \
    --values dinov3=outputs/value_dinov3/values.npy --open
```

Then open http://127.0.0.1:8765/. `--help` lists all options.

- **Datasets**: pass one or more dataset folders and switch between them in the header.
- **Value sources**: `recap/labels_r*.parquet` files are picked up automatically (their `value`, plus
  `positive`). Add others with `--values [DATASET:]NAME=PATH`, either a `values.npy` with one float per
  dataset frame in dataset-index order (what the RECAP image value function's `--values` writes) or a labels-style parquet
  (`index, value[, positive]`). Without `DATASET:` the value goes to the first dataset.
- **Episodes without a `recap/episodes.parquet` row** count as successful demos. Datasets with no
  `recap/` folder at all simply have no RL columns.

### Data on the cluster

Datasets are large (`two_r4_all` is 13 GB, mostly video), so you can also run the dashboard where the
data lives and forward the port:

```bash
ssh -L 8765:127.0.0.1:8765 <cluster-host>   # then, on the cluster, inside an env with pandas/pyarrow:
python tools/rollout_dashboard/dashboard.py /path/on/cluster/datasets/recap/two_r4_all
```

A partial local copy works too (`meta/`, `data/`, `recap/` and any subset of `videos/`). Episodes whose
mp4 is missing show a placeholder instead of video.

## How it scales

At startup the dashboard reads only metadata: the episode table and one float32 per frame for each value
source (about 5 MB per million frames). `two_r4_all` (1460 episodes, 1.22 M frames) loads in 0.1 s with
230 MB peak RSS. Per-episode data (state, action, task, values, labels, step counter) is read on
selection with a filtered parquet read (about 25 ms) and cached for 64 episodes. Only the `progress`
column of `recap/frames` is read, never the state vectors. The episode list is virtualised and filtering
runs in the browser, so 10k+ episodes stay responsive.

Video is not decoded on the server. The page seeks an HTML5 `<video>` into the shared
`videos/<camera>/chunk-*/file-*.mp4`, served with HTTP Range, at the episode's `from_timestamp`, and the
first camera acts as the clock. If the browser can't decode the codec (AV1 in older Safari), the page
requests `/clip/...`, and the server cuts that episode with ffmpeg into a cached H.264 clip (about 1 MB
per 90 s episode, a few seconds the first time). The cache is a temp dir deleted on exit unless
`--clip-dir` is given.

## Files

| File | What |
|---|---|
| `dashboard.py` | CLI entry point (`argparse`) |
| `dataset.py` | LeRobot v3 + RECAP sidecar reader: episode table, value sources, per-episode payload |
| `server.py` | stdlib HTTP server: JSON API, ranged video, ffmpeg clip fallback |
| `static/` | page: `app.js` (player, charts wiring), `table.js` (filters, virtual list), `chart.js` (canvas line chart) |
