"""Rollout dashboard: browse thousands of LeRobot episodes / RECAP rollouts in a local web page.

    python tools/rollout_dashboard/dashboard.py datasets/recap/two_r4_all \\
        --values dinov3=outputs/value_dinov3/values.npy

Two tabs: SFT (episode list + search, both cameras, state/action and step instruction per frame) and
RL (success/source/round/policy/seed filters, value functions overlaid on one chart with a time bar).
See tools/rollout_dashboard/README.md.
"""
import argparse
import webbrowser
from pathlib import Path

from dataset import Dataset
from server import Dashboard, make_server


def parse_values(specs: list[str], dataset_names: list[str]) -> dict[str, dict[str, Path]]:
    """`[DATASET:]NAME=PATH` -> {dataset_name: {NAME: PATH}}; without DATASET the first dataset is meant."""
    out: dict[str, dict[str, Path]] = {name: {} for name in dataset_names}
    for spec in specs:
        key, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--values expects NAME=PATH, got {spec!r}")
        dataset, _, name = key.rpartition(":")
        dataset = dataset or dataset_names[0]
        if dataset not in out:
            raise SystemExit(f"--values {spec!r}: no dataset named {dataset!r} (have {dataset_names})")
        out[dataset][name] = Path(path).expanduser()
    return out


def parse_subsets(specs: list[str], dataset_names: list[str]) -> dict[str, Path]:
    """`[DATASET:]PATH` -> {dataset_name: PATH}; without a known DATASET prefix the first dataset is meant."""
    out = {}
    for spec in specs:
        prefix, sep, rest = spec.partition(":")
        dataset, path = (prefix, rest) if sep and prefix in dataset_names else (dataset_names[0], spec)
        out[dataset] = Path(path).expanduser()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", type=Path, help="LeRobot v3 dataset folders (with meta/, data/, videos/; recap/ optional)")
    parser.add_argument("--values", action="append", default=[], metavar="[DATASET:]NAME=PATH",
                        help="extra value function: a values.npy (one float per dataset frame) or a labels parquet "
                             "(index, value[, positive]). Repeatable. recap/labels_r*.parquet are added automatically.")
    parser.add_argument("--episodes", action="append", default=[], metavar="[DATASET:]PATH",
                        help="episode subset: parquet/CSV with an episode_index column (or index). Only those episodes are "
                             "listed, in the file's row order; its other numeric columns become sortable list columns.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to reach it from another machine)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--clip-dir", type=Path,
                        help="where to cache H.264 clips for browsers without AV1 (default: a temp dir, deleted on exit)")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args()

    names = [p.resolve().name for p in args.datasets]
    values = parse_values(args.values, names)
    subsets = parse_subsets(args.episodes, names)
    datasets = []
    for path, name in zip(args.datasets, names):
        dataset = Dataset(path, values[name], subsets.get(name))
        print(f"{name}: {len(dataset.episodes)} episodes, {dataset.total_frames} frames, cameras {dataset.cameras}, "
              f"values {list(dataset.values)}")
        datasets.append(dataset)

    app = Dashboard(datasets, args.clip_dir)
    server = make_server(app, args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"serving {url}  (Ctrl-C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
