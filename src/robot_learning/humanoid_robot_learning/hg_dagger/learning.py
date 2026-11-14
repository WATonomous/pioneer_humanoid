# Connect saved policies and demonstration datasets to LeRobot 0.4.
from pathlib import Path
import json

import numpy as np

from .core import Contract
from .storage import open_store


# Read the data contract stored beside a dataset or checkpoint.
def load_contract(root):
    return Contract.from_json((Path(root) / "contract.json").read_text())


# Use a local checkpoint or download one from Hugging Face.
def checkpoint_path(path):
    if Path(path).is_dir():
        return Path(path)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(path)))


# Check data shapes and attach the caller's declared scene and units.
def bind_contract(dataset_root, policy_root, contract):
    root = Path(dataset_root)
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != contract.fps or info.get("total_episodes", 0) < 1:
        raise ValueError("Dataset needs saved episodes at the configured fps")
    features = info["features"]
    for key, names in (("observation.state", contract.state_names), ("action", contract.action_names)):
        feature = features[key]
        if tuple(feature["shape"]) != (len(names),) or tuple(feature["names"]) != names:
            raise ValueError(f"Dataset {key} joint order differs from the configured robot")
    images = {k: tuple(v["shape"]) for k, v in features.items() if k.startswith("observation.images.")}
    if images != contract.images:
        raise ValueError("Dataset cameras differ from configuration")
    targets = [root]
    if policy_root:
        policy = Path(policy_root)
        cfg = json.loads((policy / "config.json").read_text())
        expected = {"observation.state": (len(contract.state_names),),
                    **{k: (3, s[0], s[1]) for k, s in contract.images.items()}}
        if {k: tuple(v["shape"]) for k, v in cfg["input_features"].items()} != expected:
            raise ValueError("Policy observations differ from configured dataset")
        if tuple(cfg["output_features"]["action"]["shape"]) != (len(contract.action_names),):
            raise ValueError("Policy actions differ from configured dataset")
        targets.append(policy)
    for target in targets:
        file = target / "contract.json"
        if file.exists() and load_contract(target) != contract:
            raise ValueError(f"Refusing to replace an existing different contract: {target}")
    for target in targets:
        (target / "contract.json").write_text(contract.json() + "\n")


# Use a saved LeRobot policy with its original observation and action processors.
class LeRobotPolicy:
    # Load a compatible policy and its saved normalization processors.
    def __init__(self, path, contract, device="cuda"):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        path = checkpoint_path(path)
        if load_contract(path) != contract:
            raise ValueError("Checkpoint contract differs from scene (robot/joints/units/cameras/fps)")
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        if cfg.type == "act":
            cfg.pretrained_backbone_weights = None
        expected = {"observation.state": (len(contract.state_names),)}
        expected.update({k: (3, shape[0], shape[1]) for k, shape in contract.images.items()})
        if {k: tuple(v.shape) for k, v in cfg.input_features.items()} != expected:
            raise ValueError("Policy observation features differ from contract")
        if tuple(cfg.output_features["action"].shape) != (len(contract.action_names),):
            raise ValueError("Policy action shape differs from contract")
        self.device = torch.device(device)
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})

    # Clear queued actions and processor state before resuming control.
    def reset(self):
        self.policy.reset()
        for processor in (self.pre, self.post):
            if hasattr(processor, "reset"):
                processor.reset()

    # Predict an action without tracking gradients.
    def action(self, observation, task):
        import torch
        from lerobot.utils.control_utils import predict_action
        with torch.inference_mode():
            return predict_action(observation, self.policy, self.device, self.pre, self.post,
                                  use_amp=False, task=task).reshape(-1).cpu().numpy()


# Build a fresh dataset from original demos and accepted human segments.
def aggregate(base_roots, correction_roots, output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sources = [Path(p).resolve() for p in [*base_roots, *correction_roots]]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Supply nonempty, distinct input directories")
    contract = load_contract(sources[0])
    if any(load_contract(p) != contract for p in sources):
        raise ValueError("All source contracts must match exactly")
    output = Path(output).resolve()
    if output.exists() or any(p == output or p in output.parents for p in sources):
        raise ValueError("Output must be new and outside source directories")
    # Open the original demonstrations only after all source contracts match.
    datasets = []
    for p in base_roots:
        if not (Path(p) / "meta/info.json").is_file():
            raise ValueError(f"Expected an existing local LeRobot dataset: {p}")
        ds = LeRobotDataset("local/hg_dagger", root=Path(p), video_backend="pyav")
        if ds.fps != contract.fps:
            raise ValueError(f"FPS mismatch: {p}")
        datasets.append((Path(p), ds))
    stores = [(Path(p), open_store(p)) for p in correction_roots]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(contract.state_names),), "names": list(contract.state_names)},
        "action": {"dtype": "float32", "shape": (len(contract.action_names),), "names": list(contract.action_names)},
        **{k: {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]}
           for k, shape in contract.images.items()},
    }
    # Write a fresh dataset to avoid append and merge indexing problems.
    writer = LeRobotDataset.create("local/hg_dagger", root=output, fps=contract.fps,
                                   features=features, robot_type=contract.robot, use_videos=bool(contract.images),
                                   image_writer_threads=2)
    manifest = []

    # Check one frame and pass it to the dataset writer.
    def add(frame, task):
        obs = {k: frame[k] for k in {"observation.state", *contract.images}}
        contract.check_observation(obs)
        writer.add_frame({**obs, "action": contract.action(frame["action"]), "task": task})

    # Save one episode and record where its frames came from.
    def finish(source, count, kind):
        writer.save_episode(parallel_encoding=False)
        manifest.append({"source": str(source.resolve()), "frames": count, "kind": kind})

    # Copy original demonstrations while keeping their episode boundaries.
    for path, ds in datasets:
        previous, count = None, 0
        for sample in ds:
            episode = int(sample["episode_index"])
            if previous is not None and episode != previous:
                finish(path, count, "original")
                count = 0
            frame = {k: sample[k].numpy() for k in ("observation.state", "action")}
            for k in contract.images:
                image = sample[k].numpy()
                frame[k] = np.rint(np.clip(image.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            add(frame, sample["task"])
            previous, count = episode, count + 1
        if count:
            finish(path, count, "original")
    # Add each accepted human intervention as its own episode.
    for path, store in stores:
        for frames, metadata in store.episodes():
            for frame in frames:
                add(frame, metadata["task"])
            finish(path, len(frames), "human_intervention")
    writer.finalize()
    if not manifest:
        raise ValueError("No demonstrations found; cannot train an empty aggregate")
    (output / "contract.json").write_text(contract.json() + "\n")
    (output / "aggregation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"episodes": len(manifest), "frames": sum(x["frames"] for x in manifest)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine original demos with accepted HG-DAgger corrections")
    parser.add_argument("--base", nargs="+", required=True, help="Original LeRobot dataset directories")
    parser.add_argument("--corrections", nargs="+", required=True, help="Accepted correction directories")
    parser.add_argument("--output", required=True, help="New aggregate dataset directory")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.base, args.corrections, args.output)))
