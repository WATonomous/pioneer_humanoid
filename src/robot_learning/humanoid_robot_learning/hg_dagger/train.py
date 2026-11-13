# Refine a saved LeRobot policy while keeping its normalization.
from pathlib import Path
import sys

from .learning import checkpoint_path, load_contract


# Keep checkpoint normalization instead of replacing it with dataset statistics.
def preserve_normalization(kwargs):
    result = dict(kwargs)
    result.pop("dataset_stats", None)
    for key, step in (("preprocessor_overrides", "normalizer_processor"),
                      ("postprocessor_overrides", "unnormalizer_processor")):
        overrides = dict(result.get(key) or {})
        overrides.pop(step, None)
        result[key] = overrides
    return result


# Check compatibility and refine the policy into a new output directory.
def main():
    from importlib.metadata import version
    if not version("lerobot").startswith("0.4."):
        raise RuntimeError("Install humanoid-robot-learning[hg-dagger] (LeRobot 0.4)")
    from lerobot.configs.policies import PreTrainedConfig
    import lerobot.scripts.lerobot_train as trainer

    # Read one required LeRobot command-line flag.
    def flag(name):
        prefix = f"--{name}="
        matches = [a[len(prefix):] for a in sys.argv[1:] if a.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Supply exactly one --{name}=... argument")
        return matches[0]

    parent = checkpoint_path(flag("policy.path"))
    root = Path(flag("dataset.root"))
    output = Path(flag("output_dir"))
    contract = load_contract(root)
    if load_contract(parent) != contract:
        raise ValueError("Parent checkpoint and aggregate contracts differ")
    if output.exists():
        raise FileExistsError("Use a new output_dir for every aggregation round")
    if not (root / "aggregation.json").is_file():
        raise ValueError("Train on a completed aggregate of original demos and corrections")
    # Check the saved action size before updating any weights.
    cfg = PreTrainedConfig.from_pretrained(parent)
    if tuple(cfg.output_features["action"].shape) != (len(contract.action_names),):
        raise ValueError("Checkpoint actions differ from contract")
    if any(a.startswith("--policy.push_to_hub=") and a != "--policy.push_to_hub=false" for a in sys.argv[1:]):
        raise ValueError("Upload model artifacts separately; training does not publish")
    if not any(a.startswith("--policy.push_to_hub=") for a in sys.argv[1:]):
        sys.argv.append("--policy.push_to_hub=false")
    original = trainer.make_pre_post_processors

    # Load the saved processors without replacing their normalization.
    def make_processors(*args, **kwargs):
        if kwargs.get("pretrained_path") is None:
            raise ValueError("A complete pretrained checkpoint is required")
        return original(*args, **preserve_normalization(kwargs))

    trainer.make_pre_post_processors = make_processors
    try:
        trainer.main()
    finally:
        trainer.make_pre_post_processors = original
        # Attach the contract to saved checkpoints even if training is interrupted.
        for config in output.glob("checkpoints/*/pretrained_model/config.json"):
            (config.parent / "contract.json").write_text(contract.json() + "\n")


if __name__ == "__main__":
    main()
