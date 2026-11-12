"""Checkpoint adapter; imports no simulator or MuJoCo dependency."""
import torch


class CargoPolicy:
    def __init__(self, path, device, contract):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from humanoid_robot_learning.policies import register

        register()
        self.device = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        # A saved ACT checkpoint already includes its complete vision backbone.
        # Avoid downloading initialization weights before loading those tensors.
        if cfg.type == "act":
            cfg.pretrained_backbone_weights = None
        for name in contract["cameras"]:
            feature = cfg.image_features.get(f"observation.images.{name}")
            if feature is None or tuple(feature.shape) != (3, contract["height"], contract["width"]):
                raise ValueError("Checkpoint cameras do not match the cargo training contract")
        if tuple(cfg.output_features["action"].shape) != (7,):
            raise ValueError("Cargo runner requires a seven-action left-arm checkpoint")
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})

    def reset(self):
        self.policy.reset()
        for processor in (self.pre, self.post):
            if hasattr(processor, "reset"):
                processor.reset()

    def action(self, obs, task):
        from lerobot.utils.control_utils import predict_action
        with torch.inference_mode():
            return predict_action(obs, self.policy, self.device, self.pre, self.post,
                                  use_amp=False, task=task).reshape(-1).cpu().numpy()
