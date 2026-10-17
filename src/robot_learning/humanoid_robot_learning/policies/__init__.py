"""Policies the installed LeRobot (0.4.x) doesn't ship. Call ``register()`` before ``make_policy`` / training."""


def register() -> None:
    """Add these policy types to LeRobot's factory (``--policy.type=multi_task_dit``, loading their checkpoints)."""
    import lerobot.policies.factory as factory

    from .multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
    from .multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
    from .multi_task_dit.processor_multi_task_dit import make_multi_task_dit_pre_post_processors

    if getattr(factory, "_humanoid_policies_registered", False):
        return
    get_cls, make_pp = factory.get_policy_class, factory.make_pre_post_processors

    def get_policy_class(name):
        return MultiTaskDiTPolicy if name == "multi_task_dit" else get_cls(name)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        if isinstance(policy_cfg, MultiTaskDiTConfig) and pretrained_path is None:
            return make_multi_task_dit_pre_post_processors(policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        return make_pp(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    factory.get_policy_class, factory.make_pre_post_processors = get_policy_class, make_pre_post_processors
    factory._humanoid_policies_registered = True
