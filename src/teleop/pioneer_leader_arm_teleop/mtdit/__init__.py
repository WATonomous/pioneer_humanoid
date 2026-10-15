"""Multi-Task DiT policy (Bryson Jones; LeRobot 0.6 ``policies/multi_task_dit``), ported to the LeRobot 0.4.x installed here.

The configuration, model and README are the upstream files with only their imports changed; the processor is rebuilt from
0.4.x steps. ``install()`` registers the policy type with LeRobot's factory so ``lerobot-train --policy.type=multi_task_dit``
and ``make_policy`` / ``make_pre_post_processors`` work.
"""
from .configuration_multi_task_dit import MultiTaskDiTConfig


def install() -> None:
    import lerobot.policies.factory as factory

    from .modeling_multi_task_dit import MultiTaskDiTPolicy
    from .processor_multi_task_dit import make_multi_task_dit_pre_post_processors

    if getattr(factory, "_mtdit_installed", False):
        return
    get_cls, make_pp = factory.get_policy_class, factory.make_pre_post_processors

    def get_policy_class(name):
        return MultiTaskDiTPolicy if name == "multi_task_dit" else get_cls(name)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        if isinstance(policy_cfg, MultiTaskDiTConfig) and pretrained_path is None:
            return make_multi_task_dit_pre_post_processors(policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        return make_pp(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    factory.get_policy_class, factory.make_pre_post_processors = get_policy_class, make_pre_post_processors
    factory._mtdit_installed = True
