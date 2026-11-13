"""Policies of our own. Call ``register()`` before ``make_policy`` or training (``--policy.type=humanoid_dit`` or ``pioneer1``)."""


def register() -> None:
    import lerobot.policies.factory as factory

    from .dit.configuration_dit import DiTConfig
    from .dit.modeling_dit import DiTPolicy
    from .dit.processor_dit import make_dit_pre_post_processors
    from .pioneer1.configuration_pioneer1 import Pioneer1Config
    from .pioneer1.modeling_pioneer1 import Pioneer1Policy
    from .pioneer1.processor_pioneer1 import make_pioneer1_pre_post_processors

    if getattr(factory, "_humanoid_policies_registered", False):
        return
    get_cls, make_pp = factory.get_policy_class, factory.make_pre_post_processors

    def get_policy_class(name):
        return {DiTPolicy.name: DiTPolicy, Pioneer1Policy.name: Pioneer1Policy}.get(name) or get_cls(name)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        if isinstance(policy_cfg, DiTConfig) and pretrained_path is None:
            return make_dit_pre_post_processors(policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        if isinstance(policy_cfg, Pioneer1Config) and pretrained_path is None:
            return make_pioneer1_pre_post_processors(policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        return make_pp(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    factory.get_policy_class, factory.make_pre_post_processors = get_policy_class, make_pre_post_processors
    factory._humanoid_policies_registered = True
