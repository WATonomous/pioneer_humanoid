"""Policies of our own. Call ``register()`` before ``make_policy`` or training (``--policy.type=humanoid_dit`` or ``humanoid_ditx``)."""


def register() -> None:
    import lerobot.policies.factory as factory

    from .dit.configuration_dit import DiTConfig
    from .dit.modeling_dit import DiTPolicy
    from .dit.processor_dit import make_dit_pre_post_processors
    from .ditx.configuration_ditx import DiTXConfig
    from .ditx.modeling_ditx import DiTXPolicy
    from .ditx.processor_ditx import make_ditx_pre_post_processors

    if getattr(factory, "_humanoid_policies_registered", False):
        return
    get_cls, make_pp = factory.get_policy_class, factory.make_pre_post_processors

    def get_policy_class(name):
        return {DiTPolicy.name: DiTPolicy, DiTXPolicy.name: DiTXPolicy}.get(name) or get_cls(name)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        if isinstance(policy_cfg, DiTConfig) and pretrained_path is None:
            return make_dit_pre_post_processors(policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        if isinstance(policy_cfg, DiTXConfig) and pretrained_path is None:
            return make_ditx_pre_post_processors(policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        return make_pp(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    factory.get_policy_class, factory.make_pre_post_processors = get_policy_class, make_pre_post_processors
    factory._humanoid_policies_registered = True
