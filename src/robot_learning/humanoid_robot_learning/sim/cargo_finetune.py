"""Warm-start cargo ACT training while retaining the checkpoint's normalization.

LeRobot's generic fine-tuner overrides saved normalizers with the new dataset's
statistics. Keeping the old affine transforms preserves the meaning of the
pretrained input/output weights during this short domain-randomization update.
"""


def preserve_normalization(kwargs):
    result = dict(kwargs)
    result.pop("dataset_stats", None)
    for key, step in (("preprocessor_overrides", "normalizer_processor"),
                      ("postprocessor_overrides", "unnormalizer_processor")):
        overrides = dict(result.get(key) or {})
        overrides.pop(step, None)
        result[key] = overrides
    return result


def main():
    from humanoid_robot_learning.policies import register
    register()
    import lerobot.scripts.lerobot_train as trainer

    original = trainer.make_pre_post_processors

    def make_processors(*args, **kwargs):
        if kwargs.get("pretrained_path") is None:
            raise ValueError("Cargo fine-tuning requires --policy.path to a complete checkpoint")
        return original(*args, **preserve_normalization(kwargs))

    trainer.make_pre_post_processors = make_processors
    trainer.main()


if __name__ == "__main__":
    main()
