"""Optional fail-fast gradient clipping for the single-process PPO launcher."""

from contextlib import contextmanager
from functools import wraps
from inspect import signature

import torch


@contextmanager
def strict_gradient_clipping(enabled: bool = True):
    """Reject non-finite norms before PPO steps; always restore Torch's function.

    This opt-in guard changes only failure handling, not finite gradient values,
    rewards, or optimizer settings. It is scoped to one training process.
    """
    if not enabled:
        yield
        return

    original = torch.nn.utils.clip_grad_norm_
    original_signature = signature(original)

    @wraps(original)
    def strict_clip(*args, **kwargs):
        bound = original_signature.bind(*args, **kwargs)
        bound.arguments["error_if_nonfinite"] = True
        return original(*bound.args, **bound.kwargs)

    torch.nn.utils.clip_grad_norm_ = strict_clip
    try:
        yield
    finally:
        torch.nn.utils.clip_grad_norm_ = original
