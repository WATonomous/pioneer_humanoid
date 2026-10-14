"""Shared data types. Agree on these first; every module depends on them."""
from .frame import Frame
from .keyframe import Keyframe

__all__ = ["Frame", "Keyframe"]
