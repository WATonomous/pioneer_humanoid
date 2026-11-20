"""Loop closure: place recognition, verification, false-loop defenses. ROS-free; see detector.py."""
from .consistency import best_consistent_set, consistent, cycle_error2
from .detector import LoopClosureDetector, LoopParams
from .gate import CHI2_6_999, plausibility
from .graph import KeyframeGraph
from .retrieval import BagOfWords, DescriptorDatabase, Vlad
from .types import LoopConstraint, LoopUpdate, Rejection, Verification
from .verification import PnPVerifier, loop_covariance

__all__ = [
    "LoopClosureDetector", "LoopParams", "LoopConstraint", "LoopUpdate", "Rejection", "Verification",
    "KeyframeGraph", "DescriptorDatabase", "Vlad", "BagOfWords", "PnPVerifier", "loop_covariance",
    "plausibility", "consistent", "cycle_error2", "best_consistent_set", "CHI2_6_999",
]
