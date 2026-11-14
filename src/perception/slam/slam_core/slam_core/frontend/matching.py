"""Descriptor matching."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
# holds the two index arrays so the beypoints line up row by row
class Matches:
    """ Index pairs: features_a[idx_a[k]] matches_b[idx_b[k]] """

    idx_a: np.ndarray  # (M,) int
    idx_b: np.ndarray  # (M,) int

    def __len__(self):
        return len(self.idx_a)

# reuses the extractors model instead of loading xfeat again


class Matcher:
    """LighterGlue on XFeat features. Reuses the extractor's model so XFeat loads once."""

    def __init__(self, extractor):
        self.model = extractor.model
        self.device = extractor.device

    def _pack(self, f, image_size):
        return {"keypoints": torch.from_numpy(f.keypoints).to(self.device),
                "descriptors": torch.from_numpy(f.descriptors).to(self.device),
                "image_size": image_size}

    def match(self, a, b, image_size):
        """a, b: Features. image_size: (width, height). Returns Matches."""
        if len(a.keypoints) == 0 or len(b.keypoints) == 0:
            return Matches(np.empty(0, np.int64), np.empty(0, np.int64))
        res = self.model.match_lighterglue(self._pack(a, image_size), self._pack(b, image_size))
        idx = np.asarray(res[2], dtype=np.int64).reshape(-1, 2)
        return Matches(idx[:, 0], idx[:, 1])
