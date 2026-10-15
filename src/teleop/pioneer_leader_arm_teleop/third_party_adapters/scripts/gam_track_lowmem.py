"""GAM.tools.track within 16 GB: CoTracker3 offline on bounded windows of the episode.

GAM's track_offline gives CoTracker3 (offline) the whole episode in one call. With the detector resident, that does
not fit in 16 GB for our episodes (455-1500 frames): the offline model encodes every frame at once. Two changes, both
only in how the tracker is called:
  * frames go in at CoTracker's own working size (interp_shape 384 x 512, to which it resizes internally anyway);
  * the episode is tracked in windows of CHUNK frames; each window's query points are the previous window's last
    positions (the last visible one for a point that is occluded at the seam).
Points, tracks and visibility come back in GAM's format and full-resolution pixels. Nothing else in GAM changes.

    python gam_track_lowmem.py --config configs/pioneer_peg.yaml [--ep-start 0 --ep-end N]   (run from the GAM repo)
"""
import sys

import cv2
import numpy as np

import GAM.tools.track as T

H, W, CHUNK = 384, 512, 120
_orig = T.track_offline


def track_offline(frames, points, ckpt, device="cuda"):
    import torch

    h, w = frames[0].shape[:2]
    sx, sy = W / w, H / h
    small = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames]
    n, k = len(small), len(points)
    tracks, visible = np.zeros((n, k, 2), np.float32), np.zeros((n, k), bool)
    query, start = [(x * sx, y * sy) for x, y in points], 0
    while start < n:
        end = min(start + CHUNK, n)
        tr, vi = _orig(small[start:end], query, ckpt, device)
        tracks[start:end], visible[start:end] = tr, vi
        last = []
        for j in range(k):                          # the next window starts from each point's last visible position
            seen = np.nonzero(vi[:, j])[0]
            last.append(tuple(tr[seen[-1], j]) if len(seen) else query[j])
        query, start = last, end if end == n else end - 1
        torch.cuda.empty_cache()
    tracks[..., 0] /= sx
    tracks[..., 1] /= sy
    return tracks, visible


T.track_offline = track_offline

if __name__ == "__main__":
    T.main()
