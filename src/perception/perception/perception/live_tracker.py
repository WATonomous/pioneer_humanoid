"""Live TrackNet: push one camera frame at a time, get that frame's shuttle hotspots back.

No ROS in here, so it can be tested on recorded clips (see the TracknetV3 repo, autolabel/).

Per frame, all on the GPU:
  1. shrink the frame to 512x288 (bicubic with antialiasing, matching the resize used in training);
  2. background = per-pixel median of the last 8 frames, so moving things stand out;
  3. TrackNet on [background, 8 frames] -> heatmap of the newest frame (past frames only, no delay);
  4. hotspots = local peaks of the heatmap, strongest first, with a sub-pixel centre.

    tracker = LiveTracker('models/TrackNet_best.pt')
    hotspots = tracker.push(frame_bgr)    # [(u, v, score), ...] in pixels of the input frame;
                                          # None until 8 frames have arrived
"""
import torch
import torch.nn.functional as F

from perception.tracknet_model import TrackNet, WIDTH, HEIGHT, SEQ_LEN


class LiveTracker:
    def __init__(self, model_path, top_k=3, min_score=0.05, peak_size=7):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = TrackNet().to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True)['model'])
        self.model.to(self.dtype)
        self.top_k, self.min_score, self.peak_size = top_k, min_score, peak_size
        self.frames = torch.zeros((SEQ_LEN, 3, HEIGHT, WIDTH), dtype=torch.uint8, device=self.device)
        self.count = 0

    @torch.no_grad()
    def push(self, frame_bgr):
        # 1. Shrink to the model's input size (BGR -> RGB)
        img = torch.from_numpy(frame_bgr).to(self.device).permute(2, 0, 1).flip(0)[None].float()
        small = F.interpolate(img, (HEIGHT, WIDTH), mode='bicubic', antialias=True, align_corners=False)
        self.frames[self.count % SEQ_LEN] = small[0].round().clamp(0, 255).byte()
        self.count += 1
        if self.count < SEQ_LEN:
            return None

        # 2. Last 8 frames, oldest first, and their median background
        oldest = self.count % SEQ_LEN
        window = torch.cat([self.frames[oldest:], self.frames[:oldest]])
        srt = torch.sort(window, dim=0).values
        background = torch.floor((srt[3].float() + srt[4].float()) / 2)

        # 3. TrackNet; keep the heatmap of the newest frame
        x = torch.cat([background[None], window.float()]) / 255.
        heatmap = self.model(x.reshape(1, 27, HEIGHT, WIDTH).to(self.dtype))[0, -1].float()

        # 4. Hotspots, scaled back to the input frame's pixels
        scale_u, scale_v = frame_bgr.shape[1] / WIDTH, frame_bgr.shape[0] / HEIGHT
        return [(u * scale_u, v * scale_v, s) for u, v, s in self._peaks(heatmap)]

    def _peaks(self, hm):
        """Local maxima above min_score, one per blob, centre = weighted mean of the 5x5 patch."""
        n = self.peak_size
        is_peak = (hm == F.max_pool2d(hm[None, None], n, stride=1, padding=n // 2)[0, 0]) & (hm > self.min_score)
        ys, xs = torch.nonzero(is_peak, as_tuple=True)
        order = torch.argsort(hm[ys, xs], descending=True)[:64]
        ys, xs = ys[order], xs[order]

        d = torch.arange(-2, 3, device=hm.device)
        py = (ys[:, None, None] + d[None, :, None]).expand(-1, 5, 5)
        px = (xs[:, None, None] + d[None, None, :]).expand(-1, 5, 5)
        patch = F.pad(hm, (2, 2, 2, 2))[py + 2, px + 2]                 # zeros outside the image
        total = patch.sum((1, 2))
        cu, cv = (patch * px).sum((1, 2)) / total, (patch * py).sum((1, 2)) / total
        rows = torch.stack([ys.float(), xs.float(), cu, cv, hm[ys, xs]], 1).cpu().tolist()

        kept = []
        for y, x, u, v, s in rows:
            # A saturated blob can have several equal maxima: keep only one
            if all(abs(y - ky) >= n or abs(x - kx) >= n for ky, kx, *_ in kept):
                kept.append((y, x, u, v, s))
            if len(kept) == self.top_k:
                break
        return [(u, v, s) for _, _, u, v, s in kept]
