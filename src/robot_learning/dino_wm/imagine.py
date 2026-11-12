"""Side-by-side video: a new random-motion episode in the sim (left) vs the world model imagining it (right) from
the first 2 frames and the same actions, decoded to pixels.

  MUJOCO_GL=egl python imagine.py <wm.pt> <out.mp4> --seed 40000
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import collect_random  # noqa: E402
from model import Decoder, DinoEncoder, Predictor  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("wm", help="train.py output (wm.pt)")
p.add_argument("out")
p.add_argument("--seed", type=int, required=True, help="use one outside the training seeds (30000-30799)")
p.add_argument("--synthetic-dir", type=Path, default=collect_random.SYNTHETIC)
args = p.parse_args()
collect_random.SYNTHETIC = args.synthetic_dir

dev = "cuda"
ck = torch.load(args.wm)
TOK = ck["tok"]
P, D = Predictor(TOK).to(dev).eval(), Decoder(TOK).to(dev).eval()
P.load_state_dict(ck["P"])
D.load_state_dict(ck["D"])
zm, zs, sm, ss, am, as_ = (ck[k].to(dev) for k in ("zm", "zs", "sm", "ss", "am", "as"))

_, imgs, st, ac, _ = collect_random.episode(args.seed)
x = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255
z = (DinoEncoder(TOK)(x).float() - zm) / zs
S, A = torch.from_numpy(st).to(dev), torch.from_numpy(ac).to(dev)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    zh, s = z[None, :2], ((S[1] - sm) / ss)[None]
    imagined = []
    for t in range(1, len(st) - 1):  # tokens of frames 0..t -> frame t+1
        zn, s, _ = P(zh, s, ((A[t] - am) / as_)[None])
        zh = torch.stack([zh[:, -1], zn], 1)
        imagined.append(D(zn).float())
imagined = (torch.cat(imagined).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
real = (F.interpolate(x[2:], (96, 128), mode="bilinear", antialias=True).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
frames = []
for t, (r, m) in enumerate(zip(real, imagined)):
    c = Image.new("RGB", (2 * 384 + 8, 288 + 28), (24, 26, 32))
    for k, (im, title) in enumerate(((r, "Sim"), (m, "World model imagination"))):
        c.paste(Image.fromarray(im).resize((384, 288), Image.BILINEAR), (k * 392, 28))
        ImageDraw.Draw(c).text((k * 392 + 6, 6), title, font=font, fill=(235, 235, 235))
    ImageDraw.Draw(c).text((2 * 384 - 60, 6), f"{(t + 1) * 0.2:4.1f} s", font=font, fill=(160, 160, 160))
    frames.append(np.asarray(c))
imageio.mimsave(args.out, frames, fps=5, quality=7, macro_block_size=8)
err = np.abs(real.astype(np.int16) - imagined.astype(np.int16)).mean((1, 2, 3))
print(f"saved {args.out}: {len(frames)} frames at 5 fps; "
      f"pixel error /255 at 1 / 3 / 6 / 11 s: {[round(float(err[i]), 1) for i in (4, 14, 29, len(err) - 1)]}")
