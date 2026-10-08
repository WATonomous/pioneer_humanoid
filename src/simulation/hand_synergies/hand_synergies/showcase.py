"""Stitch the rendered GIFs in out/ into one showcase MP4 (a label in the corner of each segment;
--cards adds a title card before each).

Render the pieces first (see README: pick / actions / mpc / synergies with --gif), then

    pip install imageio imageio-ffmpeg
    python -m hand_synergies.showcase --out out/showcase.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

W, H, FPS = 640, 480, 25
CARD_S = 1.8

# (gif in out/, title, subtitle) -- missing files are skipped
SEGMENTS = (
    ("pc1.gif", "Grasp synergies", "PCA of 2543 stable grasps: PC1, ring + pinky curl (37%)"),
    ("pc2.gif", "Grasp synergies", "PC2: thumb opposition vs thumb curl (24%)"),
    ("pick.gif", "Pick", "arm-like wrist tilted 30 deg, descend, close, lift 10 cm"),
    ("place.gif", "Place", "carry 15 cm, lower, open gradually"),
    ("stack.gif", "Stack", "onto a 5 cm box"),
    ("push.gif", "Push", "backs of the fingers sweep it along the table"),
    ("press.gif", "Press", "point gesture onto a spring button (100% over 100 trials)"),
    ("type_multi.gif", "Type, four fingers", "26 keys; per key the finger needing the least wrist travel"),
    ("type.gif", "Type", "26-key keyboard, index finger, exact up to +-6 mm aiming error"),
    ("gestures.gif", "Gestures", "open, fist, point, thumbs-up, peace, OK"),
    ("mpc_joint.gif", "In-hand: spin a cube", "sampling MPC, no training"),
    ("mpc_yaw.gif", "In-hand: cube to a goal angle", "the Isaac in-hand task, green ghost = goal"),
    ("mpc_roll.gif", "In-hand: roll a ball", "to random targets on the palm"),
)


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def card(title: str, subtitle: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    draw.text((W / 2, H / 2 - 22), title, font=_font(40), fill=(240, 240, 240), anchor="mm")
    draw.text((W / 2, H / 2 + 28), subtitle, font=_font(18), fill=(170, 170, 175), anchor="mm")
    return np.asarray(img)


def gif_frames(path: Path, label: str) -> list[np.ndarray]:
    """Frames resampled to FPS (GIF frame durations honoured), letterboxed to W x H, labelled."""
    from PIL import Image, ImageDraw, ImageSequence

    out, t, clock = [], 0.0, 0.0
    for frame in ImageSequence.Iterator(Image.open(path)):
        dur = frame.info.get("duration", 40) / 1000.0
        img = frame.convert("RGB")
        scale = min(W / img.width, H / img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
        ImageDraw.Draw(canvas).text((W - 12, H - 12), label, font=_font(16), fill=(220, 220, 220), anchor="rd")
        arr = np.asarray(canvas)
        t += dur
        while clock < t:
            out.append(arr)
            clock += 1.0 / FPS
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default="out")
    p.add_argument("--out", default="out/showcase.mp4")
    p.add_argument("--cards", action="store_true", help="title card before each segment")
    args = p.parse_args()
    import imageio.v2 as imageio

    src = Path(args.src)
    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=6, macro_block_size=16)
    if args.cards:
        intro = card("Pioneer hand in MuJoCo", "CPU only, no training -- WATonomous humanoid")
        for _ in range(int(2.5 * FPS)):
            writer.append_data(intro)
    used = 0
    for name, title, subtitle in SEGMENTS:
        path = src / name
        if not path.exists():
            print(f"skip {name} (not rendered)")
            continue
        if args.cards:
            c = card(title, subtitle)
            for _ in range(int(CARD_S * FPS)):
                writer.append_data(c)
        for f in gif_frames(path, title):
            writer.append_data(f)
        used += 1
    writer.close()
    print(f"wrote {args.out} ({used} segments)")


if __name__ == "__main__":
    main()
