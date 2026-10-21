"""Procedural textures for the workshop scenes (MuJoCo 2D textures, PNG): concrete floor, butcher-block wood,
pegboard, painted wall, dark rubber mat, brushed aluminium, foam.

    python make_textures.py          # writes textures/*.png
"""
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "textures")
rng = np.random.default_rng(7)


def noise(n, scales=(256, 64, 16, 4), weights=(0.5, 0.25, 0.15, 0.1)):
    """Smooth value noise, tileable, in [0, 1]."""
    out = np.zeros((n, n))
    for s, w in zip(scales, weights):
        k = max(2, n // s)
        g = rng.random((k, k))
        g = np.kron(g, np.ones((n // k + 1, n // k + 1)))[:n, :n]
        # blur a little for smoothness (box blur, wraps)
        r = max(1, n // k // 2)
        for ax in (0, 1):
            g = sum(np.roll(g, i, axis=ax) for i in range(-r, r + 1)) / (2 * r + 1)
        out += w * g
    out -= out.min()
    return out / out.max()


def save(name, rgb):
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).save(os.path.join(OUT, name + ".png"))


def concrete(n=1024):
    a = noise(n)
    fine = rng.random((n, n)) * 0.06
    v = 0.50 + 0.12 * (a - 0.5) + fine - 0.03
    spots = rng.random((n, n)) < 0.002
    v[spots] -= 0.12
    rgb = np.stack([v * 1.0, v * 0.99, v * 0.96], -1)
    save("concrete", rgb)


def butcher_block(n=1024, strips=12):
    y = np.arange(n)
    rgb = np.zeros((n, n, 3))
    w = n // strips
    for i in range(strips):
        tone = rng.uniform(-0.06, 0.06)
        base = np.array([0.72, 0.55, 0.36]) + tone
        xs = np.arange(n)[:, None]
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.05, 0.09)
        grain = 0.5 + 0.5 * np.sin(freq * xs + 3 * np.sin(0.004 * y[None, :] * rng.uniform(0.5, 1.5)) + phase)
        g = 0.88 + 0.12 * grain ** 3 + 0.03 * (rng.random((n, n)) - 0.5)
        sl = slice(i * w, (i + 1) * w)
        rgb[sl, :, :] = (base[None, None, :] * g[sl, :, None])
        rgb[i * w, :, :] *= 0.8                                   # the glue line
    save("butcher_block", rgb)


def pegboard(n=1024, pitch=32):
    a = noise(n, scales=(128, 32), weights=(0.6, 0.4))
    v = 0.66 + 0.06 * (a - 0.5)
    rgb = np.stack([v * 1.0, v * 0.86, v * 0.68], -1)
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.hypot((xx % pitch) - pitch / 2, (yy % pitch) - pitch / 2)
    hole = d < pitch * 0.13
    rgb[hole] = [0.13, 0.11, 0.09]
    rim = (d >= pitch * 0.13) & (d < pitch * 0.18)
    rgb[rim] *= 0.85
    save("pegboard", rgb)


def wall(n=512):
    a = noise(n, scales=(128, 16), weights=(0.7, 0.3))
    v = 0.80 + 0.04 * (a - 0.5)
    save("wall", np.stack([v, v * 0.985, v * 0.96], -1))


def rubber(n=512):
    v = 0.12 + 0.03 * rng.random((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    ribs = ((xx // 6) % 2 == 0)
    v = v + 0.02 * ribs
    save("rubber_mat", np.stack([v, v, v * 1.05], -1))


def brushed(n=512):
    streak = rng.random((1, n))
    streak = np.repeat(streak, n, 0)
    for _ in range(3):
        streak = (streak + np.roll(streak, 1, 1) + np.roll(streak, -1, 1)) / 3
    v = 0.70 + 0.10 * (streak - 0.5) + 0.03 * rng.random((n, n))
    save("brushed_alu", np.stack([v, v * 1.01, v * 1.03], -1))


def foam(n=512):
    v = 0.30 + 0.05 * rng.random((n, n)) + 0.04 * (noise(n, scales=(64, 8), weights=(0.5, 0.5)) - 0.5)
    save("foam", np.stack([v, v, v * 1.02], -1))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (concrete, butcher_block, pegboard, wall, rubber, brushed, foam):
        f()
        print(f.__name__)
