"""Video of a scripted demo, in two passes: the demo records the simulation state at the video rate (``Tape``),
then ``render_tape`` draws the frames in parallel worker processes (CPU rendering is ~0.5 s a 1280x720 frame with
shadows, the physics ~0.06 s per simulated second) and joins them into one MP4 with a caption bar.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Shot:
    """A free camera: lookat, distance, azimuth, elevation (MuJoCo's)."""
    lookat: tuple
    distance: float
    azimuth: float
    elevation: float

    def blend(self, other: "Shot", s: float) -> "Shot":
        s = s * s * (3 - 2 * s)
        da = (other.azimuth - self.azimuth + 180) % 360 - 180
        return Shot(tuple(np.add(self.lookat, s * (np.subtract(other.lookat, self.lookat)))),
                    self.distance + s * (other.distance - self.distance), self.azimuth + s * da,
                    self.elevation + s * (other.elevation - self.elevation))


@dataclass
class Tape:
    """Simulation states sampled at ``fps``: ``WholeBody(on_frame=tape.frame)``."""
    fps: float = 30.0
    t: list = field(default_factory=list)
    qpos: list = field(default_factory=list)
    caption: list = field(default_factory=list)
    focus: list = field(default_factory=list)      # where the action is (for the camera), per frame
    caption_fn: object = None
    focus_fn: object = None

    def frame(self, wb) -> None:
        self.t.append(wb.t)
        self.qpos.append(wb.d.qpos.copy())
        self.caption.append(self.caption_fn() if self.caption_fn else ("", ""))
        f = None if self.focus_fn is None else self.focus_fn()
        self.focus.append(None if f is None else np.asarray(f, float))

    def save(self, path: str) -> None:
        np.savez_compressed(path, t=np.array(self.t), qpos=np.array(self.qpos),
                            caption=np.array(self.caption, dtype=object), fps=self.fps,
                            focus=np.array([f if f is not None else [np.nan] * 3 for f in self.focus]))

    @classmethod
    def load(cls, path: str) -> "Tape":
        z = np.load(path, allow_pickle=True)
        tape = cls(float(z["fps"]))
        tape.t, tape.qpos, tape.caption = list(z["t"]), list(z["qpos"]), [tuple(c) for c in z["caption"]]
        tape.focus = list(z["focus"])
        return tape


def _font(size):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _render_chunk(args):
    scene, tape_path, shots_mod, frames, out, size, title, inset = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("LP_NUM_THREADS", "1")     # one llvmpipe thread per worker: the workers are the parallelism
    import importlib

    import mujoco
    from PIL import Image, ImageDraw

    from . import make_model
    model = make_model(scene)
    data = mujoco.MjData(model)
    tape = Tape.load(tape_path)
    shot_fn = getattr(importlib.import_module(shots_mod[0]), shots_mod[1])
    w, h = size
    r = mujoco.Renderer(model, h, w)
    r_in = mujoco.Renderer(model, h // 4, w // 4) if inset else None
    cam = mujoco.MjvCamera()
    font_t, font_s = _font(max(16, h // 28)), _font(max(14, h // 40))
    ffmpeg = shutil.which("ffmpeg")
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                             "-r", str(tape.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                             "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in frames:
        data.qpos[:] = tape.qpos[i]
        mujoco.mj_forward(model, data)
        s = shot_fn(tape.t[i], tape, i)
        cam.lookat[:] = s.lookat
        cam.distance, cam.azimuth, cam.elevation = s.distance, s.azimuth, s.elevation
        r.update_scene(data, cam)
        img = r.render().copy()
        if r_in is not None:
            r_in.update_scene(data, inset)
            sub = r_in.render()
            hh, ww = sub.shape[:2]
            y0, x0 = h - hh - 18, w - ww - 18
            img[y0 - 3:y0 + hh + 3, x0 - 3:x0 + ww + 3] = 30
            img[y0:y0 + hh, x0:x0 + ww] = sub
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")
        cap_title, step = tape.caption[i]
        dr.rectangle([0, 0, w, int(h * 0.105)], fill=(18, 20, 24, 175))
        dr.text((22, 12), title, font=font_t, fill=(245, 245, 245, 255))
        dr.text((22, 14 + h // 26), step or cap_title, font=font_s, fill=(250, 176, 80, 255))
        dr.text((w - 130, 14), f"{tape.t[i]:6.1f} s", font=font_s, fill=(210, 210, 210, 255))
        if inset:
            dr.text((w - w // 4 - 18, h - h // 4 - 44), "head camera", font=font_s, fill=(235, 235, 235, 255))
        proc.stdin.write(np.asarray(im).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render_tape(scene: str, tape_path: str, out: str, shots_mod: tuple[str, str], title: str, size=(1280, 720),
                workers: int = 4, inset: str | None = None, frames=None) -> str:
    """Render the tape with ``shots_mod`` = (module, function(t, tape, i) -> Shot) into ``out`` (MP4)."""
    tape = Tape.load(tape_path)
    idx = list(range(len(tape.t))) if frames is None else list(frames)
    chunks = [c for c in np.array_split(idx, workers) if len(c)]
    tmp = tempfile.mkdtemp(prefix="wb_render_")
    jobs = [(scene, tape_path, shots_mod, list(map(int, c)), os.path.join(tmp, f"part{k:02d}.mp4"), size, title, inset)
            for k, c in enumerate(chunks)]
    with ProcessPoolExecutor(len(jobs)) as ex:
        parts = list(ex.map(_render_chunk, jobs))
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def stills(scene: str, qposes, path: str, shot: Shot, size=(480, 360)) -> None:
    """A row of stills of the given states (debugging)."""
    import mujoco
    from PIL import Image

    from . import make_model
    model = make_model(scene)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = shot.lookat, shot.distance, shot.azimuth, shot.elevation
    imgs = []
    for q in qposes:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        r.update_scene(data, cam)
        imgs.append(r.render().copy())
    cols = min(4, len(imgs))
    rows = [np.concatenate(imgs[i:i + cols] + [np.zeros_like(imgs[0])] * (cols - len(imgs[i:i + cols])), 1)
            for i in range(0, len(imgs), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)
