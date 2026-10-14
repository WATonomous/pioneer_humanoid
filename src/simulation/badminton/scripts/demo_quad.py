"""Four-camera demo: one receive at a time, seen from four angles at once,
tiled 2x2 and streamed as MJPEG to a plain web page (and/or recorded to a
video). The page keeps the picture's aspect ratio at any window size.

  MUJOCO_GL=egl uv run scripts/demo_quad.py --checkpoint-file <model.pt> \\
      [--task Mjlab-Badminton-Receive-Student-PPO] [--port 8080] \\
      [--speed 0.5] [--record runs/demo.mp4 --seconds 40] \\
      [--preview runs/preview.png]
  # live: open http://localhost:8080 through an SSH tunnel;
  # scripts/slurm_demo.sh submits the job and prints the tunnel line.

How it runs: a batch of rallies (--batch, default 128) is simulated first
with the policy in the loop, each a fresh random serve from the launcher
bank, and their joint states are recorded (each env's next natural episode,
never the one right after a global reset - see simulate_batch). Playback then draws those
recorded states at real time (or --speed times it) with full-quality
rendering. Stepping a single Warp environment costs ~60 ms per 20 ms
control tick, so simulating live would play at 1/3 speed; recording the
batch (all rallies in parallel, ~10 s) and replaying gives smooth real
time. When the batch is used up a new one is simulated.

Cameras: broadcast side view (whole flight, serve to landing), behind the
robot (the approach), the opponent's view (the return coming over), and a
high view (where it lands vs the serve origin, marked on the floor). A
status bar carries rally count, time, hit/miss, the arm the policy picked
and the predicted landing error.
"""

import argparse
import io
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import humanoid_badminton  # noqa: F401  (registers the tasks)
from humanoid_badminton import mdp  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.viewer.offscreen_renderer import OffscreenRenderer  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402

GOOD, BAD, WARN = (12, 163, 12), (208, 59, 59), (230, 160, 30)
INK, PANEL, GAP = (252, 252, 251), (11, 11, 11), (40, 40, 40)
WORLD = ViewerConfig.OriginType.WORLD

# MuJoCo free camera: azimuth 90 = looking along +y (from behind the robot
# toward the net), 270 = from the far court back at the robot, 180 = from the
# -x sideline. The arm stands at y = -2, the net at y = 0, serves come from
# y = 3..6. The stand is centred on x = 0 with an arm on each side.
CAMERAS = {
    "broadcast":    dict(azimuth=180.0, elevation=-12.0, distance=7.4,
                         lookat=(0.0, 0.9, 0.55)),
    "behind robot": dict(azimuth=90.0, elevation=-13.0, distance=4.6,
                         lookat=(0.0, -0.9, 1.45)),
    "opponent":     dict(azimuth=270.0, elevation=-12.0, distance=5.2,
                         lookat=(0.0, -1.6, 0.95)),
    "high":         dict(azimuth=90.0, elevation=-58.0, distance=9.0,
                         lookat=(0.0, 0.9, 0.0)),
}


# -- web page ---------------------------------------------------------------

class FrameFeed:
    """Latest JPEG frame, handed to every connected browser as MJPEG."""

    def __init__(self) -> None:
        self._jpeg = b""
        self._seq = 0
        self._cv = threading.Condition()

    def publish(self, jpeg: bytes) -> None:
        with self._cv:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cv.notify_all()

    def wait_next(self, seen: int, timeout: float = 1.0):
        with self._cv:
            self._cv.wait_for(lambda: self._seq != seen, timeout=timeout)
            return self._jpeg, self._seq


PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<title>badminton receive demo</title><style>
html,body{margin:0;height:100%;background:#0b0b0b;overflow:hidden}
img{width:100vw;height:100vh;object-fit:contain;display:block}
</style></head><body><img src="/stream"></body></html>"""


def serve(feed: FrameFeed, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        jpeg, seen = feed.wait_next(seen)
                        if not jpeg:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                         + jpeg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# -- simulation -------------------------------------------------------------

@dataclass
class Rally:
    qpos: torch.Tensor       # (T, nq) on the sim device
    qvel: torch.Tensor       # (T, nv)
    mocap_pos: torch.Tensor  # (1, nmocap, 3)
    mocap_quat: torch.Tensor # (1, nmocap, 4)
    origin: tuple            # serve origin (x, y): the aim point
    hit_tick: int | None     # first face contact, or None (miss)
    landing: tuple | None    # predicted landing (x, y) of the return
    landing_ok: bool         # clears the net into the far court
    err: float               # landing distance to the origin (m)
    arm: str | None          # arm the policy locked (right / left)


class RecordedData:
    """Stand-in for sim.data that the offscreen renderer can read. Mocap
    bodies are static in this scene, so one snapshot per rally suffices."""
    nworld = 1

    def __init__(self, qpos, qvel, mocap_pos, mocap_quat) -> None:
        self.qpos, self.qvel = qpos, qvel
        self.mocap_pos, self.mocap_quat = mocap_pos, mocap_quat


def load_policy(task: str, checkpoint: str, device: str, num_envs: int):
    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    from rsl_rl.runners import DistillationRunner
    distilled = isinstance(runner, DistillationRunner)
    runner.load(checkpoint,
                load_cfg={"student": True} if distilled else {"actor": True},
                strict=True, map_location=device)
    return env, runner.get_inference_policy(device=device)


@torch.no_grad()
def simulate_batch(env, policy) -> list[Rally]:
    """Record one full rally per env.

    Recording starts at each env's next *natural* reset rather than after a
    global env.reset(): the first episode after a synchronous global reset
    hits measurably less (92-97% vs 99% in continuous stepping, see
    TRAINING_LOG), so the demo records the same conditions the bank eval
    measures. Envs whose recording has finished keep stepping (their
    later episodes are ignored) until the slowest env is done."""
    uenv = env.unwrapped
    store = uenv._badminton
    n = uenv.num_envs
    obs = env.get_observations()
    dev = uenv.device
    started = torch.zeros(n, dtype=torch.bool, device=dev)
    start_tick = torch.full((n,), -1, dtype=torch.long, device=dev)
    done_tick = torch.full((n,), -1, dtype=torch.long, device=dev)
    hit_tick = torch.full((n,), -1, dtype=torch.long, device=dev)
    origin = torch.zeros(n, 2, device=dev)
    landing = torch.zeros(n, 2, device=dev)
    landing_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    err = torch.zeros(n, device=dev)
    arm = torch.full((n,), -1, dtype=torch.long, device=dev)
    qpos_hist, qvel_hist = [], []
    t = 0
    while bool((done_tick < 0).any()):
        prev_hit = store["hit"].clone()
        qpos_hist.append(uenv.sim.data.qpos.clone())
        qvel_hist.append(uenv.sim.data.qvel.clone())
        obs, _, dones, _ = env.step(policy(obs))
        d = dones.bool()
        live = started & (done_tick < 0)
        # the lock is cleared by the reset inside a done step; read it from
        # the live envs before the next step can reset them
        arm = torch.where(live & store["committed"], store["assign"], arm)
        first = store["first"] & ~prev_hit & live
        if bool(first.any()):
            pos, vel = mdp._shuttle_state(uenv)
            xy, ok = mdp.predict_landing(pos[first], vel[first])
            landing[first] = xy
            landing_ok[first] = ok
            err[first] = (xy - origin[first]).norm(dim=-1)
            hit_tick[first] = t
        done_tick[d & live] = t
        # an env's recording begins on the tick after its reset: the state
        # captured at the top of the next loop iteration is the new episode
        begin = d & ~started
        started |= begin
        start_tick[begin] = t + 1
        origin[begin] = store["p0_xy"][begin]
        arm[begin] = -1
        t += 1
    qpos_hist.append(uenv.sim.data.qpos.clone())
    qvel_hist.append(uenv.sim.data.qvel.clone())
    Q = torch.stack(qpos_hist)          # (T+1, n, nq)
    V = torch.stack(qvel_hist)
    mp, mq = uenv.sim.data.mocap_pos.clone(), uenv.sim.data.mocap_quat.clone()
    out = []
    for i in range(n):
        # the state captured after the done step is already the next
        # episode (reset happens inside that step), so stop one tick short
        a, b = int(start_tick[i]), int(done_tick[i])
        h = int(hit_tick[i])
        out.append(Rally(
            qpos=Q[a:b + 1, i].contiguous(), qvel=V[a:b + 1, i].contiguous(),
            mocap_pos=mp[i:i + 1], mocap_quat=mq[i:i + 1],
            origin=tuple(origin[i].tolist()),
            hit_tick=(h - a) if h >= 0 else None,
            landing=tuple(landing[i].tolist()) if h >= 0 else None,
            landing_ok=bool(landing_ok[i]), err=float(err[i]),
            arm={0: "right", 1: "left"}.get(int(arm[i]))))
    return out


# -- rendering --------------------------------------------------------------

def make_renderers(uenv, width: int, height: int, ss: int) -> list[OffscreenRenderer]:
    # rendered at ss x the panel size and downscaled (Lanczos) for
    # antialiased edges and lines
    base = replace(uenv.cfg.viewer, width=width * ss, height=height * ss,
                   origin_type=WORLD, entity_name=None, body_name=None,
                   max_extra_envs=0, enable_shadows=True,
                   enable_reflections=True)
    out = []
    for cam in CAMERAS.values():
        r = OffscreenRenderer(model=uenv.sim.mj_model, cfg=replace(base, **cam),
                              scene=uenv.scene, sim_model=uenv.sim.model,
                              expanded_fields=uenv.sim.expanded_fields)
        r.initialize()
        out.append(r)
    return out


def label(frame: np.ndarray, text: str, font) -> np.ndarray:
    """Camera name, bottom-left (the status bar owns the top-left)."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    h = frame.shape[0]
    # textbbox is measured at the draw position (glyph ascent offsets the
    # box from the anchor), so anchor first, then size the box from it
    y = h - font.size - 12
    box = draw.textbbox((7, y), text, font=font)
    draw.rectangle((0, box[1] - 5, box[2] + 8, h), fill=PANEL + (200,))
    draw.text((7, y), text, fill=INK, font=font)
    return np.asarray(img)


def wall(frames: list[np.ndarray], status: str, colour, font) -> np.ndarray:
    """2x2 tile under a status strip: output is (bar + 2h, 2w)."""
    h, w = frames[0].shape[:2]
    bar = font.size + 18
    out = np.zeros((bar + 2 * h, 2 * w, 3), dtype=np.uint8)
    out[:bar] = PANEL
    for i, f in enumerate(frames):
        r, c = divmod(i, 2)
        y0 = bar + r * h
        out[y0:y0 + h, c * w:(c + 1) * w] = f
    out[bar + h - 1:bar + h + 1, :] = GAP
    out[bar:, w - 1:w + 1] = GAP
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    draw.text((14, (bar - font.size) // 2 - 2), status, fill=colour, font=font)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Badminton-Receive-Student-PPO")
    ap.add_argument("--checkpoint-file", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--batch", type=int, default=128,
                    help="rallies simulated per batch (all in parallel)")
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--panel-height", type=int, default=360)
    ap.add_argument("--supersample", type=int, default=1,
                    help="render panels at this multiple and downscale "
                         "(2 for a recording; costs ~4x render time)")
    ap.add_argument("--crf", type=int, default=18,
                    help="x264 quality for --record (lower = better; 16 is "
                         "visually lossless)")
    ap.add_argument("--fps", type=int, default=25,
                    help="frame rate of the page / the recording")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed (0.5 = slow motion)")
    ap.add_argument("--hold", type=float, default=0.8,
                    help="seconds the final frame of a rally stays up")
    ap.add_argument("--record", default=None, help="video path (mp4)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this much playback time (default: run "
                         "until killed; with --record defaults to 40)")
    ap.add_argument("--no-web", "--no-viser", dest="no_web", action="store_true",
                    help="record only, do not start the web page")
    ap.add_argument("--debug", action="store_true",
                    help="print render/encode timings to stderr")
    ap.add_argument("--preview", default=None,
                    help="write one frame per camera at two moments of a "
                         "rally to this PNG and exit (camera tuning)")
    args = ap.parse_args()
    if args.record and args.seconds is None:
        args.seconds = 40.0
    if args.preview:
        args.no_web, args.seconds, args.batch = True, 3.0, 4

    env, policy = load_policy(args.task, args.checkpoint_file, args.device,
                              args.batch)
    uenv = env.unwrapped
    renderers = make_renderers(uenv, args.panel_width, args.panel_height,
                               args.supersample)
    panel = (args.panel_width, args.panel_height)
    font = ImageFont.load_default(size=max(14, args.panel_height // 22))
    step_dt = uenv.step_dt

    feed, httpd = None, None
    if not args.no_web:
        feed = FrameFeed()
        httpd = serve(feed, args.port)
        print(f"[demo] web page on port {args.port}", flush=True)
    writer = None
    if args.record:
        import imageio
        writer = imageio.get_writer(
            args.record, fps=args.fps, codec="libx264", quality=None,
            pixelformat="yuv420p", macro_block_size=1,
            ffmpeg_params=["-crf", str(args.crf), "-preset", "slow"])

    marks = {"origin": None, "landing": None, "landing_ok": True}

    def draw_marks(vis) -> None:
        if marks["origin"] is not None:
            x, y = marks["origin"]
            vis.add_cylinder(np.array([x, y, 0.0]), np.array([x, y, 0.02]),
                             0.25, (0.2, 0.45, 1.0, 0.9))
        if marks["landing"] is not None:
            x, y = marks["landing"]
            c = (0.05, 0.75, 0.05, 0.9) if marks["landing_ok"] else (0.9, 0.6, 0.1, 0.9)
            vis.add_sphere(np.array([x, y, 0.08]), 0.08, c)

    def render_wall(data, status: str, colour) -> np.ndarray:
        frames = []
        for name, r in zip(CAMERAS, renderers):
            r.update(data, debug_vis_callback=draw_marks)
            f = r.render()
            if args.supersample > 1:
                f = np.asarray(Image.fromarray(f).resize(panel, Image.LANCZOS))
            frames.append(label(f, name, font))
        return wall(frames, status, colour, font)

    def emit(img: np.ndarray, t_play: float) -> None:
        nonlocal n_frames, last_render_wall
        if writer is not None:
            writer.append_data(img)
        if feed is not None:
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "JPEG", quality=args.jpeg_quality)
            feed.publish(buf.getvalue())
            # pace playback to wall clock
            lag = t_play / args.speed - (time.perf_counter() - t_wall0)
            if lag > 0:
                time.sleep(lag)
        n_frames += 1

    n_frames, last_render_wall = 0, 0.0
    last_frame_wall = time.perf_counter()
    frame_dt = 1.0 / args.fps           # playback seconds per frame
    t_play, rally_no, hits = 0.0, 0, 0
    prev_hit_flag = False
    previews = []
    t_wall0 = time.perf_counter()
    try:
        while args.seconds is None or t_play < args.seconds:
            if feed is not None:
                sd = uenv.sim.data
                emit(render_wall(RecordedData(sd.qpos[:1], sd.qvel[:1],
                                              sd.mocap_pos[:1], sd.mocap_quat[:1]),
                                 f"simulating {args.batch} new rallies...", INK),
                     t_play)
            t0 = time.perf_counter()
            rallies = simulate_batch(env, policy)
            random.shuffle(rallies)
            print(f"[demo] simulated {len(rallies)} rallies in "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
            t_wall0 = time.perf_counter() - t_play / args.speed
            last_frame_wall = time.perf_counter()
            for rl in rallies:
                if args.seconds is not None and t_play >= args.seconds:
                    break
                rally_no += 1
                if rally_no > 1 and prev_hit_flag:
                    hits += 1
                prev_hit_flag = rl.hit_tick is not None
                marks["origin"], marks["landing"] = rl.origin, None
                status, colour = "serving...", INK
                T = rl.qpos.shape[0]
                total = (T - 1) * step_dt + args.hold
                t_r = 0.0
                while t_r < total:
                    k = min(int(round(t_r / step_dt)), T - 1)
                    if rl.hit_tick is not None and k >= rl.hit_tick and marks["landing"] is None:
                        marks["landing"], marks["landing_ok"] = rl.landing, rl.landing_ok
                        who = f" ({rl.arm} arm)" if rl.arm else ""
                        if rl.landing_ok:
                            status = f"HIT{who}   return lands {rl.err:.1f} m from the serve origin"
                            colour = GOOD
                        else:
                            status, colour = f"HIT{who}   return short / into the net", WARN
                    if k == T - 1 and rl.hit_tick is None:
                        status, colour = "MISS" + (f" ({rl.arm} arm)" if rl.arm else ""), BAD
                    data = RecordedData(rl.qpos[k:k + 1], rl.qvel[k:k + 1],
                                        rl.mocap_pos, rl.mocap_quat)
                    tr0 = time.perf_counter()
                    done_rallies = rally_no - 1
                    tally = f"   |   {hits}/{done_rallies} hit" if done_rallies else ""
                    img = render_wall(data, f"rally {rally_no}   {min(t_r, (T - 1) * step_dt):4.1f} s   {status}{tally}",
                                      colour)
                    tr1 = time.perf_counter()
                    if args.preview and (abs(t_r - 1.0) < frame_dt / 2 or abs(t_r - 2.0) < frame_dt / 2):
                        previews.append(img)
                    emit(img, t_play)
                    if args.debug and n_frames % 50 == 1:
                        print(f"[demo] rally {rally_no} t={t_r:.2f} render "
                              f"{1e3 * (tr1 - tr0):.0f} ms  wall-lag "
                              f"{t_play / args.speed - (time.perf_counter() - t_wall0):+.2f} s",
                              file=sys.stderr, flush=True)
                    if feed is not None and writer is None:
                        # live: advance by the wall time the frame took
                        # (x speed) so playback stays real time at whatever
                        # frame rate the GPU delivers; --fps is the ceiling
                        now = time.perf_counter()
                        adv = max(frame_dt, (now - last_frame_wall) * args.speed)
                        last_frame_wall = now
                    else:
                        adv = frame_dt
                    t_r += adv
                    t_play += adv
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
            print(f"[demo] wrote {args.record}", flush=True)
        if args.preview and previews:
            Image.fromarray(np.concatenate(previews[:2], axis=0)).save(args.preview)
            print(f"[demo] wrote {args.preview}", flush=True)
        for r in renderers:
            r.close()
        if httpd is not None:
            httpd.shutdown()
        env.close()


if __name__ == "__main__":
    main()
