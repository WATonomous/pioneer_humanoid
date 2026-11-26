"""HTTP server for the rollout dashboard: static page, JSON API, and video.

Routes:
  /                               the page (static/index.html)
  /static/<file>                  page assets
  /api/datasets                   dataset descriptions (cameras, joints, tasks, value sources)
  /api/episodes?ds=I              episode table of dataset I (column-oriented JSON)
  /api/episode?ds=I&ep=N          per-frame data of one episode (state, action, task, values, labels, progress)
  /video/I/<videos/...mp4>        the shared mp4, served with HTTP Range so the browser seeks into it
  /clip/I/<ep>/<camera>.mp4       one episode cut and re-encoded to H.264 by ffmpeg (for browsers without AV1)
"""
import json
import mimetypes
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from dataset import Dataset

STATIC_DIR = Path(__file__).parent / "static"
COPY_CHUNK = 1 << 20  # bytes per write when streaming video
EPISODE_CACHE = 64  # episodes kept in memory (a 90 s episode is ~0.5 MB of JSON)
CLIP_CRF = 23  # x264 quality for fallback clips; visually clean at 640x480


class Dashboard:
    """State shared by all request threads: datasets, an episode cache and the clip cache dir."""

    def __init__(self, datasets: list[Dataset], clip_dir: Path | None):
        self.datasets = datasets
        self.temp_clips = clip_dir is None  # a temp cache is deleted on shutdown; an explicit --clip-dir is kept
        self.clip_dir = Path(clip_dir or tempfile.mkdtemp(prefix="rollout_dashboard_clips_"))
        self.clip_locks: dict[str, threading.Lock] = {}
        self.lock = threading.Lock()
        self.episode_json = lru_cache(maxsize=EPISODE_CACHE)(self._episode_json)

    def close(self):
        if self.temp_clips:
            shutil.rmtree(self.clip_dir, ignore_errors=True)

    def _episode_json(self, ds: int, ep: int) -> bytes:
        return json.dumps(self.datasets[ds].episode_payload(ep)).encode()

    def clip(self, ds: int, ep: int, camera: str) -> Path:
        """Cut [from, to) of the episode's segment out of the shared mp4 and re-encode it (cached on disk)."""
        out = self.clip_dir / f"{ds}_{ep}_{camera}.mp4"
        with self.lock:
            lock = self.clip_locks.setdefault(out.name, threading.Lock())
        with lock:
            if not out.exists():
                dataset = self.datasets[ds]
                seg = dataset.video_segments(dataset.row(ep))[camera]
                tmp = out.with_suffix(".part.mp4")
                cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{seg['from']:.3f}", "-i", str(dataset.root / seg["path"]),
                       "-t", f"{seg['to'] - seg['from']:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
                       "-crf", str(CLIP_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
                subprocess.run(cmd, check=True)
                tmp.rename(out)
        return out


class Handler(BaseHTTPRequestHandler):
    app: Dashboard  # set by make_server

    def log_message(self, fmt, *args):  # keep the terminal quiet: video range requests are very chatty
        pass

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        parts = [unquote(p) for p in url.path.strip("/").split("/")]
        try:
            self.route(parts, query)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser cancels range requests whenever the user seeks
        except (KeyError, IndexError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
            self.send_bytes(json.dumps({"error": str(e)}).encode(), "application/json", HTTPStatus.NOT_FOUND)

    def route(self, parts: list[str], query: dict):
        head = parts[0] if parts else ""
        if head in ("", "index.html"):
            self.send_file(STATIC_DIR / "index.html")
        elif head == "static":
            self.send_file(self.safe_join(STATIC_DIR, "/".join(parts[1:])))
        elif head == "api":
            self.route_api(parts[1], query)
        elif head == "video":
            dataset = self.app.datasets[int(parts[1])]
            self.send_file(self.safe_join(dataset.root / "videos", "/".join(parts[3:])), ranged=True)
        elif head == "clip":
            self.send_file(self.app.clip(int(parts[1]), int(parts[2]), parts[3].removesuffix(".mp4")), ranged=True)
        else:
            raise FileNotFoundError(self.path)

    def route_api(self, name: str, query: dict):
        if name == "datasets":
            body = json.dumps([d.describe() for d in self.app.datasets]).encode()
        elif name == "episodes":
            body = json.dumps(self.app.datasets[int(query["ds"])].episode_table()).encode()
        elif name == "episode":
            body = self.app.episode_json(int(query["ds"]), int(query["ep"]))
        else:
            raise FileNotFoundError(name)
        self.send_bytes(body, "application/json")

    @staticmethod
    def safe_join(base: Path, rel: str) -> Path:
        path = (base / rel).resolve()
        if not path.is_relative_to(base.resolve()):
            raise FileNotFoundError(rel)
        return path

    def send_bytes(self, body: bytes, content_type: str, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, ranged: bool = False):
        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range", "")
        if ranged and header.startswith("bytes="):
            first, _, last = header[len("bytes="):].split(",")[0].partition("-")
            start = int(first) if first else max(size - int(last), 0)
            end = min(int(last), size - 1) if first and last else end
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache" if path.suffix in (".html", ".js", ".css") else "max-age=3600")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_server(app: Dashboard, host: str, port: int) -> ThreadingHTTPServer:
    if shutil.which("ffmpeg") is None:
        print("note: ffmpeg not found; browsers without AV1 support will not be able to play AV1 videos")
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
