"""Watch a trained LeRobot policy (ACT, Diffusion Policy, ...) act in tidy_table live, in the browser. Sim only.

    cd src/robot_learning/humanoid_il/act
    MUJOCO_GL=egl python live_view.py <checkpoint>/pretrained_model --device cuda [--port 8080]

then open http://localhost:8080 (forward the port from the node). Runs the same loop as eval_sim.py, at real time,
and streams the policy's own cameras (top | wrist_left) as MJPEG, with the current step and each episode's result.
Episodes loop over seeds from --seed until Ctrl+C.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from eval_sim import load_policy, observation  # also puts the repo packages on sys.path
from tidy_sim import CONTROL_DT, RECORD_EVERY, TidySim

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>tidy_table live</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px monospace}img{display:block;max-width:100%;margin:auto}
#s{padding:8px 16px;white-space:pre-wrap}</style></head><body><img src="/stream"><div id="s"></div>
<script>setInterval(async()=>{try{const r=await fetch('/status');const j=await r.json();
document.getElementById('s').textContent=j.now+'\\n\\n'+j.results.join('\\n')}catch(e){}},500)</script></body></html>"""


class Shared:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.now, self.results = b"", "starting", []

    def publish(self, frame: np.ndarray, now: str) -> None:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=80)
        with self.cond:
            self.jpeg, self.now = buf.getvalue(), now
            self.cond.notify_all()


def serve(shared: Shared, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with shared.cond:
                            shared.cond.wait(timeout=1.0)
                            jpeg = shared.jpeg
                        if jpeg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif self.path == "/status":
                body = json.dumps({"now": shared.now, "results": shared.results}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[LIVE] open http://localhost:{port}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="…/checkpoints/<step>/pretrained_model")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max_seconds", type=float, default=90.0, help="sim time per episode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--speed", type=float, default=1.0, help="speed vs real time (0 = as fast as possible)")
    args = parser.parse_args()

    import torch

    policy, pre, post = load_policy(args.checkpoint, args.device)
    image_keys = [k for k in policy.config.input_features if k.startswith("observation.images.")]
    shapes = {k[len("observation.images."):]: tuple(policy.config.input_features[k].shape[1:]) for k in image_keys}
    sim = TidySim(shapes)
    from humanoid_mujoco_scenes.tidy_table import scene as S

    shared = Shared()
    serve(shared, args.port)
    frame_dt = CONTROL_DT * RECORD_EVERY / args.speed if args.speed > 0 else 0.0
    seed = args.seed
    try:
        while True:
            sim.reset(seed)
            policy.reset()
            action = sim.action()
            next_frame = time.monotonic()
            for step in range(int(args.max_seconds / CONTROL_DT)):
                if step % RECORD_EVERY == 0:
                    index, total, task = sim.step_info()
                    obs, images = observation(sim, image_keys, task)
                    with torch.inference_mode():
                        action = post(policy.select_action(pre(obs))).squeeze(0).cpu().numpy()
                    views = [images[k] for k in ("top", "wrist_left") if k in images] or list(images.values())
                    shared.publish(np.concatenate(views, axis=1),
                                   f"seed {seed}  t={step * CONTROL_DT:5.1f}s  step {index}/{total}  {task}")
                    next_frame += frame_dt
                    time.sleep(max(0.0, next_frame - time.monotonic()))
                sim.act(action)
                index, total, _ = sim.step_info()
                if index >= total and np.linalg.norm(sim.state()[:6] - sim.home) < 0.05 and step % RECORD_EVERY == 0:
                    break                                  # all binned and back home
            st = S.episode_status(sim.model, sim.data)
            line = (f"seed {seed}: steps {st['step']}/{st['total']} tidiness {st['tidiness']:.0%} "
                    f"success {st['success']} dropped {st['dropped']} toppled {st['toppled']}")
            shared.results.insert(0, line)
            print(f"[LIVE] {line}", flush=True)
            seed += 1
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
