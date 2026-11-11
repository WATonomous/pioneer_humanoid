"""WebXR transport for the always-on Isaac controller scene."""
import asyncio
import json
from pathlib import Path
import ssl
import threading
import time

import numpy as np

from control import Input, quaternion_matrix


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.packet = None
        self.received = 0.
        self.sequence = 0
        self.interrupted = {"left": True, "right": True}
        self.reset_sequence = 0
        self.feedback = {"status": "Enter VR; tracked controllers move the arms"}
        self.images = {}
        self.stop = threading.Event()

    def clear(self):
        with self.lock:
            self.packet = None
            self.received = 0.
            self.sequence += 1
            self.interrupted = {"left": True, "right": True}


def parse_sample(value):
    """Validate only the ordering/reference metadata needed for safe reanchoring."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Invalid controller metadata")
    result = {}
    for key in ("sequence", "epoch"):
        number = value.get(key)
        if type(number) is not int or not 0 <= number <= 2**53-1:
            raise ValueError(f"Invalid controller {key}")
        result[key] = number
    stamp = value.get("time_ms")
    if type(stamp) not in (int, float) or not np.isfinite(stamp) or stamp < 0:
        raise ValueError("Invalid controller timestamp")
    result["time_ms"] = float(stamp)
    return result


def validate_packet(raw):
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Expected controller protocol version 1")
    head = raw.get("head")
    head_rotation = quaternion_matrix(head) if head is not None else None
    return {"head": head_rotation, "left": Input.parse(raw.get("left")),
            "right": Input.parse(raw.get("right")), "sample": parse_sample(raw.get("sample"))}


async def serve(shared, args):
    from aiohttp import web, WSMsgType
    owner = None
    # Static bytes avoid FileResponse's executor-backed filesystem operations
    # in the embedded Isaac path. All handler data is ordinary in-memory data.
    page = Path(__file__).with_name("index.html").read_bytes()

    async def index(request):
        return web.Response(body=page, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def camera(request):
        with shared.lock:
            image = shared.images.get(request.match_info["name"])
        if image is None:
            raise web.HTTPServiceUnavailable(text="Camera not ready")
        return web.Response(body=image, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def status(request):
        with shared.lock:
            feedback = dict(shared.feedback)
        return web.json_response(feedback, headers={"Cache-Control": "no-store"})

    async def socket(request):
        nonlocal owner
        if owner is not None:
            raise web.HTTPConflict(text="Another Quest already owns control")
        ws = web.WebSocketResponse(heartbeat=2., max_msg_size=8192)
        owner = ws
        reset_armed = False
        reset_down = False
        previous_epoch = None
        previous_sequence = None
        previous_time = None
        try:
            await ws.prepare(request)
            shared.clear()
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        raw = json.loads(message.data)
                        packet = validate_packet(raw)
                    except (ValueError, TypeError, KeyError):
                        shared.clear()
                        await ws.close(code=1008, message=b"Invalid controller packet")
                        break
                    with shared.lock:
                        now = time.monotonic()
                        sample = packet.get("sample")
                        if sample is not None:
                            if previous_sequence is not None and (sample["sequence"] <= previous_sequence
                                    or sample["time_ms"] < previous_time):
                                continue
                            previous_sequence = sample["sequence"]
                            previous_time = sample["time_ms"]
                        # A recenter/session change must reanchor even if the
                        # explicit null-pose notification was skipped by the mailbox.
                        epoch = sample["epoch"] if sample is not None else None
                        if epoch != previous_epoch:
                            shared.interrupted = {"left": True, "right": True}
                        previous_epoch = epoch
                        left = packet["left"]
                        down = left is not None and left.reset_scene
                        if left is None or packet["head"] is None:
                            reset_armed = False
                        if left is not None and not down:
                            reset_armed = packet["head"] is not None
                        if down and not reset_down and reset_armed:
                            shared.reset_sequence += 1
                        reset_down = down
                        for side in ("left", "right"):
                            # Latch losses even when the latest-value mailbox skips a frame.
                            if packet[side] is None or packet["head"] is None:
                                shared.interrupted[side] = True
                        shared.packet = packet
                        shared.received = now
                        shared.sequence += 1
                        feedback = dict(shared.feedback)
                    await ws.send_json(feedback)
        finally:
            shared.clear()
            owner = None
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", socket)
    app.router.add_get("/camera/{name}.jpg", camera)
    app.router.add_get("/status", status)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    context = None
    if args.cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
    try:
        await web.TCPSite(runner, args.host, args.port, ssl_context=context).start()
        print(f"[Quest controllers] Open {'https' if context else 'http'}://localhost:{args.port}", flush=True)
        while not shared.stop.is_set():
            await asyncio.sleep(.1)
    finally:
        if owner is not None:
            await owner.close()
        await runner.cleanup()


def load_pivots(path):
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    pivots = {}
    for side in ("left", "right"):
        if side in raw:
            offset = np.asarray(raw[side], dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .3:
                raise ValueError(f"Invalid {side} wrist calibration in {path}")
            pivots[side] = offset
    return pivots


