"""Every pose of the controller mounted on the gripper, from the Quest teleop's headset page.

Reuses quest_controller_teleop's index.html (Meta Browser, immersive-vr, a blank dark scene: wear the headset on your
forehead and look under it) and its packet validation. The teleop keeps only the latest packet; a dataset needs every
one, so this server queues each sample with its headset time (``time_ms``) and host arrival time.

Buttons are read on the mounted controller only, so one hand runs a session:
  primary   (B on the right controller, Y on the left)   start a take / save it
  secondary (A / X)                                       discard the take
  stick     (thumbstick click)                            calibrate_tcp.py: hold while pivoting
"""
from __future__ import annotations

import asyncio
import json
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from umi_gripper import geometry
from umi_gripper._paths import PIONEER_HUMANOID, QUEST_TELEOP, add
from umi_gripper.take import PoseSample

add(QUEST_TELEOP, PIONEER_HUMANOID)
from transport import validate_packet  # noqa: E402  (quest_controller_teleop)

PAGE = QUEST_TELEOP / "index.html"
BUTTONS = ("primary", "secondary", "stick")
OWNER_TIMEOUT_S = 0.5


@dataclass
class ButtonEvent:
    name: str   # one of BUTTONS
    host_s: float


def mounted_buttons(controller, side: str) -> dict[str, bool]:
    """The mounted controller's button states from a validated Input (None: untracked, nothing pressed)."""
    if controller is None:
        return {name: False for name in BUTTONS}
    primary = controller.record if side == "right" else controller.reset_scene
    return {"primary": bool(primary), "secondary": bool(controller.precision), "stick": bool(controller.calibrate)}


class PacketHandler:
    """Packet -> PoseSample + button edges for one side. No I/O, so tests drive it directly."""

    def __init__(self, side: str):
        self.side = side
        self.samples: deque[PoseSample] = deque()
        self.events: deque[ButtonEvent] = deque()
        self.lock = threading.Lock()
        self._down = {name: False for name in BUTTONS}
        self._previous_sequence = None
        self.latest: PoseSample | None = None
        self.received = 0

    def handle(self, raw: dict, host_s: float) -> bool:
        """Queue the packet's sample; False if it is out of order (dropped). Raises ValueError if malformed."""
        packet = validate_packet(raw)
        sample = packet.get("sample")
        if sample is None:
            raise ValueError("packet has no sample timing (old index.html?)")
        if self._previous_sequence is not None and sample["sequence"] <= self._previous_sequence:
            return False
        self._previous_sequence = sample["sequence"]
        controller = packet[self.side]
        tracked = controller is not None and packet["head"] is not None
        pose = PoseSample(host_s=host_s, client_ms=sample["time_ms"], epoch=sample["epoch"], tracked=tracked)
        if tracked:
            pose.position = np.asarray(controller.position, dtype=float)
            pose.quat_xyzw = geometry.matrix_to_quat(controller.rotation)
            pose.trigger = float(controller.trigger)
        buttons = mounted_buttons(controller if tracked else None, self.side)
        with self.lock:
            self.samples.append(pose)
            self.latest = pose
            self.received += 1
            for name, down in buttons.items():
                if down and not self._down[name]:
                    self.events.append(ButtonEvent(name, host_s))
                self._down[name] = down
        return True

    def drain(self) -> list[PoseSample]:
        with self.lock:
            out = list(self.samples)
            self.samples.clear()
        return out

    def poll_events(self) -> list[ButtonEvent]:
        with self.lock:
            out = list(self.events)
            self.events.clear()
        return out

    def reset_stream(self) -> None:
        """A new page session restarts the headset's sequence numbers."""
        with self.lock:
            self._previous_sequence = None


class QuestTracker:
    """aiohttp server on its own thread: GET / serves the headset page, /ws takes its pose packets."""

    def __init__(self, side: str, host: str = "127.0.0.1", port: int = 8444, cert: str | None = None,
                 key: str | None = None):
        self.handler = PacketHandler(side)
        self.host, self.port, self.cert, self.key = host, port, cert, key
        self.status = {"status": "UMI gripper: waiting for the headset"}
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="quest-tracker", daemon=True)

    # The page shows feedback["status"]; collect.py keeps it current (REC 3.2 s, saved, ...).
    def set_status(self, text: str) -> None:
        self.status = {"status": text}

    def start(self) -> "QuestTracker":
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("Quest pose server did not start")
        if self._error is not None:
            raise RuntimeError(f"Quest pose server failed: {self._error}") from self._error
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def drain(self) -> list[PoseSample]:
        return self.handler.drain()

    def poll_events(self) -> list[ButtonEvent]:
        return self.handler.poll_events()

    @property
    def latest(self) -> PoseSample | None:
        return self.handler.latest

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # surfaced by start()
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        from aiohttp import WSMsgType, web
        page = PAGE.read_bytes()
        owner = {"ws": None, "seen": 0.0}

        async def index(request):
            return web.Response(body=page, content_type="text/html", headers={"Cache-Control": "no-store"})

        async def status(request):
            return web.json_response(self.status, headers={"Cache-Control": "no-store"})

        async def socket(request):
            # Same rule as the teleop: only Meta Browser streams; the tab streaming head poses owns the gripper,
            # and another tab takes over only after the owner has been silent OWNER_TIMEOUT_S.
            if "OculusBrowser" not in request.headers.get("User-Agent", ""):
                raise web.HTTPForbidden(text="Pose streaming is reserved for Meta Browser on the Quest")
            ws = web.WebSocketResponse(heartbeat=2.0, max_msg_size=8192)
            await ws.prepare(request)
            try:
                async for message in ws:
                    if message.type != WSMsgType.TEXT:
                        continue
                    now = time.monotonic()
                    try:
                        raw = json.loads(message.data)
                    except ValueError:
                        raw = None
                    if not isinstance(raw, dict):
                        await ws.close(code=1008, message=b"Invalid packet")
                        break
                    if owner["ws"] is not ws:
                        if raw.get("head") is None or (owner["ws"] is not None
                                                       and now - owner["seen"] < OWNER_TIMEOUT_S):
                            await ws.send_json({"status": "Another headset tab is streaming"})
                            continue
                        owner["ws"] = ws
                        self.handler.reset_stream()
                    owner["seen"] = now
                    try:
                        self.handler.handle(raw, now)
                    except (ValueError, TypeError, KeyError):
                        await ws.close(code=1008, message=b"Invalid controller packet")
                        break
                    await ws.send_json(self.status)
            finally:
                if owner["ws"] is ws:
                    owner["ws"] = None
            return ws

        app = web.Application()
        app.router.add_get("/", index)
        app.router.add_get("/ws", socket)
        app.router.add_get("/status", status)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        context = None
        if self.cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.cert, self.key)
        await web.TCPSite(runner, self.host, self.port, ssl_context=context).start()
        print(f"[UMI] Headset page: {'https' if context else 'http'}://localhost:{self.port}/ "
              f"(adb reverse tcp:{self.port} tcp:{self.port})", flush=True)
        self._ready.set()
        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        finally:
            await runner.cleanup()


def from_config(cfg: dict) -> QuestTracker:
    server = cfg["server"]
    return QuestTracker(cfg["controller"], server["host"], int(server["port"]), server.get("cert"), server.get("key"))


def _main() -> int:
    """Bring-up: print the mounted controller's pose and button presses."""
    import argparse
    from umi_gripper.config import load_config
    parser = argparse.ArgumentParser(description="Print the mounted controller's pose (no recording)")
    parser.add_argument("--controller", choices=("left", "right"), default=None)
    args = parser.parse_args()
    cfg = load_config(overrides={"controller": args.controller} if args.controller else None)
    tracker = from_config(cfg).start()
    tracker.set_status("UMI bring-up: streaming poses")
    try:
        last = time.monotonic()
        count = 0
        while True:
            time.sleep(0.5)
            samples = tracker.drain()
            count += len(samples)
            for event in tracker.poll_events():
                print(f"  button: {event.name}")
            now = time.monotonic()
            pose = tracker.latest
            rate = count / (now - last)
            last, count = now, 0
            if pose is None:
                print("waiting for the headset (enter VR in Meta Browser) ...")
            elif not pose.tracked:
                print(f"{rate:5.1f} Hz  {cfg['controller']} controller NOT tracked")
            else:
                print(f"{rate:5.1f} Hz  pos {np.array2string(pose.position, precision=3)}  trigger {pose.trigger:.2f}")
    except KeyboardInterrupt:
        return 0
    finally:
        tracker.stop()


if __name__ == "__main__":
    raise SystemExit(_main())
